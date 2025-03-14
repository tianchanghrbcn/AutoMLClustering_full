#!/usr/bin/env python
# coding: utf-8

import sys
import os
import math
import json
import time
import subprocess
import mysql.connector
import pandas as pd
import numpy as np  # 用于识别并转换 numpy 类型
import warnings

########################################
# ========== 重定向stderr到文件 =============
########################################

stderr_file = open("grpc_err.log", "w", encoding="utf-8")
sys.stderr = stderr_file

warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable",
    category=UserWarning,
)

########################################
# ========== 数据库配置 ================
########################################

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "5ZSL45ZS28uvI3^#zv#l",  # 修改成你的 MySQL 密码
    "database": "mydb"
}

TABLE_NAME = 'big_table'

########################################
# ========== 第1步: 导入 CSV =============
########################################

def create_table_and_import_csv(csv_file):
    """
    读取 csv_file 并导入 MySQL 表 'big_table'；
    第一列 'index' 做主键，其余列 TEXT；
    强制 dtype=str 防止 NaN 被转换为 float('nan')，
    空或 'NaN' 的列名改为 'unnamed_col'。
    """
    df = pd.read_csv(csv_file, dtype=str, keep_default_na=False, na_filter=False)
    new_cols = []
    for col in df.columns:
        c = str(col).strip()
        if not c or c.lower() == 'nan':
            c = 'unnamed_col'
        new_cols.append(c)
    df.columns = new_cols
    df = df.fillna('')

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    col_names = df.columns.tolist()
    col_defs = []
    for i, col in enumerate(col_names):
        if i == 0 and col.lower() == "index":
            col_defs.append("`index` INT PRIMARY KEY")
        else:
            col_defs.append(f"`{col}` TEXT")
    create_sql = f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} ({','.join(col_defs)})"
    cursor.execute(create_sql)
    cursor.execute(f"TRUNCATE TABLE {TABLE_NAME}")

    insert_cols = ",".join([f"`{c}`" for c in col_names])
    placeholders = ",".join(["%s"] * len(col_names))
    insert_sql = f"INSERT INTO {TABLE_NAME} ({insert_cols}) VALUES({placeholders})"
    records = df.values.tolist()
    cursor.executemany(insert_sql, records)
    conn.commit()
    cursor.close()
    conn.close()
    print(f"[Import CSV] 已插入 {len(df)} 行进入表 '{TABLE_NAME}'。")

########################################
# ========== 辅助函数 ==================
########################################

def get_table_rowcount():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
    row_count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return row_count

def fetch_chunk_df(offset, chunk_size):
    conn = mysql.connector.connect(**DB_CONFIG)
    query = f"SELECT * FROM {TABLE_NAME} ORDER BY `index` LIMIT {chunk_size} OFFSET {offset}"
    df_chunk = pd.read_sql(query, con=conn)
    conn.close()
    return df_chunk

def fetch_whole_df():
    conn = mysql.connector.connect(**DB_CONFIG)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} ORDER BY `index`", con=conn)
    conn.close()
    return df

def all_detections_to_csv(detections, out_csv="all_detections.csv"):
    """
    将检测结果字典 (global_row -> {col: error_desc}) 转成 CSV，
    列: global_row, column_name, error_desc
    """
    rows = []
    for global_row_str, col_errs in detections.items():
        for col_name, err_desc in col_errs.items():
            rows.append([global_row_str, col_name, err_desc])
    df_out = pd.DataFrame(rows, columns=["global_row", "column_name", "error_desc"])
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[Detection] 生成 {out_csv} (共 {len(rows)} 条错误记录)")

########################################
# ========== 分块检测（串行） ============
########################################

def detection_chunk(chunk_idx, offset, df_chunk):
    """
    对单个分块（最多500行）执行检测：
      1) 将 df_chunk 写入临时 CSV 文件 temp_chunk_{chunk_idx}.csv；
      2) 调用检测脚本 google_gemini_detection.py，并传入该 CSV 文件路径；
      3) 脚本生成 gemini_detections.json 后，重命名为 gemini_detections_{chunk_idx}.json；
      4) 解析该文件，将局部行号转换为全局行号，并返回结果字典。
    """
    temp_csv = f"temp_chunk_{chunk_idx}.csv"
    df_chunk.to_csv(temp_csv, index=False)

    python_exec = os.path.join(os.environ.get("VIRTUAL_ENV", ""), "Scripts", "python.exe")
    if not os.path.exists(python_exec):
        python_exec = "python"

    # 调用检测脚本，并将 temp_csv 作为参数传入
    cmd = [python_exec, "google_gemini_detection.py", temp_csv]
    print(f"[Detection] Chunk {chunk_idx} invoking script: {cmd}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(f"[Detection] Chunk {chunk_idx} stdout:\n{proc.stdout}")
    if proc.stderr:
        print(f"[Detection] Chunk {chunk_idx} stderr:\n{proc.stderr}")
    if proc.returncode != 0:
        print(f"[Detection] Chunk {chunk_idx} failed, returncode={proc.returncode}")
        return {}

    detect_json = f"gemini_detections_{chunk_idx}.json"
    if os.path.exists("gemini_detections.json"):
        os.rename("gemini_detections.json", detect_json)
    else:
        print(f"[Warning] Chunk {chunk_idx} did not generate gemini_detections.json")

    res = {}
    if os.path.exists(detect_json):
        with open(detect_json, "r", encoding="utf-8") as f:
            local_det = json.load(f)
        for local_row_str, col_errs in local_det.items():
            global_row = offset + int(local_row_str)
            res[str(global_row)] = col_errs
        os.remove(detect_json)
    return res

def run_detection_serial(chunk_size=500):
    """
    串行检测：将数据按500行分块，对每块调用检测函数，
    合并所有检测结果，并写入 gemini_detections.json 和 all_detections.csv。
    """
    row_count = get_table_rowcount()
    total_chunks = math.ceil(row_count / chunk_size)
    detections = {}

    for i in range(total_chunks):
        offset = i * chunk_size
        df_chunk = fetch_chunk_df(offset, chunk_size)
        print(f"[Detection] Processing chunk {i} (global rows {offset} to {offset+len(df_chunk)-1})")
        chunk_det = detection_chunk(i, offset, df_chunk)
        detections.update(chunk_det)

    with open("gemini_detections.json", "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=4, ensure_ascii=False)
    print("[Detection] Detection phase completed => gemini_detections.json")
    all_detections_to_csv(detections, "all_detections.csv")

########################################
# ========== 分块修复（串行） ============
########################################

def run_repair_serial_by_chunk(chunk_size=500):
    """
    串行修复：按500行分块，
      对每个分块，根据全局检测结果构造局部错误映射（local_error_map），
      将对应的 CSV 文件和局部错误映射传给修复脚本 google_gemini_correction.py，
      解析修复结果，更新 DataFrame，并更新数据库。
    """
    if not os.path.exists("gemini_detections.json"):
        print("[Repair] Missing gemini_detections.json; skipping repair.")
        return

    with open("gemini_detections.json", "r", encoding="utf-8") as f:
        detections = json.load(f)
    if not detections:
        print("[Repair] gemini_detections.json is empty; no errors to repair.")
        return

    row_count = get_table_rowcount()
    total_chunks = math.ceil(row_count / chunk_size)

    for i in range(total_chunks):
        offset = i * chunk_size
        df_chunk = fetch_chunk_df(offset, chunk_size)
        partial_det = {}
        # 构造局部映射：局部行号 -> 对应的错误（从全局检测结果中取出）
        for local_idx in range(len(df_chunk)):
            global_idx = offset + local_idx
            g_str = str(global_idx)
            if g_str in detections:
                partial_det[str(local_idx)] = detections[g_str]
        if not partial_det:
            print(f"[Repair] Chunk {i} has no errors; skipping.")
            continue
        temp_csv = f"temp_chunk_{i}.csv"
        df_chunk.to_csv(temp_csv, index=False)
        with open("gemini_detections.json", "w", encoding="utf-8") as f:
            json.dump(partial_det, f, indent=4, ensure_ascii=False)
        print(f"[Repair] Chunk {i}: invoking repair script with {temp_csv} and gemini_detections.json")
        python_exec = os.path.join(os.environ.get("VIRTUAL_ENV", ""), "Scripts", "python.exe")
        if not os.path.exists(python_exec):
            python_exec = "python"
        # 修复脚本需要两个参数：CSV 文件路径和错误 JSON 文件路径
        cmd = [python_exec, "google_gemini_correction.py", temp_csv, "gemini_detections.json"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.stdout:
            print(f"[Repair] Chunk {i} stdout:\n{proc.stdout}")
        if proc.stderr:
            print(f"[Repair] Chunk {i} stderr:\n{proc.stderr}")
        if proc.returncode != 0:
            print(f"[Repair] Chunk {i} repair script failed, returncode={proc.returncode}")
            continue
        corr_json = f"gemini_corrections_{i}.json"
        if os.path.exists("gemini_corrections.json"):
            os.rename("gemini_corrections.json", corr_json)
        else:
            print(f"[Warning] Chunk {i} did not generate gemini_corrections.json")
            continue
        # 解析修复结果并应用到 DataFrame
        with open(corr_json, "r", encoding="utf-8") as f:
            local_corr = json.load(f)
        for local_row_str, colvals in local_corr.items():
            loc = int(local_row_str)
            for col, val in colvals.items():
                if col in df_chunk.columns:
                    df_chunk.at[loc, col] = val
        os.remove(corr_json)
        # 更新数据库
        update_back_to_db(df_chunk)
        print(f"[Repair] Chunk {i} repaired and DB updated.")

def update_back_to_db(df):
    """
    Update the database with the corrected DataFrame.
    Converts numpy types to native Python types.
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    col_names = df.columns.tolist()
    for i in range(len(df)):
        row_data = df.iloc[i]
        real_idx = row_data["index"]
        if isinstance(real_idx, np.generic):
            real_idx = real_idx.item()
        else:
            real_idx = int(real_idx)
        set_exprs = []
        vals = []
        for c in col_names:
            if c.lower() == 'index':
                continue
            set_exprs.append(f"`{c}`=%s")
            cell_val = row_data[c]
            if isinstance(cell_val, np.generic):
                cell_val = cell_val.item()
            cell_val = str(cell_val) if cell_val is not None else None
            vals.append(cell_val)
        set_clause = ", ".join(set_exprs)
        sql = f"UPDATE {TABLE_NAME} SET {set_clause} WHERE `index`=%s"
        vals.append(real_idx)
        cursor.execute(sql, vals)
    conn.commit()
    cursor.close()
    conn.close()

########################################
# ========== 导出结果 ================
########################################

def export_db_to_csv(out_csv="final_corrected.csv"):
    conn = mysql.connector.connect(**DB_CONFIG)
    df = pd.read_sql(f"SELECT * FROM `{TABLE_NAME}` ORDER BY `index`", con=conn)
    conn.close()
    df.to_csv(out_csv, index=False)
    print(f"[Export] Exported table '{TABLE_NAME}' to {out_csv}")

########################################
# ========== 主函数 ==================
########################################

def main():
    # ========== 新增：命令行参数解析 ==========
    if len(sys.argv) < 2:
        print("Usage: python big_pipeline.py <input_csv> [<output_dir>]")
        sys.exit(1)

    input_csv = sys.argv[1]
    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]
    else:
        output_dir = "."

    start_time = time.time()

    # 1) 导入 CSV 到 MySQL
    create_table_and_import_csv(input_csv)

    # 2) 分块检测，生成 gemini_detections.json 和 all_detections.csv
    run_detection_serial(chunk_size=500)

    # 3) 分块修复，每块调用修复脚本，并更新数据库
    run_repair_serial_by_chunk(chunk_size=500)

    # 4) 导出最终结果为 CSV，若未指定 output_dir 则导出到当前目录
    final_csv_path = os.path.join(output_dir, "repaired_data.csv")
    export_db_to_csv(final_csv_path)

    cost = time.time() - start_time
    print(f"\n=== All processes completed in {cost:.2f} seconds ===")

if __name__ == "__main__":
    main()

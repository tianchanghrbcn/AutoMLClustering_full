#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import numpy as np

# ========== 全局配置 ==========
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets", "train")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "eigenvectors.json")
K_VALUE = 5  # 固定 K=5

def compute_missing_rate(df: pd.DataFrame) -> float:
    """
    计算缺失值占比。
    """
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    return missing_cells / total_cells if total_cells > 0 else 0.0


def compute_noise_rate(df: pd.DataFrame) -> float:
    """
    基于IQR检测数值型离群点，返回占比。
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return 0.0

    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).sum().sum()
    total_elements = numeric_df.size
    noise_rate = outliers / total_elements if total_elements > 0 else 0.0
    return noise_rate


def parse_error_rate_from_filename(file_name: str) -> float:
    """
    从文件名中提取错误率数值，例如 "10%.csv" -> 10.0, "15.5%.csv" -> 15.5。
    如果是 clean.csv，返回固定错误率 0.01%。
    """
    if file_name == "clean.csv":
        return 0.01
    base = file_name.replace(".csv", "")
    if "%" in base:
        rate_str = base.replace("%", "")
        try:
            return float(rate_str)
        except ValueError:
            pass
    return 0.0


def process_single_file(csv_path: str, dataset_name: str, dataset_id: str) -> dict:
    """
    读取 CSV 文件，计算特征向量并返回字典。
    """
    df = pd.read_csv(csv_path)

    # 基础信息
    num_samples = df.shape[0]
    num_features = df.shape[1]

    # 计算
    missing_rate = compute_missing_rate(df)
    noise_rate = compute_noise_rate(df)

    file_name = os.path.basename(csv_path)
    error_rate = parse_error_rate_from_filename(file_name)

    feature_vector = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "csv_file": file_name,
        "error_rate": error_rate,
        "K": K_VALUE,
        "missing_rate": missing_rate,
        "noise_rate": noise_rate,
        "m": num_features,
        "n": num_samples,
    }
    return feature_vector


def main():
    """
    每次运行时都重新扫描 data_dir 下的子文件夹，从 D1 开始编号，
    并覆盖写 eigenvectors.json。
    """
    existing_data = []

    if not os.path.isdir(DATA_DIR):
        print(f"错误: DATA_DIR {DATA_DIR} 不存在或不是文件夹。")
        return

    dataset_id_counter = 0

    for dataset_name in os.listdir(DATA_DIR):
        sub_folder = os.path.join(DATA_DIR, dataset_name)
        if not os.path.isdir(sub_folder):
            continue

        csv_files = [f for f in os.listdir(sub_folder) if f.endswith(".csv")]
        if not csv_files:
            print(f"警告: 数据集 {dataset_name} 文件夹下无 CSV 文件，跳过。")
            continue

        for csv_file in csv_files:
            csv_path = os.path.join(sub_folder, csv_file)

            # 将 dataset_id 设置为递增的整数值，无引号
            dataset_id = dataset_id_counter
            dataset_id_counter += 1

            feature_vector = process_single_file(csv_path, dataset_name, dataset_id)
            existing_data.append(feature_vector)

            print(f"[{dataset_id}] 完成处理: {dataset_name}/{csv_file} => error_rate={feature_vector['error_rate']}%")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

    print(f"\n所有数据处理完成，共 {len(existing_data)} 条记录已写入: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

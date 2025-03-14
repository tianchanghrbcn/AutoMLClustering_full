import argparse
import os
import pandas as pd
import numpy as np
import random
import string


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Inject multiple types of errors into a dataset, skipping the first column (primary key).")
    parser.add_argument("--input", required=True, help="Path to the clean CSV file.")
    parser.add_argument("--output", required=True, help="Directory to store corrupted CSVs.")
    parser.add_argument("--task_name", required=True,
                        help="Used for naming: output => {task_name}_{num}.csv, plus a {task_name}_explanation.txt.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional).")
    return parser.parse_args()


def main():
    args = parse_arguments()
    input_path = args.input
    output_dir = args.output
    task_name = args.task_name
    seed = args.seed

    # 固定随机种子(如需要可复现)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 1. 读取数据
    df_clean = pd.read_csv(input_path)
    df_clean = df_clean.reset_index(drop=True)
    print(f"Loaded dataset from {input_path}, shape={df_clean.shape}")

    # 2. 第一列当作主键, 不做注入
    if df_clean.shape[1] < 2:
        print("Warning: <2 columns in dataset, can't do injection. Exiting.")
        return

    primary_key_col = df_clean.columns[0]
    print(f"Primary key column: '{primary_key_col}' (no injection).")

    # 其余列才可注入
    other_cols = df_clean.columns[1:].tolist()

    # 区分数值列 vs. 非数值列
    numeric_cols = df_clean[other_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_clean[other_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"Numeric columns (except primary key): {numeric_cols}")
    print(f"Non-numeric columns (except primary key): {cat_cols}")

    # 3. 定义注入方案
    numeric_scenarios = ["only_anomaly", "only_missing", "half_half"]
    cat_scenarios = ["only_format", "only_knowledge", "half_half"]
    error_rates = [0.05, 0.10]  # 5%, 10%

    # 准备输出目录 & 说明文件
    os.makedirs(output_dir, exist_ok=True)
    explanation_path = os.path.join(output_dir, f"{task_name}_explanation.txt")
    explanation_lines = []
    combo_counter = 0

    # 4. 遍历3×3×2=18种组合
    for num_scenario in numeric_scenarios:
        for cat_scenario in cat_scenarios:
            for e_rate in error_rates:
                combo_counter += 1

                # 拷贝一份干净数据
                df_corrupted = df_clean.copy()

                # 数值列注入
                if numeric_cols:
                    inject_errors_numeric(df_corrupted, numeric_cols, e_rate, num_scenario)

                # 非数值列注入
                if cat_cols:
                    inject_errors_categorical(df_corrupted, cat_cols, e_rate, cat_scenario)

                # 输出文件: "{task_name}_{combo_counter}.csv"
                out_filename = f"{task_name}_{combo_counter}.csv"
                out_path = os.path.join(output_dir, out_filename)
                df_corrupted.to_csv(out_path, index=False)

                # 记录说明
                desc = (f"{combo_counter} => num_scenario={num_scenario}, "
                        f"cat_scenario={cat_scenario}, rate={int(e_rate * 100)}%")
                explanation_lines.append(desc)
                print(f"[{combo_counter:02d}] Generated: {out_filename} | {desc}")

    # 5. 写明细到 explanation.txt
    with open(explanation_path, "w", encoding="utf-8") as f:
        f.write(f"Explanation of error combos for task: {task_name}\n\n")

        # 为每个 combo 行补充更细致的比例说明
        for line in explanation_lines:
            # line example: "1 => num_scenario=only_anomaly, cat_scenario=only_format, rate=5%"
            # 可解析出 numeric scenario / cat scenario / rate
            # 再做相应的错误比例追加, 具体实现见下文
            combo_num, rest = line.split(" => ", 1)
            scenario_part, rate_part = rest.rsplit(", rate=", 1)
            # scenario_part: "num_scenario=only_anomaly, cat_scenario=only_format"
            # rate_part: "5%" (string)
            numeric_part, cat_part = scenario_part.split(", ")
            # numeric_part: "num_scenario=only_anomaly"
            # cat_part:    "cat_scenario=only_format"
            num_type = numeric_part.split("=")[1]
            cat_type = cat_part.split("=")[1]
            e_rate_str = rate_part.strip()

            # 解析 e_rate 纯数字
            e_value = e_rate_str.replace("%", "")  # 5 or 10
            e_float = float(e_value)

            # 计算四种错误(异常/缺失/格式/知识库)各占多少%
            # 假设 numeric/cat 各占一半
            # numeric_scenario: if only_anomaly => 100% anomaly, 0% missing
            # if half_half => 50% anomaly, 50% missing
            # cat_scenario: if only_format => 100% format, 0% knowledge ...
            # total = e_float
            # => numeric half => e_float/2, cat half => e_float/2
            # for half_half => e_float/2 * 50% => e_float/4
            total_error = e_float

            numeric_half = total_error / 2
            cat_half = total_error / 2

            # numeric side
            if num_type == "only_anomaly":
                anomaly_percent = numeric_half
                missing_percent = 0.0
            elif num_type == "only_missing":
                anomaly_percent = 0.0
                missing_percent = numeric_half
            else:  # half_half
                anomaly_percent = numeric_half / 2
                missing_percent = numeric_half / 2

            # cat side
            if cat_type == "only_format":
                format_percent = cat_half
                knowledge_percent = 0.0
            elif cat_type == "only_knowledge":
                format_percent = 0.0
                knowledge_percent = cat_half
            else:  # half_half
                format_percent = cat_half / 2
                knowledge_percent = cat_half / 2

            # 组装更详细信息
            detail_str = (f"  => anomaly={anomaly_percent:.2f}%, missing={missing_percent:.2f}%, "
                          f"format={format_percent:.2f}%, knowledge={knowledge_percent:.2f}%")

            f.write(line + "\n")
            f.write(detail_str + "\n\n")

    print(f"\nAll {combo_counter} corrupted CSVs saved to: {output_dir}")
    print(f"Explanation file: {explanation_path}")


# ---------------------- 数值列注入 ----------------------
def inject_errors_numeric(df, numeric_cols, error_rate, scenario):
    """
    对数值列注入:
      - only_anomaly: 全部异常值
      - only_missing: 全部缺失值
      - half_half   : 一半异常, 一半缺失
    """
    if not numeric_cols or error_rate <= 0:
        return

    n_total = df.shape[0] * len(numeric_cols)
    n_inject = int(n_total * error_rate)
    if n_inject <= 0:
        return

    # 分配
    if scenario == "only_anomaly":
        anomaly_count = n_inject
        missing_count = 0
    elif scenario == "only_missing":
        anomaly_count = 0
        missing_count = n_inject
    else:  # half_half
        anomaly_count = n_inject // 2
        missing_count = n_inject - anomaly_count

    # 注入异常值
    if anomaly_count > 0:
        anomaly_cells = pick_random_cells(df, numeric_cols, anomaly_count)
        for (r_i, col) in anomaly_cells:
            orig_val = df.at[r_i, col]
            df.at[r_i, col] = generate_anomaly_value(orig_val)

    # 注入缺失值
    if missing_count > 0:
        missing_cells = pick_random_cells(df, numeric_cols, missing_count)
        for (r_i, col) in missing_cells:
            df.at[r_i, col] = np.nan


def generate_anomaly_value(orig_val):
    """
    简易: 将原值放大3~6倍
    """
    if pd.isnull(orig_val):
        orig_val = 1.0
    factor = random.uniform(3, 6)
    return orig_val * factor


# ---------------------- 非数值列注入 --------------------
def inject_errors_categorical(df, cat_cols, error_rate, scenario):
    """
    对非数值列:
      - only_format      (全部格式错误)
      - only_knowledge   (全部跨列互换知识库错误)
      - half_half        (格式错误+知识库错误各半)
    """
    if not cat_cols or error_rate <= 0:
        return

    n_total = df.shape[0] * len(cat_cols)
    n_inject = int(n_total * error_rate)
    if n_inject <= 0:
        return

    if scenario == "only_format":
        fmt_count = n_inject
        kn_count = 0
    elif scenario == "only_knowledge":
        fmt_count = 0
        kn_count = n_inject
    else:  # half_half
        fmt_count = n_inject // 2
        kn_count = n_inject - fmt_count

    # 格式错误
    if fmt_count > 0:
        fmt_cells = pick_random_cells(df, cat_cols, fmt_count)
        for (r_i, col) in fmt_cells:
            orig_val = df.at[r_i, col]
            new_val = create_format_error(orig_val)
            df.at[r_i, col] = new_val

    # 知识库错误(跨列互换)
    if kn_count > 0:
        kn_cells = pick_random_cells(df, cat_cols, kn_count)
        for (r_i, col) in kn_cells:
            # 跨列获取值
            new_val = create_knowledge_error_crosscol(df, col, cat_cols)
            df.at[r_i, col] = new_val


# ---------------------- pick_random_cells ----------------
def pick_random_cells(df, columns, count):
    """
    从df的指定列集合中随机挑选 count个(row,col)
    """
    candidates = []
    rows = df.index.tolist()
    for c in columns:
        for r_i in rows:
            candidates.append((r_i, c))
    if count > len(candidates):
        count = len(candidates)
    return random.sample(candidates, count)


# ---------------------- 格式错误(多样化破坏) -------------
def create_format_error(orig_val):
    """
    更真实: 随机选择一种操作(插入符号 / 去空格 / 裁剪 / 反转 /拼写错误)
    """
    if pd.isnull(orig_val):
        val_str = ""
    else:
        val_str = str(orig_val)

    transformations = [
        insert_random_special_chars,
        remove_all_spaces,
        substring_cut,
        reverse_random_segment,
        random_typo_injection
    ]
    func = random.choice(transformations)
    return func(val_str)


def insert_random_special_chars(val_str):
    specs = ["#", "$", "%", "??", "%%", "^^", "###", "##??", "~!"]
    times = random.randint(1, 3)
    s = val_str
    for _ in range(times):
        ch = random.choice(specs)
        pos = random.randint(0, len(s))
        s = s[:pos] + ch + s[pos:]
    return s


def remove_all_spaces(val_str):
    return val_str.replace(" ", "")


def substring_cut(val_str):
    ln = len(val_str)
    if ln <= 1:
        return val_str
    start_i = random.randint(0, ln - 1)
    end_i = random.randint(start_i + 1, ln)
    return val_str[start_i:end_i]


def reverse_random_segment(val_str):
    ln = len(val_str)
    if ln <= 2:
        return val_str
    start_i = random.randint(0, ln - 2)
    end_i = random.randint(start_i + 1, ln)
    seg = val_str[start_i:end_i]
    rev = seg[::-1]
    return val_str[:start_i] + rev + val_str[end_i:]


def random_typo_injection(val_str):
    arr = list(val_str)
    if not arr:
        return val_str
    times = random.randint(1, min(3, len(val_str)))
    indices = random.sample(range(len(val_str)), times)
    letters = string.ascii_letters
    for idx in indices:
        arr[idx] = random.choice(letters)
    return "".join(arr)


# ---------------------- 知识库错误(跨列互换) -------------
def create_knowledge_error_crosscol(df, target_col, cat_cols):
    """
    在非数值列 cat_cols 中,随机选一个 source_col != target_col,
    再从source_col取 distinct 值(去掉NaN),随机挑一个注入 target_col 。
    """
    possible_sources = [c for c in cat_cols if c != target_col]
    if not possible_sources:
        # 如果只有1个非数值列,则没法跨列
        return "INVALID"
    source_col = random.choice(possible_sources)
    distinct_vals = df[source_col].dropna().unique().tolist()
    if not distinct_vals:
        return "INVALID"
    new_val = random.choice(distinct_vals)
    return new_val


if __name__ == "__main__":
    main()

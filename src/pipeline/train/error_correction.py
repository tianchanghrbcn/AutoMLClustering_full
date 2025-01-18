# src/pipeline/train/error_correction.py

import os
import time
import pandas as pd

# 使用绝对导入(前提: 顶层包叫 "src", 并在 sys.path 或 PYTHONPATH 中)
from src.cleaning.mode.correction_with_mode import run_mode_cleaning

def run_error_correction(
        dataset_name: str,
        csv_path: str,
        cleaning_algo: str,
        work_dir: str,
        clean_csv_path: str = None
) -> (str, str, float):
    """
    统一的清洗函数入口:
      - dataset_name: 数据集名称
      - csv_path:     待清洗的(脏)CSV文件
      - cleaning_algo: "mode" / "raha_baran" / ...
      - work_dir:     工作目录
      - clean_csv_path: 如果算法需要对照干净数据，则传入其路径

    返回: (cleaned_csv_path, cleaning_algo_label, cleaning_time)
    """

    if cleaning_algo == "mode":
        # mode算法需要 dirty_path + clean_path
        if not clean_csv_path:
            raise ValueError("Mode cleaning requires 'clean_csv_path'.")

        # 执行清洗并记录时间
        start_time = time.time()
        repaired_df, algo_time = run_mode_cleaning(
            dirty_path=csv_path,
            clean_path=clean_csv_path,
            task_name=dataset_name
        )
        total_time = time.time() - start_time

        # 构造清洗后的文件路径
        dataset_relative_path = os.path.relpath(csv_path, start=work_dir)
        dataset_folder, csv_file = os.path.split(dataset_relative_path)
        error_rate = os.path.splitext(csv_file)[0]

        cleaned_csv_path = os.path.join(
            work_dir, "results", "cleaned_data", dataset_folder, f"repaired_{dataset_name}_{error_rate}.csv"
        )

        print(f"[Cleaning - Mode] Dataset={dataset_name}, TimeUsed={total_time:.2f}s => {cleaned_csv_path}")
        return cleaned_csv_path, cleaning_algo, total_time

    else:
        raise ValueError(f"Unknown cleaning algorithm: {cleaning_algo}")

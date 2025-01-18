# src/pipeline/train/train_pipeline.py

import os
import json
from src.pipeline.train.error_correction import run_error_correction

def main():
    """
    清洗阶段：对指定数据集运行所有清洗策略。
    """
    # ========== 配置部分 ==========
    work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    eigenvectors_path = os.path.join(work_dir, "results", "eigenvectors.json")

    if not os.path.exists(eigenvectors_path):
        print(f"未找到 {eigenvectors_path}, 请先运行 pre-processing.py.")
        return

    # 从 eigenvectors.json 读取所有记录
    with open(eigenvectors_path, "r", encoding="utf-8") as f:
        all_records = json.load(f)

    if not all_records:
        print("eigenvectors.json 文件为空或没有记录.")
        return

    # 遍历所有数据集
    for record_idx, record in enumerate(all_records):
        # 仅测试第一个数据集的第一个 CSV 文件
        if record_idx > 0:  # 调整此处条件以启用完整测试
            break

        dataset_name = record["dataset_name"]
        csv_file = record["csv_file"]  # 例如 "10%.csv"
        error_rate = record["error_rate"]

        print(f"[INFO] 准备处理数据集: {dataset_name} (CSV: {csv_file}, error_rate={error_rate}%)")

        # ========== 确定文件路径 ==========
        dataset_folder = os.path.join(work_dir, "dataset", "train", dataset_name)
        if not os.path.exists(dataset_folder):
            print(f"数据集目录 {dataset_folder} 不存在.")
            continue

        csv_path = os.path.join(dataset_folder, csv_file)
        clean_csv_path = os.path.join(dataset_folder, "clean.csv")

        if not os.path.exists(csv_path):
            print(f"脏数据文件 {csv_path} 不存在.")
            continue

        if not os.path.exists(clean_csv_path):
            print(f"干净数据文件 {clean_csv_path} 不存在.")
            continue

        # ========== 运行清洗策略 ==========
        strategies = ["mode", "raha_baran"]  # 可扩展
        for algo in strategies:
            print(f"[INFO] 正在运行清洗策略: {algo}")
            cleaned_csv_path, used_algo, cleaning_time = run_error_correction(
                dataset_name=dataset_name,
                csv_path=csv_path,
                cleaning_algo=algo,
                work_dir=work_dir,
                clean_csv_path=clean_csv_path
            )
            print("=" * 50)
            print(f"清洗完成: Dataset={dataset_name}, Algo={used_algo}")
            print(f"输出文件: {cleaned_csv_path}")
            print(f"耗时: {cleaning_time:.2f} 秒")
            print("=" * 50)

if __name__ == "__main__":
    main()

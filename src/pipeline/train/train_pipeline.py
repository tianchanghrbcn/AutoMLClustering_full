import os
import json

from src.pipeline.train.cluster_methods import ClusterMethod
from src.pipeline.train.error_correction import run_error_correction
from src.pipeline.train.cluster_methods import run_clustering
from src.pipeline.train.clustered_analysis import generate_training_data

def main():

    # ========== 配置部分 ==========
    work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    eigenvectors_path = os.path.join(work_dir, "results", "eigenvectors.json")
    cleaned_results_path = os.path.join(work_dir, "results", "cleaned_results.json")
    clustered_results_path = os.path.join(work_dir, "results", "clustered_results.json")
    analyzed_results_path = os.path.join(work_dir, "results", "analyzed_results.json")

    if not os.path.exists(eigenvectors_path):
        print(f"未找到 {eigenvectors_path}, 请先运行 pre-processing.py.")
        return

    # 从 eigenvectors.json 读取所有记录
    with open(eigenvectors_path, "r", encoding="utf-8") as f:
        all_records = json.load(f)

    if not all_records:
        print("eigenvectors.json 文件为空或没有记录.")
        return

    cleaned_results = []  # 用于保存清洗结果
    clustered_results = []  # 用于保存聚类结果

    # 遍历所有数据集
    for record_idx, record in enumerate(all_records):
        # 仅测试第一个数据集的第一个 CSV 文件
        if record_idx > 0:  # 调整此处条件以启用完整测试
            break

        dataset_id = record_idx  # 使用记录的编号作为 dataset_id
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
        strategies = ["mode", "raha_baran"]  # 新增策略
        for algo in strategies:
            print(f"[INFO] 正在运行清洗策略: {algo}")
            new_file_path, runtime = run_error_correction(
                dataset_path=csv_path,
                dataset_id=dataset_id,  # 使用编号作为 dataset_id
                algorithm_id=2 if algo == "raha_baran" else 1,  # 使用编号区分算法
                clean_csv_path=clean_csv_path,  # 直接传递完整的 clean_csv_path
                output_dir=os.path.join(work_dir, "results", dataset_name, algo),
            )

            if new_file_path and runtime:
                print(f"清洗完成: Dataset={dataset_name}, Algo={algo}")
                print(f"结果文件路径: {new_file_path}")
                print(f"运行时间: {runtime:.2f} 秒")

                # 保存清洗结果到记录中
                cleaned_results.append({
                    "dataset_id": dataset_id,
                    "algorithm": algo,
                    "algorithm_id": 1,
                    "cleaned_file_path": new_file_path,
                    "runtime": runtime
                })

                # ========== 运行聚类算法 ==========
                for cluster_method_id in range(6):  # 聚类方法从 0 到 5
                    cluster_output_dir, cluster_runtime = run_clustering(
                        dataset_id=dataset_id,
                        algorithm=algo,
                        cluster_method_id=cluster_method_id,
                        cleaned_file_path=new_file_path
                    )

                    if cluster_output_dir and cluster_runtime:
                        clustered_results.append({
                            "dataset_id": dataset_id,
                            "cleaning_algorithm": algo,
                            "cleaning_runtime": runtime,
                            "clustering_algorithm": cluster_method_id,
                            "clustering_name": ClusterMethod(cluster_method_id).name,
                            "clustering_runtime": cluster_runtime,
                            "clustered_file_path": cluster_output_dir,
                        })
                        print(f"[INFO] 聚类完成: {ClusterMethod(cluster_method_id).name}, 运行时间: {cluster_runtime:.2f} 秒")
                    else:
                        print(f"[ERROR] 聚类算法 {ClusterMethod(cluster_method_id).name} 运行失败")

            print("=" * 50)

    # 将清洗结果保存为 JSON 文件
    with open(cleaned_results_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_results, f, ensure_ascii=False, indent=4)

    print(f"清洗结果已保存到 {cleaned_results_path}")

    # 将聚类结果保存为 JSON 文件
    with open(clustered_results_path, "w", encoding="utf-8") as f:
        json.dump(clustered_results, f, ensure_ascii=False, indent=4)

    print(f"聚类结果已保存到 {clustered_results_path}")

    # ========== 分析聚类结果 ==========
    print("[INFO] 开始分析聚类结果")
    analyzed_results = generate_training_data(clustered_results_path)

    # 保存分析后的结果
    with open(analyzed_results_path, "w", encoding="utf-8") as f:
        json.dump(analyzed_results, f, ensure_ascii=False, indent=4)

    print(f"分析结果已保存到 {analyzed_results_path}")

if __name__ == "__main__":
    main()

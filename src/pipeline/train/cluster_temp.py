import os
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from src.pipeline.train.cluster_methods import ClusterMethod
from src.pipeline.train.cluster_methods import run_clustering

def process_record(record_idx, record, work_dir):
    """
    仅执行聚类逻辑：
    1) 通过 dataset_id 找到修复后的 CSV 文件（repaired_{dataset_id}.csv）。
    2) 调用 run_clustering 进行聚类。
    3) 收集聚类结果并返回。
    """
    # 这里不再做清洗，因此 cleaned_results 为空
    cleaned_results = []
    clustered_results = []

    dataset_id = record_idx
    dataset_name = record["dataset_name"]
    csv_file = record.get("csv_file", "unknown.csv")
    error_rate = record.get("error_rate", -1)

    print(f"[INFO] 准备处理数据集: {dataset_name} (CSV: {csv_file}, error_rate={error_rate}%)")

    # 如果 error_rate 大约等于 1% (干净数据集)，可直接跳过
    if abs(error_rate - 0.01) < 1e-12:
        print(f"[INFO] 检测到 clean 数据集 {dataset_name}，跳过聚类")
        print("=" * 50)
        return cleaned_results, clustered_results

    # 这里列出所有清洗算法名称，对应 cleaned_data/{algo_name}/repaired_{dataset_id}.csv
    strategies = {
        1: "mode",
        2: "baran",
        3: "holoclean",
        4: "bigdansing",
        5: "boostclean",
        6: "horizon",
        7: "scared",
        8: "Unified",
    }

    # 遍历所有清洗算法，拼出 "cleaned_data/{algo_name}/repaired_{dataset_id}.csv"
    for algo_id, algo_name in strategies.items():
        repaired_file_path = os.path.join(
            work_dir,
            "results/cleaned_data",       # 你本地存放清洗后文件的目录
            algo_name,
            f"repaired_{dataset_id}.csv"
        )

        if not os.path.exists(repaired_file_path):
            print(f"[WARNING] 文件不存在: {repaired_file_path}, 跳过算法 {algo_name}")
            continue

        print(f"[INFO] 使用 {algo_name} 的清洗结果进行聚类: {repaired_file_path}")

        # 由于没有清洗步骤，这里先默认清洗运行时间为 0
        cleaning_runtime = 0.0

        # 示例中只有 1 种聚类算法。如果有多种，修改此 range 或自行遍历
        for cluster_method_id in range(1):
            cluster_output_dir, cluster_runtime = run_clustering(
                dataset_id=dataset_id,
                algorithm=algo_name,
                cluster_method_id=cluster_method_id,
                cleaned_file_path=repaired_file_path
            )

            if cluster_output_dir and cluster_runtime:
                clustered_results.append({
                    "dataset_id": dataset_id,
                    "cleaning_algorithm": algo_name,
                    "cleaning_runtime": cleaning_runtime,
                    "clustering_algorithm": cluster_method_id,
                    "clustering_name": ClusterMethod(cluster_method_id).name,
                    "clustering_runtime": cluster_runtime,
                    "clustered_file_path": cluster_output_dir,
                })
                print(f"[INFO] 聚类完成: {ClusterMethod(cluster_method_id).name}, 运行时间: {cluster_runtime:.2f} 秒")
            else:
                print(f"[ERROR] 聚类算法 {ClusterMethod(cluster_method_id).name} 运行失败")

        print("=" * 50)

    return cleaned_results, clustered_results


def main():
    # 你的项目根目录 (根据实际情况修改)
    work_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )

    # 这些路径根据你的需求设置
    eigenvectors_path = os.path.join(work_dir, "results", "eigenvectors.json")
    clustered_results_path = os.path.join(work_dir, "results", "clustered_results.json")
    analyzed_results_path = os.path.join(work_dir, "results", "analyzed_results.json")

    # 检查 eigenvectors.json 是否存在
    if not os.path.exists(eigenvectors_path):
        print(f"[ERROR] 未找到 {eigenvectors_path}, 请先运行 pre-processing.py.")
        return

    with open(eigenvectors_path, "r", encoding="utf-8") as f:
        all_records = json.load(f)

    if not all_records:
        print("[ERROR] eigenvectors.json 文件为空或没有记录.")
        return

    # 只准备保存聚类结果
    clustered_results = []

    # 如果你想并行处理多个数据集，可以保留此并发写法
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as executor:
        futures = [
            executor.submit(process_record, record_idx, record, work_dir)
            for record_idx, record in enumerate(all_records)
        ]
        for future in futures:
            try:
                # process_record 返回 (cleaned_results, clustered_results_part)
                # 这里前者恒为空，不再使用
                _, result_clustered = future.result()
                clustered_results.extend(result_clustered)
            except Exception as e:
                print(f"[ERROR] 处理数据集时发生异常: {e}", flush=True)

    # 将最终聚类结果写入 JSON
    with open(clustered_results_path, "w", encoding="utf-8") as f:
        json.dump(clustered_results, f, ensure_ascii=False, indent=4)

    print(f"[INFO] 聚类结果已保存到 {clustered_results_path}")

    # 如果后续需要分析聚类结果
    from src.pipeline.train.clustered_analysis import save_analyzed_results
    print("[INFO] 开始分析聚类结果")
    save_analyzed_results(
        preprocessing_file_path=os.path.join("pre-processing.py"),  # 如果需要
        eigenvectors_path=eigenvectors_path,
        clustered_results_path=clustered_results_path,
        output_path=analyzed_results_path
    )
    print(f"[INFO] 分析结果已保存到 {analyzed_results_path}")

    # 如果后续需要生成训练数据
    from src.pipeline.train.classifier_preparation import generate_training_data
    print("[INFO] 开始生成训练数据")
    try:
        generate_training_data()
        print("[INFO] 训练数据已成功生成并保存")
    except Exception as e:
        print(f"[ERROR] 生成训练数据时发生错误: {e}")


if __name__ == "__main__":
    main()

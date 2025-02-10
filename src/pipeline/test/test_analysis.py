import json
from collections import defaultdict
from src.pipeline.train.clustered_analysis import parse_cluster_file

def save_test_analyzed_results(
        eigenvectors_path: str,
        clustered_results_path: str,
        output_path: str
):
    # 1) 获取 r 值
    r_value = 3
    print(f"[INFO] Top-r 值: {r_value}")

    # 2) 读取 eigenvectors.json
    try:
        with open(eigenvectors_path, "r", encoding="utf-8") as f:
            eigenvectors_list = json.load(f)
    except Exception as e:
        print(f"[ERROR] 无法读取 {eigenvectors_path}: {e}")
        return

    # 这里假设 eigenvectors_list 中的每个元素都有 "dataset_id" 键
    dataset_ids = [item["dataset_id"] for item in eigenvectors_list]

    # 3) 读取 clustered_results.json
    try:
        with open(clustered_results_path, "r", encoding="utf-8") as f:
            clustered_results = json.load(f)
    except Exception as e:
        print(f"[ERROR] 无法读取 {clustered_results_path}: {e}")
        return

    dataset_methods = defaultdict(list)
    for method_info in clustered_results:
        dataset_id = method_info.get("dataset_id")
        if dataset_id is not None:
            dataset_methods[dataset_id].append(method_info)

    # 4) 遍历每个 dataset_id 的方法
    analyzed_results = []
    for dataset_id in dataset_ids:
        if dataset_id not in dataset_methods:
            print(f"[WARNING] dataset_id {dataset_id} 在 test_clustered_results.json 中未找到记录，跳过。")
            continue

        strategy_list = []
        for method_info in dataset_methods[dataset_id]:
            cleaning_alg = method_info.get("cleaning_algorithm", "unknown_cleaning")
            # 修改这里：使用 "clustering_algorithm" 而非 "clustering_name"
            clustering_alg = method_info.get("clustering_algorithm", "unknown_clustering")
            directory_path = method_info.get("clustered_file_path", "")

            # 使用 dataset_id 定位具体的 repaired 文件
            best_params, final_score = parse_cluster_file(directory_path, dataset_id)
            strategy_list.append([cleaning_alg, clustering_alg, best_params, final_score])

        # 根据综合得分排序，取前 r_value 个策略
        strategy_list_sorted = sorted(strategy_list, key=lambda x: x[3], reverse=True)
        top_r = strategy_list_sorted[:r_value]

        analyzed_results.append({
            "dataset_id": dataset_id,
            "top_r": top_r
        })

    # 5) 保存结果
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analyzed_results, f, ensure_ascii=False, indent=4)
        print(f"[INFO] 分析结果已保存到 {output_path}")
    except Exception as e:
        print(f"[ERROR] 无法保存分析结果: {e}")

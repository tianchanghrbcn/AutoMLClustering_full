import json
from collections import defaultdict
from src.pipeline.train.clustered_analysis import parse_cluster_file

def save_test_analyzed_results(
        eigenvectors_path: str,
        clustered_results_path: str,
        output_path: str
):
    # 1) 获取 r 值（这里 r_value 用于初始候选数，但最终输出保证仅有3项）
    r_value = 7
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

    # 4) 遍历每个 dataset_id 的方法，构造 analyzed_results
    analyzed_results = []
    for dataset_id in dataset_ids:
        if dataset_id not in dataset_methods:
            print(f"[WARNING] dataset_id {dataset_id} 在 clustered_results 中未找到记录，跳过。")
            continue

        # 4.1) 先收集全部策略
        strategy_list = []
        for method_info in dataset_methods[dataset_id]:
            cleaning_alg = method_info.get("cleaning_algorithm", "unknown_cleaning")
            clustering_alg = method_info.get("clustering_algorithm", "unknown_clustering")
            directory_path = method_info.get("clustered_file_path", "")

            # 使用 dataset_id 定位具体的 repaired 文件，获取最佳参数和综合得分
            best_params, final_score = parse_cluster_file(directory_path, dataset_id)

            # 将信息存入列表
            strategy_list.append([cleaning_alg, clustering_alg, best_params, final_score])

        # 4.2) 对策略根据综合得分进行降序排序
        strategy_list_sorted = sorted(strategy_list, key=lambda x: x[3], reverse=True)

        # 4.3) 按你原本的逻辑，忽略 score >= 3.0 的策略，但只忽略 2 个
        selected = []
        ignored_count = 0
        for s in strategy_list_sorted:
            if s[3] >= 3.0 and ignored_count < 2:
                ignored_count += 1
                continue
            selected.append(s)
            if len(selected) == r_value:  # 这里仍是初步截取 r_value = 7
                break

        # 4.4) 如果经过过滤后不足 r_value=7 项，就继续从剩余里补
        if len(selected) < r_value:
            for s in strategy_list_sorted:
                if s not in selected:
                    selected.append(s)
                    if len(selected) == r_value:
                        break

        # ---- 到此为止，selected 大小最多是 7，最少也有 7 或更少(如果策略都不够) ----

        # 4.5) 去重：同一 (cleaning_alg, clustering_alg, best_params) 只留一次
        deduped = []
        seen = set()
        for s in selected:
            # 定义一个“唯一码 key”，比如把 params 转成 JSON 并排序
            # 注意把 covariance type 或清洗算法字符串中大小写等，都保持一致性
            param_key = json.dumps(s[2], sort_keys=True)
            unique_key = (s[0], s[1], param_key)
            if unique_key not in seen:
                deduped.append(s)
                seen.add(unique_key)

        # 4.6) 最终只保留 3 条，如果不足3条则用占位补齐
        top3 = deduped[:3]
        while len(top3) < 3:
            top3.append(["unknown_cleaning", "unknown_clustering", {}, 0])

        # 4.7) 将这 3 条存入结果
        analyzed_results.append({
            "dataset_id": dataset_id,
            "top_r": top3  # 这里就只有 3 项了
        })

    # 5) 保存结果到 output_path
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analyzed_results, f, ensure_ascii=False, indent=4)
        print(f"[INFO] 分析结果已保存到", output_path)
    except Exception as e:
        print(f"[ERROR] 无法保存分析结果: {e}")

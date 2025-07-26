#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_only.py ── 仅执行聚类 + 结果汇总 & 分析

逻辑概述
--------
1. 预先读取 results/cleaned_results.json，建立 {dataset_id: runtime} 对照表。
2. 并行遍历 eigenvectors.json 中的所有数据集：
   • 按 strategies 查找 repaired_{dataset_id}.csv
   • 对每个聚类算法 run_clustering(...)
   • 将步骤 1 中查到的 runtime 作为 cleaning_runtime 写入结果
3. 汇总 clustered_results.json，并调用 save_analyzed_results(...)
"""

import os
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from src.pipeline.train.cluster_methods import ClusterMethod, run_clustering


# --------------------------------------------------------------------------- #
# 1. 单个数据集的处理逻辑
# --------------------------------------------------------------------------- #
def process_record(record_idx, record, work_dir, runtime_lookup):
    """
    仅执行聚类逻辑：
      1) 通过 dataset_id 找到修复后的 CSV 文件（repaired_{dataset_id}.csv）
      2) 调用 run_clustering 进行聚类
      3) 收集聚类结果并返回

    Parameters
    ----------
    record_idx : int
        eigenvectors.json 中的索引
    record : dict
        该数据集的原始元信息
    work_dir : str
        项目根目录绝对路径
    runtime_lookup : dict[int, float]
        {dataset_id: cleaning_runtime} 查表
    """
    cleaned_results = []          # 本脚本不再生成新清洗结果
    clustered_results = []

    dataset_id = record_idx       # eigenvectors.json 的顺序即 id
    dataset_name = record["dataset_name"]
    csv_file     = record.get("csv_file", "unknown.csv")
    error_rate   = record.get("error_rate", -1)

    print(f"[INFO] 准备处理数据集: {dataset_name} (CSV: {csv_file}, error_rate={error_rate}%)")

    # 对于近似 1% 的干净数据集可跳过
    if abs(error_rate - 0.01) < 1e-12:
        print(f"[INFO] 检测到 clean 数据集 {dataset_name}，跳过聚类")
        print("=" * 50)
        return cleaned_results, clustered_results

    # ==== NEW ==== ––– 由 cleaned_results.json 查表
    cleaning_runtime = runtime_lookup.get(dataset_id, 0.0)
    if cleaning_runtime == 0.0:
        print(f"[WARNING] 未找到 dataset_id={dataset_id} 的清洗 runtime，置 0.0 秒")

    # 待聚类的清洗策略
    strategies = {
        #1: "mode",
        2: "baran",
        #3: "holoclean",
        #4: "bigdansing",
        #5: "boostclean",
        #6: "horizon",
        #7: "scared",
        #8: "Unified",
    }

    for algo_id, algo_name in strategies.items():
        repaired_file_path = os.path.join(
            work_dir, "results", "cleaned_data", algo_name, f"repaired_{dataset_id}.csv"
        )
        if not os.path.exists(repaired_file_path):
            print(f"[WARNING] 文件不存在: {repaired_file_path}, 跳过算法 {algo_name}")
            continue

        print(f"[INFO] 使用 {algo_name} 的清洗结果进行聚类: {repaired_file_path}")

        for cluster_method_id in range(6):
            cluster_output_dir, cluster_runtime = run_clustering(
                dataset_id        = dataset_id,
                algorithm         = algo_name,
                cluster_method_id = cluster_method_id,
                cleaned_file_path = repaired_file_path,
            )

            if cluster_output_dir and cluster_runtime:
                clustered_results.append({
                    "dataset_id"          : dataset_id,
                    "cleaning_algorithm"  : algo_name,
                    "cleaning_runtime"    : cleaning_runtime,       # ==== NEW ====
                    "clustering_algorithm": cluster_method_id,
                    "clustering_name"     : ClusterMethod(cluster_method_id).name,
                    "clustering_runtime"  : cluster_runtime,
                    "clustered_file_path" : cluster_output_dir,
                })
                print(f"[INFO] 聚类完成: {ClusterMethod(cluster_method_id).name}, "
                      f"运行时间: {cluster_runtime:.2f} 秒")
            else:
                print(f"[ERROR] 聚类算法 {ClusterMethod(cluster_method_id).name} 运行失败")

        print("=" * 50)

    return cleaned_results, clustered_results


# --------------------------------------------------------------------------- #
# 2. 主入口
# --------------------------------------------------------------------------- #
def main():
    # 项目根目录
    work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    # 关键文件路径
    eigenvectors_path     = os.path.join(work_dir, "results", "eigenvectors.json")
    clustered_results_path = os.path.join(work_dir, "results", "clustered_results.json")
    analyzed_results_path  = os.path.join(work_dir, "results", "analyzed_results.json")
    cleaned_results_path   = os.path.join(work_dir, "results", "cleaned_results.json")  # ==== NEW ====

    # 读取 eigenvectors.json
    if not os.path.exists(eigenvectors_path):
        print(f"[ERROR] 未找到 {eigenvectors_path}，请先运行 pre-processing.py。")
        return
    with open(eigenvectors_path, "r", encoding="utf-8") as f:
        all_records = json.load(f)
    if not all_records:
        print("[ERROR] eigenvectors.json 文件为空或没有记录。")
        return

    # ==== NEW ==== ––– 读取 cleaned_results.json 并构建 runtime_lookup
    runtime_lookup = {}
    if os.path.exists(cleaned_results_path):
        with open(cleaned_results_path, "r", encoding="utf-8") as f:
            cleaned_items = json.load(f)
        runtime_lookup = {item["dataset_id"]: item["runtime"] for item in cleaned_items}
        print(f"[INFO] 已加载 {len(runtime_lookup)} 条清洗 runtime。")
    else:
        print(f"[WARNING] 未找到 {cleaned_results_path}，所有 cleaning_runtime 将置 0.0。")

    # 并行聚类
    clustered_results = []
    with ProcessPoolExecutor(max_workers=mp.cpu_count(),
                             mp_context=mp.get_context("spawn")) as executor:
        futures = [
            executor.submit(process_record, idx, rec, work_dir, runtime_lookup)
            for idx, rec in enumerate(all_records)
        ]
        for future in futures:
            try:
                _, clustered_part = future.result()
                clustered_results.extend(clustered_part)
            except Exception as e:
                print(f"[ERROR] 处理数据集时发生异常: {e}", flush=True)

    # 保存聚类结果
    with open(clustered_results_path, "w", encoding="utf-8") as f:
        json.dump(clustered_results, f, ensure_ascii=False, indent=4)
    print(f"[INFO] 聚类结果已保存到 {clustered_results_path}")

    # 后续分析
    from src.pipeline.train.clustered_analysis import save_analyzed_results
    print("[INFO] 开始分析聚类结果")
    save_analyzed_results(
        preprocessing_file_path=os.path.join("pre-processing.py"),  # 如需可改
        eigenvectors_path       = eigenvectors_path,
        clustered_results_path  = clustered_results_path,
        output_path             = analyzed_results_path,
    )
    print(f"[INFO] 分析结果已保存到 {analyzed_results_path}")


if __name__ == "__main__":
    main()

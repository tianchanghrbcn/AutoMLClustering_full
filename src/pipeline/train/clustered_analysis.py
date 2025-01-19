import os
import json
import re
from typing import List, Dict, Any

# 读取 K_VALUE 的函数
def get_k_value(preprocessing_file_path: str) -> int:
    """从 pre-processing.py 文件的第 12 行读取常量 K_VALUE 的值"""
    k_value = 5  # 默认值
    try:
        with open(preprocessing_file_path, "r", encoding="utf-8") as f:
            # 直接定位到第 12 行
            lines = f.readlines()
            if len(lines) >= 12:
                line = lines[11]  # 第 12 行（索引从 0 开始）
                match = re.search(r"K_VALUE\s*=\s*(\d+)", line)
                if match:
                    k_value = int(match.group(1))
            else:
                print("[ERROR] pre-processing.py 文件内容不足 12 行")
    except Exception as e:
        print(f"[ERROR] 无法读取 K_VALUE: {e}")
    return k_value

# 解析聚类结果文件的函数
def parse_clustering_results(file_path: str) -> Dict[str, float]:
    """从聚类结果文件中提取相关指标"""
    metrics = {
        "number_of_clusters": None,
        "combined_score": None,
        "silhouette_score": None,
        "davies_bouldin_score": None
    }
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if "Number of clusters" in line:
                    metrics["number_of_clusters"] = int(re.search(r"(\d+)", line).group(1))
                elif "Final Combined Score" in line:
                    metrics["combined_score"] = float(re.search(r"([\d.]+)", line).group(1))
                elif "Final Silhouette Score" in line:
                    metrics["silhouette_score"] = float(re.search(r"([\d.]+)", line).group(1))
                elif "Final Davies-Bouldin score" in line:
                    metrics["davies_bouldin_score"] = float(re.search(r"([\d.]+)", line).group(1))
    except Exception as e:
        print(f"[ERROR] 无法解析聚类结果文件 {file_path}: {e}")
    return metrics


# 生成训练数据的函数
def generate_training_data(clustered_results: List[Dict[str, Any]], k_value: int) -> List[Dict[str, Any]]:
    """基于聚类结果生成训练数据"""
    training_data = []
    for result in clustered_results:
        try:
            # 从聚类结果文件中提取指标
            metrics = parse_clustering_results(result["clustered_file_path"])

            if metrics["combined_score"] is None:
                print(f"[WARNING] 数据集 {result['dataset_id']} 的聚类结果缺少 combined_score，跳过。")
                continue

            # 模拟计算 S(D^{(i)}, \omega)，这里直接使用 combined_score
            combined_score = metrics["combined_score"]

            # 使用 combined_score 模拟计算 Top-K 策略
            top_k_strategies = [{
                "dataset_id": result["dataset_id"],
                "cleaning_algorithm": result["cleaning_algorithm"],
                "cleaning_runtime": result["cleaning_runtime"],
                "clustering_algorithm": result["clustering_algorithm"],
                "clustering_name": result["clustering_name"],
                "clustering_runtime": result["clustering_runtime"],
                "score": combined_score
            }]

            # 按分数降序排序，并截取前 K 项
            top_k_strategies = sorted(top_k_strategies, key=lambda x: x["score"], reverse=True)[:k_value]

            # 将结果加入训练数据
            training_data.extend(top_k_strategies)

        except Exception as e:
            print(f"[ERROR] 数据集 {result['dataset_id']} 的训练数据生成失败: {e}")

    return training_data


# 保存分析结果的函数
def save_analyzed_results(analyzed_results: List[Dict[str, Any]], output_path: str):
    """将分析结果保存为 JSON 文件"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analyzed_results, f, ensure_ascii=False, indent=4)
        print(f"[INFO] 分析结果已保存到 {output_path}")
    except Exception as e:
        print(f"[ERROR] 分析结果保存失败: {e}")

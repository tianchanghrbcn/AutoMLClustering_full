import os
import json
import re
from typing import List, Dict, Any
from collections import defaultdict

def get_k_value(preprocessing_file_path: str) -> int:
    """
    从 pre-processing.py 文件的第 12 行读取常量 K_VALUE 的值。
    """
    k_value = 5  # 默认值
    try:
        with open(preprocessing_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) >= 12:
                line = lines[11]
                match = re.search(r"K_VALUE\s*=\s*(\d+)", line)
                if match:
                    k_value = int(match.group(1))
            else:
                print("[ERROR] pre-processing.py 文件内容不足 12 行")
    except Exception as e:
        print(f"[ERROR] 无法读取 K_VALUE: {e}")
    return k_value


def parse_clustering_results(file_path: str) -> Dict[str, float]:
    """
    从聚类结果文件中提取相关指标。
    假设文件为 JSON 格式，包含以下字段:
        - number_of_clusters
        - combined_score
        - silhouette_score
        - davies_bouldin_score
    """
    metrics = {
        "number_of_clusters": None,
        "combined_score": None,
        "silhouette_score": None,
        "davies_bouldin_score": None
    }
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            metrics["number_of_clusters"] = data.get("number_of_clusters", None)
            metrics["combined_score"] = data.get("combined_score", None)
            metrics["silhouette_score"] = data.get("silhouette_score", None)
            metrics["davies_bouldin_score"] = data.get("davies_bouldin_score", None)
    except Exception as e:
        print(f"[ERROR] 无法解析聚类结果文件 {file_path}: {e}")
    return metrics


def generate_training_data(
    clustered_results: List[Dict[str, Any]],
    eigenvectors_path: str,
    k_value: int
) -> List[Dict[str, Any]]:
    """
    基于聚类结果生成训练数据 (花体 M)。
    每条记录包含:
        - dataset_id
        - x (特征向量)
        - labels (多标签集合)
    """
    # 读取 eigenvectors.json
    try:
        with open(eigenvectors_path, "r", encoding="utf-8") as f:
            eigenvectors_data = {item["dataset_id"]: item["x"] for item in json.load(f)}
    except Exception as e:
        print(f"[ERROR] 无法读取 eigenvectors.json: {e}")
        return []

    # 处理每个 dataset_id 的聚类结果
    dataset_strategies = defaultdict(list)
    for result in clustered_results:
        try:
            dataset_id = result["dataset_id"]
            metrics = parse_clustering_results(result["clustered_file_path"])
            combined_score = metrics["combined_score"]

            if combined_score is None:
                print(f"[WARNING] 数据集 {dataset_id} 的聚类结果缺少 combined_score，跳过。")
                continue

            strategy_label = f"cleaning={result['cleaning_algorithm']}__clustering={result['clustering_algorithm']}"
            dataset_strategies[dataset_id].append({
                "label": strategy_label,
                "score": combined_score
            })
        except Exception as e:
            print(f"[ERROR] 数据集 {result.get('dataset_id', 'unknown')} 处理失败: {e}")

    # 构建训练数据
    training_data = []
    for dataset_id, strategies in dataset_strategies.items():
        if dataset_id not in eigenvectors_data:
            print(f"[WARNING] 在 eigenvectors.json 中找不到 dataset_id={dataset_id} 的特征向量，跳过。")
            continue

        # Top-K 策略
        strategies_sorted = sorted(strategies, key=lambda x: x["score"], reverse=True)[:k_value]
        labels = [s["label"] for s in strategies_sorted]

        training_data.append({
            "dataset_id": dataset_id,
            "x": eigenvectors_data[dataset_id],
            "labels": labels
        })

    return training_data


def save_training_data(training_data: List[Dict[str, Any]], output_path: str):
    """
    保存训练数据 (花体 M) 到 JSON 文件。
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(training_data, f, ensure_ascii=False, indent=4)
        print(f"[INFO] 训练数据已保存到 {output_path}")
    except Exception as e:
        print(f"[ERROR] 无法保存训练数据: {e}")


def main():
    # 工作目录路径
    work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    # 各种文件路径
    preprocessing_file_path = os.path.join(work_dir, "scripts", "pre-processing.py")
    eigenvectors_path = os.path.join(work_dir, "results", "eigenvectors.json")
    clustered_results_path = os.path.join(work_dir, "results", "clustered_results.json")
    analyzed_results_path = os.path.join(work_dir, "results", "analyzed_results.json")

    # 获取 K_VALUE
    k_value = get_k_value(preprocessing_file_path)
    print(f"[INFO] Top-K 值: {k_value}")

    # 读取 clustered_results.json
    try:
        with open(clustered_results_path, "r", encoding="utf-8") as f:
            clustered_results = json.load(f)
    except Exception as e:
        print(f"[ERROR] 无法读取 {clustered_results_path}: {e}")
        return

    # 生成训练数据 (花体 M)
    training_data = generate_training_data(clustered_results, eigenvectors_path, k_value)

    # 保存训练数据到 analyzed_results.json
    save_training_data(training_data, analyzed_results_path)


if __name__ == "__main__":
    main()

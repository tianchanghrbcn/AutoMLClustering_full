import os
import json
import re
import subprocess
import pandas as pd
import shutil
import time

# 路径配置：当前脚本位于 src/pipeline/test 目录下
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIG_JSON_PATH = os.path.join(BASE_DIR, "../../../results/clustered_results.json")
TEST_JSON_PATH = os.path.join(BASE_DIR, "../../../results/test_clustered_results.json")
OUTPUT_JSON_PATH = os.path.join(BASE_DIR, "../../../results/computed_results.json")

def parse_combined_score(file_path):
    """
    从 file_path 指向的文本文件中解析 "Final Combined Score"
    假设文件包含类似行：
      Final Combined Score: 0.7334520292607687
    若找不到，则返回 None
    """
    score_pattern = re.compile(r"Final Combined Score:\s*([\d.]+)")
    if not os.path.isfile(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    match = score_pattern.search(content)
    if match:
        return float(match.group(1))
    return None

def main():
    # 1. 读取原始 JSON 文件
    if not os.path.exists(ORIG_JSON_PATH):
        print(f"Error: {ORIG_JSON_PATH} not found")
        return
    if not os.path.exists(TEST_JSON_PATH):
        print(f"Error: {TEST_JSON_PATH} not found")
        return

    with open(ORIG_JSON_PATH, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 2. 将数据按 key=(dataset_id, cleaning_algorithm, clustering_name) 索引，
    # 并存储总耗时、综合得分及结果文件路径。
    original_dict = {}
    for item in original_data:
        ds_id = item["dataset_id"]
        cleaning = item["cleaning_algorithm"]
        c_name = item["clustering_name"]
        total_time = item["cleaning_runtime"] + item["clustering_runtime"]
        file_path = item.get("clustered_file_path", "")
        combined_score = parse_combined_score(file_path)
        key = (ds_id, cleaning, c_name)
        original_dict[key] = {
            "time": total_time,
            "score": combined_score,
            "file_path": file_path
        }

    test_dict = {}
    for item in test_data:
        ds_id = item["dataset_id"]
        cleaning = item["cleaning_algorithm"]
        # 此处 JSON 中聚类算法的标识可能为字符串，如 "KMEANS", "HC", 等
        c_name = item["clustering_algorithm"]
        total_time = item["cleaning_runtime"] + item["clustering_runtime"]
        file_path = item.get("clustered_file_path", "")
        combined_score = parse_combined_score(file_path)
        key = (ds_id, cleaning, c_name)
        test_dict[key] = {
            "time": total_time,
            "score": combined_score,
            "file_path": file_path
        }

    # 3. 计算每个匹配组合的损失率和加速比
    # 定义：
    #   loss_rate = 1 - (score_test / score_orig)
    #   acc_ratio = (score_test / score_orig) * (time_orig / time_test)
    results = []
    for key, orig_info in original_dict.items():
        if key not in test_dict:
            continue
        test_info = test_dict[key]
        time_orig = orig_info["time"]
        score_orig = orig_info["score"]
        time_test = test_info["time"]
        score_test = test_info["score"]

        # 若得分解析失败或原始得分为 0，则跳过
        if (score_orig is None) or (score_test is None) or (score_orig == 0.0):
            continue
        if time_test <= 0:
            continue

        loss_rate = 1.0 - (score_test / score_orig)
        acc_ratio = (score_test / score_orig) * (time_orig / time_test)

        results.append({
            "dataset_id": key[0],
            "cleaning_algorithm": key[1],
            "clustering_algorithm": key[2],
            "original_time": time_orig,
            "original_score": score_orig,
            "test_time": time_test,
            "test_score": score_test,
            "loss_rate": loss_rate,
            "acc_ratio": acc_ratio
        })

    # 4. 将结果保存到 JSON 文件
    try:
        os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"[SUCCESS] 计算结果已保存到 {OUTPUT_JSON_PATH}")
    except Exception as e:
        print(f"[ERROR] 保存结果失败: {e}")

    # 同时也可以打印部分结果
    for r in results:
        print(f"[DATASET={r['dataset_id']}, CLEAN={r['cleaning_algorithm']}, CLUSTER={r['clustering_algorithm']}]")
        print(f"  original_time={r['original_time']:.2f}, original_score={r['original_score']:.4f}")
        print(f"  test_time={r['test_time']:.2f}, test_score={r['test_score']:.4f}")
        print(f"  loss_rate={r['loss_rate']:.4f}, acc_ratio={r['acc_ratio']:.4f}")
        print("")

# 若作为模块被调用，则上层代码直接导入 run_test_error_correction 和 main 等接口
# 这里保留 __main__ 保护仅用于独立测试
if __name__ == "__main__":
    main()

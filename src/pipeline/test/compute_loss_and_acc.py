import os
import json
import re

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在路径
ORIG_JSON_PATH = os.path.join(BASE_DIR, "../../../results/clustered_results.json")
TEST_JSON_PATH = os.path.join(BASE_DIR, "../../../results/test_clustered_results.json")


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
    # 1. 读取两个 JSON
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

    # 2. 将数据按 key=(dataset_id, cleaning_algorithm, clustering_name) 索引
    #    并存储耗时和 "clustered_file_path"
    #    Key: (dataset_id, cleaning_algorithm, clustering_name)
    #    Value: { "time": float, "score": float, "file_path": str }
    original_dict = {}
    for item in original_data:
        ds_id = item["dataset_id"]
        cleaning = item["cleaning_algorithm"]
        c_name = item["clustering_name"]
        total_time = item["cleaning_runtime"] + item["clustering_runtime"]
        file_path = item.get("clustered_file_path", "")
        # 从文件解析得分
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
        c_name = item["clustering_algorithm"]  # 这里 json 中就是 "KMEANS", "HC"...
        total_time = item["cleaning_runtime"] + item["clustering_runtime"]
        file_path = item.get("clustered_file_path", "")
        # 同样解析
        combined_score = parse_combined_score(file_path)

        key = (ds_id, cleaning, c_name)
        test_dict[key] = {
            "time": total_time,
            "score": combined_score,
            "file_path": file_path
        }

    # 3. 计算损失率与加速比
    results = []
    for key, orig_info in original_dict.items():
        if key not in test_dict:
            # 说明自动化方法没有这个 key
            continue

        test_info = test_dict[key]

        # 原始方法
        time_orig = orig_info["time"]
        score_orig = orig_info["score"]
        # 自动化
        time_test = test_info["time"]
        score_test = test_info["score"]

        # 如果没有解析到得分，就跳过
        if (score_orig is None) or (score_test is None) or (score_orig == 0.0):
            continue

        # 损失率 = 1 - (score_test / score_orig)
        loss_rate = 1.0 - (score_test / score_orig)

        # 加速比 = (1 - loss_rate) * (time_orig / time_test)
        #        = (score_test / score_orig) * (time_orig / time_test)
        # 若 time_test 或 score_test = 0 => 需判断
        if (time_test <= 0):
            continue

        acc_ratio = (1.0 - loss_rate) * (time_orig / time_test)
        # or: acc_ratio = (score_test / score_orig) * (time_orig / time_test)

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

    # 4. 输出或写入文件
    #    这里只是示例打印
    for r in results:
        print(f"[DATASET={r['dataset_id']}, CLEAN={r['cleaning_algorithm']}, CLUSTER={r['clustering_algorithm']}]")
        print(f"  original_time={r['original_time']:.2f}, original_score={r['original_score']:.4f}")
        print(f"  test_time={r['test_time']:.2f}, test_score={r['test_score']:.4f}")
        print(f"  loss_rate={r['loss_rate']:.4f}, acc_ratio={r['acc_ratio']:.4f}")
        print("")


if __name__ == "__main__":
    main()

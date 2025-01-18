import os
import re
import pandas as pd

# 数据目录
input_dir = r"D:\algorithm paper\ML algorithms codes\data_experiments\results\3_analyzed_data\overall_results"

# 数据集和错误率设置
dataset_name = "rayyan"
error_rate = "52.73%"

# 清洗算法和聚类算法
data_cleaning_algorithms = ["mode", "raha-baran"]
cluster_methods = ["AffinityPropagation", "DBSCAN", "GMM", "HC", "KMeans", "OPTICS"]

# 收集结果
results = []

for clean_algorithm in data_cleaning_algorithms:
    for cluster_method in cluster_methods:
        # 构造文件路径
        file_name = f"repaired_{dataset_name}_{error_rate}_{cluster_method}.txt"
        file_path = os.path.join(input_dir, f"{clean_algorithm}_overall", f"clustered_{clean_algorithm}_{dataset_name}",
                                 file_name)

        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue

        # 读取文件并提取分数
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if "Final Combined Score:" in line:
                        # 使用正则表达式提取行中第一个浮点数
                        match = re.search(r"[-+]?\d*\.\d+|\d+", line)
                        if match:
                            score = float(match.group(0))  # 提取并转换为浮点数
                            results.append({
                                "Cleaning Algorithm": clean_algorithm,
                                "Clustering Method": cluster_method,
                                "Score": score
                            })
                        else:
                            print(f"未找到分数: {file_path}")
                        break
        except Exception as e:
            print(f"读取文件失败: {file_path}, 错误: {e}")

# 将结果转化为DataFrame并排序
results_df = pd.DataFrame(results)
sorted_results = results_df.sort_values(by="Score", ascending=False)

# 打印和保存结果
output_file = os.path.join(input_dir, f"analysis_{dataset_name}_{error_rate}.csv")
sorted_results.to_csv(output_file, index=False)

print("分析结果已保存到:", output_file)
print(sorted_results)

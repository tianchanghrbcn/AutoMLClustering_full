import os
import shutil

# 根目录
root_dir = r"D:\algorithm paper\ML algorithms codes\data_experiments\results\2_clustered_data"

# 获取用户输入
clean_algorithm_options = ["mlnclean", "mode", "raha-baran"]
dataset_options = ["beers", "flights", "hospital", "rayyan"]

try:
    clean_algorithm_index = int(input(f"请输入清洗算法编号 (0-2): {clean_algorithm_options}："))
    dataset_index = int(input(f"请输入数据集编号 (0-3): {dataset_options}："))
    if clean_algorithm_index not in range(3) or dataset_index not in range(4):
        raise ValueError("输入的编号超出范围！")
except ValueError as e:
    print(f"输入错误: {e}")
    exit(1)

clean_algorithm = clean_algorithm_options[clean_algorithm_index]
dataset_name = dataset_options[dataset_index]

# 输出目录
output_dir = os.path.join(root_dir, f"{clean_algorithm}_overall", f"clustered_{clean_algorithm}_{dataset_name}")
os.makedirs(output_dir, exist_ok=True)

# 聚类方法列表
cluster_methods = ["AffinityPropagation", "DBSCAN", "GMM", "HC", "KMeans", "OPTICS"]

# 遍历实验目录和文件
for i in range(1, 11):  # 循环实验编号 1-10
    base_path = os.path.join(root_dir, f"{clean_algorithm}_{i}", f"clustered_{clean_algorithm}_{dataset_name}")
    print(f"实验路径: {base_path}")

    if not os.path.exists(base_path):
        print(f"路径不存在: {base_path}")
        continue

    # 提取错误率
    error_rates = set()
    for file_name in os.listdir(base_path):
        if file_name.endswith(".txt"):
            print(f"检测到文件: {file_name}")
            try:
                error_rate = file_name.split("_")[2]
                error_rates.add(error_rate)
            except IndexError:
                print(f"文件名解析失败: {file_name}")
                continue

    # 遍历错误率和聚类方法
    for error_rate in error_rates:
        for cluster_method in cluster_methods:
            max_score = None
            max_file = None

            # 遍历 1-10 次实验
            for j in range(1, 11):
                experiment_path = os.path.join(root_dir, f"{clean_algorithm}_{j}", f"clustered_{clean_algorithm}_{dataset_name}")
                txt_path = os.path.join(experiment_path, f"repaired_{dataset_name}_{error_rate}_{cluster_method}.txt")
                print(f"尝试读取文件: {txt_path}")

                if not os.path.exists(txt_path):
                    print(f"文件不存在: {txt_path}")
                    continue

                # 查找 "Final Combined Score:"
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if "Final Combined Score:" in line:
                                # 提取关键部分并尝试解析为浮点数
                                try:
                                    score_str = line.split(":")[-1][:10].strip()  # 提取可能的数值部分（前10个字符）
                                    score = float(score_str)
                                    if max_score is None or score > max_score:
                                        max_score = score
                                        max_file = txt_path
                                except ValueError:
                                    print(f"解析分数失败，跳过: {line.strip()}")
                                break
                except Exception as e:
                    print(f"读取文件失败: {txt_path}, 错误: {e}")
                    continue

            # 保存最大分数文件
            if max_score is not None and max_file is not None:
                output_file_name = f"repaired_{dataset_name}_{error_rate}_{cluster_method}.txt"
                output_path = os.path.join(output_dir, output_file_name)

                try:
                    shutil.copy(max_file, output_path)
                    print(f"已保存文件: {output_path}")
                except Exception as e:
                    print(f"保存文件失败: {output_path}, 错误: {e}")

print(f"所有聚类算法和错误率的结果已保存到：{output_dir}")

import os
import time
import pandas as pd
import numpy as np
import optuna
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# 获取 CSV 文件路径和环境变量
csv_file_path = os.getenv("CSV_FILE_PATH")
dataset_id = os.getenv("DATASET_ID")
algorithm_name = os.getenv("ALGO")

if not csv_file_path:
    print("Error: CSV file path is not provided. Set 'CSV_FILE_PATH' environment variable.")
    exit(1)

# 规范化路径以确保跨平台兼容性
csv_file_path = os.path.normpath(csv_file_path)

# 读取 CSV 文件
try:
    df = pd.read_csv(csv_file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: File '{csv_file_path}' not found. Please check the path and try again.")
    exit(1)

start_time = time.time()

# 初始化输出存储
output_txt = []

# 排除包含 'id' 的列
excluded_columns = [col for col in df.columns if 'id' in col.lower()]
print(f"Excluded columns with 'id': {excluded_columns}")

# 使用所有非排除列作为特征
remaining_columns = df.columns.difference(excluded_columns)
X = df[remaining_columns]
print(f"Using all columns for clustering: {list(remaining_columns)}")

# 对类别型特征进行频率编码
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'category':
        X[col] = X[col].map(X[col].value_counts(normalize=True))

# 删除包含 NaN 的行
X = X.dropna()

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 定义 alpha 和 beta 权重
alpha, beta = 0.75, 0.25

# 定义优化目标函数
def objective(trial):
    damping = trial.suggest_float("damping", 0.5, 0.9)
    preference = trial.suggest_int("preference", -500, -100)

    ap = AffinityPropagation(damping=damping, preference=preference, random_state=0)
    labels = ap.fit_predict(X_scaled)
    n_clusters = len(np.unique(labels))

    # 忽略不合理的聚类结果
    if n_clusters <= 1 or n_clusters >= len(X_scaled):
        return float('-inf')

    silhouette_avg = silhouette_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)
    db_score = 1e-6 if db_score == 0 else db_score  # 防止除零
    combined_score = alpha * (1 / db_score) + beta * silhouette_avg
    return combined_score  # 返回正值以最大化综合得分


# 使用 Optuna 进行参数搜索
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# 获取最佳参数
best_params = study.best_params
best_damping = best_params["damping"]
best_preference = best_params["preference"]

# 使用最佳参数进行 AP 聚类
best_ap = AffinityPropagation(damping=best_damping, preference=best_preference, random_state=0)
best_labels = best_ap.fit_predict(X_scaled)
best_n_clusters = len(np.unique(best_labels))

# 计算最终得分
final_silhouette_score = silhouette_score(X_scaled, best_labels)
final_db_score = davies_bouldin_score(X_scaled, best_labels)
final_combined_score = alpha * (1 / final_db_score) + beta * final_silhouette_score

# 保存输出和可视化
base_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
output_dir = os.path.join(os.getcwd(), "..", "..", "..", "results", "clustered_data", "AP", algorithm_name,
                          f"clustered_{dataset_id}")
os.makedirs(output_dir, exist_ok=True)
output_txt_file = os.path.join(output_dir, f"{base_filename}.txt")

with open(output_txt_file, 'w', encoding='utf-8') as f:
    output_txt.append(f"Best parameters: damping={best_damping}, preference={best_preference}")
    output_txt.append(f"Number of clusters: {best_n_clusters}")
    output_txt.append(f"Final Combined Score: {final_combined_score}")
    output_txt.append(f"Final Silhouette Score: {final_silhouette_score}")
    output_txt.append(f"Final Davies-Bouldin score: {final_db_score}")
    f.write("\n".join(output_txt))

print(f"Text output saved to {output_txt_file}")

end_time = time.time()
print(f"Program completed in: {end_time - start_time} seconds")

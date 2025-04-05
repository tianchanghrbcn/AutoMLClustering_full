import os
import time
import numpy as np
import pandas as pd
import optuna
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

start_time = time.time()

# 获取 CSV 文件路径和环境变量
csv_file_path = os.getenv("CSV_FILE_PATH")
dataset_id = os.getenv("DATASET_ID")
algorithm_name = os.getenv("ALGO")

if not csv_file_path:
    print("Error: CSV file path is not provided. Set 'CSV_FILE_PATH' environment variable.")
    exit(1)

# 规范路径
csv_file_path = os.path.normpath(csv_file_path)

# 读取数据集
try:
    df = pd.read_csv(csv_file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: File '{csv_file_path}' not found. Please check the path and try again.")
    exit(1)

# 排除包含 'id' 的列
excluded_columns = [col for col in df.columns if 'id' in col.lower()]
print(f"Excluded columns with 'id': {excluded_columns}")
remaining_columns = df.columns.difference(excluded_columns)
print(f"Using all columns for clustering: {list(remaining_columns)}")

# 使用所有非排除列作为特征
X = df[remaining_columns]

# 对类别型特征进行频率编码
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'category':
        X.loc[:, col] = X[col].map(X[col].value_counts(normalize=True))

# 删除包含 NaN 的行并标准化数据
X = X.dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 定义 Optuna 的目标函数
def objective(trial):
    # 参数范围
    eps = trial.suggest_float("eps", 0.1, 2.0, step=0.01)
    min_samples = trial.suggest_int("min_samples", 5, 50)

    # 初始化 DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = dbscan.fit_predict(X_scaled)

    # 确保形成多个有效簇
    n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return float('-inf')

    # 计算噪声点比例
    noise_ratio = np.sum(labels == -1) / len(labels)
    noise_penalty = max(0, 1 - noise_ratio)  # 动态惩罚，噪声越高得分越低

    # 计算评分
    try:
        silhouette_avg = silhouette_score(X_scaled, labels) if n_clusters > 1 else 0
        db_score = davies_bouldin_score(X_scaled, labels) if n_clusters > 1 else float('inf')
        db_score = 1e-6 if db_score == 0 else db_score  # 防止除零
    except ValueError:
        return float('-inf')  # 跳过异常情况

    alpha, beta = 0.5, 0.5
    combined_score = (alpha * (1 / db_score) + beta * silhouette_avg) * noise_penalty
    return combined_score

# 使用 Optuna 优化参数
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)

# 获取最佳参数
best_params = study.best_params
best_combined_score = study.best_value

# 使用最佳参数运行 DBSCAN
best_dbscan = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples'], metric='euclidean')
best_labels = best_dbscan.fit_predict(X_scaled)
n_clusters_final = len(np.unique(best_labels)) - (1 if -1 in best_labels else 0)

# 计算最终得分
if n_clusters_final > 1:
    silhouette_avg = silhouette_score(X_scaled, best_labels)
    db_score = davies_bouldin_score(X_scaled, best_labels)
else:
    silhouette_avg = float('nan')
    db_score = float('nan')

# 保存结果
base_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
output_dir = os.path.join(os.getcwd(), "..", "..", "..", "results", "clustered_data", "DBSCAN", algorithm_name,
                          f"clustered_{dataset_id}")
os.makedirs(output_dir, exist_ok=True)

output_txt_file = os.path.join(output_dir, f"{base_filename}.txt")
with open(output_txt_file, 'w', encoding='utf-8') as f:
    f.write(f"Best parameters: eps={best_params['eps']}, min_samples={best_params['min_samples']}\n")
    f.write(f"Number of clusters: {n_clusters_final}\n")
    f.write(f"Final Combined Score: {best_combined_score}\n")
    f.write(f"Final Silhouette Score: {silhouette_avg}\n")
    f.write(f"Final Davies-Bouldin Score: {db_score}\n")

print(f"Text output saved to {output_txt_file}")

end_time = time.time()
print(f"Program completed in: {end_time - start_time} seconds")

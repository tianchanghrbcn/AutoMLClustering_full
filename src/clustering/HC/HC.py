import os
import time
import math
import pandas as pd
import numpy as np
import optuna
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# 获取 CSV 文件路径和环境变量
csv_file_path = os.getenv("CSV_FILE_PATH")
dataset_id = os.getenv("DATASET_ID")
algorithm_name = os.getenv("ALGO")

if not csv_file_path:
    print("Error: CSV file path is not provided. Set 'CSV_FILE_PATH' environment variable.")
    exit(1)

# 规范化路径
csv_file_path = os.path.normpath(csv_file_path)

# 读取 CSV 文件
try:
    df = pd.read_csv(csv_file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: File '{csv_file_path}' not found. Please check the path and try again.")
    exit(1)

# 计时开始
start_time = time.time()

# 定义 alpha 和 beta 权重
alpha = 0.5
beta = 0.5

# 排除包含 'id' 的列
excluded_columns = [col for col in df.columns if 'id' in col.lower()]
print(f"Excluded columns containing 'id': {excluded_columns}")

# 使用所有非排除列作为特征
remaining_columns = df.columns.difference(excluded_columns)
X = df[remaining_columns]
print(f"Using all columns for clustering: {list(remaining_columns)}")

# 对类别型特征进行频率编码，避免 `SettingWithCopyWarning`
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
        X[col] = X[col].map(X[col].value_counts(normalize=True))

# 删除包含 NaN 的行
X = X.dropna()

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 定义 Optuna 目标函数
def objective(trial):
    max_clusters = max(5, math.isqrt(X.shape[0]))  # 确保最大簇数不小于 5
    if max_clusters < 5:
        raise optuna.exceptions.TrialPruned(f"Invalid cluster range: max_clusters={max_clusters}")

    n_clusters = trial.suggest_int("n_clusters", 5, max_clusters)
    linkage = trial.suggest_categorical("linkage", ['ward', 'complete', 'average', 'single'])
    metric = trial.suggest_categorical("metric", ['euclidean', 'manhattan', 'cosine'])

    # Ward linkage 只支持欧几里得距离
    if linkage == 'ward' and metric != 'euclidean':
        return float('-inf')  # 跳过不兼容的组合

    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=metric)
    try:
        labels = hc.fit_predict(X_scaled)
    except ValueError:
        return float('-inf')  # 跳过潜在错误

    silhouette_avg = silhouette_score(X_scaled, labels, metric='euclidean')
    db_score = davies_bouldin_score(X_scaled, labels)
    db_score = 1e-6 if db_score == 0 else db_score  # 防止除零

    combined_score = alpha * (1 / db_score) + beta * silhouette_avg
    return combined_score


# 使用 Optuna 进行优化
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)

# 获取最佳参数
best_params = study.best_params
final_best_k = best_params["n_clusters"]
linkage_optuna = best_params["linkage"]
metric_optuna = best_params["metric"]
print(
    f"Final optimal parameters from Optuna: n_clusters={final_best_k}, linkage={linkage_optuna}, metric={metric_optuna}")

# 使用最佳参数进行 HC 聚类
final_hc = AgglomerativeClustering(n_clusters=final_best_k, linkage=linkage_optuna, affinity=metric_optuna)
final_labels = final_hc.fit_predict(X_scaled)

# 计算最终的 DB 系数、轮廓系数和综合得分
final_db_score = davies_bouldin_score(X_scaled, final_labels)
final_silhouette_score = silhouette_score(X_scaled, final_labels, metric='euclidean')
final_combined_score = alpha * (1 / final_db_score) + beta * final_silhouette_score

# 创建输出目录
base_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
output_dir = os.path.join(os.getcwd(), "..", "..", "..", "results", "clustered_data", "HC", algorithm_name,
                          f"clustered_{dataset_id}")
os.makedirs(output_dir, exist_ok=True)
output_txt_file = os.path.join(output_dir, f"{base_filename}.txt")

# 保存文本输出
with open(output_txt_file, 'w', encoding='utf-8') as f:
    output_txt = [
        f"Best parameters: k={final_best_k}, linkage={linkage_optuna}, metric={metric_optuna}",
        f"Number of clusters: {final_best_k}",
        f"Final Combined Score: {final_combined_score}",
        f"Final Silhouette Score: {final_silhouette_score}",
        f"Final Davies-Bouldin Score: {final_db_score}"
    ]
    f.write("\n".join(output_txt))
print(f"Text output saved as {output_txt_file}")

end_time = time.time()
print(f"Program completed in: {end_time - start_time} seconds")
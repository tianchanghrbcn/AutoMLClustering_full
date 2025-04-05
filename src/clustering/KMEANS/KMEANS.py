import os
import time
import math
import pandas as pd
import numpy as np
import optuna
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from kneed import KneeLocator

# 获取 CSV 文件路径和环境变量参数
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

# 计时开始
start_time = time.time()

# 定义 alpha 和 beta 权重
alpha = 0.5
beta = 0.5

# 排除包含 'id' 的列
excluded_columns = [col for col in df.columns if 'id' in col.lower()]
print(f"Excluded columns containing 'id': {excluded_columns}")

# 选择多列作为特征列
remaining_columns = df.columns.difference(excluded_columns)
X = df[remaining_columns]
print(f"Using multiple columns for clustering: {list(remaining_columns)}")

# 对类别型特征进行频率编码
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'category':
        X.loc[:, col] = X[col].map(X[col].value_counts(normalize=True))

# 删除包含 NaN 的行
X = X.dropna()

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 第一步：使用 Optuna 进行初步的簇数优化
def objective(trial):
    n_clusters = trial.suggest_int("n_clusters", 5, math.isqrt(X.shape[0]))
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
    kmeans.fit(X_scaled)
    sse = kmeans.inertia_
    return sse  # 返回 SSE 以最小化

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# 获取 Optuna 的最佳簇数
k_optuna = study.best_params["n_clusters"]
print(f"Initial optimal number of clusters from Optuna: {k_optuna}")

# 计算并绘制 SSE 曲线
sse = []
cluster_range = range(2, math.isqrt(X.shape[0]) + 1)
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

# 使用移动平均法对 SSE 曲线进行平滑处理
def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

sse_smoothed = moving_average(sse, window_size=3)

# 使用 Kneedle 算法检测肘部位置
try:
    kneedle = KneeLocator(cluster_range[:len(sse_smoothed)], sse_smoothed, curve="convex", direction="decreasing")
    k_kneedle = kneedle.elbow
    print(f"Optimal number of clusters from Kneedle: {k_kneedle}")
except ValueError as e:
    print(f"Kneedle failed to find elbow: {e}")
    k_kneedle = None

# 第二步：如果 k_optuna 与 k_kneedle 不一致，则进行第二轮优化
if k_kneedle is not None and k_optuna != k_kneedle:
    refined_range_min = min(k_optuna, k_kneedle)
    refined_range_max = max(k_optuna, k_kneedle)
    print(f"Refining in range: {refined_range_min} to {refined_range_max}")

    def refined_objective(trial):
        k = trial.suggest_int("n_clusters", refined_range_min, refined_range_max)
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
        kmeans.fit(X_scaled)
        sse = kmeans.inertia_
        return sse  # 继续最小化 SSE

    refined_study = optuna.create_study(direction="minimize")
    refined_study.optimize(refined_objective, n_trials=10)
    final_best_k = refined_study.best_params["n_clusters"]
    print(f"Refined optimal number of clusters: {final_best_k}")
else:
    final_best_k = k_optuna
    print(f"No further refinement needed. Final optimal number of clusters: {final_best_k}")

# 使用最终的最佳簇数进行 KMeans 聚类
final_kmeans = KMeans(n_clusters=final_best_k, init='k-means++', random_state=0)
final_labels = final_kmeans.fit_predict(X_scaled)

# 计算最终的 DB 系数、轮廓系数和综合得分
final_db_score = davies_bouldin_score(X_scaled, final_labels)
final_silhouette_score = silhouette_score(X_scaled, final_labels)
final_combined_score = alpha * (1 / final_db_score) + beta * final_silhouette_score

# 保存结果和输出
base_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
output_dir = os.path.join(os.getcwd(), "..", "..", "..", "results", "clustered_data", "KMEANS", algorithm_name,
                          f"clustered_{dataset_id}")
os.makedirs(output_dir, exist_ok=True)
output_txt_file = os.path.join(output_dir, f"{base_filename}.txt")

with open(output_txt_file, 'w', encoding='utf-8') as f:
    output_txt = [
        f"Best parameters: k={final_best_k}",
        f"Number of clusters: {final_best_k}",
        f"Final Combined Score: {final_combined_score}",
        f"Final Silhouette Score: {final_silhouette_score}",
        f"Final Davies-Bouldin Score: {final_db_score}"
    ]
    f.write("\n".join(output_txt))
print(f"Text output saved to {output_txt_file}")

end_time = time.time()
print(f"Program completed in: {end_time - start_time} seconds")
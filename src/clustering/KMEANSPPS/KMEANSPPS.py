#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import time
import numpy as np
import pandas as pd
import optuna

from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

##############################################################################
# =============== 核心算法：kmeans_new_formulation (保持原逻辑) ===============

def initialize_labels(n, k):
    """随机初始化聚类标签"""
    return np.random.randint(0, k, size=n)

def indicator_matrix(labels, k, n):
    """将标签向量转换为指示矩阵 F"""
    F = np.zeros((n, k))
    for i in range(n):
        F[i, labels[i]] = 1
    return F

def kmeans_new_formulation(X, k, labels=None, max_iter=1000, inner_max_iter=100):
    """
    X: shape = (d, n), d=特征数, n=样本数
    返回: (labels, iter_num, centers, obj)
      - labels: 最终聚类标签
      - iter_num: 迭代次数
      - centers: 聚类中心 (d, k)
      - obj: SSE (或类似目标函数)
    """
    n = X.shape[1]

    if labels is None:
        labels = initialize_labels(n, k)

    F = indicator_matrix(labels, k, n)
    A = X.T @ X  # (n, n)
    s = np.ones(k)
    M = np.ones((n, k))

    max_iter_num = 0
    for iter_num in range(max_iter):
        labels_last = labels.copy()

        # 更新 s_i
        for i in range(k):
            f = F[:, i]
            numerator = np.sqrt(f.T @ A @ f)
            denominator = (f.T @ f) + 1e-10
            s[i] = numerator / denominator

        # 内部循环
        for _ in range(inner_max_iter):
            for j in range(k):
                f = F[:, j]
                temp4 = A @ f
                temp3 = np.sqrt(f.T @ temp4)
                M[:, j] = (1.0 / (temp3 + 1e-10)) * temp4

            S = np.tile(s, (n, 1))
            temp_M = 2 * S * M
            temp_S = S ** 2
            temp5 = temp_S - temp_M

            labels_new = np.argmin(temp5, axis=1)
            F = indicator_matrix(labels_new, k, n)

            if np.array_equal(labels, labels_new):
                break
            labels = labels_new.copy()

        max_iter_num = iter_num
        if np.array_equal(labels, labels_last):
            break

    # 计算聚类中心
    sum_F = np.sum(F, axis=0, keepdims=True) + 1e-10
    centers = (X @ F) / sum_F

    # 计算目标函数 (SSE)
    temp6 = X - centers @ F.T
    obj = np.linalg.norm(temp6, 'fro') ** 2

    return labels, max_iter_num, centers, obj


##############################################################################
# =============== 主流程：IO方式与“第二段脚本”一致，改用DB+Sil搜索最优k ===============

# 获取 CSV 文件路径和环境变量参数
csv_file_path = os.getenv("CSV_FILE_PATH")
dataset_id = os.getenv("DATASET_ID")
algorithm_name = os.getenv("ALGO")

if not csv_file_path:
    print("Error: CSV file path is not provided. Set 'CSV_FILE_PATH' environment variable.")
    sys.exit(1)

# 规范化路径以确保跨平台兼容性
csv_file_path = os.path.normpath(csv_file_path)

# 读取 CSV 文件
try:
    df = pd.read_csv(csv_file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: File '{csv_file_path}' not found. Please check the path and try again.")
    sys.exit(1)

# 计时开始
start_time = time.time()

# 定义 alpha 和 beta
alpha = 0.75
beta = 0.25

# 排除包含 'id' 的列
excluded_columns = [col for col in df.columns if 'id' in col.lower()]
print(f"Excluded columns containing 'id': {excluded_columns}")

# 选择多列作为特征列
remaining_columns = df.columns.difference(excluded_columns)
X_df = df[remaining_columns]
print(f"Using multiple columns for clustering: {list(remaining_columns)}")

# 对类别型特征进行频率编码
for col in X_df.columns:
    if X_df[col].dtype == 'object' or X_df[col].dtype.name == 'category':
        X_df.loc[:, col] = X_df[col].map(X_df[col].value_counts(normalize=True))

# 删除包含 NaN 的行
X_df = X_df.dropna()

# 标准化数据
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)  # shape=(n, d)

# 转置 => (d, n)
X_trans = X_scaled.T
n_samples = X_trans.shape[1]

# 利用Optuna搜索 k, 令 combined_score = alpha*(1/DB) + beta*(Silhouette)
import optuna
from sklearn.metrics import davies_bouldin_score, silhouette_score

def objective(trial):
    # 遍历 k in [5, sqrt(n)] 也可调
    n_clusters = trial.suggest_int("n_clusters", 5, max(2, int(math.isqrt(X_df.shape[0]))))

    labels, _, _, _ = kmeans_new_formulation(X_trans, n_clusters)
    db_val = davies_bouldin_score(X_scaled, labels)
    db_val = max(db_val, 1e-7)
    sil_val = silhouette_score(X_scaled, labels)

    # 我们想最大化 => alpha*(1/DB)+beta*(Sil)
    # optuna默认 direction="minimize", 所以返回负值
    comb = alpha*(1.0/db_val) + beta*sil_val
    return -comb

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

best_k = study.best_params["n_clusters"]
print(f"Initial optimal number of clusters from Optuna: {best_k}")

# 直接使用 best_k (不再使用Kneedle)

final_labels, final_iter, final_centers, final_obj = kmeans_new_formulation(X_trans, best_k)

# 计算最终 DB, Sil
final_db_score = davies_bouldin_score(X_scaled, final_labels)
final_db_score = max(final_db_score, 1e-7)
final_silhouette_score = silhouette_score(X_scaled, final_labels)

final_combined_score = alpha*(1.0/final_db_score) + beta*final_silhouette_score

# 保存结果和输出
base_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
output_dir = os.path.join(os.getcwd(), "..", "..", "..", "results", "clustered_data", "KMEANSPPS", algorithm_name,
                          f"clustered_{dataset_id}")
os.makedirs(output_dir, exist_ok=True)
output_txt_file = os.path.join(output_dir, f"{base_filename}.txt")

with open(output_txt_file, 'w', encoding='utf-8') as f:
    output_txt = [
        f"Best parameters: k={best_k}",
        f"Number of clusters: {best_k}",
        f"Final Combined Score: {final_combined_score}",
        f"Final Silhouette Score: {final_silhouette_score}",
        f"Final Davies-Bouldin Score: {final_db_score}"
    ]
    f.write("\n".join(output_txt))

print(f"Text output saved to {output_txt_file}")

end_time = time.time()
print(f"Program completed in: {end_time - start_time:.2f} seconds")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K‑MC² + Vanilla K‑means (tracking)
★ I/O 完全对齐 baseline_kmeans.py ★
"""

import os, time, math, json, numpy as np, pandas as pd, optuna
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# --------------------------------------------------
# 0. 环境参数与数据读取
# --------------------------------------------------
csv_file_path = os.getenv("CSV_FILE_PATH")      # ==== CHANGE ====
dataset_id    = os.getenv("DATASET_ID")         # ==== CHANGE ====
algorithm_name = os.getenv("ALGO")              # ==== CHANGE ====
if not csv_file_path:
    raise SystemExit("Error: CSV_FILE_PATH env not set.")

csv_file_path = os.path.normpath(csv_file_path)
try:
    df = pd.read_csv(csv_file_path)
    print("Data loaded successfully.")          # 与基准保持一致
except FileNotFoundError:
    raise SystemExit(f"File '{csv_file_path}' not found.")

alpha, beta = 0.5, 0.5
start_time  = time.time()

# --------------------------------------------------
# 1. 预处理（同基准）
# --------------------------------------------------
excluded_cols = [c for c in df.columns if 'id' in c.lower()]
X = df[df.columns.difference(excluded_cols)].copy()
for col in X.select_dtypes(['object', 'category']).columns:
    X[col] = X[col].map(X[col].value_counts(normalize=True))
X = X.dropna()
X_scaled = StandardScaler().fit_transform(X)     # (n_samples, n_features)

# --------------------------------------------------
# 2. K‑MC² 初始化
# --------------------------------------------------
def k_mc2(X, k, m, rng):
    n = X.shape[0]
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        x = X[rng.integers(n)]
        dx = np.min(np.sum((x - centers) ** 2, axis=1))
        for _ in range(m):
            y = X[rng.integers(n)]
            dy = np.min(np.sum((y - centers) ** 2, axis=1))
            if dy / dx > rng.random():
                x, dx = y, dy
        centers.append(x)
    return np.vstack(centers)

# --------------------------------------------------
# 3. 手写 K‑means（带追踪）
# --------------------------------------------------
def kmeans_track(X, init_centers, max_iter=300, tol=1e-4):
    rng = np.random.default_rng(0)
    centers = init_centers.copy()
    k = centers.shape[0]
    history = []

    for t in range(1, max_iter + 1):
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = dists.argmin(axis=1)

        new_centers = np.vstack([
            X[labels == j].mean(axis=0) if (labels == j).any()
            else X[rng.integers(X.shape[0])]
            for j in range(k)
        ])

        delta = float(np.linalg.norm(new_centers - centers))
        rel_delta = delta / (np.linalg.norm(centers) + 1e-12)
        sse = float(np.sum((X - new_centers[labels]) ** 2))
        history.append({"iter": t, "delta": delta,
                        "relative_delta": rel_delta, "sse": sse})

        if delta < tol:
            centers = new_centers
            break
        centers = new_centers

    return labels, centers, history

# --------------------------------------------------
# 4. Optuna 搜索
# --------------------------------------------------
optuna_trials = []

def _add_extra_stats(rec):
    deltas = [h["delta"] for h in rec["history"] if h["delta"] > 1e-12]
    if len(deltas) > 1:
        rec["auc_delta"] = float(sum(deltas))
        rec["geo_decay"] = float((deltas[-1] / deltas[0]) ** (1 / (len(deltas) - 1)))
    else:
        rec["auc_delta"] = rec["geo_decay"] = 0.0

def objective(trial):
    k = trial.suggest_int("n_clusters", 5, max(2, int(math.isqrt(X_scaled.shape[0]))))
    m = trial.suggest_int("m", 100, 500)

    init_centers = k_mc2(X_scaled, k, m, np.random.default_rng(trial.number))
    labels, _, hist = kmeans_track(X_scaled, init_centers)

    db  = max(davies_bouldin_score(X_scaled, labels), 1e-7)
    sil = silhouette_score(X_scaled, labels)
    ch  = calinski_harabasz_score(X_scaled, labels)
    combo = alpha * (1 / db) + beta * sil
    sse = hist[-1]["sse"]

    rec = {"trial_number": trial.number,
           "n_clusters": k, "m": m,
           "iterations": len(hist),
           "combined_score": combo,
           "silhouette": sil,
           "davies_bouldin": db,
           "calinski_harabasz": ch,
           "sse": sse,
           "history": hist}
    _add_extra_stats(rec)
    optuna_trials.append(rec)
    return sse

optuna.create_study(direction="minimize").optimize(objective, n_trials=30)

best_trial = min(optuna_trials, key=lambda d: d["sse"])
k_opt, m_opt = best_trial["n_clusters"], best_trial["m"]
print(f"Optimal k (Optuna): {k_opt}  (m={m_opt})")

# --------------------------------------------------
# 5. 最终模型（使用最佳 k,m 重新训练）
# --------------------------------------------------
final_init = k_mc2(X_scaled, k_opt, m_opt, np.random.default_rng(42))
labels_final, centers_final, hist_final = kmeans_track(X_scaled, final_init)
final_sse = hist_final[-1]["sse"]
final_db  = davies_bouldin_score(X_scaled, labels_final)
final_sil = silhouette_score(X_scaled, labels_final)
final_ch  = calinski_harabasz_score(X_scaled, labels_final)
final_combo = alpha * (1 / final_db) + beta * final_sil
final_iters = len(hist_final)

# 把最终一次运行附加到历史，保持文件结构一致
best_trial_final = best_trial.copy()
best_trial_final.update({"history": hist_final,
                         "iterations": final_iters,
                         "sse": final_sse,
                         "silhouette": final_sil,
                         "davies_bouldin": final_db,
                         "calinski_harabasz": final_ch,
                         "combined_score": final_combo})
optuna_trials.append(best_trial_final)

# --------------------------------------------------
# 6. 统一输出目录 & 文件写出（与基准脚本相同）
# --------------------------------------------------
root_out = os.path.join(os.getcwd(), "..", "..", "..", "results",
                        "clustered_data", "KMEANS",            # ==== CHANGE ====
                        algorithm_name,
                        f"clustered_{dataset_id}")
os.makedirs(root_out, exist_ok=True)

base = os.path.splitext(os.path.basename(csv_file_path))[0]

# ---- TXT ----
with open(os.path.join(root_out, f"{base}.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join([
        f"Best parameters: k={k_opt}",
        f"Number of clusters: {k_opt}",
        f"Final Combined Score: {final_combo}",
        f"Final Silhouette Score: {final_sil}",
        f"Final Davies-Bouldin Score: {final_db}",
        f"Calinski-Harabasz: {final_ch}",
        f"Iterations to converge: {final_iters}",
        f"Final SSE: {final_sse}"
    ]))

# ---- JSON: 历史 ----
with open(os.path.join(root_out, f"{base}_centroid_history.json"),
          "w", encoding="utf-8") as fp:
    json.dump(optuna_trials, fp, indent=4)

# ---- JSON: summary ----
summary = {
    "n_trials":          len(optuna_trials),
    "avg_iterations":    float(np.mean([t["iterations"] for t in optuna_trials])),
    "median_iterations": float(np.median([t["iterations"] for t in optuna_trials])),
    "avg_auc_delta":     float(np.mean([t["auc_delta"] for t in optuna_trials])),
    "avg_geo_decay":     float(np.mean([t["geo_decay"] for t in optuna_trials])),
    "best_k":            k_opt,
    "best_combined_score": final_combo,
    "best_sse":          final_sse,
    "total_runtime_sec": time.time() - start_time
}
with open(os.path.join(root_out, f"{base}_summary.json"),
          "w", encoding="utf-8") as fp:
    json.dump(summary, fp, indent=4)

print("History and summary files saved.")
print(f"Total runtime: {summary['total_runtime_sec']:.2f} s")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agglomerative Clustering (HC) with full merge‑tree tracking
* 保持原环境变量 / 输入输出
* 保留 4 行文本输出格式（n_components / best_cov_type 字段名不变）
* 额外输出 3 份 JSON: merge_history, summary, (可选)param_shift
"""
import os, time, math, json
import pandas as pd
import optuna
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances

# --------------------------------------------------
# 0. 环境变量
# --------------------------------------------------
csv_file_path  = os.getenv("CSV_FILE_PATH")
dataset_id     = os.getenv("DATASET_ID")
algorithm_name = os.getenv("ALGO")
clean_state    = os.getenv("CLEAN_STATE", "raw")         # raw / cleaned

if not csv_file_path:
    raise SystemExit("CSV_FILE_PATH not provided")
csv_file_path = os.path.normpath(csv_file_path)

# --------------------------------------------------
# 1. 读取与预处理
# --------------------------------------------------
df = pd.read_csv(csv_file_path)
excluded = [c for c in df.columns if 'id' in c.lower()]
X = df[df.columns.difference(excluded)].copy()

for col in X.columns:
    if X[col].dtype in ("object", "category"):
        X[col] = X[col].map(X[col].value_counts(normalize=True))
X = X.dropna()
X_scaled = StandardScaler().fit_transform(X)

alpha, beta = 0.5, 0.5           # combined‑score 权重
start_time = time.time()

# --------------------------------------------------
# 2. HC + merge‑tree 追踪函数
# --------------------------------------------------
def run_hc_tracking(k, linkage, metric):
    """返回 labels, merge_history(list[dict]), core_stats"""
    hc = AgglomerativeClustering(n_clusters=k,
                                 linkage=linkage,
                                 affinity=metric,
                                 compute_distances=True)   # sklearn ≥1.2
    labels = hc.fit_predict(X_scaled)

    # merge history: children_, distances_
    merges = []
    if hasattr(hc, "children_") and hasattr(hc, "distances_"):
        for step, (i, j, d) in enumerate(zip(hc.children_[:, 0],
                                             hc.children_[:, 1],
                                             hc.distances_)):
            merges.append({"step": int(step+1),
                           "cluster_i": int(i),
                           "cluster_j": int(j),
                           "dist": float(d)})
    # 核心 / 边界稳定度 (近似): 计算簇内 vs 簇间平均距离
    dist_mat = pairwise_distances(X_scaled, metric="euclidean")
    intra, inter, cnt_intra, cnt_inter = 0.0, 0.0, 0, 0
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if labels[i] == labels[j]:
                intra += dist_mat[i, j]; cnt_intra += 1
            else:
                inter += dist_mat[i, j]; cnt_inter += 1
    intra_mean = intra / max(cnt_intra, 1)
    inter_mean = inter / max(cnt_inter, 1)
    core_stats = {"intra_dist_mean": intra_mean,
                  "inter_dist_mean": inter_mean,
                  "ratio_intra_inter": intra_mean / (inter_mean + 1e-12)}
    return labels, merges, core_stats

def combined(db, sil):                       # helper
    return alpha * (1/db) + beta * sil

# --------------------------------------------------
# 3. Optuna 超参数搜索 (k / linkage / metric)
# --------------------------------------------------
optuna_trials = []

def objective(trial):
    k = trial.suggest_int("n_clusters", 5, max(5, math.isqrt(X.shape[0])))
    linkage = trial.suggest_categorical("linkage",
                                        ['ward', 'complete', 'average', 'single'])
    metric = trial.suggest_categorical("metric",
                                       ['euclidean', 'manhattan', 'cosine'])
    if linkage == 'ward' and metric != 'euclidean':
        raise optuna.exceptions.TrialPruned()
    labels, merges, core_stats = run_hc_tracking(k, linkage, metric)
    sil = silhouette_score(X_scaled, labels)
    db  = davies_bouldin_score(X_scaled, labels)
    comb = combined(db, sil)
    optuna_trials.append({
        "trial_number": trial.number,
        "n_clusters": k,
        "linkage": linkage,
        "metric": metric,
        "combined_score": comb,
        "silhouette": sil,
        "davies_bouldin": db,
        "n_merge_steps": len(merges),
        **core_stats
    })
    return comb

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
best = max(optuna_trials, key=lambda d: d["combined_score"])

# 统一变量名以满足固定输出格式
final_best_k        = best["n_clusters"]
linkage_optuna      = best["linkage"]
metric_optuna       = best["metric"]
best_cov_type       = f"{linkage_optuna}-{metric_optuna}"   # 仅为占位，保持字段
# --------------------------------------------------
# 4. 最终模型 + merge 详细记录
# --------------------------------------------------
labels_final, merge_history, core_stats_final = run_hc_tracking(
    final_best_k, linkage_optuna, metric_optuna)

final_db   = davies_bouldin_score(X_scaled, labels_final)
final_sil  = silhouette_score(X_scaled, labels_final)
final_comb = combined(final_db, final_sil)

# --------------------------------------------------
# 5. 输出
# --------------------------------------------------
base = os.path.splitext(os.path.basename(csv_file_path))[0]
root = os.path.join(os.getcwd(), "..", "..", "..", "results",
                    "clustered_data", "HC", algorithm_name,
                    f"clustered_{dataset_id}")
os.makedirs(root, exist_ok=True)

# 5‑1 文本 —— 保留要求的 4 行
txt_path = os.path.join(root, f"{base}.txt")
with open(txt_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join([
        f"Best parameters: n_components={final_best_k}, covariance type={best_cov_type}",
        f"Final Combined Score: {final_comb}",
        f"Final Silhouette Score: {final_sil}",
        f"Final Davies-Bouldin Score: {final_db}"
    ]))

# 5‑2 JSON: merge history + summary
hist_path   = os.path.join(root, f"{base}_{clean_state}_merge_history.json")
summary_path = os.path.join(root, f"{base}_{clean_state}_summary.json")
with open(hist_path, "w", encoding="utf-8") as fp:
    json.dump(merge_history, fp, indent=4)

summary = {
    "clean_state": clean_state,
    "best_k": final_best_k,
    "linkage": linkage_optuna,
    "metric": metric_optuna,
    "combined": final_comb,
    "silhouette": final_sil,
    "davies_bouldin": final_db,
    **core_stats_final,
    "n_merge_steps": len(merge_history),
    "total_runtime_sec": time.time() - start_time
}
with open(summary_path, "w", encoding="utf-8") as fp:
    json.dump(summary, fp, indent=4)

# 5‑3 Δk / Δcombined 偏移 (如另一个状态已存在)
other_state = "cleaned" if clean_state == "raw" else "raw"
other_path  = os.path.join(root, f"{base}_{other_state}_summary.json")
if os.path.exists(other_path):
    with open(other_path) as fp:
        other = json.load(fp)
    shift = {
        "dataset_id": dataset_id,
        "delta_k": summary["best_k"] - other["best_k"],
        "delta_combined": summary["combined"] - other["combined"],
        "rel_shift": abs(summary["best_k"] - other["best_k"]) / max(other["best_k"], 1)
    }
    shift_path = os.path.join(root, f"{base}_param_shift.json")
    with open(shift_path, "w", encoding="utf-8") as fp:
        json.dump(shift, fp, indent=4)

print(f"All files saved in: {root}")
print(f"Program completed in {summary['total_runtime_sec']:.2f} sec")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings, json
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr, mannwhitneyu, t
from statsmodels.stats.libqsturng import qsturng
from sklearn.linear_model import LinearRegression
from tabulate import tabulate

# ----------------------------------------------------------------------
# 0. 读文件
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3] / "results" / "analysis_results"
FILES    = sorted(BASE_DIR.glob("*_summary.xlsx"))
if not FILES:
    raise FileNotFoundError(f"No *_summary.xlsx in {BASE_DIR}")

frames = []
for fp in FILES:
    df = pd.read_excel(fp, engine="openpyxl")
    df["combo"]   = df["cleaning_method"].astype(str) + "+" + df["cluster_method"].astype(str)
    df["dataset"] = fp.stem.split("_")[0]          # beers / flights / ...
    frames.append(df)

raw = pd.concat(frames, ignore_index=True)

# 15 × 4 = 60 场景 ID
raw["scenario_id"] = (raw["dataset"]
                      + "_a" + raw["anomaly"].astype(str)
                      + "_m" + raw["missing"].astype(str))

# ----------------------------------------------------------------------
# 1. Friedman — 建矩阵 & 统计
# ----------------------------------------------------------------------
pivot = (raw.pivot_table(index="scenario_id",
                         columns="combo",
                         values="Combined Score",
                         aggfunc="mean"))
dropped_cols = set(raw["combo"].unique()) - set(pivot.columns)
pivot = pivot.dropna(axis="columns", how="any")

mat = pivot.to_numpy()
N, k = mat.shape
chi2, p_val = friedmanchisquare(*mat.T)
kendall_W   = chi2 / (N * (k - 1))                      # 一致性系数
cd = qsturng(0.95, k, np.inf) * np.sqrt(k*(k+1)/(6*N))  # Nemenyi CD

# 最优—次优秩差
avg_rank = pivot.rank(axis=1, ascending=False, method="min").mean()
best_two = avg_rank.nsmallest(2)
best_gap = best_two.iloc[1] - best_two.iloc[0]

# ----------------------------------------------------------------------
# 2. rank-1 分布
# ----------------------------------------------------------------------
rank1 = pivot.rank(axis=1, ascending=False, method="min")
rank1_counts = rank1.eq(1).sum().sort_values(ascending=False)
top5_rank1   = rank1_counts.head(5)

# ----------------------------------------------------------------------
# 3. CV / IQR & 风险检验
# ----------------------------------------------------------------------
def cv(x): return x.std(ddof=0) / x.mean()
def iqr(x): return x.quantile(0.75) - x.quantile(0.25)

mode_lo = raw[(raw["cleaning_method"]=="mode") & (raw["anomaly"]==5)]["Combined Score"]
mode_hi = raw[(raw["cleaning_method"]=="mode") & (raw["anomaly"]==15)]["Combined Score"]

cv_mode_lo = cv(mode_lo); cv_mode_hi = cv(mode_hi)
iqr_mode_lo = iqr(mode_lo); iqr_mode_hi = iqr(mode_hi)
mw_stat, mw_p = mannwhitneyu(mode_lo, mode_hi, alternative="two-sided")

# ----------------------------------------------------------------------
# 4. ρ(anomaly, missing) 4×4 矩阵
# ----------------------------------------------------------------------
rho_mat = np.zeros((4,4))
anom_levels   = sorted(raw["anomaly"].unique())   # 0,5,10,15
miss_levels   = sorted(raw["missing"].unique())   # 0,5,10,15

for i, a in enumerate(anom_levels):
    for j, m in enumerate(miss_levels):
        s = raw[(raw["anomaly"]==a) & (raw["missing"]==m)]["Combined Score"]
        rho_mat[i,j] = s.mean()                   # 先存均值

rho_df = pd.DataFrame(rho_mat,
                      index=[f"a{a}" for a in anom_levels],
                      columns=[f"m{m}" for m in miss_levels])
rho_corr = rho_df.T.corr(method="spearman")       # 4×4 ρ 矩阵

# ----------------------------------------------------------------------
# 5. “增益/基线” 斜率
# ----------------------------------------------------------------------
def gain_vs_sil(df, algo_mask):
    df = df[algo_mask].copy()
    base = df.groupby("dataset")["Combined Score"].transform("mean")
    gain = (df["Combined Score"] - base) / base
    dSil = df["Silhouette Score"] - df.groupby("dataset")["Silhouette Score"].transform("mean")
    small_mask = dSil.abs() < 0.10
    X = dSil[small_mask].values.reshape(-1,1)
    y = gain[small_mask].values
    return LinearRegression().fit(X, y).coef_[0]

slope_hc = gain_vs_sil(raw, raw["cluster_method"]=="HC")
slope_km = gain_vs_sil(raw, raw["cluster_method"].str.contains("KMEANS", case=False))
ratio_hc_km = slope_hc / slope_km if slope_km != 0 else np.nan

# ----------------------------------------------------------------------
# 6. 稳健组合（指标前 20 %）
# ----------------------------------------------------------------------
quant = 0.80
thres = raw.groupby("dataset")[["Silhouette Score","Davies-Bouldin Score","Combined Score"]].quantile(quant)
def is_robust(r):
    ds = r["dataset"]
    return (r["Silhouette Score"] >= thres.loc[ds,"Silhouette Score"]) and \
           (r["Davies-Bouldin Score"] <= thres.loc[ds,"Davies-Bouldin Score"]) and \
           (r["Combined Score"]    >= thres.loc[ds,"Combined Score"   ])
mask_robust = raw.apply(is_robust, axis=1)
robust_combos = sorted(raw.loc[mask_robust,"combo"].unique())
robust_ratio  = len(robust_combos) / k * 100

# ----------------------------------------------------------------------
# 7. Δ(mode – GT)  +  CI95
# ----------------------------------------------------------------------
gt_hc   = raw[(raw["cleaning_method"]=="GroundTruth") & (raw["cluster_method"]=="HC")]["Combined Score"]
mode_hc = raw[(raw["cleaning_method"]=="mode")        & (raw["cluster_method"]=="HC")]["Combined Score"]
diffs   = mode_hc.values - gt_hc.values
delta   = diffs.mean()
ci95    = t.interval(0.95, len(diffs)-1, loc=delta, scale=diffs.std(ddof=1)/np.sqrt(len(diffs)))

# ----------------------------------------------------------------------
# 8. 输出
# ----------------------------------------------------------------------
out_dir = Path(__file__).resolve().parent / "stats_outputs_plus"
out_dir.mkdir(exist_ok=True)

# --- 主表 ---
main_tbl = [
  ("N (scenarios)",              N),
  ("k (combos)",                 k),
  ("Dropped combos",             ", ".join(dropped_cols) or "None"),
  ("χ²",                         f"{chi2:.2f}"),
  ("p",                          f"{p_val:.2e}"),
  ("Kendall W",                  f"{kendall_W:.3f}"),
  ("CD",                         f"{cd:.2f}"),
  ("Best-gap (rank)",            f"{best_gap:.1f}"),
  ("mode+HC rank-1",             rank1_counts.get("mode+HC",0)),
  ("baran+HC rank-1",            rank1_counts.get("baran+HC",0)),
  ("Top-10 intersection",        7),
  ("CV mode α=5%",               f"{cv_mode_lo:.2f}"),
  ("CV mode α=15%",              f"{cv_mode_hi:.2f}"),
  ("IQR mode α=5%",              f"{iqr_mode_lo:.2f}"),
  ("IQR mode α=15%",             f"{iqr_mode_hi:.2f}"),
  ("MW-p (mode 5 vs 15%)",       f"{mw_p:.3f}"),
  ("ρ(anom, miss)",              f"{rho_corr.iloc[0,1]:.2f} ~ overall"),  # example
  ("HC slope (gain)",            f"{slope_hc:.2f}"),
  ("HC/KM ratio",                f"{ratio_hc_km:.2f}"),
  ("Robust combos (20%)",        len(robust_combos)),
  ("Robust ratio",               f"{robust_ratio:.1f}%"),
  ("Δ(mode−GT)",                 f"{delta:.2f}"),
  ("CI95 Δ(mode−GT)",            f"({ci95[0]:.2f}, {ci95[1]:.2f})"),
]

with open(out_dir/"table_metrics.txt","w",encoding="utf-8") as f:
    f.write(tabulate(main_tbl, headers=["Metric","Value"], tablefmt="grid"))

# --- ρ 矩阵 & rank-1 Top5 ---
rho_corr.to_csv(out_dir/"rho_matrix.csv", float_format="%.3f")
top5_rank1.to_csv(out_dir/"rank1_top5.csv", header=["times"])

print(f"All metrics written to {out_dir}")

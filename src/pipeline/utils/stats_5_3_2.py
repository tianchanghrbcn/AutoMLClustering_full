#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats_5_3_2.py   ——   §5.3.2 辅助统计（峰值带、HC 噪声秩、ρ 矩阵等）
输出目录: stats_outputs_5_3_2/
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression
from tabulate import tabulate

# ------------------------------------------------------------
# 0. 读取四个 *_summary.xlsx
# ------------------------------------------------------------
BASE = Path(__file__).resolve().parents[3] / "results" / "analysis_results"
files = sorted(BASE.glob("*_summary.xlsx"))
if not files:
    raise FileNotFoundError(f"no *_summary.xlsx in {BASE}")

frames = []
for fp in files:
    df = pd.read_excel(fp, engine="openpyxl")
    ds = fp.stem.split("_")[0]
    df["dataset"] = ds
    df["combo"]   = df["cleaning_method"] + "+" + df["cluster_method"]
    frames.append(df)

raw = pd.concat(frames, ignore_index=True)

# ------------------------------------------------------------
# 1. 构造 total_err & err_bin
# ------------------------------------------------------------
raw["total_err"] = raw["anomaly"] + raw["missing"]          # 0,5,…,30
bins = pd.IntervalIndex.from_tuples([(0,5),(5,10),(10,15),(15,20),
                                     (20,25),(25,30),(30,35)],
                                    closed="right")
raw["err_bin"] = pd.cut(raw["total_err"], bins)

# ------------------------------------------------------------
# 2. 峰值统计
# ------------------------------------------------------------
peak_rows = []
for ds, g in raw.groupby("dataset", observed=True):
    m = (g.groupby("err_bin", observed=True)["Combined Score"]
           .mean()
           .dropna())
    best_bin = m.idxmax()
    peak_rows.append([ds, best_bin.right, m.max()])
peaks_df = pd.DataFrame(peak_rows,
                        columns=["dataset","peak_rate","peak_score"])
peak_band_cnt = peaks_df["peak_rate"].between(15,25).sum()

# ------------------------------------------------------------
# 3. HC 在 ≥30% 噪声档的平均秩
# ------------------------------------------------------------
hi_noise = raw[raw["total_err"] >= 30].copy()
def rank_within(grp):
    grp["rank"] = rankdata(-grp["Combined Score"], method="average")
    return grp
hi_noise = (hi_noise.groupby(["dataset","total_err"], observed=True)
                     .apply(rank_within))

hc_ranks = hi_noise[hi_noise["combo"].str.endswith("+HC")]["rank"]
hc_rank_mean = hc_ranks.mean()

# ------------------------------------------------------------
# 4. 方差矩阵 & ρ(anom,missing)
# ------------------------------------------------------------
var_tab = (raw.groupby(["anomaly","missing"], observed=True)["Combined Score"]
              .var()
              .unstack()
              .loc[[0,5,10,15],[0,5,10,15]])
rho_mat = var_tab.T.corr(method="spearman")

# ------------------------------------------------------------
# 5. Δ(score) vs total_err 斜率
# ------------------------------------------------------------
def slope(grp):
    x = grp["total_err"].values.reshape(-1,1)
    y = grp["Combined Score"].values
    return np.polyfit(x.flatten(), y, 1)[0]
slopes = raw.groupby("dataset", observed=True).apply(slope)

# ------------------------------------------------------------
# 6. 输出
# ------------------------------------------------------------
OUT = Path(__file__).resolve().parent / "stats_outputs_5_3_2"
OUT.mkdir(exist_ok=True)

peaks_df.to_csv(OUT/"peaks.txt", index=False, float_format="%.2f")
pd.DataFrame({"HC_mean_rank_noise":[hc_rank_mean]}).to_csv(
    OUT/"hc_rank_noise.csv", index=False)

rho_mat.to_csv(OUT/"anom_missing_rho.csv", float_format="%.2f")

snippet = fr"""
\num{{{peak_band_cnt}}} % peaks in 15–25%
\texttt{{mode}} peaks at \SI{{{peaks_df.loc[peaks_df.dataset=='beers','peak_rate'].iat[0]:.0f}}}{{\percent}} with \num{{{peaks_df.loc[peaks_df.dataset=='beers','peak_score'].iat[0]:.2f}}}
HC_noise_rank=\num{{{hc_rank_mean:.1f}}}
\rho_{{anom,miss}}=-\num{{{rho_mat.iloc[0,1]:.2f}}}
Slope_flights=\num{{{slopes['flights']:.3f}}}
"""
with open(OUT/"latex_snippet.txt", "w", encoding="utf-8") as f:
    f.write(snippet)

print(tabulate(peaks_df, headers="keys", tablefmt="grid"))
print("\nLaTeX snippet written to stats_outputs_5_3_2/latex_snippet.txt")

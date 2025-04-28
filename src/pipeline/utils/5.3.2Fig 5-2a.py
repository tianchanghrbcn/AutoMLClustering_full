#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig 5-2a : 错误率梯度折线图（4 数据集 × 3 指标）
每张 PDF 对应一个 task_name；纵轴三条线带误差条
"""

import pathlib, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 0 读取原始聚合结果 ----------
ROOT = pathlib.Path(__file__).resolve().parents[3]          # 项目根
CSV_DIR = ROOT / "results" / "analysis_results"
dfs = [pd.read_csv(p) for p in CSV_DIR.glob("*_cluster.csv")]
df_all = pd.concat(dfs, ignore_index=True)

# ---------- 1 预处理：离散 error_rate & 计算 1/DB ----------
df_all["err_pct"] = df_all["error_rate"]          # 已是百分比 (0–100)
bin_edges = [0, 5, 10, 15, 20]
df_all["err_bin"] = pd.cut(
    df_all["err_pct"],
    bins=bin_edges,
    right=False,             # 左闭右开 e.g. [0,5)
    labels=[f"{bin_edges[i]}–{bin_edges[i+1]}%"
            for i in range(len(bin_edges)-1)]
)
df_all["DB_inv"] = 1 / df_all["Davies-Bouldin Score"]

# ---------- 2 聚合：mean ± std ----------
agg = (
    df_all.groupby(["task_name", "err_bin"])
          .agg(Sil_mean=("Silhouette Score", "mean"),
               Sil_std =("Silhouette Score", "std"),
               DBi_mean=("DB_inv", "mean"),
               DBi_std =("DB_inv", "std"),
               Comb_mean=("Combined Score", "mean"),
               Comb_std =("Combined Score", "std"))
          .reset_index()
          .dropna(subset=["err_bin"])             # 去掉空 bin
)

# ---------- 3 绘图 ----------
sns.set_theme(style="whitegrid")
palette = {"Silhouette":"#1f77b4", "1/DB":"#ff7f0e", "Combined":"#2ca02c"}
SAVE_DIR = ROOT / "task_progress" / "figures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

for task, sub in agg.groupby("task_name"):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    # 按 x 轴顺序确保线条连续
    sub = sub.sort_values("err_bin")
    x = np.arange(len(sub["err_bin"]))

    # ① Silhouette
    ax.errorbar(
        x, sub["Sil_mean"], yerr=sub["Sil_std"],
        marker="o", color=palette["Silhouette"], label="Silhouette"
    )
    # ② 1/DB
    ax.errorbar(
        x, sub["DBi_mean"], yerr=sub["DBi_std"],
        marker="s", color=palette["1/DB"], label="1/DB"
    )
    # ③ Combined
    ax.errorbar(
        x, sub["Comb_mean"], yerr=sub["Comb_std"],
        marker="^", color=palette["Combined"], label="Combined"
    )

    # 坐标 & 标题
    ax.set_xticks(x)
    ax.set_xticklabels(sub["err_bin"], rotation=0)
    ax.set_xlabel("Injected error-rate bin (%)")
    ax.set_ylabel("Metric value (mean ±1σ)")
    ax.set_title(f"{task} : score vs. error-rate")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    # 保存 (PDF + EPS)
    for ext in ("pdf", "eps"):
        fig.savefig(SAVE_DIR / f"err_grad_{task}.{ext}", dpi=450 if ext=="pdf" else None)
    plt.close(fig)

print(f"✅ 4 张 err_grad_<task>.pdf / .eps 已保存至 {SAVE_DIR}")

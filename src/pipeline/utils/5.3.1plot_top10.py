#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/pipeline/utils/plot_top10_per_dataset.py
# -------------------------------------------------------------
# 为 beers / flights / hospital / rayyan 分别绘制
#  Top-10 组合的 (rel_mean ± SD) 条形图
#  —— 柱子更窄、画布更窄，颜色深→好
#  使用 Matplotlib “tight” 布局
# -------------------------------------------------------------
import pathlib, matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

UTIL_DIR = pathlib.Path(__file__).resolve().parent
RES_DIR  = UTIL_DIR / ".." / ".." / ".." / "results" / "analysis_results"

# ---------- ❶ 读取并健壮地将字符列转换为数值 ------------------------
numeric_cols = ["Combined Score"]                       # 需要数值化的关键列
dfs = []
for p in RES_DIR.glob("*.csv"):
    df = pd.read_csv(
        p,
        na_values=["", "NA", "N/A", "-", "null"],        # 常见缺失标记
        keep_default_na=True
    )
    # 对每个待数值化的列进行逐项清洗 → 数值
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)                             # 统一成字符串
                .str.replace(r"[^\d\.\-eE+]", "", regex=True)  # 去掉 %, 千位逗号、空格等
                .pipe(pd.to_numeric, errors="coerce")    # 无法解析 → NaN
            )
    dfs.append(df)

assert dfs, f"未找到聚合结果文件于 {RES_DIR}"
df_all = pd.concat(dfs, ignore_index=True)

# ---------- ❷ 计算相对得分（GT 作为分母） -------------------------
gt = (
    df_all.query("cleaning_method == 'GroundTruth'")
          .groupby(["task_name", "cluster_method"])["Combined Score"]
          .mean()
          .rename("GT_score")
)
df_rel = (
    df_all.merge(gt, on=["task_name", "cluster_method"])
          .assign(rel_score=lambda d: 100 * d["Combined Score"] / d["GT_score"])
          .query("cleaning_method != 'GroundTruth'")
)

# ---------- ❸ 逐数据集绘制 Top-10 条形图 --------------------------
for task, grp in df_rel.groupby("task_name"):
    stats = (
        grp.groupby(["cleaning_method", "cluster_method"])
           .agg(rel_mean=("rel_score", "mean"),
                var      =("rel_score", "var"))
           .reset_index()
           .assign(label=lambda d: d["cleaning_method"] + " + " + d["cluster_method"])
    )
    top10 = stats.nlargest(10, "rel_mean")

    # --- 绘制 ---
    x = np.arange(len(top10))
    colors = sns.color_palette("tab10", len(top10))

    # ★ 使用 tight 布局：tight_layout=True
    plt.figure(figsize=(6, 5), tight_layout=True)

    plt.bar(
        x,
        top10["rel_mean"],
        yerr=np.sqrt(top10["var"]),
        width=0.6,
        capsize=4,
        color=colors,
        edgecolor="none"
    )
    plt.axhline(100, ls="--", lw=1, c="black")

    plt.xticks(x, top10["label"], rotation=18, ha="right", fontsize=12)
    plt.yticks(fontsize=14)
    plt.ylabel("Relative mean score (% of GT)", fontsize=15)
    plt.title(f"Top-10 combinations on “{task}” (mean ± SD)", fontsize=16)

    # tight_layout 已在 figure() 设置，不必再调用

    # --- 保存 ---
    FIG_DIR = pathlib.Path("../../../task_progress/figures")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / f"top10_bar_error_{task}.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / f"top10_bar_error_{task}.eps", format="eps",
                bbox_inches="tight")
    plt.close()

print("Top-10 bar charts saved.")

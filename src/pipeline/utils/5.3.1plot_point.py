#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ROOT_DIR = pathlib.Path(__file__).resolve().parents[3]          # project root
CSV_DIR  = ROOT_DIR / "results" / "analysis_results"
csv_paths = CSV_DIR.glob("*.csv")
dfs = [pd.read_csv(p) for p in csv_paths]
df_all = pd.concat(dfs, ignore_index=True)

# GroundTruth 作为归一化基准
gt = (
    df_all.query("cleaning_method == 'GroundTruth'")
          .groupby(["task_name", "cluster_method"])["Combined Score"]
          .mean()
          .rename("GT_score")
)
df = (
    df_all.merge(gt, on=["task_name", "cluster_method"])
          .assign(rel_score=lambda d: 100 * d["Combined Score"] / d["GT_score"])
)

stats = (
    df.groupby(["task_name", "cleaning_method", "cluster_method"])
      .agg(rel_mean=("rel_score", "mean"),
           var=("Combined Score", "var"))
      .reset_index()
)

# ------------------------------------------------------------------
# ❷ 样式映射
cluster_markers = {
    "KMEANS": "o",
    "KMEANSNF": "s",
    "KMEANSPPS": "P",
    "GMM": "D",
    "DBSCAN": "^",
    "HC": "v",
}

cleaner_palette = sns.color_palette("tab10", n_colors=9)
cleaner_list = sorted(stats["cleaning_method"].unique())
cleaner_colors = {c: cleaner_palette[i % 10] for i, c in enumerate(cleaner_list)}

# ------------------------------------------------------------------
# ❸ 逐 task 绘制
SAVE_DIR = ROOT_DIR / "task_progress" / "figures"
for task, sub in stats.groupby("task_name"):
    fig, ax = plt.subplots(figsize=(6, 5))

    # 1) 散点
    for _, row in sub.iterrows():
        ax.scatter(row["rel_mean"], row["var"],
                   marker=cluster_markers.get(row["cluster_method"], "o"),
                   color=cleaner_colors[row["cleaning_method"]],
                   s=120, alpha=.85,
                   edgecolor="k", linewidth=.4)

    # 2) 参考线
    ax.axvline(100, color="grey", lw=.8, ls="--")
    ax.axhline(sub["var"].median(), color="grey", lw=.8, ls="--")

    # 3) 标题 & 轴标签
    ax.set_title(f"Mean–Variance plot · {task}", fontsize=18)
    ax.set_xlabel("Relative mean score  (% of  GT)", fontsize=18)
    ax.set_ylabel("Score variance", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)

    # --------------------------------------------------------------
    # ❹ 自定义图例（横向并排）
    handles_clean = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cleaner_colors[c], markersize=9,
                   label=c)
        for c in cleaner_list
    ]
    handles_cluster = [
        plt.Line2D([0], [0], marker=cluster_markers[k], color='k',
                   markersize=9, linestyle='', label=k)
        for k in cluster_markers
    ]

    # Cleaning 图例（左）
    leg1 = ax.legend(
        handles=handles_clean,
        title="Cleaning",
        loc='upper left',
        bbox_to_anchor=(0.02, 0.98),
        borderpad=0.5,
        frameon=True,
        fontsize=12.5,
        title_fontsize=12,
    )
    ax.add_artist(leg1)

    # Cluster 图例（右，与 Cleaning 同高）
    ax.legend(
        handles=handles_cluster,
        title="Cluster",
        loc='upper left',
        bbox_to_anchor=(0.38, 0.98),   # 横向右移
        borderpad=0.5,
        frameon=True,
        fontsize=12.5,
        title_fontsize=12,
    )

    # 5) 布局 & 保存
    fig.tight_layout()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("eps", "pdf"):
        fig.savefig(SAVE_DIR / f"mean_var_scatter_{task}.{ext}", format=ext)
    plt.close(fig)

print(f"Figures saved to {SAVE_DIR}")

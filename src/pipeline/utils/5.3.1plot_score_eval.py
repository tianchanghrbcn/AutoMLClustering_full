#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, pathlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------- 0. 目标数据集列表 ----------
ALL_TASKS = ["beers", "flights", "hospital", "rayyan"]
tasks_cli = sys.argv[1:]                       # 0 或 1 个参数
if not tasks_cli:
    tasks = ALL_TASKS
elif tasks_cli[0] in ALL_TASKS:
    tasks = [tasks_cli[0]]
else:
    raise SystemExit(f"❌ 无效 task_name={tasks_cli[0]}，应为 {ALL_TASKS}")

# ---------- 1. 读取聚合结果 ----------
UTIL_DIR = pathlib.Path(__file__).resolve().parent
RES_DIR  = UTIL_DIR / ".." / ".." / ".." / "results" / "analysis_results"
csvs = sorted(RES_DIR.glob("*.csv"))
if not csvs:
    raise SystemExit(f"❌ 找不到 csv 于 {RES_DIR}")
df_all = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)

# ---------- 2. 共用的绘图函数（横纵轴已对调） ----------
def draw_heatmap(task, df_task):
    # -- 计算相对得分均值 --
    gt_ref = (df_task.query("cleaning_method == 'GroundTruth'")
                      .groupby("cluster_method")["Combined Score"]
                      .mean()
                      .rename("gt"))
    df_rel = (df_task.merge(gt_ref, on="cluster_method")
                      .assign(rel=lambda d: d["Combined Score"] / d["gt"]))
    stats = (df_rel.groupby(["cleaning_method","cluster_method"])
                      .agg(rel_mean=("rel","mean"))
                      .reset_index())

    clean_order   = ["mode","bigdansing","boostclean","holoclean","horizon",
                     "scared","baran","Unified","GroundTruth"]
    cluster_order = ["KMEANS","KMEANSNF","KMEANSPPS","GMM","HC","DBSCAN"]

    # 横纵轴对调：行→ cluster_method，列→ cleaning_method
    heat = (stats.pivot(index="cluster_method",
                        columns="cleaning_method",
                        values="rel_mean")
                  .reindex(index=cluster_order, columns=clean_order)
                  .fillna(0))

    # -- 自定义颜色：0->灰，其余 RdBu_r --
    base_cmap = sns.color_palette("RdBu_r", 256)
    cmap      = ListedColormap(["#e5e5e5"] + base_cmap.as_hex())
    vals      = heat.values.flatten()
    vmin_pos  = vals[vals>0].min()
    vmax      = vals.max()

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(6, 5))   # 宽高稍微调大一点，读数更清晰
    sns.heatmap(
        heat, cmap=cmap, vmin=vmin_pos, vmax=vmax,
        annot=True, fmt=".2f",
        annot_kws={"size":15, "weight":"bold"},
        linewidths=.4, linecolor="grey",
        square=True, cbar=False, ax=ax)

    # 斜体刻度；去轴标题
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right",
                       fontsize=15, fontstyle="italic")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                       fontsize=15, fontstyle="italic")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_title(f"{task} — Relative Mean Score", fontsize=18, pad=10)
    ax.set_aspect("equal")
    plt.tight_layout()

    for ext in ("eps", "pdf", "png"):
        plt.savefig(
            f"../../../task_progress/figures/heatmap_rel_{task}.{ext}",
            dpi=450 if ext == "png" else None,
            bbox_inches="tight",  # ← 关键：裁掉周围空白
            pad_inches=0          # ← 关键：不留 padding
        )
    plt.close()

# ---------- 3. 批量绘制 ----------
for task in tasks:
    subset = df_all.query("task_name == @task")
    if subset.empty:
        print(f"⚠️  跳过 {task} — 无数据")
        continue
    draw_heatmap(task, subset)
    print(f"✅ heatmap_rel_{task}.[eps|pdf|png] 已生成")

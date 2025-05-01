#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_error_type_heatmap_max.py   ·   (0,0) 参与色阶
-----------------------------------------------------------------
• (0,0) = GroundTruth 最大 Combined Score，并参与 vmin/vmax 计算
• 其余 15 格 = 非 GroundTruth 清洗算法下的最大 Combined Score
• CSV 中若数字带 %, 逗号, 中文字符, 科学计数等可自动解析
• 打印 4×4 分数矩阵；保存 EPS + PDF
• 更新要点：
  1) 增大总标题、坐标标题、刻度标签、网格文字字体
  2) 图像保存、显示均使用 tight 布局
  3) 缩小画布尺寸以减小网格物理尺寸
"""

from pathlib import Path
import re
import subprocess
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ──────────── 配置 ────────────
TASKS     = ["beers", "flights", "hospital", "rayyan"]
CSV_ROOT  = Path("../../../results/analysis_results")
FIG_ROOT  = Path("../../../task_progress/figures/5.3.2graph")
FIG_ROOT.mkdir(parents=True, exist_ok=True)

LEVELS    = [0, 5, 10, 15]                  # (%) 刻度
CMAP_BASE = plt.colormaps["coolwarm"]       # Matplotlib ≥ 3.7
NUM_RE    = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
# ──────────────────────────────


def parse_numeric(series: pd.Series) -> pd.Series:
    """从任意字符串提取首个数字片段并转成 float"""
    def _extract(x):
        m = NUM_RE.search(str(x))
        return float(m.group()) if m else np.nan
    return series.apply(_extract).astype(float)


def best_gt_score(df: pd.DataFrame) -> Optional[float]:
    mask = df["cleaning_method"].str.lower().str.contains("ground")
    return df.loc[mask, "Combined Score"].max() if mask.any() else None


def print_matrix(name: str, mat: pd.DataFrame):
    print(f"\n=== {name} ===")
    hdr = "Anom\\Miss | " + " | ".join(f"{c:>9}%" for c in mat.columns)
    print(hdr); print("-" * len(hdr))
    for a in mat.index:
        row = " | ".join(f"{mat.loc[a,m]:>9.4f}"
                         if not np.isnan(mat.loc[a,m]) else "    NaN   "
                         for m in mat.columns)
        print(f"{a:>10}% | {row}")


def draw_one(task: str):
    csv = CSV_ROOT / f"{task}_cluster.csv"
    if not csv.is_file():
        print(f"[WARN] {csv} 不存在，跳过 {task}")
        return

    df = pd.read_csv(csv)

    # 1) 关键列字符串 → float
    for col in ["anomaly", "missing", "Combined Score"]:
        if col not in df.columns:
            raise KeyError(f"{csv} 缺少列 {col}")
        df[col] = parse_numeric(df[col])
    df = df.dropna(subset=["anomaly", "missing", "Combined Score"])

    # 2) GroundTruth 最大分
    gt_val = best_gt_score(df)

    # 3) 非 GroundTruth 下的最大 Combined Score
    non_gt = df[~df["cleaning_method"].str.lower().str.contains("ground")]
    pivot = (non_gt.groupby(["anomaly", "missing"])["Combined Score"]
                   .max()
                   .reset_index()
                   .pivot(index="anomaly", columns="missing",
                          values="Combined Score")
                   .reindex(index=LEVELS, columns=LEVELS))

    if gt_val is not None:
        pivot.loc[0, 0] = gt_val

    # 4) 打印 4×4
    print_matrix(task, pivot)

    # 5) 色阶 —— 直接用全部 16 格
    vals = pivot.values[~np.isnan(pivot.values)]
    vmin, vmax = vals.min(), vals.max()
    if np.isclose(vmin, vmax):
        vmin, vmax = vmin - 1e-3, vmax + 1e-3

    # 6) 绘图
    cmap = CMAP_BASE.copy(); cmap.set_bad("white")

    # 缩小画布尺寸(宽×高)以减小格子，dpi保持高分辨率
    fig = plt.figure(figsize=(6.5, 4.5), dpi=300)
    ax = fig.add_subplot(111)

    im = ax.imshow(pivot, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")

    labels = [f"{x}%" for x in LEVELS]
    ax.set_xticks(range(len(LEVELS)), labels)
    ax.set_yticks(range(len(LEVELS)), labels)

    # 坐标标题 + 刻度字体
    ax.set_xlabel("Missing Rate", fontsize=16)
    ax.set_ylabel("Anomaly Rate", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # 总标题
    ax.set_title(f"{task.capitalize()} – Error-Type Heatmap",
                 fontsize=18, pad=10)

    # 网格
    ax.set_xticks(np.arange(-.5, len(LEVELS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(LEVELS), 1), minor=True)
    ax.grid(which="minor", color="w", linewidth=0.4)

    # 网格内数值字体进一步增大
    for i, a in enumerate(LEVELS):
        for j, m in enumerate(LEVELS):
            v = pivot.loc[a, m]
            if not np.isnan(v):
                tc = "black"  # ★ 固定为黑色
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=16, weight="bold", color=tc)

    # 颜色条
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Combined Score", fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    # tight 布局
    fig.tight_layout()

    # 7) 保存（同样使用紧凑边界）
    eps = FIG_ROOT / f"{task}_heatmap.eps"
    pdf = FIG_ROOT / f"{task}_heatmap.pdf"
    fig.savefig(eps, format="eps", bbox_inches="tight")
    try:
        subprocess.run(
            ["epstopdf", str(eps), "--outfile", str(pdf)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        fig.savefig(pdf, format="pdf", bbox_inches="tight")

    plt.close(fig)
    print(f"[OK] {task}: 图像保存为 {eps.name} / {pdf.name}")


if __name__ == "__main__":
    for task in TASKS:
        draw_one(task)

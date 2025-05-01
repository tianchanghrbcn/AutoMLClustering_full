#!/usr/bin/env python3
# make_radar.py  —— 4-in-1 雷达图（字体 & 标签位置再调）
# -----------------------------------------------------------------
from __future__ import annotations
import os, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ========= 0. 配置 =========================================================
DATA_DIR   = "../../../results/analysis_results"
OUT_DIR    = "../../../task_progress/figures/6.4.3graph"
TASKS      = ["beers", "flights", "hospital", "rayyan"]
RADAR_COLS = ["precision", "recall", "F1",
              "EDR", "Sil_relative", "DB_relative", "Comb_relative"]

matplotlib.rc('font', family='Times New Roman')
os.makedirs(OUT_DIR, exist_ok=True)

# ========= 1. 数据 =========================================================
def load_all() -> pd.DataFrame:
    dfs = []
    for t in TASKS:
        fp = os.path.join(DATA_DIR, f"{t}_summary.xlsx")
        if not os.path.isfile(fp):
            print(f"[WARN] 跳过 {t}: {fp} 不存在")
            continue
        df = pd.read_excel(fp)
        df["task_name"] = t
        dfs.append(df)
    if not dfs:
        raise RuntimeError("❌ 没找到任何 XLSX！")
    return pd.concat(dfs, ignore_index=True)

def norm01(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if lo != hi else 0.5

def prep_task(df_t):
    mid = df_t.groupby(["cluster_method","cleaning_method"])[RADAR_COLS].median().reset_index()
    agg = mid.groupby("cleaning_method")[RADAR_COLS].median().reset_index()
    agg.insert(0, "label", agg["cleaning_method"])
    ranges = {c: (agg[c].min(), agg[c].max()) for c in RADAR_COLS}
    for c in RADAR_COLS:
        agg[c] = norm01(agg[c])
    return agg, ranges

# ========= 2. 绘图 =========================================================
def draw_radar(ax, cats, df_lines, ranges, cmap, fs=16):
    N = len(cats)
    angs = [n/N*2*math.pi for n in range(N)] + [0]

    # 背景灰
    ax.set_facecolor("#f5f5f5")
    ax.patch.set_alpha(1)

    # 极坐标设置
    ax.spines["polar"].set_visible(False)
    ax.set_theta_offset(math.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angs[:-1])
    ax.set_yticks(np.linspace(0,1,6)[1:])
    ax.set_ylim(0,1)

    # 同心圆网格：浅灰 & 最外圈更深
    ax.grid(alpha=0.45, lw=0.6, color="gray")
    ax.plot(angs, [1]*(N+1), lw=1.0, color="gray", alpha=0.85)  # 外圈加深

    # 轴标签 + 区间
    labels = [f"{c}\n[{ranges[c][0]:.2f}, {ranges[c][1]:.2f}]" for c in cats]
    ax.set_xticklabels(labels, fontsize=fs, ha="center", va="center")
    ax.set_yticklabels([])

    for _, row in df_lines.iterrows():
        vals = row[cats].tolist() + [row[cats[0]]]
        col  = cmap[row["label"]]
        ax.plot(angs, vals, lw=2.0, color=col)
        ax.fill(angs, vals, alpha=0.07, color=col)

# ========= 3. 主 ===========================================================
def main():
    df_all = load_all()

    # 颜色统一
    methods = sorted(df_all["cleaning_method"].unique())
    base_cmap = plt.cm.get_cmap("tab10", len(methods))
    COLORS = {m: base_cmap(i) for i, m in enumerate(methods)}

    fig, axes = plt.subplots(
        1, 4, subplot_kw={"projection": "polar"},
        figsize=(30, 8), facecolor="none"
    )

    handles, lbls = [], []
    for i, task in enumerate(TASKS):
        ax = axes[i]
        agg, ranges = prep_task(df_all[df_all["task_name"] == task])
        draw_radar(ax, RADAR_COLS, agg, ranges, COLORS, fs=22)

        # 更靠近图像，字体更大
        letter = chr(ord('a')+i)
        ax.text(0.5, -0.16, f"({letter}) {task}", transform=ax.transAxes,
                ha="center", va="center", fontsize=34)

        if i == 0:
            for _, row in agg.iterrows():
                h = ax.plot([], [], color=COLORS[row["label"]], lw=2)[0]
                handles.append(h); lbls.append(row["label"])

    # 顶部单行图例（更大字体）
    fig.legend(handles, lbls, ncol=len(lbls),
               loc="upper center", bbox_to_anchor=(0.5, 1.14),
               frameon=False, handlelength=3, fontsize=26)

    plt.tight_layout(rect=[0,0,1,1.08])

    for fmt in ("eps","pdf"):
        fp = os.path.join(OUT_DIR, f"radar_four_in_one.{fmt}")
        fig.savefig(fp, format=fmt, bbox_inches="tight")
    plt.close(fig)
    print("✅ 已重新生成 radar_four_in_one.eps / .pdf")

# ========= 入口 ============================================================
if __name__ == "__main__":
    main()

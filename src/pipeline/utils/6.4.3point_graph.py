#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import matplotlib
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# 0. 显式注册 Times New Roman 字体
# ------------------------------------------------------------------
ttf_candidates = [
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\Times New Roman.ttf",
    r"/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
]
for p in ttf_candidates:
    if os.path.exists(p):
        fm.fontManager.addfont(p)
        break
else:
    print("[WARN] Times New Roman.ttf not found – fallback to default serif.")

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "TimesNewRomanPSMT", "Times New Roman PS"],
    "pdf.fonttype": 42,
})
sns.set_style("ticks")

# ------------------------------------------------------------------
# 仅保存 PDF（矢量 + 透明度）
# ------------------------------------------------------------------
def save_pdf(fig, pdf_path):
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")

def main():
    # ---------- 1. 读数据 ----------
    all_tasks = ["beers", "rayyan", "flights", "hospital"]
    data_dir   = os.path.join("..", "..", "..", "results", "analysis_results")

    frames = []
    for task in all_tasks:
        fp = os.path.join(data_dir, f"{task}_summary.xlsx")
        if not os.path.isfile(fp):
            print(f"[WARN] {fp} missing, skip")
            continue
        df_tmp = pd.read_excel(fp)
        df_tmp["task_name"] = task
        frames.append(df_tmp)

    if not frames:
        sys.exit("No data loaded – check paths.")
    df = pd.concat(frames, ignore_index=True)

    # ---------- 2. 列检查 ----------
    need = ["dataset_id", "cluster_method", "task_name",
            "EDR", "F1", "Comb_relative", "Sil_relative", "DB_relative"]
    for c in need:
        if c not in df.columns:
            sys.exit(f"[ERROR] missing column: {c}")
    df["dataset_id"] = pd.to_numeric(df["dataset_id"])

    # ---------- 3. 每数据集一组 & 指标组合 ----------
    task_groups = [([t], t[:2].upper()) for t in all_tasks]
    combos = [("EDR", "Comb_relative"), ("F1", "Sil_relative")]

    out_dir = os.path.join("..", "..", "..", "task_progress", "figures", "6.4.3graph")
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 4. 主循环 ----------
    for tasks, tag in task_groups:
        df_sub = df[df["task_name"].isin(tasks)].copy()
        if df_sub.empty:
            continue

        # 4.1 将 dataset_id → 紧凑坐标
        pos_map, centers, boundaries = {}, [], []
        current_offset, group_gap = 0, 0.6

        for idx, t in enumerate(tasks):        # 实际上每轮只有一个 t
            local_ids = sorted(df_sub[df_sub.task_name == t]["dataset_id"].unique())
            for j, ds in enumerate(local_ids):
                pos_map[ds] = current_offset + j
            start, end = current_offset, current_offset + len(local_ids) - 1
            centers.append((start + end) / 2)
            if idx < len(tasks) - 1:
                boundary = end + group_gap / 2
                boundaries.append(boundary)
                current_offset = end + group_gap
            else:
                current_offset = end + 1

        # 4.2 绘图
        for xm, ym in combos:
            rs = []
            for (ds, cm), sub in df_sub.groupby(["dataset_id", "cluster_method"]):
                r_val = np.nan if len(sub) < 2 else pearsonr(sub[xm], sub[ym])[0]
                rs.append({"dataset_id": ds, "cluster_method": cm, "r": r_val})
            pv = (pd.DataFrame(rs)
                    .pivot(index="dataset_id", columns="cluster_method", values="r")
                    .sort_index())
            long = (pv.stack().reset_index()
                    .rename(columns={0: "r"})
                    .fillna({"r": 0.0}))

            long["xpos"] = long["dataset_id"].map(pos_map)
            long["cluster_method"] = pd.Categorical(
                long["cluster_method"],
                sorted(long["cluster_method"].unique()),
                ordered=True
            )

            # ---- 绘图 ----------------------------------------------------
            fig = plt.figure(figsize=(6.5, 4.5))
            ax  = sns.scatterplot(
                data=long,
                x="xpos", y="cluster_method",
                size=long["r"].abs(), hue="r",
                palette="RdBu", sizes=(60, 600),
                alpha=0.85, edgecolor="black", legend=False
            )
            ax.axis("tight")

            for x in boundaries:
                ax.axvline(x, ls="--", lw=1, c="gray")

            # 轴刻度 & 标签
            ax.set_xticks([])                     # 不显示 x 轴刻度
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelsize=16)

            for tick in ax.get_yticklabels():
                tick.set_rotation(0)
                tick.set_va("center")
                tick.set_ha("right")

            # 色条
            sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-1, vmax=1), cmap="RdBu")
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                                pad=0.12, fraction=0.035, aspect=30)
            cbar.set_label("Pearson r", fontsize=16)
            cbar.ax.tick_params(labelsize=16)

            # —— 不再添加标题 ——
            plt.tight_layout()

            pdf_path = os.path.join(out_dir, f"{tag}_{xm}_vs_{ym}.pdf")
            save_pdf(fig, pdf_path)
            plt.close()
            print(f"saved → {pdf_path}")

    print("✅ 8 figures regenerated without titles and with smaller y-tick fonts.")

if __name__ == "__main__":
    main()

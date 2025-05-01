#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw error-rate vs. combined-score line charts (lines = cluster_method).

Input : ../../../results/analysis_results/{task}_cluster.csv
Output: ../../../task_progress/figures/{task}_combined_score.svg + .pdf

⚙️ 变动：启用 “tight” 绘图模式
        • 全局设置 plt.rcParams['savefig.bbox'] = 'tight'
        • 保存 SVG 时显式 bbox_inches='tight'
其它逻辑保持不变。
"""
import subprocess, shutil, importlib
from pathlib import Path

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------- 全局 tight 模式 ------------------------------- ★
plt.rcParams['savefig.bbox'] = 'tight'

# ------------ 1. constants ---------------------------------
TASK_NAMES = ["beers", "flights", "hospital", "rayyan"]
CSV_ROOT   = Path("../../../results/analysis_results")
FIG_ROOT   = Path("../../../task_progress/figures/5.3.2graph")
FIG_ROOT.mkdir(parents=True, exist_ok=True)

BIN_EDGES  = [0, 5, 10, 15, 20, 25, 30, float("inf")]
BIN_LABELS = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", ">=30"]

# high-contrast colour pool
COLOR_LIST = (
    list(mpl.colormaps["tab10"].colors) +
    list(mpl.colormaps["Set1"].colors)  +
    list(mpl.colormaps["Dark2"].colors) +
    list(mpl.colormaps["tab20"].colors) +
    list(mpl.colormaps["tab20b"].colors)+
    list(mpl.colormaps["tab20c"].colors)
)
MARKERS = ["o","s","D","^","v",">","<","h","p","X","8","*","P"]

# ---------- 2. build style map ------------------------------
clusterers = set()
for t in TASK_NAMES:
    fp = CSV_ROOT / f"{t}_cluster.csv"
    if fp.exists():
        clusterers |= set(pd.read_csv(fp, usecols=["cluster_method"])
                          ["cluster_method"].unique())

if len(clusterers) > len(COLOR_LIST):
    COLOR_LIST *= (len(clusterers) // len(COLOR_LIST) + 1)

STYLE_MAP = {clu: (COLOR_LIST[i], MARKERS[i % len(MARKERS)])
             for i, clu in enumerate(sorted(clusterers))}

# ---------- 3. helper：SVG → PDF ----------------------------
def svg_to_pdf(svg_path: Path):
    pdf_path = svg_path.with_suffix(".pdf")
    # ① Inkscape CLI
    if shutil.which("inkscape"):
        try:
            subprocess.run(
                ["inkscape", "--export-type=pdf", "-o", pdf_path, svg_path],
                check=True, capture_output=True
            )
            return pdf_path
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Inkscape failed: {e.stderr.decode().strip()}")
    # ② cairosvg fallback
    try:
        cairosvg = importlib.import_module("cairosvg")
        cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
        return pdf_path
    except ModuleNotFoundError:
        pass
    except Exception as e:
        print(f"[WARN] cairosvg failed: {e}")
    print("[INFO] No SVG→PDF converter found – SVG kept only.")
    return None

# ---------- 4. main loop ------------------------------------
for task in TASK_NAMES:
    csv = CSV_ROOT / f"{task}_cluster.csv"
    if not csv.exists():
        print(f"[WARN] {csv} missing – skipped");  continue

    df = pd.read_csv(csv)
    need_cols = {"error_rate", "Combined Score", "cleaning_method", "cluster_method"}
    if not need_cols <= set(df.columns):
        print(f"[ERROR] {csv} lacks required columns");  continue

    # 按 error_rate 分箱
    df["error_bin"] = pd.cut(df["error_rate"], BIN_EDGES,
                             labels=BIN_LABELS, right=False, include_lowest=True)

    # 每个 cluster_method & error_bin 取最高 Combined Score
    best = (df.groupby(["cluster_method", "error_bin"])["Combined Score"]
              .max().reset_index())

    # --------- 绘图 -----------------------------------------
    plt.figure(figsize=(6.5, 4.5))
    for clu, sub in best.groupby("cluster_method"):
        y = (sub.set_index("error_bin")
                 .reindex(BIN_LABELS)["Combined Score"].values)
        color, marker = STYLE_MAP[clu]
        plt.plot(BIN_LABELS, y, label=clu,
                 color=color, marker=marker,
                 linewidth=1.8, markersize=9)

    plt.title(f"{task.capitalize()} – Combined Score vs. Error Rate",
              fontsize=18, pad=6)
    plt.xlabel("Error-Rate Range (%)", fontsize=16)
    plt.ylabel("Combined Score",      fontsize=16)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)

    leg = plt.legend(title="Clustering Method",
                     fontsize=12, loc="upper right", framealpha=0.4)
    frame = leg.get_frame()
    frame.set_edgecolor("0.5"); frame.set_linewidth(0.8)

    plt.tight_layout()

    # 保存 SVG（tight bbox）
    svg_path = FIG_ROOT / f"{task}_combined_score_cluster.svg"
    plt.savefig(svg_path, format="svg", bbox_inches='tight')     # ★
    plt.close()

    svg_to_pdf(svg_path)

print("✅ New SVG & PDF (if converter present) saved to:", FIG_ROOT.resolve())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw error-rate vs. combined-score line charts (one per dataset).

Input : ../../../results/analysis_results/{task}_cluster.csv
Output: ../../../task_progress/figures/{task}_combined_score.svg + .pdf

改动要点
--------
* 保存图像时使用 **bbox_inches="tight"**，确保真正的 tight 模式。
* 其余逻辑完全保持不变。
"""

import subprocess, shutil, itertools, importlib
from pathlib import Path

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ------------ 1. constants ---------------------------------------------------
TASK_NAMES = ["beers", "flights", "hospital", "rayyan"]
CSV_ROOT   = Path("../../../results/analysis_results")
FIG_ROOT   = Path("../../../task_progress/figures")
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

# ------------ 2. global style map -------------------------------------------
cleaners = set()
for t in TASK_NAMES:
    p = CSV_ROOT / f"{t}_cluster.csv"
    if p.exists():
        cleaners |= set(pd.read_csv(p, usecols=["cleaning_method"])
                        ["cleaning_method"].unique())

if len(cleaners) > len(COLOR_LIST):
    COLOR_LIST *= len(cleaners) // len(COLOR_LIST) + 1

STYLE_MAP = {cl: (COLOR_LIST[i], MARKERS[i % len(MARKERS)])
             for i, cl in enumerate(sorted(cleaners))}

# ------------ 3. helpers -----------------------------------------------------
def svg_to_pdf(svg_path: Path):
    """Convert SVG→PDF using inkscape or cairosvg; return Path or None."""
    pdf_path = svg_path.with_suffix(".pdf")
    # try inkscape CLI
    if shutil.which("inkscape"):
        try:
            subprocess.run(
                ["inkscape", "--export-type=pdf", "-o", pdf_path, svg_path],
                check=True, capture_output=True
            )
            return pdf_path
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Inkscape failed: {e.stderr.decode().strip()}")
    # try cairosvg python library
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

# ------------ 4. main loop ---------------------------------------------------
for task in TASK_NAMES:
    csv = CSV_ROOT / f"{task}_cluster.csv"
    if not csv.exists():
        print(f"[WARN] {csv} missing – skipped");  continue

    df = pd.read_csv(csv)
    if not {"error_rate","Combined Score","cleaning_method"} <= set(df.columns):
        print(f"[ERROR] {csv} lacks required columns");  continue

    df["error_bin"] = pd.cut(df["error_rate"], BIN_EDGES, labels=BIN_LABELS,
                             right=False, include_lowest=True)
    best = (df.groupby(["cleaning_method","error_bin"])["Combined Score"]
              .max().reset_index())

    plt.figure(figsize=(6.5,4.5))
    for cln, sub in best.groupby("cleaning_method"):
        y = (sub.set_index("error_bin")
                 .reindex(BIN_LABELS)["Combined Score"].values)
        color, marker = STYLE_MAP[cln]
        plt.plot(BIN_LABELS, y, label=cln,
                 color=color, marker=marker,
                 linewidth=1.8, markersize=6)

    # larger fonts
    plt.title(f"{task.capitalize()} - Combined Score vs. Error Rate",
              fontsize=18, pad=6)
    plt.xlabel("Error-Rate Range (%)", fontsize=16)
    plt.ylabel("Combined Score",      fontsize=16)
    plt.xticks(fontsize=16);  plt.yticks(fontsize=16)

    # legend: upper-right, semi-transparent
    leg = plt.legend(title="Cleaning Method",
                     fontsize=11,
                     loc="upper right",
                     framealpha=0.4)
    frame = leg.get_frame()
    frame.set_edgecolor("0.5"); frame.set_linewidth(0.8)

    plt.tight_layout()                         # layout 紧凑

    svg_path = FIG_ROOT / f"{task}_combined_score_cleaning.svg"
    # ★ tight 模式保存
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()

    svg_to_pdf(svg_path)

print("✅ SVG & PDF (if converter present) generated in:", FIG_ROOT.resolve())

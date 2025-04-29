#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 生成各数据集的 mean–variance 散点图
#
import pathlib, subprocess, shutil, importlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ---------- 0. 目录 ----------------------------------------------------------
ROOT_DIR = pathlib.Path(__file__).resolve().parents[3]
CSV_DIR  = ROOT_DIR / "results" / "analysis_results"
SVG_DIR  = ROOT_DIR / "task_progress" / "figures"
SVG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 1. 读取 + 聚合（显式数值化，健壮处理） ----------------------------
numeric_cols = ["Combined Score"]                         # 关键数值列
dfs = []
for p in CSV_DIR.glob("*.csv"):
    df = pd.read_csv(
        p,
        na_values=["", "NA", "N/A", "-", "null"],         # 常见缺失标记
        keep_default_na=True
    )
    # → 将字符型数值强制转换为 float，去掉 %, 千位逗号、空格等
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[^\d.\-eE+]", "", regex=True)  # 仅保留数字/符号
                .pipe(pd.to_numeric, errors="coerce")         # 失败 → NaN
            )
    dfs.append(df)

if not dfs:
    raise SystemExit(f"❌ 找不到 csv 于 {CSV_DIR}")

df_all = pd.concat(dfs, ignore_index=True)

# ---------- 2. 计算统计量 -----------------------------------------------------
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

# ---------- 3. 样式映射 -------------------------------------------------------
cluster_markers = {
    "KMEANS": "o", "KMEANSNF": "s", "KMEANSPPS": "P",
    "GMM": "D", "DBSCAN": "^", "HC": "v",
}

cleaner_palette = sns.color_palette("tab10", n_colors=9)
cleaner_list = sorted(stats["cleaning_method"].unique())
cleaner_colors = {c: cleaner_palette[i % 10] for i, c in enumerate(cleaner_list)}

# ---------- 4. SVG → PDF 转换辅助 --------------------------------------------
def svg_to_pdf(svg_path: pathlib.Path):
    pdf_path = svg_path.with_suffix(".pdf")
    if shutil.which("inkscape"):
        try:
            subprocess.run(
                ["inkscape", "--export-type=pdf", "-o", pdf_path, svg_path],
                check=True, capture_output=True
            )
            return
        except subprocess.CalledProcessError:
            pass
    try:
        cairosvg = importlib.import_module("cairosvg")
        cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
        return
    except Exception:
        print("[INFO] No SVG→PDF converter found – kept SVG only.")

# ---------- 5. 逐 task 绘制 ---------------------------------------------------
for task, sub in stats.groupby("task_name"):
    fig, ax = plt.subplots(figsize=(6, 5))

    # (1) 散点
    for _, row in sub.iterrows():
        ax.scatter(row["rel_mean"], row["var"],
                   marker=cluster_markers.get(row["cluster_method"], "o"),
                   color=cleaner_colors[row["cleaning_method"]],
                   s=120, alpha=.85,
                   edgecolor="k", linewidth=.4)

    # (2) 参考线
    ax.axvline(100, color="grey", lw=.8, ls="--")
    ax.axhline(sub["var"].median(), color="grey", lw=.8, ls="--")

    # (3) 标题 & 轴
    ax.set_title(f"Mean–Variance plot · {task}", fontsize=18)
    ax.set_xlabel("Relative mean score  (% of  GT)", fontsize=18)
    ax.set_ylabel("Score variance", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)

    # (4) 自定义图例
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

    leg1 = ax.legend(handles=handles_clean, title="Cleaning",
                     loc='upper left', bbox_to_anchor=(0.02, 0.98),
                     borderpad=0.5, frameon=True, framealpha=0.4,
                     fontsize=12.5, title_fontsize=12)
    leg1.get_frame().set_edgecolor("0.5"); leg1.get_frame().set_linewidth(0.8)
    ax.add_artist(leg1)

    leg2 = ax.legend(handles=handles_cluster, title="Cluster",
                     loc='upper left', bbox_to_anchor=(0.38, 0.98),
                     borderpad=0.5, frameon=True, framealpha=0.4,
                     fontsize=12.5, title_fontsize=12)
    leg2.get_frame().set_edgecolor("0.5"); leg2.get_frame().set_linewidth(0.8)

    fig.tight_layout()

    svg_path = SVG_DIR / f"mean_var_scatter_{task}.svg"
    fig.savefig(svg_path, format="svg")
    plt.close(fig)
    svg_to_pdf(svg_path)

print(f"Figures saved to {SVG_DIR.resolve()}  (PDFs generated if converter present)")

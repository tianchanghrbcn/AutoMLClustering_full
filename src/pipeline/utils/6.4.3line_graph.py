#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rc('font', family='Times New Roman')
# sns.set_style("whitegrid")  # 如需网格，可取消注释

def main():
    #--------------------------------------------------------------------
    # A) 读取并合并
    #--------------------------------------------------------------------
    task_names = ["beers", "rayyan", "flights", "hospital"]
    data_dir   = os.path.join("..", "..", "..", "results", "analysis_results")
    out_dir    = os.path.join("..", "..", "..", "task_progress", "figures", "6.4.3graph")

    dfs = []
    for t in task_names:
        fp = os.path.join(data_dir, f"{t}_summary.xlsx")
        if not os.path.isfile(fp):
            print(f"[WARN] missing {fp}, skip {t}")
            continue
        tmp = pd.read_excel(fp)
        if "task_name" not in tmp.columns:
            tmp["task_name"] = t
        dfs.append(tmp)

    if not dfs:
        print("[ERROR] No data loaded."); sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    df["error_rate"] = df["error_rate"].astype(float)

    #--------------------------------------------------------------------
    # B) 错误率分箱
    #--------------------------------------------------------------------
    bins   = [0, 5, 10, 15, 20, 25, 30, 9e9]
    labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "≥30"]
    df["error_rate_bin"] = pd.cut(df["error_rate"], bins=bins,
                                  labels=labels, right=False)

    #--------------------------------------------------------------------
    # C) 输出目录
    #--------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)

    for task in sorted(df["task_name"].unique()):
        sub = df[df["task_name"] == task]
        if sub.empty:
            continue

        # 计算 CEGR
        rec = []
        for (ds, ebin, cm), g in sub.groupby(
                ["dataset_id", "error_rate_bin", "cluster_method"]):
            if len(g) < 2:
                continue
            best  = g.loc[g["EDR"].idxmax()]
            worst = g.loc[g["EDR"].idxmin()]
            d_edr = best["EDR"] - worst["EDR"]
            if abs(d_edr) < 1e-10:
                continue
            rec.append({
                "error_rate_bin": ebin,
                "cluster_method": cm,
                "CEGR": (best["Comb_relative"] - worst["Comb_relative"]) / d_edr
            })

        ratio_df = pd.DataFrame(rec)
        if ratio_df.empty:
            print(f"[WARN] no CEGR for {task}"); continue

        agg = (ratio_df.groupby(["error_rate_bin", "cluster_method"], as_index=False)
               .CEGR.median()
               .rename(columns={"CEGR": "CEGR_median"}))
        agg["error_rate_bin"] = agg["error_rate_bin"].astype(
            pd.CategoricalDtype(labels, ordered=True))
        agg = agg.sort_values(["error_rate_bin", "cluster_method"])

        #----------------------------------------------------------------
        # E) 保存图像对应的数据表格  ← 新增
        #----------------------------------------------------------------
        table_path = os.path.join(out_dir, f"CEGR_5pct_{task}.xlsx")
        agg.to_excel(table_path, index=False)
        print(f"[INFO] saved {table_path}")

        #----------------------------------------------------------------
        # D) 绘图
        #----------------------------------------------------------------
        plt.figure(figsize=(6.5, 4.5))
        sns.lineplot(
            data=agg,
            x="error_rate_bin",
            y="CEGR_median",
            hue="cluster_method",
            style="cluster_method",
            markers=True,
            dashes=False,
            linewidth=2,
            markersize=13
        )

        plt.xlabel("Error-Rate Bin (%)",  fontsize=18)
        plt.ylabel("Median CEGR",         fontsize=18)
        # plt.title(f"{task}: CEGR vs Error-Rate", fontsize=18, pad=6)

        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=18)

        # ★ 图例移入图内右下角并设置透明度
        plt.legend(title="Cluster Method",
                   fontsize=12, title_fontsize=13,
                   loc="lower right", frameon=True, framealpha=0.5)

        plt.tight_layout()
        out_pdf = os.path.join(out_dir, f"CEGR_5pct_{task}.pdf")
        plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] saved {out_pdf}")

    print("[INFO] Done. Each task_name ⇒ one PDF chart and one data table.")

if __name__ == "__main__":
    main()

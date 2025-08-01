#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties   # ★

# ── 字体设置 ──────────────────────────────────────────────
matplotlib.rc('font', family='Times New Roman')  # 英文默认
# 中文用宋体；若没有宋体请改为 SimHei / Microsoft YaHei 等
cn_font         = FontProperties(family='SimSun')           # 轴标签
cn_font_title   = FontProperties(family='SimSun', size=16)  # 备用标题
cn_font_legend  = FontProperties(family='SimSun', size=14)  # 图例

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
        tmp["task_name"] = t
        dfs.append(tmp)

    if not dfs:
        print("[ERROR] No data loaded."); sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    df["error_rate"] = df["error_rate"].astype(float)

    #--------------------------------------------------------------------
    # B) 最近 5 的倍数 error_bin
    #--------------------------------------------------------------------
    df["error_rate_bin"] = ((df["error_rate"] / 5).round() * 5).astype(int)
    df = df.sort_values("error_rate_bin")

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
                ["dataset_id", "error_rate_bin", "cluster_method"], observed=False):
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
                "CEGR": (best["Combined Score"] - worst["Combined Score"]) / d_edr
            })

        ratio_df = pd.DataFrame(rec)
        if ratio_df.empty:
            print(f"[WARN] no CEGR for {task}"); continue

        agg = (ratio_df.groupby(["error_rate_bin", "cluster_method"], observed=False, as_index=False)
               .CEGR.median()
               .rename(columns={"CEGR": "CEGR_median"})
               .sort_values(["error_rate_bin", "cluster_method"]))

        #----------------------------------------------------------------
        # 数据表保存
        #----------------------------------------------------------------
        table_path = os.path.join(out_dir, f"CEGR_5pct_{task}.xlsx")
        agg.to_excel(table_path, index=False)
        print(f"[INFO] saved {table_path}")

        #----------------------------------------------------------------
        # 绘图
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

        # 中文轴标签
        plt.xlabel("错误率",     fontsize=18, fontproperties=cn_font)   # ★
        plt.ylabel("CEGR 中位数", fontsize=18, fontproperties=cn_font)  # ★

        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=18)

        # 中文图例
        leg = plt.legend(title="聚类方法",                 # ★
                         fontsize=12, title_fontsize=13,
                         loc="lower right", frameon=True, framealpha=0.5)
        for txt in leg.get_texts():
            txt.set_fontproperties(cn_font_legend)
        leg.get_title().set_fontproperties(cn_font_legend)

        plt.tight_layout()
        out_pdf = os.path.join(out_dir, f"CEGR_5pct_{task}.pdf")
        plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] saved {out_pdf}")

    print("[INFO] Done. Each task ⇒ PDF chart + data table.")

if __name__ == "__main__":
    main()

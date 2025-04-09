#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# 设置字体为 Times New Roman
matplotlib.rc('font', family='Times New Roman')

def load_data(task_names, data_dir):
    all_data = []
    for task in task_names:
        fpath = os.path.join(data_dir, f"{task}_summary.xlsx")
        if not os.path.isfile(fpath):
            print(f"[WARN] missing {fpath}, skip {task}")
            continue
        tempdf = pd.read_excel(fpath)
        if "task_name" not in tempdf.columns:
            tempdf["task_name"] = task
        all_data.append(tempdf)
    if not all_data:
        print("[ERROR] No data loaded. Please check files.")
        return None
    df = pd.concat(all_data, ignore_index=True)
    return df

##############################################################################
# 2) 雷达图核心绘制函数
##############################################################################
def plot_radar_chart(ax, categories, df_vals, title="Radar Chart"):
    """
    ax : matplotlib 的 polar 坐标子图
    categories : 要在雷达图上的维度列表
    df_vals : 包含多行，每行代表一个清洗方法 / radar line
              row["cleaning_method"] + row[categories] (各列数值)
    title : 子图标题
    """
    N = len(categories)
    angles = [n / float(N) * 2*math.pi for n in range(N)]
    angles.append(angles[0])  # 闭合雷达图

    ax.set_title(title, pad=20, fontsize=12)

    for idx, row in df_vals.iterrows():
        method = row["cleaning_method"]
        values = row[categories].values.tolist()
        values.append(values[0])  # 闭合

        ax.plot(angles, values, label=method, linewidth=2, alpha=0.7)
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)

    # 根据数据范围可自行调整
    # 例如 0..1
    ax.set_ylim(0,1)

##############################################################################
# 3) 可选的一些辅助函数(截断负值, minmax normalize)
##############################################################################
def ensure_positive(series):
    """若允许负值则截断成0, 避免雷达图出现奇怪效果"""
    return series.clip(lower=0)

def minmax_normalize(series):
    mi, ma = series.min(), series.max()
    if mi == ma:
        return pd.Series([0.5]*len(series), index=series.index)
    return (series - mi)/(ma - mi)

##############################################################################
# 4) 主流程
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="radar_plots", help="Output directory for radar images.")
    args = parser.parse_args()

    # user config
    task_names = ["beers","rayyan","flights","hospital"]  # 可自行修改
    data_dir   = os.path.join("..","..","..","results","analysis_results")

    # 雷达图上要展示的指标
    radar_cols = ["precision","recall","F1","EDR","Sil_relative","DB_relative","Comb_relative"]

    df = load_data(task_names, data_dir)
    if df is None or df.empty:
        print("[ERROR] No data found. Exiting.")
        return

    # 确保脚本内需要以下列
    needed = [
        "cluster_method","cleaning_method"
    ] + radar_cols
    for col in needed:
        if col not in df.columns:
            print(f"[ERROR] Missing column => {col}")
            return

    #------------------------------------------
    # 取全部 cluster_method
    #------------------------------------------
    all_methods = sorted(df["cluster_method"].unique())

    # 建立输出目录
    os.makedirs(args.output, exist_ok=True)

    #------------------------------------------
    # 5) 对每个 cluster_method 计算 "所有数据集上" 的中位数
    #    => groupby([cluster_method, cleaning_method]) => median
    #    => 之后每个 cluster_method 画 1 张雷达图
    #------------------------------------------
    # 先 groupby => median
    group_cols = ["cluster_method","cleaning_method"]
    sub_agg = df.groupby(group_cols)[radar_cols].median().reset_index()

    #------------------------------------------
    # 6) 处理负值 + 归一化
    #------------------------------------------
    for col in radar_cols:
        # 截断负值
        sub_agg[col] = ensure_positive(sub_agg[col])
    for col in radar_cols:
        # minmax 0..1
        sub_agg[col] = minmax_normalize(sub_agg[col])

    #------------------------------------------
    # 7) 针对每个 cluster_method => 取 sub => plot radar
    #------------------------------------------
    for cm in all_methods:
        sub_cm = sub_agg[sub_agg["cluster_method"]==cm].copy()
        if sub_cm.empty:
            print(f"[INFO] cluster_method={cm} has no data, skip.")
            continue

        # 做单个雷达图(只有 1 axes)
        fig = plt.figure(figsize=(6,6))
        ax  = fig.add_subplot(111, polar=True)

        # 画
        plot_radar_chart(ax, radar_cols, sub_cm, title=f"{cm} (median across all data)")

        # legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.2,1.05), fontsize=8)

        plt.tight_layout()
        outfname = f"radar_{cm}.png"
        outpath  = os.path.join(data_dir, outfname)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVE] => {outpath}")

    print("[INFO] Done. Each cluster_method => 1 radar chart with cleaning_method lines.")


if __name__ == "__main__":
    main()

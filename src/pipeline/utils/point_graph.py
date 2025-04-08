#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# 在所有绘图代码执行之前，设置字体为 Times New Roman
matplotlib.rc('font', family='Times New Roman')
# 如果需要对 seaborn 的默认样式做进一步调整，也可以加:
# sns.set_style("whitegrid")  # 如果想保留 seaborn 的白色网格风格
# sns.set(font="Times New Roman")  # 进一步确保 seaborn 中也用 Times New Roman

def main():
    # -------------------------------------------
    # 1) 读取Excel并合并
    # -------------------------------------------
    task_names = ["beers","rayyan","flights","hospital"]  # 如需可自行修改
    data_dir = os.path.join("..","..","..","results","analysis_results")  # 示例路径

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
        print("No data loaded. Please check file paths.")
        sys.exit(1)

    df = pd.concat(all_data, ignore_index=True)

    # -------------------------------------------
    # 2) 需要列
    # -------------------------------------------
    needed_cols = [
        "dataset_id","cluster_method","EDR","F1",
        "Comb_relative","Sil_relative","DB_relative"
    ]
    for c in needed_cols:
        if c not in df.columns:
            print(f"[ERROR] Column '{c}' not found in data. Please check naming => {c}")
            sys.exit(1)

    # -------------------------------------------
    # 3) 定义 (x_metric vs. y_metric) 组合 => 6 张图
    # -------------------------------------------
    combos = [
        ("EDR","Comb_relative"),
        ("EDR","Sil_relative"),
        ("EDR","DB_relative"),
        ("F1","Comb_relative"),
        ("F1","Sil_relative"),
        ("F1","DB_relative")
    ]

    out_dir = os.path.join("..","..","..","results","analysis_results")
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------
    # 4) 循环处理每个组合
    # -------------------------------------------
    for (x_metric, y_metric) in combos:
        print(f"\n=== Analyzing correlation: {x_metric} vs. {y_metric} ===")

        # (dataset_id, cluster_method) 分组 => Pearson r
        results_list = []
        grp = df.groupby(["dataset_id","cluster_method"])
        for (ds_id, clus_m), subdf in grp:
            if len(subdf) < 2:
                results_list.append({
                    "dataset_id": ds_id,
                    "cluster_method": clus_m,
                    "pearson_r": np.nan,
                    "pval": np.nan,
                    "n_points": len(subdf)
                })
                continue

            x = subdf[x_metric].values
            y = subdf[y_metric].values
            r, pval = pearsonr(x, y)

            results_list.append({
                "dataset_id": ds_id,
                "cluster_method": clus_m,
                "pearson_r": r,
                "pval": pval,
                "n_points": len(subdf)
            })

        results_df = pd.DataFrame(results_list)

        # pivot: 行=dataset_id, 列=cluster_method, 值=r
        pivot_df = results_df.pivot(index="dataset_id", columns="cluster_method", values="pearson_r")
        print(f"Pivot table of r for {x_metric} vs. {y_metric}:")
        print(pivot_df)

        # 保存 pivot
        pivot_csv = os.path.join(out_dir,f"pearson_{x_metric}_vs_{y_metric}_pivot.csv")
        pivot_df.to_csv(pivot_csv, float_format="%.3f")
        print(f"[INFO] pivot saved => {pivot_csv}")

        # 根据绝对值max => 排序 dataset_id
        row_abs = pivot_df.abs()
        ds_max_abs = row_abs.max(axis=1)
        ds_sorted = ds_max_abs.sort_values(ascending=False).index.tolist()

        # stack => 长表
        long_df = pivot_df.stack().reset_index()
        long_df.columns = ["dataset_id","cluster_method","r_value"]
        long_df["r_value"] = long_df["r_value"].fillna(0.0)

        # 设定分类顺序
        ds_cat_type = pd.CategoricalDtype(ds_sorted, ordered=True)
        long_df["dataset_id"] = long_df["dataset_id"].astype(ds_cat_type)

        cm_unique = sorted(long_df["cluster_method"].unique())
        cm_cat_type = pd.CategoricalDtype(cm_unique, ordered=True)
        long_df["cluster_method"] = long_df["cluster_method"].astype(cm_cat_type)

        # 绘图
        plt.figure(figsize=(14,6))
        ax = sns.scatterplot(
            data=long_df,
            x="dataset_id",
            y="cluster_method",
            size=long_df["r_value"].abs(),
            hue="r_value",
            palette="RdBu",
            sizes=(20,200),
            alpha=0.8,
            edgecolor="black",
            legend=False  # <--- 1) 去掉自动生成的 r_value 图例
        )

        # Colorbar
        norm = plt.Normalize(vmin=-1, vmax=1)
        sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label(f"Pearson r ({x_metric} vs. {y_metric})")

        # 2) 删除坐标轴标签
        plt.xlabel('')
        plt.ylabel('')

        # 保留主标题（如不想保留可注释掉以下行）
        plt.title(f"Pearson r: {x_metric} vs. {y_metric}\n(dot size=|r|, color=sign(r))")

        out_png = os.path.join(out_dir, f"dot_{x_metric}_vs_{y_metric}_sorted.png")
        plt.tight_layout()
        # 3) 提高图片清晰度
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"[INFO] => {out_png} saved.\n")

    print("[INFO] All combos done. 6 figures generated.")

if __name__=="__main__":
    main()

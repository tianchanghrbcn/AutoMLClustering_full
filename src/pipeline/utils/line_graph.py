#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# 设置字体为 Times New Roman
matplotlib.rc('font', family='Times New Roman')
# 如果想保留 seaborn 的网格，可以加:
# sns.set_style("whitegrid")

def main():
    #--------------------------------------------------------------------
    # A) 读取多份Excel并合并
    #--------------------------------------------------------------------
    task_names = ["beers","rayyan","flights","hospital"]  # 可自行修改
    data_dir = os.path.join("..","..","..","results","analysis_results")

    all_data = []
    for task in task_names:
        fpath = os.path.join(data_dir, f"{task}_summary.xlsx")
        if not os.path.isfile(fpath):
            print(f"[WARN] missing {fpath}, skip {task}")
            continue
        temp_df = pd.read_excel(fpath)
        if "task_name" not in temp_df.columns:
            temp_df["task_name"] = task
        all_data.append(temp_df)

    if not all_data:
        print("[ERROR] No data loaded. Please check file paths.")
        sys.exit(1)

    df = pd.concat(all_data, ignore_index=True)
    print(f"[INFO] Merged DataFrame shape=", df.shape)

    #--------------------------------------------------------------------
    # B) 确保关键列
    #--------------------------------------------------------------------
    needed_cols = [
        "task_name","dataset_id","error_rate","cluster_method",
        "cleaning_method","EDR","Comb_relative"
    ]
    for c in needed_cols:
        if c not in df.columns:
            print(f"[ERROR] Missing col: {c}")
            sys.exit(1)

    # 转为 float
    df["error_rate"] = df["error_rate"].astype(float)

    #--------------------------------------------------------------------
    # C) 按 5% 步长划分错误率区间 => [0,5), [5,10), [10,15), [15,20),
    #   [20,25), [25,30), [30, +∞)
    #--------------------------------------------------------------------
    bins = [0, 5, 10, 15, 20, 25, 30, 999999]
    labels = ["0-5","5-10","10-15","15-20","20-25","25-30",">=30"]
    df["error_rate_bin"] = pd.cut(
        df["error_rate"], bins=bins, labels=labels, right=False
    )

    print("[INFO] error_rate_bin counts:")
    print(df["error_rate_bin"].value_counts())

    #--------------------------------------------------------------------
    # D) 逐个 task_name 画图
    #--------------------------------------------------------------------
    out_dir = os.path.join(data_dir, "plots_errbin_line_5pct_CEGR_byTask")
    os.makedirs(out_dir, exist_ok=True)

    task_list = sorted(df["task_name"].unique())
    for task in task_list:
        sub_task = df[df["task_name"] == task].copy()
        if sub_task.empty:
            continue

        # E) 对 (dataset_id, error_rate_bin, cluster_method) 分组:
        #   仅考虑 "EDR" 维度 => EDR最大 vs EDR最小 => 计算 CEGR
        records = []
        grp = sub_task.groupby(["dataset_id","error_rate_bin","cluster_method"])
        for (ds_id, ebin, clus_m), subdf in grp:
            # subdf => 该场景下多个 cleaning_method
            # EDR_max vs EDR_min
            if len(subdf) < 2:
                continue

            idx_edr_best = subdf["EDR"].idxmax()
            idx_edr_worst= subdf["EDR"].idxmin()

            bestEDR_edr  = subdf.loc[idx_edr_best,"EDR"]
            worstEDR_edr = subdf.loc[idx_edr_worst,"EDR"]
            bestEDR_comb = subdf.loc[idx_edr_best,"Comb_relative"]
            worstEDR_comb= subdf.loc[idx_edr_worst,"Comb_relative"]

            delta_edr = bestEDR_edr - worstEDR_edr
            if abs(delta_edr) < 1e-10:
                # EDR相同 => skip
                continue

            # CEGR = [Comb( EDR-max ) - Comb( EDR-min )] / [ EDR-max - EDR-min ]
            CEGR_val = (bestEDR_comb - worstEDR_comb) / delta_edr

            records.append({
                "error_rate_bin": ebin,
                "cluster_method": clus_m,
                "CEGR": CEGR_val,
                "dataset_id": ds_id
            })

        ratio_df = pd.DataFrame(records)
        if ratio_df.empty:
            print(f"[WARN] => No data for task={task}. skip.")
            continue

        # F) 对( error_rate_bin, cluster_method )聚合 => median(CEGR) in all dataset_id
        grouped2 = ratio_df.groupby(["error_rate_bin","cluster_method"], as_index=False)
        agg_df = grouped2["CEGR"].median()
        agg_df.rename(columns={"CEGR":"CEGR_median"}, inplace=True)

        # error_rate_bin 设为有序分类
        cat_type = pd.CategoricalDtype(labels, ordered=True)
        agg_df["error_rate_bin"] = agg_df["error_rate_bin"].astype(cat_type)
        agg_df = agg_df.sort_values(["error_rate_bin","cluster_method"])

        if agg_df.empty:
            print(f"[WARN] => no CEGR aggregator for {task}")
            continue

        # G) 绘图 => x=error_rate_bin, y=CEGR_median, hue=cluster_method
        plt.figure(figsize=(8,6))
        sns.lineplot(
            data=agg_df,
            x="error_rate_bin",
            y="CEGR_median",
            hue="cluster_method",
            marker="o",
            linewidth=2
        )

        plt.xlabel("Error Rate Bin (%)", fontsize=13)
        plt.ylabel("Median CEGR\n[Comb(EDR-max)-Comb(EDR-min)] / [EDR-max - EDR-min]", fontsize=13)
        plt.title(f"Task={task} => 5% bin, median CEGR in EDR dimension", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Cluster Method", fontsize=11)

        plt.tight_layout()
        out_png = os.path.join(out_dir, f"CEGR_5pct_{task}.png")
        plt.savefig(out_png, dpi=300)
        plt.close()

        print(f"[INFO] => {out_png} saved for task={task}")

    print("[INFO] Done. Each task_name => one chart with CEGR approach.")


if __name__=="__main__":
    main()

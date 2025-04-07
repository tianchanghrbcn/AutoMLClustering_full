#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_metric_for_all_tasks(task_names, data_dir, metric):
    """
    对同一个指标 metric (如 'Sil_relative')，创建一个图(2x2子图)，
    分别画 4 个 task (beers, rayyan, flights, hospital)，
    在每个子图中用箱线图 x=cleaning_method, y=metric, hue=cluster_method
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{metric} for 4 tasks (boxplot)")

    # 遍历任务,把 (row,col) => i//2, i%2
    for i, task in enumerate(task_names):
        data_file = os.path.join(data_dir, f"{task}_summary.xlsx")
        ax = axs[i // 2, i % 2]

        if not os.path.isfile(data_file):
            ax.text(0.5, 0.5, f"[WARN] No file for {task}", ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{task} - No Data")
            continue

        df = pd.read_excel(data_file)
        # 过滤相关列
        df_plot = df.dropna(subset=[metric, "cleaning_method", "cluster_method"]).copy()
        if df_plot.empty:
            ax.text(0.5, 0.5, f"No valid data for {task}", ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{task} - No valid {metric}")
            continue

        sns.boxplot(
            x="cleaning_method",
            y=metric,
            hue="cluster_method",
            data=df_plot,
            ax=ax
        )
        ax.set_title(f"{task} - {metric}")
        ax.tick_params(axis='x', rotation=45)

        # 不要子图自己的legend
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # 全局添加legend (从第一个子图拿handles)
    handles, labels = axs[0,0].get_legend_handles_labels() if axs[0,0].get_legend() else (None,None)
    if handles and labels:
        fig.legend(handles, labels, loc='upper right')

    fig.tight_layout(rect=[0,0,1,0.96])
    out_fig = os.path.join(data_dir, f"figure_relative_{metric}_4tasks.png")
    plt.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {metric} distribution for all 4 tasks => {out_fig}")


def main():
    # 假设要处理的任务
    task_names = ["beers", "rayyan", "flights", "hospital"]
    # 数据所在目录: ../../../results/analysis_results
    data_dir = os.path.join("..", "..", "..", "results", "analysis_results")

    # 我们需要绘制的3个指标
    metrics = ["Sil_relative", "DB_relative", "Comb_relative"]

    for metric in metrics:
        plot_metric_for_all_tasks(task_names, data_dir, metric)

if __name__ == "__main__":
    main()

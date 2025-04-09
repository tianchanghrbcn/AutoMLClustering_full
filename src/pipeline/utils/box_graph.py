#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# 在绘图之前，统一设置字体为 Times New Roman
matplotlib.rc('font', family='Times New Roman')

# 增大横纵坐标轴刻度数字的字体大小
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14

# 增大标题字体（axes.titlesize）大小
plt.rcParams["axes.titlesize"] = 16

def plot_metric_for_all_tasks(task_names, data_dir, metric):
    """
    对同一个指标 metric (如 'Sil_relative')，创建一个图(2x2子图)，
    分别画 4 个 task (beers, rayyan, flights, hospital)，
    在每个子图中用箱线图 x=cleaning_method, y=metric, hue=cluster_method
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    # 增大 suptitle 字体
    fig.suptitle(f"{metric} for 4 tasks (boxplot)", fontsize=20)

    # 用于记录全局图例的句柄与标签
    global_handles, global_labels = None, None

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

        # 绘制箱线图
        sns.boxplot(
            x="cleaning_method",
            y=metric,
            hue="cluster_method",
            data=df_plot,
            ax=ax
        )

        ax.set_title(f"{task} - {metric}")
        ax.tick_params(axis='x', rotation=45)
        # 移除子图自己的 x、y 轴标签
        ax.set_xlabel('')
        ax.set_ylabel('')

        # 如果当前子图产生了图例，就获取其 handles 和 labels
        if ax.get_legend() is not None:
            handles, labels = ax.get_legend_handles_labels()

            # 如果全局还没保存过 handles, labels，则保存第一个有效子图的
            if not global_handles and not global_labels:
                global_handles, global_labels = handles, labels

            # 移除子图自身的 legend（防止子图重复显示 legend）
            ax.get_legend().remove()

    # 如果成功获取到全局句柄和标签，则在主标题下方展示一个全局横向图例
    if global_handles and global_labels:
        fig.legend(
            global_handles,
            global_labels,
            loc='upper center',
            # bbox_to_anchor 的 (x=0.5, y=0.92) 表示图例居中且稍微贴近主标题
            bbox_to_anchor=(0.5, 0.92),
            ncol=len(global_labels),
            title='Cluster Method',
            fontsize=12,          # 适当增大图例字体
            title_fontsize=14     # 适当增大图例标题字体
        )

    # 调整布局以防止重叠
    # rect=[left, bottom, right, top], 让出一点顶部空间给 suptitle 和 legend
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_fig = os.path.join(data_dir, f"figure_relative_{metric}_4tasks.png")
    # 增大 dpi 提高清晰度
    plt.savefig(out_fig, dpi=300)
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

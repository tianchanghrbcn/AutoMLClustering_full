import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_cleaning_effect_sub(ax, df, task_name):
    """
    在给定子图 ax 上绘制“不同清洗方法 vs (Precision, Recall, F1, EDR)”的并列柱状图
    使用 median() 聚合
    """
    metrics = ["precision", "recall", "F1", "EDR"]
    # 用中位数
    agg_df = df.groupby("cleaning_method")[metrics].median().reset_index()

    cleaning_methods = agg_df["cleaning_method"].tolist()
    x = np.arange(len(cleaning_methods))
    bar_width = 0.2

    # 逐列画4个bar
    for i, metric in enumerate(metrics):
        ax.bar(
            x + i*bar_width,
            agg_df[metric],
            width=bar_width,
            label=metric  # 确保每个bar都有自己的label
        )

    ax.set_xticks(x + bar_width*(len(metrics)-1)/2)
    ax.set_xticklabels(cleaning_methods, rotation=45, ha="right")

    # Y轴范围: 若 EDR 有负值，需要 min(0, ...)
    min_val = min(0, agg_df[metrics].min().min())
    max_val = 1.1 * agg_df[metrics].max().max()
    ax.set_ylim(min_val, max_val)

    ax.set_title(f"{task_name} (cleaning, median)")

def plot_cluster_result_sub(ax, df, task_name):
    """
    在给定子图 ax 上绘制“不同聚类算法 vs (Silhouette, DB, Combined Score)”的并列柱状图
    使用 median() 聚合 + minmax 归一化(对DB做1-).
    """
    metrics = ["Silhouette Score", "Davies-Bouldin Score", "Combined Score"]
    agg_df = df.groupby("cluster_method")[metrics].median().reset_index()

    if agg_df.empty:
        ax.text(0.5, 0.5, f"No cluster_method data for {task_name}", ha='center', va='center')
        return

    # min-max 归一化
    min_sil = agg_df["Silhouette Score"].min()
    max_sil = agg_df["Silhouette Score"].max()
    min_db  = agg_df["Davies-Bouldin Score"].min()
    max_db  = agg_df["Davies-Bouldin Score"].max()
    min_comb= agg_df["Combined Score"].min()
    max_comb= agg_df["Combined Score"].max()

    def minmax_scale(x, mn, mx):
        if mx - mn < 1e-12:
            return 0.5
        return (x - mn)/(mx - mn)

    sil_norm = []
    db_norm  = []
    comb_norm= []
    for _, row in agg_df.iterrows():
        raw_sil = minmax_scale(row["Silhouette Score"], min_sil, max_sil)
        raw_db  = minmax_scale(row["Davies-Bouldin Score"], min_db, max_db)
        inv_db  = 1.0 - raw_db
        raw_comb= minmax_scale(row["Combined Score"], min_comb, max_comb)

        sil_norm.append(raw_sil)
        db_norm.append(inv_db)
        comb_norm.append(raw_comb)

    agg_df["Sil_norm"] = sil_norm
    agg_df["DB_norm"]  = db_norm
    agg_df["Comb_norm"]= comb_norm

    cluster_methods = agg_df["cluster_method"].tolist()
    x = np.arange(len(cluster_methods))
    bar_width = 0.2

    # 3 个并列bar
    ax.bar(x,             agg_df["Sil_norm"],  width=bar_width, label="Silhouette")
    ax.bar(x+bar_width,   agg_df["DB_norm"],   width=bar_width, label="DB(1-minmax)")
    ax.bar(x+2*bar_width, agg_df["Comb_norm"], width=bar_width, label="Combined")

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(cluster_methods, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(f"{task_name} (cluster, median)")

def main():
    # 要处理的任务
    task_names = ["beers","rayyan","flights","hospital"]
    data_dir = os.path.join("..","..","..","results","analysis_results")

    # 图1: 不同清洗方法(4子图)
    fig1, axs1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle("Figure 1: Different Cleaning Methods (Median of Precision/Recall/F1/EDR)")

    # 图2: 不同聚类算法(4子图)
    fig2, axs2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle("Figure 2: Different Cluster Methods (Median => Sil, DB(1-), Combined)")

    for i, task in enumerate(task_names):
        data_file = os.path.join(data_dir, f"{task}_summary.xlsx")
        if not os.path.isfile(data_file):
            print(f"[WARN] {data_file} not found, skip {task}.")
            continue
        df = pd.read_excel(data_file)

        row_id = i // 2
        col_id = i % 2

        # 子图1
        plot_cleaning_effect_sub(axs1[row_id, col_id], df, task)
        # 子图2
        plot_cluster_result_sub(axs2[row_id, col_id], df, task)

    # 提取图例并放右上角 => 图1
    handles1, labels1 = axs1[0,0].get_legend_handles_labels()
    fig1.legend(handles1, labels1, loc='upper right')
    fig1.tight_layout(rect=[0,0,1,0.96])

    # 图2
    handles2, labels2 = axs2[0,0].get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc='upper right')
    fig2.tight_layout(rect=[0,0,1,0.96])

    out_fig1 = os.path.join(data_dir, "figure1_cleaning_4tasks_median.png")
    out_fig2 = os.path.join(data_dir, "figure2_cluster_4tasks_median.png")
    fig1.savefig(out_fig1, dpi=150)
    fig2.savefig(out_fig2, dpi=150)
    plt.close(fig1)
    plt.close(fig2)

    print(f"[INFO] Saved 2 big figures with 4 subplots each:\n - {out_fig1}\n - {out_fig2}")

if __name__ == "__main__":
    main()

import os
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt


##############################################################################
# 1) 读取多个Excel并合并
##############################################################################

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
# 2) 绘制单个雷达图(子图) 的核心函数
##############################################################################

def plot_radar_chart(ax, categories, df_vals, title="Radar Chart"):

    N = len(categories)
    angles = [n / float(N) * 2*math.pi for n in range(N)]
    angles.append(angles[0])  # 闭合

    ax.set_title(title, pad=20, fontsize=10)

    for idx, row in df_vals.iterrows():
        method = row["cleaning_method"]
        values = row[categories].values.tolist()
        values.append(values[0])  # 闭合
        ax.plot(angles, values, label=method)
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)

    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=7)
    ax.set_ylim(0,1)

    # legend可放在外部统一
    # 也可以放这里 but subplot多,易重叠
    # ax.legend(loc='best', fontsize=6)

##############################################################################
# 3) 辅助: 归一化/保证非负
##############################################################################

def ensure_positive(series):
    """截断负值到0"""
    return series.clip(lower=0)

def minmax_normalize(series):
    mi, ma = series.min(), series.max()
    if mi==ma:
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
    data_dir = os.path.join("..","..","..","results","analysis_results")
    # 期望7个列
    radar_cols = ["precision","recall","F1","EDR","Sil_relative","DB_relative","Comb_relative"]

    df = load_data(task_names, data_dir)
    if df is None or df.empty:
        print("[ERROR] No data found. Exiting.")
        return

    os.makedirs(args.output, exist_ok=True)

    # 取全部task_name
    all_tasks = df["task_name"].unique()
    for task in all_tasks:
        # 先在 outputDir 下创建 taskName 文件夹
        task_dir = os.path.join(data_dir, "radar", task)
        os.makedirs(task_dir, exist_ok=True)

        # 筛选只取当前task
        sub_task = df[df["task_name"]==task].copy()
        if len(sub_task)==0:
            continue

        all_dsid = sub_task["dataset_id"].unique()

        for dsid in all_dsid:
            # 取该 dataset
            sub_ds = sub_task[sub_task["dataset_id"]==dsid].copy()
            if len(sub_ds)==0:
                continue

            # 找到该 dataset 下有哪些 cluster_method
            # 例如 5 个: KMEANS, GMM, HC, KMEANSNF, KMEANSPPS...
            all_cms = sub_ds["cluster_method"].unique()

            # 先决定 subplot 布局: 5 or fewer => 2x3  (最多6 subplot)
            # user can do: n_sub = len(all_cms), grid?
            n_sub = len(all_cms)
            # for simplicity do row=2, col=3 => up to 6 subplots
            n_rows = 2
            n_cols = 3
            fig = plt.figure(figsize=(10,8))

            # sort cluster_method to keep stable order
            all_cms_sorted = sorted(all_cms)

            # subplot index
            for i, cm in enumerate(all_cms_sorted):
                r = i // n_cols
                c = i % n_cols
                ax = fig.add_subplot(n_rows, n_cols, i+1, polar=True)

                # 取 (dsid, cm)
                sub_combo = sub_ds[sub_ds["cluster_method"]==cm].copy()
                if len(sub_combo)==0:
                    # empty
                    ax.set_title(f"{cm} (no data)", fontsize=10)
                    continue

                # group by cleaning_method => 7列 => mean
                group_cols = ["cleaning_method"]
                sub_agg = sub_combo.groupby(group_cols)[radar_cols].mean().reset_index()

                # 处理负值 & 归一化
                for col in radar_cols:
                    sub_agg[col] = ensure_positive(sub_agg[col])
                for col in radar_cols:
                    sub_agg[col] = minmax_normalize(sub_agg[col])
                # DB_relative如需(1-x)可加: sub_agg["DB_relative"] = 1.0 - sub_agg["DB_relative"]

                # 画子图
                sub_title = f"{cm}"
                plot_radar_chart(ax, radar_cols, sub_agg, title=sub_title)

                # 显示legend只在最后或某特定子图, 避免重复
                if i==(n_sub-1):
                    ax.legend(loc='upper right', bbox_to_anchor=(1.45,1.2), fontsize=8)

            # 整个figure标题
            fig.suptitle(f"Radar for task={task}, dsid={dsid}", fontsize=12, y=1.02)

            plt.tight_layout()
            # 保存
            outfname = f"ds_{dsid}.png"
            outpath = os.path.join(task_dir, outfname)
            plt.savefig(outpath, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[SAVE] {outpath}")

    print("[INFO] All done.")

if __name__ == "__main__":
    main()

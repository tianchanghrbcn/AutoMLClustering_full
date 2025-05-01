import os
import argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# 1) 将字体改为 Times New Roman
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
# 2) 绘制单个雷达图(子图) 的核心函数
##############################################################################

def plot_radar_chart(ax, categories, df_vals, title="Radar Chart"):
    import math
    N = len(categories)
    angles = [n / float(N) * 2*math.pi for n in range(N)]
    angles.append(angles[0])  # 闭合

    ax.set_title(title, pad=20, fontsize=10)

    for idx, row in df_vals.iterrows():
        method = row["cleaning_method"]
        values = row[categories].values.tolist()
        values.append(values[0])  # 闭合

        # 3) 增大线条宽度并调高透明度，让对比更明显
        ax.plot(angles, values, label=method, linewidth=2, alpha=0.7)
        # 填充内部，可适当减小alpha，保证线条对比度
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)

    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=7)
    ax.set_ylim(0,1)

##############################################################################
# 3) 辅助: 归一化/保证非负
##############################################################################

def ensure_positive(series):
    """截断负值到0"""
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
    data_dir = os.path.join("..","..","..","results","analysis_results")
    radar_cols = ["precision","recall","F1","EDR","Sil_relative","DB_relative","Comb_relative"]

    df = load_data(task_names, data_dir)
    if df is None or df.empty:
        print("[ERROR] No data found. Exiting.")
        return

    os.makedirs(args.output, exist_ok=True)

    # 取全部 task_name
    all_tasks = df["task_name"].unique()
    for task in all_tasks:
        task_dir = os.path.join(data_dir, "radar", task)
        os.makedirs(task_dir, exist_ok=True)

        # 筛选只取当前task
        sub_task = df[df["task_name"] == task].copy()
        if len(sub_task) == 0:
            continue

        all_dsid = sub_task["dataset_id"].unique()

        for dsid in all_dsid:
            sub_ds = sub_task[sub_task["dataset_id"]==dsid].copy()
            if len(sub_ds) == 0:
                continue

            all_cms = sub_ds["cluster_method"].unique()
            n_sub = len(all_cms)
            n_rows = 2
            n_cols = 3
            fig = plt.figure(figsize=(10,8))

            all_cms_sorted = sorted(all_cms)

            for i, cm in enumerate(all_cms_sorted):
                r = i // n_cols
                c = i % n_cols
                ax = fig.add_subplot(n_rows, n_cols, i+1, polar=True)

                sub_combo = sub_ds[sub_ds["cluster_method"] == cm].copy()
                if len(sub_combo) == 0:
                    ax.set_title(f"{cm} (no data)", fontsize=10)
                    continue

                group_cols = ["cleaning_method"]
                sub_agg = sub_combo.groupby(group_cols)[radar_cols].mean().reset_index()

                # 处理负值 & 归一化
                for col in radar_cols:
                    sub_agg[col] = ensure_positive(sub_agg[col])
                for col in radar_cols:
                    sub_agg[col] = minmax_normalize(sub_agg[col])

                # DB_relative如需(1-x)可再加:
                # sub_agg["DB_relative"] = 1.0 - sub_agg["DB_relative"]

                plot_radar_chart(ax, radar_cols, sub_agg, title=cm)

                # 只在最后一个子图绘制 legend
                if i == (n_sub - 1):
                    ax.legend(loc='upper right', bbox_to_anchor=(1.45,1.2), fontsize=8)

            fig.suptitle(f"Radar for task={task}, dsid={dsid}", fontsize=12, y=1.02)

            # 使用高分辨率
            plt.tight_layout()
            outfname = f"ds_{dsid}.png"
            outpath = os.path.join(task_dir, outfname)
            plt.savefig(outpath, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[SAVE] {outpath}")

    print("[INFO] All done.")

if __name__ == "__main__":
    main()

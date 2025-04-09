import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

def plot_heatmap_for_each_task(df, data_dir, value_col='Comb_relative'):
    """
    在传入的 DataFrame 中，通过 (cluster_method, cleaning_method) 对value_col做 median 聚合，
    按 task_name 分别绘制热力图，保存到 data_dir 目录下.

    其中:
    - df: 已包含 (task_name, cluster_method, cleaning_method, value_col)
    - data_dir: 与 xlsx 相同的文件夹, 用于保存 PNG
    - value_col: 热力图中显示的数值列, e.g. 'Comb_relative'
    """

    required_cols = {'task_name', 'cluster_method', 'cleaning_method', value_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame缺少以下列: {missing}")

    # 设置字体为Times New Roman
    # 注: 需要系统已安装Times New Roman，如在Windows默认有，若在Linux可先安装相应字体
    matplotlib.rcParams['font.family'] = 'Times New Roman'

    # 按 task_name 分组，分别绘制图
    task_names = df['task_name'].unique()
    for tname in task_names:
        df_sub = df[df['task_name'] == tname].copy()
        if df_sub.empty:
            print(f"[WARN] task_name={tname}没有数据，跳过。")
            continue

        # 计算 median
        pivot_df = (df_sub
                    .groupby(['cluster_method', 'cleaning_method'], as_index=False)[value_col]
                    .median())

        # pivot成 行=cluster_method, 列=cleaning_method, 值= median(value_col)
        heatmap_data = pivot_df.pivot(index='cluster_method',
                                      columns='cleaning_method',
                                      values=value_col)

        if heatmap_data.empty:
            print(f"[WARN] task_name={tname} 计算 pivot 后无数据。")
            continue

        # 开始绘图
        plt.figure(figsize=(10, 6), dpi=300)  # dpi=300 提高清晰度

        # 如果 Comb_relative=1表示基准，可设 center=1.0
        # 为加大颜色对比度, 可设置 vmin/vmax 或使用更明显的配色
        sns.heatmap(heatmap_data,
                    annot=True,
                    fmt=".3f",
                    cmap='RdYlGn',    # 红-黄-绿, 高值更绿, 低值更红
                    center=1.0,       # 若1.0为基准
                    vmin=0.5, vmax=1.5,  # 视实际分布可根据调节, 让色彩对比更明显
                    )

        plt.title(f"Heatmap of {value_col} for task_name={tname}", fontsize=14)
        plt.xlabel("Cleaning Method", fontsize=12)
        plt.ylabel("Cluster Method", fontsize=12)

        # 保存到与 xlsx 相同的目录
        outfile = os.path.join(data_dir, f"heatmap_{tname}_{value_col}.png")
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)  # 再次声明 dpi以防
        plt.close()
        print(f"[INFO] 为 task_name={tname} 绘制的图已保存到: {outfile}")


def main():
    # Step 1: 读取多个 task_name 的 summary 文件并合并
    task_names = ["beers", "rayyan", "flights", "hospital"]  # 可调整
    data_dir = os.path.join("..", "..", "..", "results", "analysis_results")

    all_data = []
    for task in task_names:
        fpath = os.path.join(data_dir, f"{task}_summary.xlsx")
        if not os.path.isfile(fpath):
            print(f"[WARN] missing {fpath}, skip {task}")
            continue

        temp_df = pd.read_excel(fpath)
        # 若 'task_name' 不在列里，就加上
        if "task_name" not in temp_df.columns:
            temp_df["task_name"] = task

        all_data.append(temp_df)

    if not all_data:
        print("[ERROR] 没有读取到任何数据，程序终止。")
        return

    df_all = pd.concat(all_data, ignore_index=True)

    # Step 2: 检查 df_all 是否包含必需列
    needed_cols = ['task_name', 'cluster_method', 'cleaning_method', 'Comb_relative']
    for col in needed_cols:
        if col not in df_all.columns:
            print(f"[ERROR] df_all中缺少必需列 {col}")
            return

    # Step 3: 调用绘图函数(这里传 data_dir 进去, 让保存到相同目录)
    plot_heatmap_for_each_task(df_all, data_dir, value_col='Comb_relative')


if __name__ == "__main__":
    main()

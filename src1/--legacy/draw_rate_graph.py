import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_absolute_scores(
        algo1,
        dataset,
        base_dir=r"D:\algorithm paper\ML algorithms codes\data_experiments\results\3_analyzed_data\analysis_original_results"
):

    dataset_path = os.path.join(base_dir, dataset)

    # 仅收集以“_relative.csv”结尾的文件
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith("_relative.csv")]
    if not csv_files:
        print(f"在路径 {dataset_path} 下未找到任何 '_relative.csv' 文件，请检查目录或文件名。")
        return

    # 定义需要绘制的聚类算法
    clustering_methods = ["HC", "AffinityPropagation", "GMM", "KMeans", "DBSCAN", "OPTICS"]
    # 为不同聚类算法设置“Marker”形状
    method_markers = {
        "HC": "o",  # 圆
        "AffinityPropagation": "^",  # 上三角
        "GMM": "s",  # 正方形
        "KMeans": "D",  # 菱形
        "DBSCAN": "x",  # 叉号
        "OPTICS": "+"  # 加号
    }

    # 用于存放“错误率”和不同聚类算法在该错误率下的“绝对得分”
    scores_dict = {m: [] for m in clustering_methods}
    error_rates = []

    # 读取各个 CSV 文件，提取信息
    for csv_file in csv_files:
        file_path = os.path.join(dataset_path, csv_file)

        # 解析文件名，获取错误率 (error_rate)
        # 假设形如：analysis_beers_46.73%_relative.csv
        file_name_noext = csv_file.replace("_relative.csv", "")
        parts = file_name_noext.split("_")
        # 假设最后一个部分就是 "46.73%"（含%）
        err_str = parts[-1]
        # 将 "46.73%" -> 46.73 (浮点数)
        err_float = float(err_str.replace("%", ""))
        error_rates.append(err_float)

        # 读取 CSV 到 DataFrame
        df = pd.read_csv(file_path)

        # 仅保留当前清洗算法 (algo1) 的行
        subset_df = df[df["Cleaning Algorithm"] == algo1]

        # 如果没有该清洗算法
        if subset_df.empty:
            for method in clustering_methods:
                scores_dict[method].append(0.0)
            continue

        for method in clustering_methods:
            row = subset_df[subset_df["Clustering Method"] == method]
            if not row.empty:
                score = row["Score"].values[0]
                # 超过 3.0 则截断为 3.0
                if score > 3.0:
                    score = 3.0
            else:
                # 若无此组合则记为0
                score = 0.0
            scores_dict[method].append(score)

    # 对错误率进行升序排序
    sorted_indices = sorted(range(len(error_rates)), key=lambda i: error_rates[i])
    sorted_error_rates = [error_rates[i] for i in sorted_indices]
    for method in clustering_methods:
        scores_dict[method] = [scores_dict[method][i] for i in sorted_indices]

    # x 轴离散等分：直接用 0,1,2,... 来表示
    x_positions = list(range(len(sorted_error_rates)))

    # 开始绘图
    plt.figure(figsize=(10, 6))

    # 每种聚类算法绘制一条折线（等距 x_positions）
    for method in clustering_methods:
        plt.plot(
            x_positions,
            scores_dict[method],
            marker=method_markers[method],
            markersize=10,  # Marker 大小
            label=method
        )

    # 标题
    plt.title(
        f"Combined Scores of Clustering Algorithms vs. Error Rates on '{dataset}' Dataset after '{algo1}' Cleaning")

    # x 轴显示错误率
    plt.xticks(x_positions, [f"{er:.2f}%" for er in sorted_error_rates])
    plt.xlabel("Error Rate (%)")
    plt.ylabel("Score")

    # 稍稍超过 3.0，防止顶部截断
    plt.ylim([0, 3.1])

    # 去掉图像内部的横竖网格线
    plt.grid(False)

    # (如需在错误率=25%处画线和标记，可保留以下代码，否则可删除)
    target_value = 25.0
    vertical_line_x = None
    if target_value in sorted_error_rates:
        i_25 = sorted_error_rates.index(target_value)
        if i_25 < len(sorted_error_rates) - 1:
            vertical_line_x = i_25 + 0.5
        else:
            vertical_line_x = i_25 + 0.25
    else:
        for i in range(len(sorted_error_rates) - 1):
            if sorted_error_rates[i] < target_value < sorted_error_rates[i + 1]:
                vertical_line_x = i + 0.5
                break
    if vertical_line_x is not None:
        plt.axvline(vertical_line_x, linestyle='--', color='gray')
        plt.scatter([vertical_line_x], [3.0], marker='*', s=200, color='red', zorder=5)

    # 图例位置可根据需要进行微调
    plt.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        borderaxespad=0,
        frameon=True
    )

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 显示或保存图像
    plt.show()
    # plt.savefig(f"D:\\algorithm paper\\ML algorithms codes\\data_experiments\\results\\4_final_results\\graphs\\error_rate_graph\\{algo1}_{dataset}_combined_scores.png")


if __name__ == "__main__":
    for algo in ["mode", "raha-baran"]:
        for ds in ["beers", "flights", "hospital", "rayyan"]:
            plot_absolute_scores(algo, ds)

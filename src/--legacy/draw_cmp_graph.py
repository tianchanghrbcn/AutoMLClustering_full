import pandas as pd
import matplotlib.pyplot as plt
import itertools
from matplotlib import cm

# 更新的数据
data = {
    "Algorithm Combination": [
        "mode + HC", "raha-baran + HC", "GroundTruth + AffinityPropagation",
        "mode + GMM", "mode + AffinityPropagation", "mode + KMeans",
        "GroundTruth + KMeans", "GroundTruth + GMM", "raha-baran + AffinityPropagation",
        "raha-baran + GMM", "raha-baran + KMeans", "GroundTruth + OPTICS",
        "GroundTruth + DBSCAN", "mode + DBSCAN", "mode + OPTICS",
        "raha-baran + DBSCAN", "raha-baran + OPTICS"
    ],
    "Average Score(%)": [100.1395, 91.0383, 55.3176, 53.0168, 52.1821, 51.9696,
                         48.0923, 47.8710, 44.8849, 40.4636, 37.3249, 32.8866,
                         29.7827, 28.0060, 22.8670, 21.2534, 15.5436],
    "Percentage Score Standard Deviation": [74.2006, 20.0367, 17.3633, 39.0547,
                                            55.8460, 37.9887, 19.5221, 15.1519,
                                            19.7885, 13.3241, 16.7098, 21.9937,
                                            20.4866, 26.8614, 15.3506, 13.8799, 5.8002],
    "Average Combined Score": [2.1066, 2.2062, 1.3390, 1.1672, 1.2076, 1.1493,
                               1.1502, 1.1962, 1.0720, 0.9884, 0.8719, 0.7535,
                               0.6181, 0.5542, 0.5017, 0.4467, 0.3616],
    "Combined Score Standard Deviation": [0.9391, 0.6724, 0.5278, 0.6404,
                                          0.9316, 0.6504, 0.4252, 0.5420,
                                          0.4655, 0.4143, 0.4138, 0.4951,
                                          0.1899, 0.3314, 0.2101, 0.1335, 0.0845],
    "Average Deviation from 100%": [54.0506, 18.0464, 44.6824, 56.3417,
                                    69.1682, 55.9357, 51.9077, 52.1290,
                                    55.1778, 59.5364, 62.6751, 67.1134,
                                    70.2173, 71.9940, 77.1330, 78.7466, 84.4564]
}

# 创建数据框并移除 GT + HC
df = pd.DataFrame(data)
df = df[df["Algorithm Combination"] != "GroundTruth + HC"]

# 指标和颜色映射
metrics = [
    "Average Score(%)", "Average Combined Score", "Percentage Score Standard Deviation",
    "Combined Score Standard Deviation", "Average Deviation from 100%"
]
limits = [150, 2.0, 100, 1, 100]
lower_is_better_metrics = ["Percentage Score Standard Deviation", "Combined Score Standard Deviation", "Average Deviation from 100%"]

# 绘制图表
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

# 定义多个调色板
color_palettes = [cm.tab20, cm.Dark2, cm.Paired]

# 生成颜色循环
colors = itertools.cycle([palette(i % palette.N) for palette in color_palettes for i in range(palette.N)])

# 分配给每个算法组合
algorithm_colors = {algo: next(colors) for algo in df["Algorithm Combination"].unique()}
# 标题
titles = [
    "Average Score (%) Across Algorithm Combinations",
    "Average Combined Score Across Algorithm Combinations",
    "Standard Deviation of Percentage Score",
    "Standard Deviation of Combined Score",
    "Average Deviation from Reference (100%)"
]

for i, metric in enumerate(metrics):
    ax = axes[i if i < 2 else i + 1]  # 第一行两个图，第二行三个图
    df.sort_values(by=metric, ascending=(metric in lower_is_better_metrics), inplace=True)
    bars = ax.bar(
        df["Algorithm Combination"],
        df[metric],
        color=[algorithm_colors[alg] for alg in df["Algorithm Combination"]],
        alpha=0.8
    )
    ax.set_title(titles[i], fontsize=14, pad=15)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["Algorithm Combination"], rotation=90, fontsize=10)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_ylim(0, limits[i])
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# 图例放置在空白区域
ax_legend = axes[2]
ax_legend.axis("off")
handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in algorithm_colors.values()]
labels = algorithm_colors.keys()
ax_legend.legend(handles, labels, loc="center", fontsize=12, title="Algorithm Combinations", title_fontsize=14)

# 调整布局
fig.suptitle("Comprehensive Analysis of Algorithm Combinations Across Metrics (Lower Error Rate Datasets)", fontsize=16, y=0.95)
fig.tight_layout(rect=[0, 0, 1, 0.92])

plt.show()

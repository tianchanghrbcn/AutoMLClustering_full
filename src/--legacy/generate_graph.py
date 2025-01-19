import matplotlib.pyplot as plt
import pandas as pd

def plot_from_csv(file_path, output_path, algorithm, dataset):
    # 读取CSV文件
    df = pd.read_csv(file_path, encoding='utf-8')

    # 计算倒数的DB分数
    df['Inverse DB Score'] = 1 / df['Best Davies-Bouldin Score (lower better)']

    # 创建图像
    fig, ax1 = plt.subplots(figsize=(19.2, 10.8))

    # 绘制倒数的DB分数折线图
    color = 'tab:blue'
    ax1.set_xlabel('Error Rate (%)', fontsize=16)  # 设置 x 轴字体大小
    ax1.set_ylabel('Inverse Davies-Bouldin Score', color=color, fontsize=16)  # 设置 y 轴字体大小
    ax1.plot(df['Error rate (%)'], df['Inverse DB Score'], 'o-', color=color, label='Inverse DB Score')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=16)  # 设置 y 轴刻度字体大小
    ax1.set_ylim([0, 2])  # 设置倒数的DB分数范围

    # 创建双轴，绘制轮廓系数折线图
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Silhouette Score', color=color, fontsize=16)  # 设置 y 轴字体大小
    ax2.plot(df['Error rate (%)'], df['Best Silhouette Score (higher better)'], 'o-', color=color, label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=16)  # 设置 y 轴刻度字体大小
    ax2.set_ylim([-1, 1])  # 设置轮廓系数范围

    # 设置X轴错误率范围和刻度，留出一些空隙
    ax1.set_xlim([-2, 60])  # 错误率范围为0%到60%，留出一些空间
    ax1.set_xticks([0, 10, 20, 30, 40, 50, 60])  # 错误率刻度
    ax1.tick_params(axis='x', labelsize=16)  # 设置 x 轴刻度字体大小

    # 设置标题，包括第二行
    fig.suptitle(
        f'Clustering Analysis: Error Rate vs DB and Silhouette Scores\nAlgorithm: {algorithm}, Dataset: {dataset}',
        fontsize=20  # 设置标题字体大小
    )

    # 设置图例
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # 让标题与图像之间有足够空间
    plt.grid(True)

    # 保存图像
    plt.savefig(output_path, dpi=300)

    # 显示图像
    plt.show()

# 主函数，读取CSV文件并绘图
if __name__ == "__main__":
    # 设定CSV文件路径和保存图像的路径
    file_path = r'D:\algorithm paper\ML algorithms codes\data_experiments\results\clustered_analysis\restaurants_DBSCAN_analysis\restaurants_DBSCAN_analysis.csv'
    output_path = r'D:\algorithm paper\ML algorithms codes\data_experiments\results\final_results\graphs\restaurants_DBSCAN_analysis.png'

    # 设定算法和数据集名称
    algorithm = "DBSCAN"  # 替换为你要设定的算法名称
    dataset = "restaurants"  # 替换为你要设定的数据集名称

    # 绘制并保存图表
    plot_from_csv(file_path, output_path, algorithm, dataset)

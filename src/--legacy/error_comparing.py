import pandas as pd
import numpy as np

def add_id_and_sort(clean_data_path):
    # 去除路径字符串的多余引号
    clean_data_path = clean_data_path.strip('"')

    # 读取干净数据集
    clean_data = pd.read_csv(clean_data_path)

    # 检查是否存在 'ID' 列，如果不存在则添加到最前面
    if 'ID' not in clean_data.columns:
        # 在第一列位置插入 'ID' 列，从 1 到 n 编号
        clean_data.insert(0, 'ID', range(1, len(clean_data) + 1))
        # 按 'ID' 列排序
        clean_data = clean_data.sort_values(by='ID').reset_index(drop=True)
        # 保存新的文件
        new_clean_data_path = clean_data_path.replace('.csv', '_with_ID.csv')
        clean_data.to_csv(new_clean_data_path, index=False)
        print(f"已在最前面添加 'ID' 列，并生成新的文件: {new_clean_data_path}")
    else:
        print("干净数据集已经包含 'ID' 列，不需要添加。")

def calculate_error_rate(clean_data_path, dirty_data_path):
    # 去除路径字符串的多余引号
    clean_data_path = clean_data_path.strip('"')
    dirty_data_path = dirty_data_path.strip('"')

    # 增加 'ID' 列并排序
    add_id_and_sort(clean_data_path)

    # 读取干净数据集和脏数据集
    clean_data = pd.read_csv(clean_data_path)
    dirty_data = pd.read_csv(dirty_data_path)

    # 确保两个数据集具有相同的形状
    if clean_data.shape != dirty_data.shape:
        print("错误：干净数据集和脏数据集的形状不一致。")
        return

    # 初始化计数器
    total_cells = clean_data.size
    error_cells = 0

    # 逐个单元格比较
    for row in range(clean_data.shape[0]):
        for col in range(clean_data.shape[1]):
            clean_value = clean_data.iat[row, col]
            dirty_value = dirty_data.iat[row, col]

            # 忽略两者均为 NaN 的情况
            if pd.isna(clean_value) and pd.isna(dirty_value):
                continue
            # 统计其他不匹配的情况
            elif clean_value != dirty_value:
                error_cells += 1

    # 计算错误率
    error_rate = (error_cells / total_cells) * 100

    # 打印错误率
    print(f"错误率: {error_rate:.2f}%")

if __name__ == "__main__":
    try:
        while True:
            # 用户输入文件路径
            clean_data_path = input("请输入干净数据集的路径: ")
            dirty_data_path = input("请输入脏数据集的路径: ")

            # 计算并打印错误率
            calculate_error_rate(clean_data_path, dirty_data_path)

            # 询问用户是否继续
            print("\n按 Ctrl+C 退出，或继续输入新的文件路径以计算新的错误率。\n")
    except KeyboardInterrupt:
        print("\n程序已终止。")

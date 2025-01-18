import pandas as pd
import os


def remove_id_column(file_path):
    # 读取 CSV 文件
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print("错误：文件未找到，请检查路径是否正确。")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 检查是否包含 'ID' 列并删除
    if 'ID' in data.columns:
        data = data.drop(columns=['ID'])
        print("已删除 'ID' 列。")
    else:
        print("文件中没有 'ID' 列。")

    # 构造输出文件名
    base, ext = os.path.splitext(file_path)
    output_file_path = f"{base}_remove_ID{ext}"

    # 保存新的 CSV 文件
    data.to_csv(output_file_path, index=False)
    print(f"已保存新文件: {output_file_path}")


if __name__ == "__main__":
    while True:
        # 用户输入文件路径
        file_path = input("请输入 CSV 文件的路径 (或输入 'exit' 退出): ").strip('"')

        if file_path.lower() == 'exit':
            print("程序已退出。")
            break

        remove_id_column(file_path)

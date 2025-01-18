import csv
import os

def txt_to_csv(txt_file):
    # 去除路径中的多余引号
    txt_file = txt_file.strip('"')

    # 检查输入的txt文件是否存在
    if not os.path.isfile(txt_file):
        print(f"文件 '{txt_file}' 不存在，请检查路径。")
        return

    # 设置csv文件名与路径
    csv_file = os.path.splitext(txt_file)[0] + '.csv'

    # 读取txt文件并写入csv文件
    with open(txt_file, 'r', encoding='utf-8') as txt_f:
        lines = txt_f.readlines()

    with open(csv_file, 'w', newline='', encoding='utf-8') as csv_f:
        writer = csv.writer(csv_f)
        for line in lines:
            # 使用逗号分隔txt文件中的每一行内容
            row = line.strip().split(',')
            writer.writerow(row)

    print(f"转换成功，已将 '{txt_file}' 转换为 '{csv_file}'")

# 主程序，从终端接收文件路径
if __name__ == "__main__":
    try:
        while True:
            txt_file = input("请输入要转换的txt文件路径：")
            txt_to_csv(txt_file)
            print("\n按 Ctrl+C 退出，或继续输入新的文件路径进行转换。\n")
    except KeyboardInterrupt:
        print("\n程序已终止。")

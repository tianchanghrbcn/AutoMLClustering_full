import os
import subprocess
import pandas as pd
import shutil
import time
import re  # 用于解析 stdout 输出

def run_error_correction(dataset_path, dataset_id, algorithm_id, clean_csv_path, output_dir):

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 设置任务名称
    task_name = f"dataset_{dataset_id}_algo_{algorithm_id}"
    algo_name = "mode" if algorithm_id == 1 else "baran"

    # 设置命令
    if algorithm_id == 1:
        command = [
            "python", "../../cleaning/mode/correction_with_mode.py",
            "--clean_path", clean_csv_path,
            "--dirty_path", dataset_path,
            "--task_name", task_name
        ]
    elif algorithm_id == 2:
        try:
            df = pd.read_csv(dataset_path)
            index_attribute = df.columns[0]  # 获取首列名称作为主键
        except Exception as e:
            print(f"读取数据集时出错: {e}")
            return None, None

        command = [
            "python", "../../cleaning/baran/correction_with_baran.py",
            "--dirty_path", dataset_path,
            "--clean_path", clean_csv_path,
            "--task_name", task_name,
            "--output_path", output_dir,
            "--index_attribute", index_attribute
        ]
    else:
        print(f"运行清洗算法 {algorithm_id}（其他算法占位符），数据集编号: {dataset_id}")
        return None, None

    # 执行命令并处理结果
    try:
        print(f"运行清洗算法 {algorithm_id}（{algo_name}），数据集编号: {dataset_id}")
        start_time = time.time()  # 记录开始时间
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        end_time = time.time()  # 记录结束时间
        runtime = end_time - start_time  # 计算运行时间

        print(f"算法输出:\n{result.stdout}")

        # 动态提取生成的结果文件路径
        match = re.search(r"Repaired data saved to (.+\.csv)", result.stdout)
        if match:
            repaired_file = match.group(1)
        else:
            raise FileNotFoundError("无法从算法输出中提取生成的文件路径")

        # 保存结果到指定目录
        cleaned_data_dir = os.path.join("../../../results/cleaned_data", algo_name)
        os.makedirs(cleaned_data_dir, exist_ok=True)
        new_file_path = os.path.join(cleaned_data_dir, f"repaired_{dataset_id}.csv")
        shutil.copy(repaired_file, new_file_path)

        print(f"结果文件已保存到: {new_file_path}")
        print(f"运行时间: {runtime:.2f} 秒")

        return new_file_path, runtime

    except subprocess.CalledProcessError as e:
        print(f"运行错误：{e.stderr}")
        return None, None
    except FileNotFoundError as e:
        print(f"文件错误：{e}")
        return None, None

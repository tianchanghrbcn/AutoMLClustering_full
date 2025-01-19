import os
import subprocess
import pandas as pd
import shutil
import time
import re  # 用于解析 stdout 输出

def run_error_correction(dataset_path, dataset_id, algorithm_id, clean_csv_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    task_name = f"dataset_{dataset_id}_algo_{algorithm_id}"
    algo_name = "mode" if algorithm_id == 1 else "baran"

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
            index_attribute = df.columns[0]
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

    try:
        print(f"运行清洗算法 {algorithm_id}（{algo_name}），数据集编号: {dataset_id}")
        start_time = time.time()

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        repaired_file = None
        stdout_lines = []  # 用于存储完整的标准输出

        while process.poll() is None:
            output = process.stdout.readline()
            if output:
                stdout_lines.append(output.strip())  # 实时保存输出
                print(output.strip())  # 实时打印输出

                # 检测到清洗结果路径时立即停止
                if "Repaired data saved to" in output:
                    match = re.search(r"Repaired data saved to\s+(.+\.csv)", output)
                    if match:
                        repaired_file = match.group(1).strip()
                        print(f"[INFO] 清洗结果文件路径: {repaired_file}")
                        process.terminate()
                        break

        # 处理剩余的标准输出内容
        stdout, stderr = process.communicate()
        stdout_lines.extend(stdout.splitlines())
        full_stdout = "\n".join(stdout_lines)
        print(f"子进程完整输出:\n{full_stdout}")

        if not repaired_file:
            # 如果未检测到路径，尝试从完整输出中匹配
            match = re.search(r"Repaired data saved to\s+(.+\.csv)", full_stdout, re.MULTILINE)
            if match:
                repaired_file = match.group(1).strip()

        if not repaired_file:
            print("[ERROR] 未检测到清洗结果文件路径，可能清洗未正常完成")
            return None, None

        # 正常终止时处理后续逻辑
        end_time = time.time()
        runtime = end_time - start_time

        cleaned_data_dir = os.path.join("../../../results/cleaned_data", algo_name)
        os.makedirs(cleaned_data_dir, exist_ok=True)
        new_file_path = os.path.join(cleaned_data_dir, f"repaired_{dataset_id}.csv")
        shutil.copy(repaired_file, new_file_path)

        print(f"结果文件已保存到: {new_file_path}")
        print(f"运行时间: {runtime:.2f} 秒")

        return new_file_path, runtime

    except Exception as ex:
        process.kill()  # 确保子进程被终止
        print(f"[ERROR] 清洗算法执行中断: {ex}")
        return None, None

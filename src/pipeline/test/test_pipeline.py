import os
import json
from concurrent.futures import ProcessPoolExecutor

from pre_processing_test import process_test_datasets
from test_classify import run_test_classification
from function_back import function_back

from src.pipeline.train.error_correction import run_error_correction
from src.pipeline.test.test_cluster import run_clustering_test
from src.pipeline.test.test_analysis import save_test_analyzed_results

# 全局配置
BASE_DIR = os.path.dirname(__file__)
DATASETS_DIR = os.path.join(BASE_DIR, "..", "..", "..", "datasets", "test")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "..", "..", "results")

TEST_DATA_FILE = os.path.join(RESULTS_DIR, "test_eigenvectors.json")
MODEL_FILE = os.path.join(RESULTS_DIR, "xgboost_multilabel_model.joblib")
BINARIZER_FILE = os.path.join(RESULTS_DIR, "multilabel_binarizer.joblib")
PREDICTIONS_OUTPUT_FILE = os.path.join(RESULTS_DIR, "test_predictions.json")
STRATEGIES_OUTPUT_FILE = os.path.join(RESULTS_DIR, "test_strategies.json")

TEST_CLEANED_RESULTS_PATH = os.path.join(RESULTS_DIR, "test_cleaned_results.json")
TEST_CLUSTERED_RESULTS_PATH = os.path.join(RESULTS_DIR, "test_clustered_results.json")
TEST_ANALYZED_RESULTS_PATH = os.path.join(RESULTS_DIR, "test_analyzed_results.json")


def process_test_record(record_idx, record, work_dir):

    cleaned_results = []
    clustered_results = []

    dataset_id = record.get("dataset_id")
    dataset_name = record.get("dataset_name")
    csv_file = record.get("csv_file")
    top_r_strategies = record.get("top_r", [])

    print(f"[INFO] [DatasetID={dataset_id}] 开始处理测试数据集: {dataset_name}, 文件={csv_file}")

    # 构造数据集文件路径
    dataset_folder = os.path.join(work_dir, "datasets", "test", dataset_name)
    csv_path = os.path.join(dataset_folder, csv_file)
    clean_csv_path = os.path.join(dataset_folder, "clean.csv")

    # 如果原始文件或真实 clean 文件不存在，则跳过
    if not os.path.exists(csv_path) or not os.path.exists(clean_csv_path):
        print(f"[WARNING] 数据集 {dataset_name} 的文件路径不存在，跳过处理。")
        return cleaned_results, clustered_results

    # 根据 top_r 中的策略列表，逐项进行清洗和聚类
    for idx, strategy in enumerate(top_r_strategies):
        # 每个 strategy 的格式:
        # [cleaning_algo, clustering_algo, clustering_params]
        cleaning_algo = strategy[0]         # "mode" 或 "raha_baran"
        clustering_algo = strategy[1]       # 如 "KMEANS", "AP", "HC", ...
        clustering_params = strategy[2]     # dict, 如 { "k": "> sqrt(n)" }

        # 根据清洗算法名称决定 algorithm_id
        algorithm_id = 2 if cleaning_algo == "raha_baran" else 1

        print(f"\n[INFO] 正在运行第 {idx+1}/{len(top_r_strategies)} 个策略:")
        print(f"       清洗算法: {cleaning_algo}, 聚类算法: {clustering_algo}, 参数: {clustering_params}")

        # 1. 清洗
        new_file_path, runtime = run_error_correction(
            dataset_path=csv_path,
            dataset_id=dataset_id,
            algorithm_id=algorithm_id,
            clean_csv_path=clean_csv_path,
            output_dir=os.path.join(work_dir, "results", dataset_name, cleaning_algo),
        )

        if new_file_path and runtime is not None:
            print(f"[INFO] 清洗完成 => 结果文件路径: {new_file_path}, 运行时间: {runtime:.2f} 秒")

            cleaned_results.append({
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "algorithm": cleaning_algo,
                "algorithm_id": algorithm_id,
                "cleaned_file_path": new_file_path,
                "runtime": runtime
            })

            # 2. 聚类
            cluster_output_dir, cluster_runtime = run_clustering_test(
                dataset_id=dataset_id,
                algorithm=clustering_algo,        # 聚类算法名称
                params=clustering_params,         # 聚类算法参数
                cleaned_file_path=new_file_path
            )

            if cluster_output_dir and cluster_runtime is not None:
                clustered_results.append({
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "cleaning_algorithm": cleaning_algo,
                    "cleaning_runtime": runtime,
                    "clustering_algorithm": clustering_algo,
                    "clustering_params": clustering_params,
                    "clustering_runtime": cluster_runtime,
                    "clustered_file_path": cluster_output_dir,
                })
                print(f"[INFO] 聚类完成 => 算法: {clustering_algo}, 运行时间: {cluster_runtime:.2f} 秒")
            else:
                print(f"[ERROR] 聚类算法 {clustering_algo} 运行失败")
        else:
            print(f"[ERROR] 清洗算法 {cleaning_algo} 运行失败")

        print("=" * 80)

    return cleaned_results, clustered_results


def main():
    """
    test_pipeline.py 的主流程：
    1. 预处理测试数据集
    2. 使用已有模型对测试数据分类
    3. 将分类结果映射到具体策略
    4. (新增) 根据策略执行清洗与聚类
    5. (可选) 分析聚类结果
    """
    work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    # Step 1: 预处理测试数据集
    print("[STEP 1] 开始预处理测试数据集...")
    output_file = process_test_datasets()

    if not output_file or not os.path.exists(output_file):
        print("[ERROR] 测试数据集预处理失败，流程中止。")
        exit(1)
    print("[STEP 1] 测试数据集预处理完成！")

    # Step 2: 分类测试数据
    print("[STEP 2] 开始分类测试数据...")
    try:
        run_test_classification(
            test_data_path=TEST_DATA_FILE,
            model_path=MODEL_FILE,
            binarizer_path=BINARIZER_FILE,
            output_path=PREDICTIONS_OUTPUT_FILE,
            top_r=3  # 定义 Top R 的值
        )
        print("[STEP 2] 分类测试数据完成！")
    except Exception as e:
        print(f"[ERROR] 分类测试数据失败: {e}")
        exit(1)

    # Step 3: 映射预测结果到策略
    print("[STEP 3] 开始映射预测结果到策略...")
    try:
        function_back(
            predictions_file=PREDICTIONS_OUTPUT_FILE,
            strategies_file=STRATEGIES_OUTPUT_FILE,
            eigenvectors_file=TEST_DATA_FILE
        )
        print("[STEP 3] 映射完成！")
    except Exception as e:
        print(f"[ERROR] 映射预测结果失败: {e}")
        exit(1)

    # ======================== 以下为清洗 & 聚类部分 ========================
    print("[STEP 4] 开始执行测试清洗 & 聚类流程...")

    # 读取 test_strategies.json 文件，解析其中的 top_r
    if not os.path.exists(STRATEGIES_OUTPUT_FILE):
        print(f"[ERROR] 未找到测试策略文件 {STRATEGIES_OUTPUT_FILE}, 无法执行清洗与聚类。")
        exit(1)

    with open(STRATEGIES_OUTPUT_FILE, "r", encoding="utf-8") as f:
        test_strategies = json.load(f)

    if not test_strategies:
        print("[ERROR] test_strategies.json 文件为空，无法执行清洗与聚类。")
        exit(1)

    # 准备收集所有数据集的清洗和聚类结果
    test_cleaned_results = []
    test_clustered_results = []

    # 使用多进程处理（可根据需要调整并行数）
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_test_record, idx, record, work_dir)
            for idx, record in enumerate(test_strategies)
        ]

        for future in futures:
            cleaned, clustered = future.result()
            test_cleaned_results.extend(cleaned)
            test_clustered_results.extend(clustered)

    # 将清洗结果与聚类结果分别写入文件
    with open(TEST_CLEANED_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(test_cleaned_results, f, ensure_ascii=False, indent=4)
    print(f"[STEP 4] 测试清洗结果已保存到 {TEST_CLEANED_RESULTS_PATH}")

    with open(TEST_CLUSTERED_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(test_clustered_results, f, ensure_ascii=False, indent=4)
    print(f"[STEP 4] 测试聚类结果已保存到 {TEST_CLUSTERED_RESULTS_PATH}")

    # ======================== 分析步骤（可选） ========================
    print("[STEP 5] 开始分析测试聚类结果...")

    try:
        # 分析并保存结果
        save_test_analyzed_results(
            eigenvectors_path=TEST_DATA_FILE,
            clustered_results_path=TEST_CLUSTERED_RESULTS_PATH,
            output_path=TEST_ANALYZED_RESULTS_PATH
        )
        print(f"[STEP 5] 测试聚类分析结果已保存到 {TEST_ANALYZED_RESULTS_PATH}")
    except Exception as e:
        print(f"[ERROR] 分析测试聚类结果时发生错误: {e}")

    print("[INFO] test_pipeline 全部流程完成！")


if __name__ == "__main__":
    main()

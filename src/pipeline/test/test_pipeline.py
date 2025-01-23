import os
from pre_processing_test import process_test_datasets
from test_classify import run_test_classification
from function_back import function_back

# 全局配置
BASE_DIR = os.path.dirname(__file__)
DATASETS_DIR = os.path.join(BASE_DIR, "..", "..", "..", "datasets", "test")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "..", "..", "results")

TEST_DATA_FILE = os.path.join(RESULTS_DIR, "test_eigenvectors.json")
MODEL_FILE = os.path.join(RESULTS_DIR, "xgboost_multilabel_model.joblib")
BINARIZER_FILE = os.path.join(RESULTS_DIR, "multilabel_binarizer.joblib")
PREDICTIONS_OUTPUT_FILE = os.path.join(RESULTS_DIR, "test_predictions.json")
STRATEGIES_OUTPUT_FILE = os.path.join(RESULTS_DIR, "test_strategies.json")

def main():
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
            strategies_file=STRATEGIES_OUTPUT_FILE
        )
        print("[STEP 3] 映射完成！")
    except Exception as e:
        print(f"[ERROR] 映射预测结果失败: {e}")
        exit(1)

    print("[INFO] 全部流程完成！")

if __name__ == "__main__":
    main()

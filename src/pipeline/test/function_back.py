import os
import json

# ========== 函数定义 ==========

def load_predictions(predictions_file):
    """
    加载预测结果 JSON 文件。
    """
    with open(predictions_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_strategies(strategies, output_file):
    """
    保存映射后的策略到 JSON 文件。
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(strategies, f, indent=4, ensure_ascii=False)


def parse_ap_parameters(ap_param, dataset_id):
    """
    解析 AP 参数字符串，格式应为 'damp_X-pref_Y'。
    """
    try:
        # 确保参数部分符合格式 "damp_X-pref_Y"
        if "-" in ap_param and "damp" in ap_param and "pref" in ap_param:
            damping_label, preference_label = ap_param.split("-")
            damping = "0.5-0.7" if damping_label == "damp_Low" else "0.7-0.9"
            preference = "-500 to -300" if preference_label == "pref_Low" else "-300 to -100"
            return {"damping": damping, "preference": preference}
        else:
            raise ValueError(f"AP 参数未包含预期的 'damping-preference' 格式: {ap_param}")
    except Exception as e:
        print(f"[WARNING] 无法解析 AP 参数：{ap_param} (dataset_id={dataset_id}) - {e}")
        return {"damping": "未知", "preference": "未知"}


def map_predictions_to_strategies(predictions):
    """
    将预测的标签映射回具体的策略配置。
    """
    strategies = []

    for prediction in predictions:
        dataset_id = prediction["dataset_id"]
        top_labels = prediction["top_labels"]

        dataset_strategies = {
            "dataset_id": dataset_id,
            "top_r": []
        }

        for label in top_labels:
            parts = label.split("-")
            cleaning_algorithm = parts[0]
            clustering_algorithm = parts[1]
            hyperparams = {}

            if clustering_algorithm == "KMEANS":
                if parts[2] == "k_bin1":
                    hyperparams["k"] = "≤ sqrt(n)/2"
                elif parts[2] == "k_bin2":
                    hyperparams["k"] = "(sqrt(n)/2, sqrt(n)]"
                else:
                    hyperparams["k"] = "> sqrt(n)"
            elif clustering_algorithm == "AP":
                ap_param = "-".join(parts[2:])  # 提取 AP 参数部分
                hyperparams = parse_ap_parameters(ap_param, dataset_id)
            elif clustering_algorithm == "DBSCAN":
                eps, min_samples = parts[2].split("-")
                hyperparams["eps"] = "0.1-1.0" if eps == "eps_Low" else "1.0-2.0"
                hyperparams["min_samples"] = "5-25" if min_samples == "minS_Low" else "25-50"
            elif clustering_algorithm == "OPTICS":
                min_samples, xi = parts[2].split("-")
                hyperparams["min_samples"] = "5-15" if min_samples == "minS_Low" else "15-30"
                hyperparams["xi"] = "0.01-0.05" if xi == "xi_Low" else "0.05-0.1"
            elif clustering_algorithm == "HC":
                if parts[2] == "k_bin1":
                    hyperparams["k"] = "≤ sqrt(n)/2"
                elif parts[2] == "k_bin2":
                    hyperparams["k"] = "(sqrt(n)/2, sqrt(n)]"
                else:
                    hyperparams["k"] = "> sqrt(n)"
            elif clustering_algorithm == "GMM":
                if parts[2] == "k_bin1":
                    hyperparams["k"] = "≤ sqrt(n)/2"
                elif parts[2] == "k_bin2":
                    hyperparams["k"] = "(sqrt(n)/2, sqrt(n)]"
                else:
                    hyperparams["k"] = "> sqrt(n)"
                hyperparams["covariance_type"] = parts[3].split("=")[1]

            dataset_strategies["top_r"].append([cleaning_algorithm, clustering_algorithm, hyperparams])

        strategies.append(dataset_strategies)

    return strategies


def function_back(predictions_file, strategies_file, eigenvectors_file):
    """
    主函数：将预测标签映射回具体策略，并添加 dataset_name 和 csv_file 字段。
    """
    print("[INFO] 加载预测结果...")
    predictions = load_predictions(predictions_file)

    print("[INFO] 加载特征向量...")
    with open(eigenvectors_file, "r", encoding="utf-8") as f:
        eigenvectors = json.load(f)

    # 创建 dataset_id 到 dataset_name 和 csv_file 的映射
    id_to_metadata = {
        record["dataset_id"]: {"dataset_name": record["dataset_name"], "csv_file": record["csv_file"]}
        for record in eigenvectors
    }

    print("[INFO] 映射预测标签到具体策略...")
    strategies = map_predictions_to_strategies(predictions)

    # 添加 dataset_name 和 csv_file 到策略中
    for strategy in strategies:
        dataset_id = strategy["dataset_id"]
        if dataset_id in id_to_metadata:
            strategy["dataset_name"] = id_to_metadata[dataset_id]["dataset_name"]
            strategy["csv_file"] = id_to_metadata[dataset_id]["csv_file"]

    print("[INFO] 保存映射后的策略...")
    save_strategies(strategies, strategies_file)
    print(f"[INFO] 策略映射完成，结果已保存到 {strategies_file}")


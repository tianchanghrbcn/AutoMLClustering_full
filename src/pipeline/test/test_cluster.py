import os
import math
import time
import numpy as np
import pandas as pd
import optuna

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    AffinityPropagation,
    DBSCAN,
    OPTICS
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)
from sklearn.metrics.pairwise import cosine_distances


# ---- 1) 辅助函数：解析k的特殊范围字符串 ----

def get_cluster_range(n: int, k_str: str):
    """
    根据给定的 k_str (如 '> sqrt(n)', '(sqrt(n)/2, sqrt(n)]', '≤ sqrt(n)/2')
    返回 (n_min, n_max)。若k_str不在该集合中，则抛异常或由外部处理。
    """
    # 统一给个默认值，防止出错
    n_min = 2
    n_max = n  # 可以理解为最小2，最大数据集大小

    if k_str == "> sqrt(n)":
        # 大于 sqrt(n)
        n_min = max(2, math.ceil(math.sqrt(n)) + 1)
        n_max = max(n_min, n // 5)
    elif k_str == "(sqrt(n)/2, sqrt(n)]":
        # 在 sqrt(n)/2 与 sqrt(n) 之间
        n_min = max(2, math.ceil(math.sqrt(n) / 2) + 1)
        n_max = max(n_min, math.ceil(math.sqrt(n)))
    elif k_str == "≤ sqrt(n)/2":
        # 小于等于 sqrt(n)/2
        n_min = 5
        n_max = max(2, math.floor(math.sqrt(n) / 2))
        if n_max < n_min:
            n_max = n_min

    return n_min, n_max


# ---- 2) 辅助函数：通用的字符串解析（区间） ----

def parse_range(value_str: str, is_int: bool = False):
    """
    将类似 '0.7-0.9'、'0.7 to 0.9'、'5-25'、'5 to 25' 等字符串解析成 (low, high)。
    is_int=True 表示需要解析为整数区间，否则解析为浮点区间。

    若解析失败则抛出 ValueError。
    """
    # 兼容用 " to " 作为分隔符
    if ' to ' in value_str:
        parts = value_str.split(' to ')
    elif '-' in value_str:
        parts = value_str.split('-')
    else:
        raise ValueError(f"无法解析区间参数: {value_str}")

    if len(parts) != 2:
        raise ValueError(f"解析失败，找不到正确的上下界: {value_str}")

    low_str, high_str = parts[0].strip(), parts[1].strip()

    if is_int:
        return int(low_str), int(high_str)
    else:
        return float(low_str), float(high_str)


# ---- 3) 数据预处理 ----

def preprocess_data(cleaned_file_path: str):
    """
    预处理数据：
    1. 读取CSV文件，并检查是否存在；
    2. 排除列名中带 'id' 的列；
    3. 对类别型特征做频率编码；
    4. 删除缺失数据；
    5. 标准化；
    返回：X_scaled (np.array), 数据行数 n
    """
    if not os.path.exists(cleaned_file_path):
        raise FileNotFoundError(f"[ERROR] 清洗后的文件不存在: {cleaned_file_path}")

    df = pd.read_csv(cleaned_file_path)

    # 排除包含 'id' 的列
    excluded_columns = [col for col in df.columns if 'id' in col.lower()]
    remaining_columns = df.columns.difference(excluded_columns)

    # 做一份拷贝，避免潜在警告
    X = df[remaining_columns].copy()

    # 类别型特征做频率编码
    for col in X.columns:
        if X[col].dtype == 'object' or str(X[col].dtype).startswith('category'):
            freq_map = X[col].value_counts(normalize=True)
            X[col] = X[col].map(freq_map).fillna(0)

    X.dropna(inplace=True)  # 删除含NaN的行

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, len(X)


# ---- 4) 保存结果 ----

def save_results(
        cleaned_file_path: str,
        dataset_id: str,
        algorithm: str,
        params: dict,
        final_labels: np.ndarray,
        final_db_score: float,
        final_silhouette_score: float,
        alpha: float,
        beta: float,
        start_time: float
):
    """
    计算最终综合得分并保存结果到文本文件，返回 (输出目录, 运行时长)。
    """
    # 避免除零
    final_db_score = max(final_db_score, 1e-12)

    combined_score = alpha * (1 / final_db_score) + beta * final_silhouette_score

    # 参数字符串
    params_str = ", ".join([f"{k}={v}" for k, v in params.items()])

    # 输出路径
    base_filename = os.path.splitext(os.path.basename(cleaned_file_path))[0]
    output_dir = os.path.join(
        os.getcwd(), "..", "..", "..",
        "results",
        "test_clustered_data",
        algorithm.upper(),
        f"clustered_{dataset_id}"
    )
    os.makedirs(output_dir, exist_ok=True)

    result_file = os.path.join(output_dir, f"{base_filename}_results.txt")

    # 写文件
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"Best parameters: {params_str}\n")
        f.write(f"Number of clusters: {len(set(final_labels))}\n")
        f.write(f"Final Combined Score: {combined_score}\n")
        f.write(f"Final Silhouette Score: {final_silhouette_score}\n")
        f.write(f"Final Davies-Bouldin Score: {final_db_score}\n")

    print(f"[INFO] 结果已保存到: {result_file}")

    run_time = time.time() - start_time
    return output_dir, run_time


# ---- 5) 主函数：自动聚类与调参 ----

def run_clustering_test(
        dataset_id: str,
        algorithm: str,
        params: dict,
        cleaned_file_path: str,
        alpha: float = 0.75,  # 所有算法统一使用
        beta: float = 0.25  # 所有算法统一使用
):
    """
    主函数，根据算法类型（KMEANS / GMM / HC / AP / DBSCAN / OPTICS）和对应的 params，
    自动完成超参数搜索并保存最终结果。

    :param dataset_id: 数据集ID
    :param algorithm: 算法名称 (KMEANS / GMM / HC / AP / DBSCAN / OPTICS)
    :param params: 包含算法参数的字典，例如：
        - KMeans: { "k": "(sqrt(n)/2, sqrt(n)]" or "2-10" }
        - GMM:    { "k": "> sqrt(n)" }
        - HC:     { "k": "2 to 10" }
        - AP:     { "damping": "0.7-0.9", "preference": "-300 to -100" }
        - DBSCAN: { "eps": "0.1-1.0", "min_samples": "5 to 25" }
        - OPTICS: { "min_samples": "5-15", "xi": "0.01 to 0.05", "min_cluster_size": "0.01-0.05" }
    :param cleaned_file_path: 清洗后的 CSV 文件路径
    :param alpha: 综合得分中 (1 / DB) 的权重
    :param beta: 综合得分中 Silhouette 的权重
    """
    start_time = time.time()
    algo = algorithm.upper()

    try:
        X_scaled, n = preprocess_data(cleaned_file_path)

        # -----------------------------------------------------------------
        # 1) ========== KMeans ==========
        if algo == "KMEANS":
            k_str = params.get("k", None)
            if k_str in {"> sqrt(n)", "(sqrt(n)/2, sqrt(n)]", "≤ sqrt(n)/2"}:
                n_clusters_min, n_clusters_max = get_cluster_range(n, k_str)
            elif k_str is not None:
                # 解析自定义区间，如 "2-10" 或 "2 to 10"
                n_clusters_min, n_clusters_max = parse_range(k_str, is_int=True)
            else:
                # 默认
                n_clusters_min, n_clusters_max = 2, min(n, 50)

            def objective(trial):
                k_ = trial.suggest_int("n_clusters", n_clusters_min, n_clusters_max)
                labels_ = KMeans(n_clusters=k_, init='k-means++', n_init=10, random_state=0).fit_predict(X_scaled)
                db_ = davies_bouldin_score(X_scaled, labels_)
                sil_ = silhouette_score(X_scaled, labels_)
                return alpha * (1 / max(db_, 1e-12)) + beta * sil_

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)

            best_k = study.best_params["n_clusters"]
            final_labels = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=0).fit_predict(X_scaled)

            final_db_score = davies_bouldin_score(X_scaled, final_labels)
            final_silhouette_score = silhouette_score(X_scaled, final_labels)

            return save_results(
                cleaned_file_path,
                dataset_id,
                "KMEANS",
                {"k": k_str if k_str else f"{n_clusters_min}-{n_clusters_max}", "best_n_clusters": best_k},
                final_labels,
                final_db_score,
                final_silhouette_score,
                alpha,
                beta,
                start_time
            )

        # -----------------------------------------------------------------
        # 2) ========== GMM ==========
        elif algo == "GMM":
            k_str = params.get("k", None)
            if k_str in {"> sqrt(n)", "(sqrt(n)/2, sqrt(n)]", "≤ sqrt(n)/2"}:
                n_components_min, n_components_max = get_cluster_range(n, k_str)
            elif k_str is not None:
                n_components_min, n_components_max = parse_range(k_str, is_int=True)
            else:
                n_components_min, n_components_max = 2, min(n, 50)

            covariance_type = params.get("covariance_type", "full")

            def objective(trial):
                k_ = trial.suggest_int("n_components", n_components_min, n_components_max)
                labels_ = GaussianMixture(n_components=k_, covariance_type=covariance_type, random_state=0) \
                    .fit_predict(X_scaled)
                db_ = davies_bouldin_score(X_scaled, labels_)
                sil_ = silhouette_score(X_scaled, labels_)
                return alpha * (1 / max(db_, 1e-12)) + beta * sil_

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)

            best_n_components = study.best_params["n_components"]
            final_labels = GaussianMixture(n_components=best_n_components, covariance_type=covariance_type,
                                           random_state=0).fit_predict(X_scaled)

            final_db_score = davies_bouldin_score(X_scaled, final_labels)
            final_silhouette_score = silhouette_score(X_scaled, final_labels)

            return save_results(
                cleaned_file_path,
                dataset_id,
                "GMM",
                {"k": k_str if k_str else f"{n_components_min}-{n_components_max}",
                 "best_n_components": best_n_components,
                 "covariance_type": covariance_type},
                final_labels,
                final_db_score,
                final_silhouette_score,
                alpha,
                beta,
                start_time
            )

        # -----------------------------------------------------------------
        # 3) ========== 层次聚类 (HC) ==========
        elif algo == "HC":
            k_str = params.get("k", None)
            if k_str in {"> sqrt(n)", "(sqrt(n)/2, sqrt(n)]", "≤ sqrt(n)/2"}:
                n_clusters_min, n_clusters_max = get_cluster_range(n, k_str)
            elif k_str is not None:
                n_clusters_min, n_clusters_max = parse_range(k_str, is_int=True)
            else:
                n_clusters_min, n_clusters_max = 2, min(n, 50)

            linkage = params.get("linkage", "ward")
            metric = params.get("metric", "euclidean")

            if linkage == "ward" and metric != "euclidean":
                raise ValueError("[ERROR] Ward linkage only supports 'euclidean' metric.")

            def objective(trial):
                k_ = trial.suggest_int("n_clusters", n_clusters_min, n_clusters_max)
                labels_ = AgglomerativeClustering(n_clusters=k_, linkage=linkage, affinity=metric) \
                    .fit_predict(X_scaled)
                db_ = davies_bouldin_score(X_scaled, labels_)
                sil_ = silhouette_score(X_scaled, labels_)
                return alpha * (1 / max(db_, 1e-12)) + beta * sil_

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)

            best_k = study.best_params["n_clusters"]
            final_labels = AgglomerativeClustering(n_clusters=best_k, linkage=linkage, affinity=metric) \
                .fit_predict(X_scaled)

            final_db_score = davies_bouldin_score(X_scaled, final_labels)
            final_silhouette_score = silhouette_score(X_scaled, final_labels)

            return save_results(
                cleaned_file_path,
                dataset_id,
                "HC",
                {"k": k_str if k_str else f"{n_clusters_min}-{n_clusters_max}",
                 "best_n_clusters": best_k,
                 "linkage": linkage,
                 "metric": metric},
                final_labels,
                final_db_score,
                final_silhouette_score,
                alpha,
                beta,
                start_time
            )

        # -----------------------------------------------------------------
        # 4) ========== AffinityPropagation (AP) ==========
        elif algo == "AP":
            # damping, preference 都是字符串形式
            damping_str = params.get("damping", "0.5-0.9")  # 默认范围
            pref_str = params.get("preference", "-500 to -100")

            # 解析范围
            damping_low, damping_high = parse_range(damping_str, is_int=False)
            pref_low, pref_high = parse_range(pref_str, is_int=True)  # preference 通常是整型

            n_trials = params.get("n_trials", 50)

            def objective(trial):
                damping_ = trial.suggest_float("damping", damping_low, damping_high)
                preference_ = trial.suggest_int("preference", pref_low, pref_high)

                ap_ = AffinityPropagation(damping=damping_, preference=preference_, random_state=0)
                labels_ = ap_.fit_predict(X_scaled)
                n_clusters_ = len(np.unique(labels_))

                if n_clusters_ <= 1 or n_clusters_ >= len(X_scaled):
                    return float('-inf')

                db_ = davies_bouldin_score(X_scaled, labels_)
                sil_ = silhouette_score(X_scaled, labels_)
                return alpha * (1 / max(db_, 1e-12)) + beta * sil_

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

            best_params_ = study.best_params
            best_damping = best_params_["damping"]
            best_preference = best_params_["preference"]

            best_ap = AffinityPropagation(damping=best_damping, preference=best_preference, random_state=0)
            final_labels = best_ap.fit_predict(X_scaled)

            final_db_score = davies_bouldin_score(X_scaled, final_labels)
            final_silhouette_score = silhouette_score(X_scaled, final_labels)

            return save_results(
                cleaned_file_path,
                dataset_id,
                "AP",
                {"damping": damping_str, "preference": pref_str,
                 "best_damping": best_damping, "best_preference": best_preference},
                final_labels,
                final_db_score,
                final_silhouette_score,
                alpha,
                beta,
                start_time
            )

        # -----------------------------------------------------------------
        # 5) ========== DBSCAN ==========
        elif algo == "DBSCAN":
            eps_str = params.get("eps", "0.1-1.0")
            min_samples_str = params.get("min_samples", "5-25")

            eps_low, eps_high = parse_range(eps_str, is_int=False)
            ms_low, ms_high = parse_range(min_samples_str, is_int=True)

            n_trials = params.get("n_trials", 50)

            def objective(trial):
                eps_ = trial.suggest_float("eps", eps_low, eps_high)
                min_s_ = trial.suggest_int("min_samples", ms_low, ms_high)

                dbscan_ = DBSCAN(eps=eps_, min_samples=min_s_, metric='euclidean')
                labels_ = dbscan_.fit_predict(X_scaled)
                n_clusters_ = len(np.unique(labels_)) - (1 if -1 in labels_ else 0)

                if n_clusters_ < 2:
                    return float('-inf')

                # 噪声点惩罚
                noise_ratio = np.mean(labels_ == -1)
                noise_penalty = max(0, 1 - noise_ratio)

                db_ = davies_bouldin_score(X_scaled, labels_)
                sil_ = silhouette_score(X_scaled, labels_)
                combined_score_ = (alpha * (1 / max(db_, 1e-12)) + beta * sil_) * noise_penalty
                return combined_score_

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

            best_params_ = study.best_params
            best_eps = best_params_["eps"]
            best_min_samples = best_params_["min_samples"]

            best_dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples, metric='euclidean')
            final_labels = best_dbscan.fit_predict(X_scaled)
            n_clusters_final = len(np.unique(final_labels)) - (1 if -1 in final_labels else 0)

            if n_clusters_final > 1:
                final_db_score = davies_bouldin_score(X_scaled, final_labels)
                final_silhouette_score = silhouette_score(X_scaled, final_labels)
            else:
                final_db_score = float('inf')
                final_silhouette_score = 0.0

            return save_results(
                cleaned_file_path,
                dataset_id,
                "DBSCAN",
                {"eps": eps_str, "min_samples": min_samples_str,
                 "best_eps": best_eps, "best_min_samples": best_min_samples},
                final_labels,
                final_db_score,
                final_silhouette_score,
                alpha,
                beta,
                start_time
            )

        # -----------------------------------------------------------------
        # 6) ========== OPTICS ==========
        elif algo == "OPTICS":
            # 示例参数：{"min_samples": "5-15", "xi": "0.01 to 0.05", "min_cluster_size": "0.01-0.05"}
            min_samples_str = params.get("min_samples", "5-15")
            xi_str = params.get("xi", "0.01-0.05")
            min_cluster_size_str = params.get("min_cluster_size", "0.01-0.05")

            ms_low, ms_high = parse_range(min_samples_str, is_int=True)
            xi_low, xi_high = parse_range(xi_str, is_int=False)
            mcs_low, mcs_high = parse_range(min_cluster_size_str, is_int=False)

            n_trials = params.get("n_trials", 50)

            # 先计算余弦距离矩阵
            X_cosine = cosine_distances(X_scaled)

            def objective(trial):
                ms_ = trial.suggest_int("min_samples", ms_low, ms_high)
                xi_ = trial.suggest_float("xi", xi_low, xi_high)
                mcs_ = trial.suggest_float("min_cluster_size", mcs_low, mcs_high)

                optics_ = OPTICS(min_samples=ms_, xi=xi_, min_cluster_size=mcs_, metric='precomputed')
                optics_.fit(X_cosine)
                labels_ = optics_.labels_

                n_clusters_ = len(np.unique(labels_)) - (1 if -1 in labels_ else 0)
                if n_clusters_ < 2:
                    return float('-inf')

                noise_ratio = np.mean(labels_ == -1)
                noise_penalty = max(0, 1 - noise_ratio)

                sil_ = silhouette_score(X_cosine, labels_, metric='precomputed')
                db_ = davies_bouldin_score(X_scaled, labels_)

                combined_score_ = (alpha * (1 / max(db_, 1e-12)) + beta * sil_) * noise_penalty
                return combined_score_

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

            best_params_ = study.best_params
            best_ms = best_params_["min_samples"]
            best_xi = best_params_["xi"]
            best_mcs = best_params_["min_cluster_size"]

            # 最终聚类
            final_optics = OPTICS(min_samples=best_ms, xi=best_xi, min_cluster_size=best_mcs, metric='precomputed')
            final_optics.fit(X_cosine)
            final_labels = final_optics.labels_

            n_clusters_final = len(np.unique(final_labels)) - (1 if -1 in final_labels else 0)
            if n_clusters_final > 1:
                final_silhouette_score = silhouette_score(X_cosine, final_labels, metric='precomputed')
                final_db_score = davies_bouldin_score(X_scaled, final_labels)
            else:
                final_silhouette_score = 0.0
                final_db_score = float('inf')

            return save_results(
                cleaned_file_path,
                dataset_id,
                "OPTICS",
                {"min_samples": min_samples_str,
                 "xi": xi_str,
                 "min_cluster_size": min_cluster_size_str,
                 "best_min_samples": best_ms,
                 "best_xi": best_xi,
                 "best_min_cluster_size": best_mcs},
                final_labels,
                final_db_score,
                final_silhouette_score,
                alpha,
                beta,
                start_time
            )

        else:
            raise ValueError(f"[ERROR] 不支持的算法类型: {algorithm}")

    except Exception as e:
        print(f"[ERROR] 聚类时出错: {e}")
        return None, None

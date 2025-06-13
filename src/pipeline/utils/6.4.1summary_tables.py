from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, spearmanr

# ────────────────────────────────────────────────
# 可选：Nemenyi 事后检验需要 scikit-posthocs
try:
    import scikit_posthocs as sp
    HAS_SPH = True
except ModuleNotFoundError:
    HAS_SPH = False
# ────────────────────────────────────────────────

DATASETS = ["beers", "flights", "hospital", "rayyan"]
BASE_DIR = Path("../../../results/analysis_results")   # 统一的结果目录
BASE_DIR.mkdir(parents=True, exist_ok=True)

# 1. 读取四张 summary.xlsx
df_dict = {}
for ds in DATASETS:
    fp = BASE_DIR / f"{ds}_summary.xlsx"
    if not fp.exists():
        raise FileNotFoundError(f"❌ 找不到数据文件：{fp}")
    df_dict[ds] = pd.read_excel(fp)

# 2. 计算“数据集 × 清洗方法”的平均 F1
mean_f1 = pd.DataFrame(
    {ds: df.groupby("cleaning_method")["F1"].mean() for ds, df in df_dict.items()}
).sort_index()
mean_f1.to_csv(BASE_DIR / "cleaning_method_F1_matrix.csv")

# 3. 仅保留四个数据集都出现过的清洗方法
common_methods = mean_f1.dropna().index.tolist()
f1_common = mean_f1.loc[common_methods]

# 4. Friedman 检验
scores = [f1_common[ds].values for ds in DATASETS]       # 每个数据集一列
chi2, p_val = friedmanchisquare(*scores)

# 5. Nemenyi 事后比较（或手动 CD）
if HAS_SPH:
    posthoc = sp.posthoc_nemenyi_friedman(f1_common.T.values)
    posthoc.index = common_methods
    posthoc.columns = common_methods
    posthoc.to_csv(BASE_DIR / "posthoc_nemenyi_friedman.csv")
else:
    # —— 手动计算临界差值（CD） ——
    ranks = f1_common.rank(ascending=False, axis=0)
    avg_ranks = ranks.mean(axis=1)
    k, N = len(common_methods), len(DATASETS)
    q_alpha = 3.314                    # α = 0.05 近似临界值
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
    avg_ranks.to_csv(BASE_DIR / "average_ranks.csv", header=["avg_rank"])
    with open(BASE_DIR / "critical_difference.txt", "w") as f:
        f.write(f"Critical Difference (α=0.05): {cd:.4f}\n")

# 6. Spearman 排名相关矩阵
rank_matrix = f1_common.rank(ascending=False, axis=0)
rho_mat = pd.DataFrame(index=DATASETS, columns=DATASETS, dtype=float)
for d1 in DATASETS:
    for d2 in DATASETS:
        rho_mat.loc[d1, d2] = spearmanr(rank_matrix[d1], rank_matrix[d2]).correlation
rho_mat.to_csv(BASE_DIR / "spearman_rank_correlation.csv")

# 7. 控制台快速摘要
print("=" * 60)
print(f"Friedman χ² = {chi2:.4f},  p = {p_val:.4g}")
if not HAS_SPH:
    print(f"⚠️  未安装 scikit_posthocs —— 已手动计算 CD 写入 critical_difference.txt")
print("\nTop-5 rows of mean F1 matrix:")
print(mean_f1.head())
print("\nSpearman ρ_rank 矩阵：")
print(rho_mat)
print("=" * 60)

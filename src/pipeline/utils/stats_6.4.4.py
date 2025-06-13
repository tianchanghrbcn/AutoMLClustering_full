#!/usr/bin/env python
# coding: utf-8
"""
Generate Table 10 (hyper-parameter drift) and Table 11 (two-way ANOVA)
for Section 6.4.4.

Author: ChatGPT  (2025-06-11)
"""

# ──────────────────── imports ──────────────────── #
import re
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ──────────────────── 路径与数据集 ──────────────────── #
DATASETS   = ["beers", "flights", "hospital", "rayyan"]

ROOT_DIR   = Path(__file__).resolve().parents[3]
INPUT_DIR  = ROOT_DIR / "results" / "analysis_results"
OUTPUT_DIR = ROOT_DIR / "task_progress" / "tables" / "6.4.4tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────── 参数解析工具函数 ──────────────────── #
K_PAT   = re.compile(r"k\s*=\s*([0-9]+)")
EPS_PAT = re.compile(r"eps\s*=\s*([0-9]*\.?[0-9]+)")
COV_PAT = re.compile(r"covariance\s+type\s*=\s*([a-zA-Z_]+)")

def parse_parameters(param_str: str) -> Tuple[Optional[int], Optional[float], Optional[str]]:
    """
    从 parameters 字段中提取:
      - k  (int)
      - eps (float)
      - covariance type (str: full / tied / spherical / diag …)

    若对应字段不存在，则返回 None.
    """
    k_match   = K_PAT.search(param_str)
    eps_match = EPS_PAT.search(param_str)
    cov_match = COV_PAT.search(param_str)

    k   = int(k_match.group(1)) if k_match else None
    eps = float(eps_match.group(1)) if eps_match else None
    cov = cov_match.group(1) if cov_match else None

    return k, eps, cov


# ──────────────────── 读取与拼接 ──────────────────── #
def load_dataset(path: Path, dataset_name: str) -> pd.DataFrame:
    """Load a single dataset xlsx and append helper columns."""
    df = pd.read_excel(path, engine="openpyxl")
    df["dataset"] = dataset_name
    # 提取 k/eps/cov
    parsed = df["parameters"].fillna("").apply(parse_parameters)
    df[["k", "eps", "cov_type"]] = pd.DataFrame(parsed.tolist(), index=df.index)

    # 离散化 error_rate
    bins   = [0, 5, 10, 15, 25, 30, np.inf]
    labels = ["0-5", "5-10", "10-15", "15-25", "25-30", "≥30"]
    df["error_bin"] = pd.cut(df["error_rate"], bins=bins, labels=labels, right=False)
    return df


def concat_all() -> pd.DataFrame:
    """Load every dataset listed in DATASETS into one big DataFrame."""
    frames = []
    for ds in DATASETS:
        # 兼容文件名写法： beers.xlsx  或  beers_results.xlsx
        file_candidates = list(INPUT_DIR.glob(f"*{ds}*.xlsx"))
        if not file_candidates:
            print(f"[WARN] No xlsx found for dataset '{ds}' in {INPUT_DIR}")
            continue
        df = load_dataset(file_candidates[0], ds)
        frames.append(df)
    if not frames:
        raise RuntimeError("No dataset files loaded – please check INPUT_DIR.")
    return pd.concat(frames, ignore_index=True)


# ──────────────────── Δ 计算 ──────────────────── #
def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    对同一 dataset × error_bin × cluster_method 计算相对漂移:
    ① 先找 GroundTruth 行（若无则退到 cleaning_method=='None'）
    ② 若仍无基准，则用同 group 的中位数代替
    Δ = 当前值 – 基准值
    """
    delta_rows = []
    grouping_cols = ["dataset", "error_bin", "cluster_method"]

    for _, group in df.groupby(grouping_cols):
        # 基准行优先级
        base = group[group["cleaning_method"].str.lower() == "groundtruth"]
        if base.empty:
            base = group[group["cleaning_method"].str.lower() == "none"]
        if base.empty:
            # fallback: 用该组中位数
            base_k   = group["k"].median()
            base_eps = group["eps"].median()
            base_cov = group["cov_type"].mode().iloc[0] if group["cov_type"].notna().any() else None
        else:
            base_k   = base["k"].iloc[0]
            base_eps = base["eps"].iloc[0]
            base_cov = base["cov_type"].iloc[0]

        for idx, row in group.iterrows():
            delta_k   = row["k"]   - base_k   if pd.notna(row["k"])   and pd.notna(base_k)   else np.nan
            delta_eps = row["eps"] - base_eps if pd.notna(row["eps"]) and pd.notna(base_eps) else np.nan
            # cov_type → 漂移定义为 0/1 (same / different)
            delta_cov = 0 if (row["cov_type"] == base_cov) else 1 if pd.notna(row["cov_type"]) else np.nan

            delta_rows.append({
                **row.to_dict(),
                "Δk":   delta_k,
                "Δε":  delta_eps,
                "Δcov": delta_cov,
            })

    return pd.DataFrame(delta_rows)


# ──────────────────── 表 10：跨清洗法的 Δ 中位数 ──────────────────── #
def make_table10(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot 成：
        index  = error_bin
        columns= cleaning_method (MultiIndex: method & metric)
        values = median Δ
    """
    # 聚合
    agg = (df.groupby(["error_bin", "cleaning_method"])
             .agg({"Δk":"median", "Δε":"median", "Δcov":"median"})
             .rename(columns={"Δk":"delta_k", "Δε":"delta_eps", "Δcov":"delta_cov"}))

    # pivot – 得到  error_bin × (cleaning_method, metric) 的宽表
    wide = agg.unstack("cleaning_method").round(3)

    # 保存 Excel
    out_path = OUTPUT_DIR / "table10_hyper_shift.xlsx"
    wide.to_excel(out_path, merge_cells=False, engine="openpyxl")
    print(f"[INFO] Table 10 written to {out_path}")
    return wide


# ──────────────────── 表 11：两因素 ANOVA ──────────────────── #
def anova_single_metric(df: pd.DataFrame, metric: str) -> Tuple[float, float, float]:
    """
    返回 (R², F, p) for  error_bin + cleaning_method + interaction.
    metric ∈ {"Δk", "Δε", "Δcov"}
    """
    # 丢掉缺失
    _df = df[[metric, "error_bin", "cleaning_method"]].dropna()
    # statsmodels 需要字符串类别
    _df["error_bin"]       = _df["error_bin"].astype(str)
    _df["cleaning_method"] = _df["cleaning_method"].astype(str)

    # ordinary least squares
    model = ols(f"{metric} ~ C(error_bin) + C(cleaning_method) + C(error_bin):C(cleaning_method)",
                data=_df, missing="drop").fit()

    # ANOVA
    aov_table = sm.stats.anova_lm(model, typ=2)

    # 主效应
    r2_total = model.rsquared
    # statsmodels 不直接给每项 R²；用 sum_sq / total_ss 近似解释度
    total_ss = aov_table["sum_sq"].sum()
    item_r2  = aov_table["sum_sq"] / total_ss

    # 提取各行结果
    main_err   = aov_table.loc["C(error_bin)"]
    main_clean = aov_table.loc["C(cleaning_method)"]
    inter      = aov_table.loc["C(error_bin):C(cleaning_method)"]

    # Return important numbers as dict row
    return {
        "Metric": metric,
        "Err_R2":   item_r2["C(error_bin)"],
        "Err_F":    main_err["F"],
        "Err_p":    main_err["PR(>F)"],
        "Clean_R2": item_r2["C(cleaning_method)"],
        "Clean_F":  main_clean["F"],
        "Clean_p":  main_clean["PR(>F)"],
        "Inter_R2": item_r2["C(error_bin):C(cleaning_method)"],
        "Inter_F":  inter["F"],
        "Inter_p":  inter["PR(>F)"],
    }


def make_table11(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in ["Δk", "Δε", "Δcov"]:
        rows.append(anova_single_metric(df, m))
    tbl = pd.DataFrame(rows)
    # 星号显著性标记
    def star(p):
        if p < 0.001: return "†"
        if p < 0.01:  return "☆"
        if p < 0.05:  return "★"
        return ""
    for col in ["Err_p", "Clean_p", "Inter_p"]:
        tbl[col.replace("_p", "_sig")] = tbl[col].apply(star)

    # 保存
    out_path = OUTPUT_DIR / "table11_anova.xlsx"
    tbl.to_excel(out_path, index=False, engine="openpyxl")
    print(f"[INFO] Table 11 written to {out_path}")
    return tbl


# ──────────────────── 主函数 ──────────────────── #
def main():
    df_raw = concat_all()
    df_delta = compute_deltas(df_raw)

    _ = make_table10(df_delta)
    _ = make_table11(df_delta)

    print("[DONE] Section 6.4.4 tables ready.")


if __name__ == "__main__":
    main()

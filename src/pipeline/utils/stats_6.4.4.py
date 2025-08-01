#!/usr/bin/env python
# coding: utf-8
"""
Generate Table 10 (hyper‑parameter drift) & Table 11 (two-way ANOVA, Sect. 6.4.4)
– 使用最近 5 的倍数作为 error_bin，Silhouette Score 映射到 [0,1].

Author: ChatGPT  (2025‑07‑30)
"""
# ── imports ─────────────────────────────────────────────────────────
import re
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ── DATASET & PATHS ────────────────────────────────────────────────
DATASETS   = ["beers", "flights", "hospital", "rayyan"]

ROOT_DIR   = Path(__file__).resolve().parents[3]
INPUT_DIR  = ROOT_DIR / "results" / "analysis_results"
OUTPUT_DIR = ROOT_DIR / "task_progress" / "tables" / "6.4.4tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── RegEx for parameter parsing ────────────────────────────────────
K_PAT   = re.compile(r"k\s*=\s*([0-9]+)")
EPS_PAT = re.compile(r"eps\s*=\s*([0-9]*\.?[0-9]+)")
COV_PAT = re.compile(r"covariance\s+type\s*=\s*([a-zA-Z_]+)")

def parse_parameters(param_str: str) -> Tuple[Optional[int], Optional[float], Optional[str]]:
    """从 parameters 字符串中提取 k / eps / cov_type."""
    k_match, eps_match, cov_match = K_PAT.search(param_str), EPS_PAT.search(param_str), COV_PAT.search(param_str)
    k   = int(k_match.group(1))       if k_match  else None
    eps = float(eps_match.group(1))   if eps_match else None
    cov = cov_match.group(1)          if cov_match else None
    return k, eps, cov

# ── load & concat ──────────────────────────────────────────────────
def load_dataset(path: Path, dataset_name: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df["dataset"] = dataset_name

    # Silhouette Score 映射到 [0,1]
    if "Silhouette Score" in df.columns:
        df["Silhouette Score"] = (df["Silhouette Score"] + 1) / 2

    # 解析 k / eps / cov
    parsed = df["parameters"].fillna("").apply(parse_parameters)
    df[["k", "eps", "cov_type"]] = pd.DataFrame(parsed.tolist(), index=df.index)

    # error_bin = 最近 5 的倍数（数值列）
    df["error_bin"] = ((df["error_rate"] / 5).round() * 5).astype(int)  # e.g. 13.2 → 15
    return df


def concat_all() -> pd.DataFrame:
    frames = []
    for ds in DATASETS:
        cand = list(INPUT_DIR.glob(f"*{ds}*.xlsx"))
        if not cand:
            print(f"[WARN] no xlsx for '{ds}'")
            continue
        frames.append(load_dataset(cand[0], ds))
    if not frames:
        raise RuntimeError("No dataset files loaded – check INPUT_DIR.")
    return pd.concat(frames, ignore_index=True)

# ── Δ computation ─────────────────────────────────────────────────
def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于 dataset × error_bin × cluster_method 计算参数漂移 Δ。
    """
    rows, grouping_cols = [], ["dataset", "error_bin", "cluster_method"]

    for _, group in df.groupby(grouping_cols, observed=False):
        base = group[group["cleaning_method"].str.lower() == "groundtruth"]
        if base.empty:
            base = group[group["cleaning_method"].str.lower() == "none"]

        if base.empty:      # fallback 中位/众数
            base_k   = group["k"].median()
            base_eps = group["eps"].median()
            base_cov = group["cov_type"].mode().iat[0] if group["cov_type"].notna().any() else None
        else:
            base_k, base_eps, base_cov = base["k"].iat[0], base["eps"].iat[0], base["cov_type"].iat[0]

        for _, row in group.iterrows():
            rows.append({
                **row.to_dict(),
                "Δk":   row["k"]   - base_k   if pd.notna(row["k"])   and pd.notna(base_k)   else np.nan,
                "Δε":  row["eps"] - base_eps if pd.notna(row["eps"]) and pd.notna(base_eps) else np.nan,
                "Δcov": 0 if row["cov_type"] == base_cov else 1 if pd.notna(row["cov_type"]) else np.nan,
            })
    return pd.DataFrame(rows)

# ── Table 10 ──────────────────────────────────────────────────────
def make_table10(df: pd.DataFrame) -> pd.DataFrame:
    agg = (df.groupby(["error_bin", "cleaning_method"], observed=False)
             .agg({"Δk": "median", "Δε": "median", "Δcov": "median"})
             .rename(columns={"Δk": "delta_k", "Δε": "delta_eps", "Δcov": "delta_cov"}))
    wide = agg.unstack("cleaning_method").round(3)
    out = OUTPUT_DIR / "table10_hyper_shift.xlsx"
    wide.to_excel(out, merge_cells=False, engine="openpyxl")
    print(f"[INFO] Table 10 written → {out}")
    return wide

# ── Table 11 ──────────────────────────────────────────────────────
def anova_single_metric(df: pd.DataFrame, metric: str) -> dict:
    _df = df[[metric, "error_bin", "cleaning_method"]].dropna()

    if (_df["cleaning_method"].nunique() < 2) or (_df["error_bin"].nunique() < 2):
        print(f"[SKIP] {metric}: levels<2")
        nan_row = {k: np.nan for k in
                   ["Err_R2","Err_F","Err_p","Clean_R2","Clean_F","Clean_p","Inter_R2","Inter_F","Inter_p"]}
        return {"Metric": metric, **nan_row}

    # error_bin 连续变量，不加 C()
    model = ols(f"{metric} ~ error_bin + C(cleaning_method) + error_bin:C(cleaning_method)",
                data=_df, missing="drop").fit()
    aov   = sm.stats.anova_lm(model, typ=2)

    total_ss = aov["sum_sq"].sum()
    item_r2  = aov["sum_sq"] / total_ss

    return {
        "Metric":    metric,
        "Err_R2":    item_r2["error_bin"],
        "Err_F":     aov.loc["error_bin", "F"],
        "Err_p":     aov.loc["error_bin", "PR(>F)"],
        "Clean_R2":  item_r2["C(cleaning_method)"],
        "Clean_F":   aov.loc["C(cleaning_method)", "F"],
        "Clean_p":   aov.loc["C(cleaning_method)", "PR(>F)"],
        "Inter_R2":  item_r2["error_bin:C(cleaning_method)"],
        "Inter_F":   aov.loc["error_bin:C(cleaning_method)", "F"],
        "Inter_p":   aov.loc["error_bin:C(cleaning_method)", "PR(>F)"],
    }

def make_table11(df: pd.DataFrame) -> pd.DataFrame:
    rows = [anova_single_metric(df, m) for m in ["Δk", "Δε", "Δcov"]]
    tbl  = pd.DataFrame(rows)

    def star(p):
        if p < 0.001: return "†"
        if p < 0.01:  return "☆"
        if p < 0.05:  return "★"
        return ""
    for col in ["Err_p", "Clean_p", "Inter_p"]:
        tbl[col.replace("_p", "_sig")] = tbl[col].apply(star)

    out = OUTPUT_DIR / "table11_anova.xlsx"
    tbl.to_excel(out, index=False, engine="openpyxl")
    print(f"[INFO] Table 11 written → {out}")
    return tbl

# ── main ──────────────────────────────────────────────────────────
def main():
    df_raw   = concat_all()
    df_delta = compute_deltas(df_raw)
    make_table10(df_delta)
    make_table11(df_delta)
    print("✅ Section 6.4.4 tables ready (error_bin 数值化 & Silhouette rescaled).")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats_corr_cov.py
=================
• 读取 4× *_summary.xlsx 结果表            （路径: ../../../results/analysis_results）
• 强制把文本型数字 → float
• 若缺 EDR / F1 列则按 TP/FP/FN 自动推算
• 按『数据集 × error_rate』(7 档) 聚合中位数 → 恢复 28 条观测
• 计算:
    – Pearson / Spearman ρ :  EDR↔Comb , F1↔Comb
    – 五指标联合协方差矩阵 Σ₅×₅   (EDR, F1, Comb, DB*, Sil*)
• 将 28 条观测 & Σ 写入  corr_cov_summary.xlsx
"""

from pathlib import Path
import re, sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# ───── 0. 基本参数 ─────────────────────────────────────────────────────────────
TASKS    = ["beers", "flights", "hospital", "rayyan"]
ROOT     = Path(__file__).resolve().parents[3]
SRC_DIR  = ROOT / "results" / "analysis_results"
OUT_DIR  = ROOT / "task_progress" / "tables" / "6.4.3stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_RE   = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def force_float(s: pd.Series) -> pd.Series:
    """将任意字符串列强制转换成 float（无法解析→NaN）"""
    return s.apply(lambda x: float(NUM_RE.search(str(x)).group()) if NUM_RE.search(str(x)) else np.nan)

# ───── 1. 读取 & 预处理 ────────────────────────────────────────────────────────
frames = []
for task in TASKS:
    fp = SRC_DIR / f"{task}_summary.xlsx"
    if not fp.is_file():
        print(f"[WARN] {fp} 缺失 – 跳过该数据集")
        continue

    df = pd.read_excel(fp, engine="openpyxl")
    df["task"] = task

    # ---------- 1.1 数值化 ----------------------------------------------------
    num_cols = ["error_rate", "EDR", "F1", "Combined Score",
                "Silhouette Score", "Davies-Bouldin Score",
                "TP_e1", "TP_e2", "FP_e1", "FP_e2",
                "FN_e1", "FN_e2"]
    for c in num_cols:
        if c in df.columns:
            df[c] = force_float(df[c])

    # ---------- 1.2 若缺 EDR / F1 → 自动推算 ----------------------------------
    if "EDR" not in df.columns or "F1" not in df.columns:
        need = ["TP_e1","TP_e2","FP_e1","FP_e2","FN_e1","FN_e2"]
        if not set(need) <= set(df.columns):
            sys.exit(f"[ERROR] {fp} 缺少 EDR/F1 且无法通过 TP/FP/FN 计算")
        tp = df["TP_e1"].fillna(0) + df["TP_e2"].fillna(0)
        fp = df["FP_e1"].fillna(0) + df["FP_e2"].fillna(0)
        fn = df["FN_e1"].fillna(0) + df["FN_e2"].fillna(0)
        with np.errstate(divide='ignore', invalid='ignore'):
            df["EDR"] = (tp - fp) / (tp + fn)
            precision = tp / (tp + fp)
            recall    = tp / (tp + fn)
            df["F1"]  = 2 * precision * recall / (precision + recall)
        print(f"[INFO] {task}: EDR / F1 由 TP/FP/FN 推算完成")

    # ---------- 1.3 生成 DB* 并保存 -------------------------------------------
    df["DB*"] = 1 / (1 + force_float(df["Davies-Bouldin Score"]))
    frames.append(df)

if not frames:
    sys.exit("❌ 未加载到任何数据，请检查目录与文件名！")

df_all = pd.concat(frames, ignore_index=True)

# ───── 2. 将 error_rate 按 5% 分箱，聚合中位数 ────────────────────────────────
bins   = [0, 5, 10, 15, 20, 25, 30, np.inf]
labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "≥30"]
df_all["err_bin"] = pd.cut(df_all["error_rate"], bins=bins,
                           labels=labels, right=False)

agg = (df_all.groupby(["task", "err_bin"])
             .agg(EDR=("EDR", "median"),
                  F1=("F1", "median"),
                  Comb=("Combined Score", "median"),
                  DB_star=("DB*", "median"),
                  Sil=("Silhouette Score", "median"))
             .reset_index())

if agg.isna().any().any():
    print("[WARN] NaN 仍然存在，某些组合缺观测")

# ───── 3. 相关系数 ────────────────────────────────────────────────────────────
def rho(x, y, kind="pearson"):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return np.nan
    if kind == "pearson":
        return pearsonr(x[mask], y[mask])[0]
    return spearmanr(x[mask], y[mask])[0]

p_edr  = rho(agg["EDR"].values, agg["Comb"].values, "pearson")
p_f1   = rho(agg["F1"].values,  agg["Comb"].values, "pearson")
s_edr  = rho(agg["EDR"].values, agg["Comb"].values, "spearman")
s_f1   = rho(agg["F1"].values,  agg["Comb"].values, "spearman")

print("\n──────── Pearson & Spearman ρ ────────")
print(f"Pearson  ρ(EDR , Comb) = {p_edr:6.3f}")
print(f"Pearson  ρ(F1  , Comb) = {p_f1:6.3f}")
print(f"Spearman ρ(EDR , Comb) = {s_edr:6.3f}")
print(f"Spearman ρ(F1  , Comb) = {s_f1:6.3f}")

# ───── 4. 全局协方差矩阵 Σ₅×₅ ────────────────────────────────────────────────
mat = agg[["EDR", "F1", "Comb", "DB_star", "Sil"]].to_numpy(float)
cov = np.cov(mat, rowvar=False)
Σ = pd.DataFrame(cov,
                 index=["EDR","F1","Comb","DB*","Sil"],
                 columns=["EDR","F1","Comb","DB*","Sil"])

print("\nΣ – 联合协方差矩阵 (n=28 行)：")
print(Σ.round(4))

# ───── 5. 导出 Excel ─────────────────────────────────────────────────────────
out_xlsx = OUT_DIR / "corr_cov_summary.xlsx"
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
    agg.to_excel(w, sheet_name="28_observations", index=False)
    Σ.to_excel(w, sheet_name="cov_matrix")
print(f"\n[OK] 结果写入：{out_xlsx}")

# ───── 6. 打印参与条目摘要 ────────────────────────────────────────────────────
print("\n=== 使用的 28 条观测（中位值）===")
print(agg[["task","err_bin","EDR","F1","Comb"]].to_string(index=False))

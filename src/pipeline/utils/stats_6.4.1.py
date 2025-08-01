"""
统计单元格级清洗准确度（细粒度版）+ 中位指标汇总
================================================
运行：
    python -m src.pipeline.utils.stats_6_4_1
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ---------- 配置 ----------
CONF_PATH = Path(__file__).resolve().parents[2] / "pipeline" / "train" / "comparison.json"
CFG_DIR   = CONF_PATH.parent
OUT_DIR   = Path(__file__).resolve().parents[3] / "results" / "analysis_results" / "stats"
OUT_CSV   = OUT_DIR / "q1_cell_level_counts.csv"
SMALL = 1e-8

# ---------- 工具 ----------
def resolve(p_str: str) -> Path:
    p = Path(p_str)
    return p if p.is_absolute() else (CFG_DIR / p).resolve()

def safe_read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)

def align(d1: pd.DataFrame, d2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """行列交集并按 dirty 顺序对齐"""
    d1_ = d1.reset_index(drop=True)
    d2_ = d2.reset_index(drop=True)
    cols = d1_.columns.intersection(d2_.columns)
    return d1_[cols], d2_[cols]

# ---------- 单条记录 ----------
def process_one_record(rec: Dict, outs: List[Dict]) -> None:
    task, did, err_rate = rec["task_name"], rec["dataset_id"], rec["error_rate"]

    clean_df = safe_read_csv(resolve(rec["paths"]["clean_csv"]))
    dirty_df = safe_read_csv(resolve(rec["paths"]["dirty_csv"]))
    dirty_df, clean_df = align(dirty_df, clean_df)

    # ---- 原始错误掩码 ----
    mask_orig_missing = dirty_df.isna() & clean_df.notna()             # e1
    mask_orig_anom    = (~dirty_df.isna()) & (dirty_df != clean_df)    # e2

    n_w_e1 = int(mask_orig_missing.values.sum())
    n_w_e2 = int(mask_orig_anom.values.sum())
    n_w    = n_w_e1 + n_w_e2
    n_raw  = dirty_df.size

    for method, rep_rel in rec["paths"]["repaired_paths"].items():
        rep_df = safe_read_csv(resolve(rep_rel))
        rep_df, _ = align(rep_df, dirty_df)

        mask_correct_after = rep_df == clean_df
        mask_changed       = rep_df != dirty_df

        # ----- TP -----
        TP_e1 = int((mask_orig_missing & mask_correct_after).values.sum())
        TP_e2 = int((mask_orig_anom    & mask_correct_after).values.sum())

        # ----- FN -----
        FN_e1 = n_w_e1 - TP_e1
        FN_e2 = n_w_e2 - TP_e2

        # ----- FP e1 -----
        fp1_case1 = (~mask_orig_missing) & (~mask_orig_anom) & (~dirty_df.isna()) & rep_df.isna()
        fp1_case2 = clean_df.isna() & dirty_df.isna() & (~rep_df.isna())
        FP_e1 = int((fp1_case1 | fp1_case2).values.sum())

        # ----- FP e2 -----
        fp2_case = (~mask_orig_missing) & (~mask_orig_anom) & (~dirty_df.isna()) \
                   & (~rep_df.isna()) & (rep_df != clean_df)
        FP_e2 = int(fp2_case.values.sum())

        # ----- TN / 其他 -----
        n_r2w = FP_e1 + FP_e2
        n_r2r = n_raw - n_w - n_r2w

        # ----- 指标 -----
        def metrics(tp: int, fp: int, fn: int):
            precision = tp / (tp + fp + SMALL)
            recall    = tp / (tp + fn + SMALL)
            f1        = 2 * precision * recall / (precision + recall + SMALL)
            edr       = (tp - fp) / (tp + fn + SMALL)
            return edr, precision, recall, f1

        edr_e1, prec_e1, rec_e1, f1_e1 = metrics(TP_e1, FP_e1, FN_e1)
        edr_e2, prec_e2, rec_e2, f1_e2 = metrics(TP_e2, FP_e2, FN_e2)

        outs.append({
            "task_name": task, "dataset_id": did, "error_rate": err_rate, "method": method,
            "n_raw": n_raw, "n_w": n_w,
            "n_w_e1": n_w_e1, "n_w_e2": n_w_e2,
            "TP_e1": TP_e1, "TP_e2": TP_e2,
            "FP_e1": FP_e1, "FP_e2": FP_e2,
            "FN_e1": FN_e1, "FN_e2": FN_e2,
            "n_r2r": n_r2r,
            "EDR_e1": edr_e1, "Precision_e1": prec_e1, "Recall_e1": rec_e1, "F1_e1": f1_e1,
            "EDR_e2": edr_e2, "Precision_e2": prec_e2, "Recall_e2": rec_e2, "F1_e2": f1_e2
        })

# ---------- 名称格式 ----------
def display_name(method: str) -> str:
    mapping = {
        "mode": "Mode",
        "bigdansing": "BigDansing",
        "boostclean": "BoostClean",
        "holoclean": "HoloClean",
        "horizon": "Horizon",
        "scared": "SCARE",
        "scare": "SCARE",
        "baran": "Raha\\textminus Baran",
        "raha-baran": "Raha\\textminus Baran",
        "unified": "Unified",
    }
    return mapping.get(method.lower(), method)

# ---------- 中位汇总 & LaTeX ----------
def generate_summary() -> None:
    metrics_e1 = ["EDR_e1", "Precision_e1", "Recall_e1", "F1_e1"]
    metrics_e2 = ["EDR_e2", "Precision_e2", "Recall_e2", "F1_e2"]
    metrics_all = metrics_e1 + metrics_e2

    if not OUT_CSV.exists():
        raise FileNotFoundError(OUT_CSV)

    df = pd.read_csv(OUT_CSV)
    df = df[df["method"] != "GroundTruth"]

    df.loc[df["n_w_e1"] == 0, metrics_e1] = 0.0
    df.loc[df["n_w_e2"] == 0, metrics_e2] = 0.0

    rows_summary = []
    for (task, method), g in df.groupby(["task_name", "method"], sort=False):
        g_e1 = g[g["n_w_e1"] > 0]
        g_e2 = g[g["n_w_e2"] > 0]
        vals_e1 = g_e1[metrics_e1].median() if not g_e1.empty else pd.Series({m: 0.0 for m in metrics_e1})
        vals_e2 = g_e2[metrics_e2].median() if not g_e2.empty else pd.Series({m: 0.0 for m in metrics_e2})
        rows_summary.append({
            "task_name": task, "method": method,
            **vals_e1.to_dict(), **vals_e2.to_dict()
        })

    summary_df = pd.DataFrame(rows_summary).round(3)
    summary_df.to_csv(OUT_DIR / "q1_metrics_summary.csv", index=False)
    print("✓ 中位指标已写入 → q1_metrics_summary.csv")

    # ---- LaTeX ----
    tex = [
        r"\begin{table*}[t]",
        r"\centering\footnotesize",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{各清洗方法在两类错误上的综合指标（中位数，排除无错记录）}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{ll*{8}{S}}",
        r"\toprule",
        r"\multicolumn{2}{l}{} &",
        r"\multicolumn{4}{c}{\bfseries 缺失值错误} &",
        r"\multicolumn{4}{c}{\bfseries 异常值错误}\\",
        r"\cmidrule(lr){3-6}\cmidrule(l){7-10}",
        r"数据集 & 方法 &"
        r"{EDR} & {Prec.} & {Rec.} & {$F_1$} &"
        r"{EDR} & {Prec.} & {Rec.} & {$F_1$}\\",
        r"\midrule"
    ]

    for task, grp in summary_df.groupby("task_name", sort=False):
        max_vals = grp[metrics_all].max(numeric_only=True)
        first = True
        for _, row in grp.iterrows():
            bold = [r"\bfseries" if np.isclose(row[m], max_vals[m]) else "" for m in metrics_all]
            tex.append(
                (rf"\multirow{{{len(grp)}}}{{*}}{{\textbf{{{task}}}}} " if first else " ")
                + f"& {display_name(row['method'])} "
                + " & ".join(f"{b} {row[m]:.3f}" for b, m in zip(bold, metrics_all))
                + r" \\"
            )
            first = False
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    tex += [
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\label{tab:q1-acc-summary}",
        r"\end{table*}"
    ]
    (OUT_DIR / "q1_metrics_table.tex").write_text("\n".join(tex), encoding="utf-8")
    print("✓ LaTeX 表格已写入 → q1_metrics_table.tex")

# ---------- 主入口 ----------
def main_all() -> None:
    with open(CONF_PATH, encoding="utf-8") as fp:
        records = json.load(fp)

    rows: List[Dict] = []
    for rec in records:
        try:
            process_one_record(rec, rows)
        except Exception as e:
            print(f"[WARN] 跳过 {rec.get('task_name')} (id={rec.get('dataset_id')}): {e}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"✓ 单元格级统计完成 → {OUT_CSV}  (共 {len(rows)} 行)")

    generate_summary()

# ---------- 执行 ----------
if __name__ == "__main__":
    main_all()

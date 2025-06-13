"""
统计单元格级清洗准确度（细粒度版）
================================
计数字段：
    n_raw              总单元格数
    n_w                原始错误总数
    n_w_e1 / n_w_e2    Missing / Dirty 计数
    n_w2r              原错→对 (TP)
    n_w2r_e1 / _e2     按类型拆分 TP
    n_r2w              原对→错 (FP)
    n_w2w              错仍错 (FN)  (= n_w - n_w2r)
    n_w2w_e1 / _e2     按类型拆分 FN
    n_r2r              原对→对 (TN)
    Precision, Recall, F1, EDR
运行：
    python -m src.pipeline.utils.stats_6_4_1
"""
from __future__ import annotations
import json, pandas as pd, numpy as np
from pathlib import Path
from typing import Dict, List

# ---------- 配置 ----------
CONF_PATH = Path(__file__).resolve().parents[2] / "pipeline" / "train" / "comparison.json"
CFG_DIR   = CONF_PATH.parent
OUT_DIR   = Path(__file__).resolve().parents[3] / "results" / "analysis_results" / "stats"
OUT_CSV   = OUT_DIR / "q1_cell_level_counts.csv"

ERR_TYPES = ["Missing", "Dirty"]      # e1,e2
SMALL = 1e-8

# ---------- 工具函数 ----------
def resolve(p_str: str) -> Path:
    p = Path(p_str)
    return p if p.is_absolute() else (CFG_DIR / p).resolve()

def detect_err_type(clean_val, dirty_val) -> str | None:
    if pd.isna(dirty_val) and not pd.isna(clean_val):
        return "Missing"
    if dirty_val != clean_val:
        return "Dirty"
    return None

def safe_read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)

def align(d1: pd.DataFrame, d2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """行列取交集并按 dirty 顺序对齐"""
    d1_ = d1.reset_index(drop=True)
    d2_ = d2.reset_index(drop=True)
    cols = d1_.columns.intersection(d2_.columns)
    return d1_[cols], d2_[cols]

# ---------- 主流程 ----------
def process_one_record(rec: Dict, outs: List[Dict]) -> None:
    task, did, err_rate = rec["task_name"], rec["dataset_id"], rec["error_rate"]

    clean_df = safe_read_csv(resolve(rec["paths"]["clean_csv"]))
    dirty_df = safe_read_csv(resolve(rec["paths"]["dirty_csv"]))
    dirty_df, clean_df = align(dirty_df, clean_df)

    # === 预生成错误类型矩阵 ===
    err_type_mat = dirty_df.copy()
    for c in dirty_df.columns:
        err_type_mat[c] = [detect_err_type(cv, dv) for cv, dv in zip(clean_df[c], dirty_df[c])]

    mask_err_tot = err_type_mat.notna()
    n_raw = dirty_df.size
    n_w   = int(mask_err_tot.values.sum())

    n_w_e1 = int((err_type_mat == "Missing").values.sum())
    n_w_e2 = n_w - n_w_e1

    for method, rep_rel in rec["paths"]["repaired_paths"].items():
        rep_df = safe_read_csv(resolve(rep_rel))
        rep_df, _ = align(rep_df, dirty_df)           # 只保证列交集

        # --- 基布尔矩阵 ---
        mask_changed = rep_df != dirty_df
        mask_correct_after = rep_df == clean_df

        # True Positives
        mask_tp = mask_correct_after & mask_err_tot
        n_w2r   = int(mask_tp.values.sum())
        n_w2r_e1 = int((mask_tp & (err_type_mat == "Missing")).values.sum())
        n_w2r_e2 = n_w2r - n_w2r_e1

        # False Positives
        mask_fp = mask_changed & (~mask_err_tot)
        n_r2w   = int(mask_fp.values.sum())

        # False Negatives
        n_w2w   = n_w - n_w2r
        n_w2w_e1 = n_w_e1 - n_w2r_e1
        n_w2w_e2 = n_w_e2 - n_w2r_e2

        # True Negatives
        n_r2r = n_raw - n_w - n_r2w

        # --- 指标 ---
        precision = n_w2r / (n_w2r + n_r2w + SMALL)
        recall    = n_w2r / (n_w + SMALL)
        f1        = 2*precision*recall / (precision + recall + SMALL)
        edr       = (n_w2r - n_r2w) / (n_w + SMALL)

        outs.append({
            # 基本
            "task_name": task, "dataset_id": did, "error_rate": err_rate, "method": method,
            # 全局计数
            "n_raw": n_raw, "n_w": n_w, "n_w_e1": n_w_e1, "n_w_e2": n_w_e2,
            "n_w2r": n_w2r, "n_w2r_e1": n_w2r_e1, "n_w2r_e2": n_w2r_e2,
            "n_r2w": n_r2w,
            "n_w2w": n_w2w, "n_w2w_e1": n_w2w_e1, "n_w2w_e2": n_w2w_e2,
            "n_r2r": n_r2r,
            # 指标
            "EDR": edr, "Precision": precision, "Recall": recall, "F1": f1
        })

def main():
    with open(CONF_PATH, "r", encoding="utf-8") as fp:
        records = json.load(fp)

    rows: List[Dict] = []
    for rec in records:
        try:
            process_one_record(rec, rows)
        except Exception as e:
            print(f"[WARN] 跳过 {rec.get('task_name')} (id={rec.get('dataset_id')}): {e}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"✓ 统计完成 → {OUT_CSV}  (共 {len(rows)} 行)")

if __name__ == "__main__":
    main()

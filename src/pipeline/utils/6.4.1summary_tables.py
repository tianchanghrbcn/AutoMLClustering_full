# q1_build_tables.py  (6.4.1 tables)
# -----------------------------------------------------------
import os, pandas as pd, numpy as np
DATA_DIR = "../../../results/analysis_results"
OUT_DIR  = "../../../task_progress/tables/6.4.1tables"
TASKS    = ["beers", "flights", "hospital", "rayyan"]

# ★ 行顺序 & 显示名映射 -------------------------------------------------
METHOD_ORDER = [
    ("mode",         "Mode"),
    ("baran",   "Raha–Baran"),
    ("holoclean",    "HoloClean"),
    ("bigdansing",   "BigDansing"),
    ("boostclean",   "BoostClean"),
    ("horizon",      "Horizon"),
    ("scared",        "SCARE"),
    ("Unified",      "Unified"),
]

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------
def load_all() -> pd.DataFrame:
    """读取四个任务的 summary.xlsx 后纵向拼接"""
    dfs = []
    for t in TASKS:
        fp = os.path.join(DATA_DIR, f"{t}_summary.xlsx")
        if not os.path.isfile(fp):
            print(f"[WARN] {fp} not found → skip")
            continue
        df         = pd.read_excel(fp)
        df["task"] = t
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("❌  没有任何 summary.xlsx 被读到！")
    return pd.concat(dfs, ignore_index=True)

# -----------------------------------------------------------
def build_error_type(df: pd.DataFrame) -> pd.Series:
    """Missing / Typo 判定（示例逻辑，可按实际修改）"""
    return np.where(df["missing"] > 0, "Missing", "Typo")

# -----------------------------------------------------------
METRICS = ["EDR", "precision", "recall", "F1"]

def agg_one_task(df_task: pd.DataFrame) -> pd.DataFrame:
    """→ (cleaner × etype×metric) 的宽表"""
    df_task          = df_task.copy()
    df_task["etype"] = build_error_type(df_task)

    # 只保留所需列
    keep = ["cleaning_method", "etype", *METRICS]
    df_melt = df_task[keep]

    # 聚合：同一 cleaner 多次结果取平均
    df_grp = (df_melt
              .groupby(["cleaning_method", "etype"])
              .mean(numeric_only=True)
              .reset_index())

    # pivot → Missing_EDR ... Typo_F1
    tbl = (df_grp
           .pivot(index="cleaning_method",
                  columns="etype")[METRICS]
           .round(3)
           .reindex(columns=["Missing", "Typo"], level=1))

    tbl.columns = [f"{etype}_{metric}"
                   for metric in METRICS
                   for etype in ["Missing", "Typo"]]

    # 行排序 & 显示名
    display_rows = []
    for key, disp in METHOD_ORDER:
        if key in tbl.index:
            row = tbl.loc[key].copy()
            row.name = disp
            display_rows.append(row)
    tbl = pd.DataFrame(display_rows)

    # 缺失值 → '--'
    tbl = tbl.where(~tbl.isna(), other="--")
    return tbl.reset_index().rename(columns={"index": "Method"})

# -----------------------------------------------------------
def to_latex_subtable(tbl: pd.DataFrame,
                      task: str,
                      width: str = "0.492\\linewidth") -> str:
    """生成带 subtable 环境的 LaTeX"""
    # 1) 数据 → tabular 字符串（不含表头）
    body = tbl.to_latex(index=False,
                        header=False,
                        escape=False,
                        float_format="%.3f".__mod__)

    # 2) 手动拼表头（两行 + cmidrule）
    hdr1 = ("\\multirow{2}{*}{Method} &"
            "\\multicolumn{4}{c}{Missing} &"
            "\\multicolumn{4}{c}{Typo}\\\\")
    hdr2 = ("\\cmidrule(lr){2-5}\\cmidrule(l){6-9}\n"
            " & EDR & Prec. & Rec. & F1"
            " & EDR & Prec. & Rec. & F1\\\\")
    # 把 pandas 生成的 tabular 拆开：第一行是 \begin{tabular}{...}
    lines = body.strip().splitlines()
    begin_tabular = lines[0]
    tab_body      = "\n".join(lines[2:-1])  # 跳过 \toprule 与 \bottomrule

    latex = fr"""\begin{{subtable}}[t]{{{width}}}
\caption{{Dataset: \textbf{{{task}}}}}
\label{{tab:q1-acc-{task}}}
\centering
{begin_tabular}
\toprule
{hdr1}
{hdr2}
\midrule
{tab_body}
\bottomrule
\end{{tabular}}
\end{{subtable}}"""
    return latex

# ------------------------ 主流程 ----------------------------
all_df = load_all()

for task in TASKS:
    sub = all_df[all_df["task"] == task]
    if sub.empty:
        continue

    tbl = agg_one_task(sub)

    # --- 1) Excel ---
    out_xlsx = os.path.join(OUT_DIR, f"{task}_q1_metrics.xlsx")
    tbl.to_excel(out_xlsx, index=False)
    print(f"[SAVE] {out_xlsx}")

    # --- 2) LaTeX subtable ---
    tex_code = to_latex_subtable(tbl, task)
    tex_path = os.path.join(OUT_DIR, f"{task}_q1_metrics.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_code)
    print(f"[SAVE] {tex_path}")

print("✅ 6.4.1 tables & LaTeX subtables ready!")

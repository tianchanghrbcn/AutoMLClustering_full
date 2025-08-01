#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats_6.4.3.py  ·  v2
———————————————————————————————
ΔComb / ΔEDR 先按 0.1 区段聚合，再求差分 →
结果即 “每 ↑0.1 EDR 带来的 ΔComb”.
写出：
  • slopes_raw        每条差分 (已统一到 0.1 单位)
  • slopes_by_bin     7 个区段平均
  • summary           最佳区段、ΔComb@ΔEDR=0.2、整体均值
"""

from pathlib import Path
import re, sys, warnings, itertools
import numpy as np
import pandas as pd

# ────────────────── 配置 ────────────────── #
TASKS   = ["beers", "flights", "hospital", "rayyan"]
ROOT    = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "results" / "analysis_results"
OUT_XLS = ROOT / "task_progress" / "tables" / "best_marginal_zone.xlsx"
OUT_XLS.parent.mkdir(parents=True, exist_ok=True)

NUM_RE   = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
BINS     = np.round(np.arange(0.0, 0.8, 0.1), 1)        # 0.0-0.7

# ────────── 工具：文本 → float ────────── #
def to_float(s: pd.Series) -> pd.Series:
    return s.apply(lambda x: float(NUM_RE.search(str(x)).group())
                             if NUM_RE.search(str(x)) else np.nan)

# ────────── 1. 读取 & 清洗 ────────── #
need = ["error_rate", "EDR", "Combined Score"]
frames = []
for t in TASKS:
    fp = SRC_DIR / f"{t}_summary.xlsx"
    if not fp.is_file():
        warnings.warn(f"{fp} 缺失，跳过")
        continue
    df = pd.read_excel(fp, engine="openpyxl")
    if not set(need) <= set(df.columns):
        sys.exit(f"[ERR] {fp} 缺列：{set(need) - set(df.columns)}")
    df = df[need].copy()
    df[need] = df[need].apply(to_float)
    df["task"] = t
    frames.append(df)

if not frames:
    sys.exit("❌ 未读取到任何 *_summary.xlsx 文件")

df_all = pd.concat(frames, ignore_index=True).dropna()

# ────────── 2. 先把 EDR 按 0.1 区段取平均 ────────── #
df_all["EDR_bin"] = (df_all["EDR"] // 0.1 * 0.1).round(1)   # 0.00,0.10…

agg = (df_all.groupby(["task", "error_rate", "EDR_bin"], as_index=False)
              .agg(EDR=("EDR", "mean"),
                   Comb=("Combined Score", "mean")))

# 使每个 error_rate 内的 bin 排序齐全（缺失补 NaN）
full_index = pd.MultiIndex.from_product(
    [agg["task"].unique(),
     agg["error_rate"].unique(),
     BINS],
    names=["task", "error_rate", "EDR_bin"]
)
agg = (agg.set_index(["task", "error_rate", "EDR_bin"])
          .reindex(full_index)
          .reset_index())

# ────────── 3. 差分：ΔComb / 0.1 ────────── #
rows = []
for (task, er), g in agg.groupby(["task", "error_rate"]):
    g = g.sort_values("EDR_bin")
    for (b1, b2) in itertools.pairwise(BINS):
        c1, c2 = g.loc[g["EDR_bin"] == b1, "Comb"].values[0], \
                 g.loc[g["EDR_bin"] == b2, "Comb"].values[0]
        if np.isnan(c1) or np.isnan(c2):
            continue
        rows.append({
            "task": task,
            "error_rate": er,
            "EDR_mid": round(b1 + 0.05, 2),            # 0.05,0.15…
            "slope": (c2 - c1)        # ΔComb already per 0.1 EDR
        })

slopes = pd.DataFrame(rows)
if slopes.empty:
    sys.exit("❌ 计算不到斜率（有效点不足）")

slopes["bin"] = pd.cut(
    slopes["EDR_mid"],
    bins=BINS,
    labels=[f"{b:.1f}" for b in BINS[:-1]],
    include_lowest=True, right=False
)

mean_by_bin = (slopes.groupby("bin")["slope"]
                        .mean()
                        .rename("avg_slope")
                        .reset_index())

best_row = mean_by_bin.loc[mean_by_bin["avg_slope"].idxmax()]
best_left  = float(best_row["bin"])
best_slope = best_row["avg_slope"]
delta_comb_02 = best_slope * 2          # ΔEDR=0.2 ⇒ 两段

overall_mean = mean_by_bin["avg_slope"].mean()

# ────────── 4. 输出 Excel ────────── #
with pd.ExcelWriter(OUT_XLS, engine="openpyxl") as w:
    slopes.to_excel(w, sheet_name="slopes_raw", index=False)
    mean_by_bin.to_excel(w, sheet_name="mean_by_bin", index=False)
    pd.DataFrame({
        "best_zone_left": [best_left],
        "best_zone_right": [best_left + 0.1],
        "best_avg_slope": [best_slope],
        "ΔComb@ΔEDR=0.2": [delta_comb_02],
        "overall_mean_slope": [overall_mean]
    }).to_excel(w, sheet_name="summary", index=False)

# ────────── 5. 控制台预览 ────────── #
print("\n=== 平均边际斜率（ΔEDR=0.1） ===")
print(mean_by_bin.round(4).to_string(index=False))

print(f"\n最佳边际收益区：EDR ≈ {best_left:.1f} – {best_left+0.1:.1f}")
print(f"   区段平均斜率  ≈ {best_slope:.4f}")
print(f"   预期 ΔComb   @ ΔEDR = 0.2 → {delta_comb_02:.4f}")
print(f"\nOverall mean slope：{overall_mean:.4f}")
print(f"[OK] 结果已写入 {OUT_XLS}\n")

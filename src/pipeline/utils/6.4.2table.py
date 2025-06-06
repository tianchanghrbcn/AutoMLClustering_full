#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 6-4-2 过程级 Δ% 表格
· ρ_noise 保留
· 新增 B1:  Δn_avg   (平均邻居数)
· 新增 B3:  ΔW_cdf   (CDF–Wasserstein, log10 放大)
用法:
    python 6.4.2table.py  [RESULT_ROOT]
"""

import sys, json, glob, statistics, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

# --------------------------------------------------
# 1. 结果根目录
# --------------------------------------------------
if len(sys.argv) > 1:
    ROOT = Path(sys.argv[1]).expanduser().resolve()
else:
    ROOT = (Path(__file__).resolve()
            .parent.parent.parent.parent / "results" / "clustered_data")
if not ROOT.exists():
    sys.exit(f"[ERR] 结果目录不存在: {ROOT}")

# --------------------------------------------------
# 2. 常量
# --------------------------------------------------
CLEANINGS = ["baran", "holoclean", "bigdansing", "boostclean",
             "horizon", "scared", "unified"]            # mode 为基线
BASELINE  = "mode"

ALG_FAMILIES = {
    "centroid": ["KMEANS", "KMEANSNF", "KMEANSPPS", "GMM"],
    "density" : ["DBSCAN"],
    "hierarch": ["HC"]
}

def _geo_decay(js: dict):
    """首选 summary['avg_geo_decay']; 如无则尝试单值 geo_decay"""
    if js is None:
        return None
    for k in ("avg_geo_decay", "ll_geo_decay", "geo_decay"):
        if k in js and js[k] is not None:
            return js[k]
    return None

METRICS = {
    "centroid": {
        "GeoDecay": (_geo_decay, True),        # 越小越好
        "AUC_Δ":    (lambda js: js.get("avg_auc_delta") or js.get("auc_ll"), True),
        "ΔSSE/NLL": (lambda js: js.get("best_sse")      or js.get("best_nll"), True),
    },
    "density": {
        # ---------- 新指标 ----------
        "Δn_avg":   (lambda js:                                   # B1 ↑
                     sum(i * c for i, c in enumerate(js["neighbor_hist"]))
                     / max(sum(js["neighbor_hist"]), 1), False),
        "ΔW_cdf":   (lambda js: js["neighbor_hist"], True),       # B3 ↓
        # ---------------------------
        "Δn_core":  (lambda js: js["core_count"],  False),
        "Δρ_noise": (lambda js: js["noise_ratio"], True),
    },
    "hierarch": {
        "Δn_merge":        (lambda js: js["n_merge_steps"],               True),
        "Δh_max":          (lambda js: js.get("h_max") or js.get("max_dist"), True),
        "ΔR_intra/inter":  (lambda js: js["ratio_intra_inter"],           True),
    }
}

# --------------------------------------------------
# 3. 工具函数
# --------------------------------------------------
def first_json(pattern: str):
    files = glob.glob(pattern)
    return files[0] if files else None

def read_json(p: str):
    with open(p, encoding="utf-8") as fp:
        return json.load(fp)

def cdf_wasserstein(hist_c, hist_m):
    """CDF–Wasserstein 距离（未放大）"""
    m = max(len(hist_c), len(hist_m))
    h_c = np.pad(hist_c, (0, m - len(hist_c)))
    h_m = np.pad(hist_m, (0, m - len(hist_m)))
    cdf_c = np.cumsum(h_c) / (h_c.sum() + 1e-12)
    cdf_m = np.cumsum(h_m) / (h_m.sum() + 1e-12)
    return wasserstein_distance(range(m), range(m), cdf_c, cdf_m)

def collect_pair(alg: str, cleaning: str, cid: int):
    """返回 (json_clean, json_mode) 或 (None, None)"""
    def _base(a, c):
        p = ROOT / a / c / f"clustered_{cid}"
        return p if p.exists() else ROOT / a / c / f"cluster_{cid}"

    base_c = _base(alg, cleaning)
    base_m = _base(alg, BASELINE)

    pat = "*_core_stats.json" if alg == "DBSCAN" else "*_summary.json"
    p_c = first_json(str(base_c / pat))
    p_m = first_json(str(base_m / pat))
    if not p_c or not p_m:
        return None, None
    return read_json(p_c), read_json(p_m)

def pct_delta(v_c, v_m):
    try:
        return 100.0 * (v_c - v_m) / (abs(v_m) + 1e-8)
    except ZeroDivisionError:
        return None

# --------------------------------------------------
# 4. 主循环
# --------------------------------------------------
records = []      # (family, metric, cleaning, Δ%)

for family, algs in ALG_FAMILIES.items():
    for cleaning in CLEANINGS:
        buf = {m: [] for m in METRICS[family]}

        for alg in algs:
            for cid in range(60):     # cluster(-ed)_0 … 59
                j_c, j_m = collect_pair(alg, cleaning, cid)
                if j_c is None:
                    continue

                for mkey, (getter, _) in METRICS[family].items():
                    v_c, v_m = getter(j_c), getter(j_m)
                    if v_c is None or v_m is None:
                        continue

                    # ---------- 新分支: ΔW_cdf ----------
                    if mkey == "ΔW_cdf":
                        raw = cdf_wasserstein(v_c, v_m)
                        delta = math.log10(1.0 + raw) * 1000.0
                        buf[mkey].append(delta)
                        continue

                    # 其它指标（含 Δn_avg 等）
                    try:
                        if np.isnan(v_c) or np.isnan(v_m):
                            continue
                    except TypeError:
                        pass

                    d = pct_delta(v_c, v_m)
                    if d is not None and not np.isnan(d):
                        buf[mkey].append(d)

        # 汇总中位数
        for mkey, lst in buf.items():
            if lst:
                records.append((family, mkey, cleaning,
                                round(float(statistics.median(lst)), 1)))

# --------------------------------------------------
# 5. 输出 CSV / LaTeX
# --------------------------------------------------
df = (pd.DataFrame(records, columns=["family", "metric", "cleaning", "Δ%"])
        .pivot_table(index=["family", "metric"],
                     columns="cleaning",
                     values="Δ%",
                     aggfunc="first")
        .sort_index())

OUT_DIR = Path(r"D:\algorithm paper\AutoMLClustering\task_progress\tables\6.4.2tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(OUT_DIR / "table6_process_delta.csv")
with open(OUT_DIR / "table6_process_delta.tex", "w", encoding="utf-8") as fh:
    fh.write(df.to_latex(float_format="%.1f",
                         column_format="@{}ll" + "r"*len(CLEANINGS) + "@{}",
                         na_rep="--"))

print("✔ 结果已写出到:", OUT_DIR / "table6_process_delta.csv")
print(df)

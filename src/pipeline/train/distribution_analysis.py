#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import numpy as np
import pandas as pd
from scipy import stats

# IsolationForest
try:
    from sklearn.ensemble import IsolationForest
    has_isolation_forest = True
except ImportError:
    has_isolation_forest = False

# POT for multi-dim Wasserstein
try:
    import ot
    has_pot = True
except ImportError:
    has_pot = False


def load_csv_drop_id(path):
    """读取 CSV, 去除 'ID' 列, 若无文件或异常返回 None."""
    if not os.path.isfile(path):
        print(f"[WARN] CSV not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        df.drop(columns=["ID"], errors="ignore", inplace=True)
        return df
    except Exception as e:
        print(f"[WARN] fail read {path}: {e}")
        return None


def parse_details(details_str):
    """解析 e.g. "anomaly=2.50%, missing=0.00%, format=2.50%, knowledge=0.00%".
       返回: {"anomaly":2.5, "missing":0.0, "format":2.5, "knowledge":0.0}
    """
    result = {"anomaly": 0.0, "missing": 0.0, "format": 0.0, "knowledge": 0.0}
    if not details_str:
        return result
    pattern = r'(\w+)\s*=\s*([\d\.]+)\%'
    matches = re.findall(pattern, details_str)
    for (k, v) in matches:
        if k in result:
            result[k] = float(v)
    return result


def numeric_cols(df):
    """返回df中数值列列表."""
    if df is None or df.empty:
        return []
    return df.select_dtypes(include=[np.number]).columns.tolist()


def basic_stats(df):
    """基础统计: mean, var, missing_rate."""
    out = {}
    if df is None or df.empty:
        return out
    num_cols = numeric_cols(df)
    if not num_cols:
        return out

    mean_ = df[num_cols].mean().to_dict()
    var_ = df[num_cols].var().to_dict()

    missing_count = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    mrate = float(missing_count / total_cells) if total_cells > 0 else 0

    out["mean"] = mean_
    out["var"] = var_
    out["missing_rate"] = mrate
    return out


def detect_outliers_iforest(df, sample_size=1000):
    """IsolationForest, 返回离群点比例. 若没安装sklearn返回None."""
    if not has_isolation_forest:
        return None
    if df is None or df.empty:
        return None
    num_cols = numeric_cols(df)
    if not num_cols:
        return None

    if df.shape[0] > sample_size:
        df_samp = df[num_cols].sample(sample_size, random_state=42).dropna()
    else:
        df_samp = df[num_cols].dropna()
    if df_samp.empty:
        return None

    iso = IsolationForest(random_state=42)
    iso.fit(df_samp)
    preds = iso.predict(df_samp)
    out_count = (preds == -1).sum()
    ratio = float(out_count / len(preds))
    return ratio


def multi_dim_wasserstein(df1, df2):
    """多维Wasserstein, using pot. if not installed => None."""
    if not has_pot:
        return None
    if df1 is None or df2 is None:
        return None
    cols1 = numeric_cols(df1)
    cols2 = numeric_cols(df2)
    shared = list(set(cols1) & set(cols2))
    if not shared:
        return None

    X1 = df1[shared].dropna()
    X2 = df2[shared].dropna()
    if X1.empty or X2.empty:
        return None

    import ot
    a = ot.unif(len(X1))
    b = ot.unif(len(X2))
    M = ot.dist(X1.values, X2.values, metric='euclidean')
    cost = ot.emd2(a, b, M)
    wd = cost ** 0.5
    return wd


def single_col_metrics(dcol, rcol, bins=10):
    """
    对单数值列: mean, var(对rcol), normality p(对rcol), kl(dcol||rcol).
    返回 {"mean":..., "var":..., "normal_p":..., "kl":...}.
    """
    out = {"mean": np.nan, "var": np.nan, "normal_p": np.nan, "kl": np.nan}
    if rcol is None or rcol.dropna().empty:
        return out

    sr = rcol.dropna()
    out["mean"] = sr.mean()
    out["var"] = sr.var()

    if len(sr) > 3:
        # Shapiro test p-value
        sample_r = sr.sample(min(len(sr), 5000), random_state=42)
        _, pval = stats.shapiro(sample_r)
        out["normal_p"] = pval

    # 计算KL(dirty||rep)
    if dcol is not None and not dcol.dropna().empty:
        sd = dcol.dropna()
        minv = min(sd.min(), sr.min())
        maxv = max(sd.max(), sr.max())
        if minv < maxv:
            be = np.linspace(minv, maxv, bins + 1)
            hd, _ = np.histogram(sd, bins=be)
            hr, _ = np.histogram(sr, bins=be)
            if hd.sum() > 0 and hr.sum() > 0:
                pd_ = hd / hd.sum()
                pr_ = hr / hr.sum()
                kl_ = 0.0
                for i in range(len(pd_)):
                    if pd_[i] > 0 and pr_[i] > 0:
                        kl_ += pd_[i] * np.log(pd_[i] / pr_[i])
                out["kl"] = kl_

    return out


def average_corr(df):
    """平均绝对相关 & 最大绝对相关(对数值列)."""
    if df is None or df.empty:
        return (np.nan, np.nan)
    cols = numeric_cols(df)
    if len(cols) < 2:
        return (np.nan, np.nan)
    cmat = df[cols].corr().abs()
    mask = np.triu(np.ones_like(cmat, dtype=bool))
    vals = cmat.where(~mask).stack()
    if vals.empty:
        return (np.nan, np.nan)
    return (vals.mean(), vals.max())


def cat_cols(df):
    """object/category columns."""
    if df is None or df.empty:
        return []
    obj_cols = df.select_dtypes(include=["object","category"]).columns
    return obj_cols.tolist()


def cat_kl(dser, rser):
    """
    简单的类别KL(d||r). 统计各值频次 => p(x), q(x).
    返回单列 KL值(可能为 nan).
    """
    if dser is None or dser.dropna().empty or rser is None or rser.dropna().empty:
        return np.nan
    f_d = dser.value_counts(dropna=False)
    f_r = rser.value_counts(dropna=False)
    allkeys = set(f_d.index) | set(f_r.index)
    sd_ = float(f_d.sum())
    sr_ = float(f_r.sum())
    kl_ = 0.0
    for val in allkeys:
        pd_ = f_d.get(val, 0) / sd_
        pr_ = f_r.get(val, 0) / sr_
        if pd_ > 0 and pr_ > 0:
            kl_ += pd_ * np.log(pd_ / pr_)
    return kl_


def cat_analysis(dirty_df, rep_df):
    """
    修改后：只生成一个 cat_kl_avg (类别列KL的平均),
    而不再生成 cat_kl_col1..cat_kl_colN.
    """
    out = {}
    if rep_df is None or rep_df.empty:
        return out
    rcols = cat_cols(rep_df)
    dcols = cat_cols(dirty_df) if (dirty_df is not None and not dirty_df.empty) else []
    shared = sorted(list(set(rcols) & set(dcols)))

    kl_values = []
    for col in shared:
        klv = cat_kl(dirty_df[col], rep_df[col])
        if not np.isnan(klv):
            kl_values.append(klv)

    if kl_values:
        out["cat_kl_avg"] = float(np.mean(kl_values))
    else:
        out["cat_kl_avg"] = np.nan
    return out


def compute_all_metrics(dirty_df, rep_df):
    """
    同时对数值列(多列) & 非数值列(多列)做:
      - numeric: mean,var,normal,多列KL => 只算平均 (kl_numeric_avg), multiWasserstein, iforest outlier, corr
      - categorical: cat_kl_avg
    """
    res = {}
    if rep_df is None or rep_df.empty:
        return res

    # 1. multi维Wasserstein
    if has_pot and dirty_df is not None:
        wd = multi_dim_wasserstein(dirty_df, rep_df)
        res["multi_wasserstein"] = wd
    else:
        res["multi_wasserstein"] = np.nan

    # 2. IsolationForest => outlier
    if has_isolation_forest:
        out_r = detect_outliers_iforest(rep_df, sample_size=1000)
        res["outlier_ratio_rep"] = out_r
    else:
        res["outlier_ratio_rep"] = np.nan

    # 3. 数值列 => mean_colX, var_colX, normal_p_colX 保留,
    #    但 K-L =>只留1个平均 => kl_numeric_avg
    num_cols_rep = numeric_cols(rep_df)
    num_cols_dir = numeric_cols(dirty_df) if dirty_df is not None else []
    shared_num = sorted(list(set(num_cols_rep)&set(num_cols_dir)))

    # 用于收集所有列的 kl
    kl_list = []

    idx = 1
    for col in shared_num:
        mets = single_col_metrics(dirty_df[col], rep_df[col], bins=10)

        # 保留mean/var/normal_p按列
        res[f"mean_col{idx}"] = mets["mean"]
        res[f"var_col{idx}"]  = mets["var"]
        res[f"normal_p_col{idx}"] = mets["normal_p"]

        # 收集 kl
        if not np.isnan(mets["kl"]):
            kl_list.append(mets["kl"])

        idx+=1

    # 数值列 kl 平均
    if kl_list:
        res["kl_numeric_avg"] = float(np.mean(kl_list))
    else:
        res["kl_numeric_avg"] = np.nan

    # 4. 列间相关 => rep
    cavg, cmax = average_corr(rep_df)
    res["corr_avg"] = cavg
    res["corr_max"] = cmax

    # 5. 类别列 => cat_kl_avg
    cat_res = cat_analysis(dirty_df, rep_df)
    # 这里cat_res只有一个字段 cat_kl_avg
    res.update(cat_res)

    return res

def main():
    comparison_file = "comparison.json"
    out_dir = "../../../results/cleaned_data/cleaned_analysis"
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(comparison_file):
        print(f"[ERROR] {comparison_file} not found.")
        return

    with open(comparison_file, 'r', encoding='utf-8') as f:
        comp_list = json.load(f)

    task_map = {}

    for entry in comp_list:
        task_name   = entry.get("task_name","")
        num         = entry.get("num","")
        dataset_id  = entry.get("dataset_id","")
        error_rate  = entry.get("error_rate",None)
        m_          = entry.get("m",None)
        n_          = entry.get("n",None)

        d_dict  = parse_details(entry.get("details",""))
        anomaly_val   = d_dict["anomaly"]
        missing_val   = d_dict["missing"]
        format_val    = d_dict["format"]
        knowledge_val = d_dict["knowledge"]

        paths    = entry.get("paths", {})
        dirty_csv= paths.get("dirty_csv","")
        df_dirty = load_csv_drop_id(dirty_csv)

        rep_map  = paths.get("repaired_paths",{})
        for method, rep_path in rep_map.items():
            df_rep = load_csv_drop_id(rep_path)
            mets = compute_all_metrics(df_dirty, df_rep)

            row = {
                "task_name":task_name,
                "num":num,
                "dataset_id":dataset_id,
                "error_rate":error_rate,
                "m":m_,
                "n":n_,
                "anomaly":anomaly_val,
                "missing":missing_val,
                "format":format_val,
                "knowledge":knowledge_val,
                "cleaning_method":method
            }
            row.update(mets)

            if task_name not in task_map:
                task_map[task_name] = []
            task_map[task_name].append(row)

    # 每个 task_name 输出 csv
    for tname, rows in task_map.items():
        df_out = pd.DataFrame(rows)
        out_csv = os.path.join(out_dir, f"{tname}_analysis.csv")
        df_out.to_csv(out_csv, index=False, encoding='utf-8')
        print(f"[INFO] Wrote => {out_csv}, rows={len(rows)}")

    print("[INFO] All done. Check CSV in", out_dir)


if __name__=="__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBMRegressor · Optuna δ-min (single-train) · GPU auto · timing log
"""

# ---------- imports ----------
import re, pickle, warnings, json, time, optuna, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMRegressor
from joblib import dump

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted",
)

# ---------- 0. files ----------
FILES = [
    "../../../results/analysis_results/beers_summary.xlsx",
    "../../../results/analysis_results/flights_summary.xlsx",
    "../../../results/analysis_results/hospital_summary.xlsx",
    "../../../results/analysis_results/rayyan_summary.xlsx",
]

# ---------- 1. columns ----------
MISS_COL, OUT_COL   = "missing", "anomaly"
CLN_METHOD_COL      = "cleaning_method"
CLUSTER_TYPE_COL    = "cluster_method"
PARAM_COL           = "parameters"
M_COL, D_COL        = "m", "n"
TARGET_COL          = "Combined Score"

# ---------- 2. load ----------
df = pd.concat([pd.read_excel(p) for p in FILES], ignore_index=True)
df = df[np.isfinite(df[TARGET_COL])].reset_index(drop=True)
print(f"[INFO] Loaded {len(df):,} samples.")

# ---------- 3. label 0-1 ----------
f_min, f_max = df[TARGET_COL].min(), df[TARGET_COL].max()
df["f_norm"] = (df[TARGET_COL] - f_min) / (f_max - f_min)

# ---------- 4. one-hot ----------
clean_ops = sorted(df[CLN_METHOD_COL].dropna().unique())
for op in clean_ops:
    df[op] = (df[CLN_METHOD_COL] == op).astype(int)

# ---------- 5. parse params ----------
def parse_param(s):
    kv = dict(re.findall(r"(\w+)=([0-9.]+)", str(s)))
    return float(kv.get("k", 0)), float(kv.get("eps", 0)), float(kv.get("minPts", 0))
df[["k", "eps", "minPts"]] = df[PARAM_COL].apply(lambda x: pd.Series(parse_param(x)))

# ---------- 6. feature engineering ----------
df["log_m"] = np.log10(df[M_COL] + 1)
df["log_d"] = np.log10(df[D_COL] + 1)
for a in ["kmeans", "dbscan", "hierarchical"]:
    m = (df[CLUSTER_TYPE_COL] == a).astype(int)
    df[f"miss_{a}"] = df[MISS_COL] * m
    df[f"out_{a}"]  = df[OUT_COL] * m

num_cols = (
    [MISS_COL, OUT_COL, "log_m", "log_d", "k", "eps", "minPts"] +
    [f"{p}_{a}" for p in ["miss", "out"] for a in ["kmeans", "dbscan", "hierarchical"]]
)
cat_cols = clean_ops + [CLUSTER_TYPE_COL]

preproc = ColumnTransformer(
    [("num", MinMaxScaler(), num_cols),
     ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
    remainder="drop"
)

X = df[num_cols + cat_cols]
y = df["f_norm"].astype(float)

# ---------- 7. stratified 10-fold ----------
bins = pd.qcut(y, 10, duplicates="drop", labels=False)
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2025)

# ---------- 8. GPU auto-detect ----------
def has_nvidia_gpu() -> bool:
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetCount() > 0
    except Exception:
        return False

USE_GPU = has_nvidia_gpu()
print("[INFO] GPU detected – using GPU" if USE_GPU else "[INFO] No GPU – fallback CPU")

gpu_params = dict(
    device_type="gpu", gpu_platform_id=0, gpu_device_id=0,
    max_bin=63, force_row_wise=False
) if USE_GPU else dict(
    device_type="cpu", max_bin=255, force_row_wise=True
)

# ---------- 9. Optuna search (single-train) ----------
best_delta, best_models = np.inf, None     # 全局容器
begin_optuna = time.time()

def objective(trial):
    global best_delta, best_models

    params = {
        "objective": "regression_l1",
        "verbosity": -1,
        "deterministic": True,
        **gpu_params,
        # search space
        "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 2000, 12000, step=1000),
        "num_leaves": trial.suggest_int("num_leaves", 63, 1023, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 2, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
    }

    oof = np.zeros(len(X))
    fold_models = []

    for fold, (tr, val) in enumerate(kf.split(X, bins)):
        model = Pipeline([
            ("prep", preproc),
            ("lgb", LGBMRegressor(**params, random_state=2025 + fold))
        ])
        model.fit(X.iloc[tr], y.iloc[tr])
        oof[val] = model.predict(X.iloc[val])
        fold_models.append(model)

    delta = np.percentile(np.abs(oof - y), 95)

    # 记录最优模型
    if delta < best_delta:
        best_delta = delta
        best_models = fold_models  # shallow copy ok

    return delta

print("[INFO] ⏳  Optuna search running …")
study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=2025))
study.optimize(objective, n_trials=100, show_progress_bar=True)
optuna_time = time.time() - begin_optuna
print(f"[TIME] Optuna search finished in {optuna_time/60:.2f} min")

print(f"[INFO] Best δ = {best_delta:.4f}")
print(json.dumps(study.best_params, indent=2, ensure_ascii=False))

# ---------- 10. 保存最佳折模型 ----------
save_start = time.time()
Path("models").mkdir(exist_ok=True)
for i, mdl in enumerate(best_models, 1):
    dump(mdl, f"models/reg_fold{i}.joblib")
save_time = time.time() - save_start
print(f"[TIME] Saved 10 models in {save_time:.1f} s")

# ---------- 11. artefacts & Predictor ----------
meta = {"delta": float(best_delta), "f_min": float(f_min), "f_max": float(f_max)}
dump(meta, "models/regressor.pkl")
dump(preproc, "models/scaler.pkl")

class Predictor:
    """10-fold soft-voting回归器 + δ-UCB"""
    def __init__(self, m):
        self.delta, self.f_min, self.f_max = m["delta"], m["f_min"], m["f_max"]
        self.children = [pickle.load(open(f"models/reg_fold{i}.joblib", "rb"))
                         for i in range(1, 11)]
    def _denorm(self, x): return x * (self.f_max - self.f_min) + self.f_min
    def _avg(self, X):    return np.mean([m.predict(X) for m in self.children], axis=0)
    def predict(self, df): return self._denorm(self._avg(df))
    def ucb(self, df):    return self._denorm(np.clip(self._avg(df) + self.delta, 0, 1))

with open("models/predictor.pkl", "wb") as fp:
    pickle.dump(Predictor(meta), fp)

total_time = time.time() - begin_optuna
print(f"[INFO] ✅ Training complete – artefacts in ./models/ (total {total_time/60:.2f} min)")

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMRegressor

from src.paths import PROCESSED_DIR, MODELS_DIR, ensure_dirs
from src.features import rmsle

def _fit_label_map(series: pd.Series) -> dict:
    # Stable, sorted mapping of stringified values -> ints
    vals = sorted(series.astype(str).unique())
    return {v: i for i, v in enumerate(vals)}

def _apply_label_map(series: pd.Series, mapping: dict) -> pd.Series:
    return series.astype(str).map(mapping).fillna(-1).astype("int32")

def main():
    ensure_dirs()
    path = PROCESSED_DIR / "train_proc.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python -m scripts.preprocess")

    df = pd.read_parquet(path)
    print("[OK] Loaded train_proc:", df.shape)

    # Time-based split: last 56 days as validation
    df = df.sort_values("date")
    cutoff = df["date"].max() - pd.Timedelta(days=56)
    trn = df[df["date"] <= cutoff].copy()
    val = df[df["date"] > cutoff].copy()
    print(f"[Info] Split cutoff={cutoff.date()}  Train={trn.shape}  Val={val.shape}")

    # Categorical columns present in this dataset
    cat_cols = [c for c in ["store_nbr","family","city","state","type","cluster"] if c in df.columns]
    mappings = {}

    for col in cat_cols:
        mapping = _fit_label_map(trn[col])
        mappings[col] = mapping
        trn[col] = _apply_label_map(trn[col], mapping)
        val[col] = _apply_label_map(val[col], mapping)

    # Feature selection
    drop_cols = {"id", "date", "sales", "description", "locale_name"} & set(df.columns)
    feature_cols = [c for c in df.columns if c not in drop_cols and c != "sales"]

    X_trn = trn[feature_cols]
    y_trn = trn["sales"].values
    X_val = val[feature_cols]
    y_val = val["sales"].values

    print(f"[Info] X_train={X_trn.shape}, X_val={X_val.shape}, features={len(feature_cols)}")

    model = LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=-1,
        random_state=42
    )

    model.fit(
        X_trn, y_trn,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse"
    )

    val_pred = np.clip(model.predict(X_val), 0, None)
    score = rmsle(y_val, val_pred)
    print(f"[OK] Validation RMSLE: {score:.6f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "features": feature_cols,
        "cat_cols": cat_cols,
        "mappings": mappings,
        "cutoff": cutoff
    }
    out_path = MODELS_DIR / "lgbm_baseline.pkl"
    joblib.dump(bundle, out_path)
    print("[OK] Saved model ->", out_path)

if __name__ == "__main__":
    main()

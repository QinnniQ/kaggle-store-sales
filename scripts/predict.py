from __future__ import annotations
import pandas as pd
import numpy as np
import joblib

from src.paths import PROCESSED_DIR, MODELS_DIR, SUBMISSIONS_DIR, ensure_dirs

def _apply_label_map(series: pd.Series, mapping: dict) -> pd.Series:
    # Use saved mapping from training; unseen values -> -1
    return series.astype(str).map(mapping).fillna(-1).astype("int32")

def main():
    ensure_dirs()
    model_path = MODELS_DIR / "lgbm_baseline.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}. Run training first (python -m scripts.train).")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["features"]
    cat_cols = bundle["cat_cols"]
    mappings = bundle["mappings"]

    test_path = PROCESSED_DIR / "test_proc.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}. Run preprocessing first (python -m scripts.preprocess).")

    test_df = pd.read_parquet(test_path).copy()

    # Apply saved categorical mappings consistently
    for col in cat_cols:
        if col in test_df.columns:
            test_df[col] = _apply_label_map(test_df[col], mappings[col])

    # Ensure all needed features exist
    for c in feature_cols:
        if c not in test_df.columns:
            test_df[c] = 0

    X_test = test_df[feature_cols]
    preds = np.clip(model.predict(X_test), 0, None)

    # Build submission using sample file for correct id order
    sample = pd.read_csv(PROCESSED_DIR.parent / "raw" / "sample_submission.csv")
    out = sample.copy()
    out["sales"] = preds

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBMISSIONS_DIR / "submission_lgbm_baseline.csv"
    out.to_csv(path, index=False)
    print("[OK] Wrote submission ->", path)

if __name__ == "__main__":
    main()

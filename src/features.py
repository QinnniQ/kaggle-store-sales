from __future__ import annotations
import pandas as pd
import numpy as np

def add_date_parts(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["day"] = out[date_col].dt.day
    out["dow"] = out[date_col].dt.dayofweek
    out["week"] = out[date_col].dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    return out

def join_oil(df: pd.DataFrame, oil: pd.DataFrame) -> pd.DataFrame:
    oil2 = oil.copy()
    oil2["date"] = pd.to_datetime(oil2["date"])
    oil2 = oil2.sort_values("date").ffill()
    return df.merge(oil2, on="date", how="left")

def _holiday_flag_table(holidays: pd.DataFrame) -> pd.DataFrame:
    h = holidays.copy()
    h["date"] = pd.to_datetime(h["date"])
    h["is_holiday"] = 1
    # Optional: mark Work Day as not holiday
    if "type" in h.columns:
        h.loc[h["type"].str.lower().eq("work day"), "is_holiday"] = 0
    h = h.groupby("date", as_index=False)["is_holiday"].max()
    return h

def join_holidays(df: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    h = _holiday_flag_table(holidays)
    out = df.merge(h, on="date", how="left")
    out["is_holiday"] = out["is_holiday"].fillna(0).astype(int)
    return out

def join_stores(df: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    return df.merge(stores, on="store_nbr", how="left")

def join_transactions(df: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    t = transactions.copy()
    t["date"] = pd.to_datetime(t["date"])
    return df.merge(t, on=["date", "store_nbr"], how="left")

def add_lags_rolls(df: pd.DataFrame, group_cols=("store_nbr","family"), target="sales") -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["store_nbr","family","date"])
    g = out.groupby(list(group_cols), group_keys=False)[target]
    for lag in [1, 7, 14, 28]:
        out[f"lag_{lag}"] = g.shift(lag)
    for win in [7, 14, 28]:
        out[f"rmean_{win}"] = g.shift(1).rolling(win).mean()
    feat_cols = [c for c in out.columns if c.startswith("lag_") or c.startswith("rmean_")]
    out[feat_cols] = out[feat_cols].fillna(0.0)
    return out

def safe_clip_predictions(preds: np.ndarray) -> np.ndarray:
    return np.clip(preds, 0, None)

def rmsle(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    return float(np.sqrt(np.mean((np.log1p(y_pred + eps) - np.log1p(y_true + eps))**2)))

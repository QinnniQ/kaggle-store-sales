from src.paths import RAW_DIR, PROCESSED_DIR, ensure_dirs
from src.features import add_date_parts, join_oil, join_holidays, join_stores, join_transactions, add_lags_rolls
import pandas as pd

def load_csv(name: str) -> pd.DataFrame:
    path = RAW_DIR / name
    df = pd.read_csv(path)
    print(f"[OK] Loaded {name} -> {df.shape}")
    return df

def main():
    ensure_dirs()
    train = load_csv("train.csv")
    test  = load_csv("test.csv")
    oil   = load_csv("oil.csv")
    hol   = load_csv("holidays_events.csv")
    stores= load_csv("stores.csv")
    trans = load_csv("transactions.csv")

    # Ensure datetime
    for df in [train, test]:
        df["date"] = pd.to_datetime(df["date"])

    # Basic date parts
    train_fe = add_date_parts(train)
    test_fe  = add_date_parts(test)

    # onpromotion to int (exists in both train/test)
    for df in [train_fe, test_fe]:
        df["onpromotion"] = df.get("onpromotion", 0).fillna(0).astype(int)

    # External joins
    train_fe = join_oil(train_fe, oil)
    test_fe  = join_oil(test_fe, oil)

    train_fe = join_holidays(train_fe, hol)
    test_fe  = join_holidays(test_fe, hol)

    train_fe = join_stores(train_fe, stores)
    test_fe  = join_stores(test_fe, stores)

    train_fe = join_transactions(train_fe, trans)
    test_fe  = join_transactions(test_fe, trans)

    # Lags & rolling means on train
    train_fe = add_lags_rolls(train_fe, group_cols=("store_nbr","family"), target="sales")

    # Save processed
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_fe.to_parquet(PROCESSED_DIR / "train_proc.parquet", index=False)
    test_fe.to_parquet(PROCESSED_DIR / "test_proc.parquet", index=False)
    print("[OK] Saved processed parquet files in:", PROCESSED_DIR)

if __name__ == "__main__":
    main()

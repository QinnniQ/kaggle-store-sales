from src.paths import RAW_DIR, ensure_dirs
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile, os

COMP = "store-sales-time-series-forecasting"

def main():
    ensure_dirs()
    api = KaggleApi()
    api.authenticate()
    print("[OK] Kaggle authenticated")

    print("[...] Downloading competition files...")
    api.competition_download_files(COMP, path=str(RAW_DIR), quiet=False)

    zips = [p for p in os.listdir(RAW_DIR) if p.endswith(".zip")]
    for z in zips:
        zpath = RAW_DIR / z
        print(f"[...] Extracting {zpath}")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(RAW_DIR)
        os.remove(zpath)
    print("[OK] Raw CSVs ready in:", RAW_DIR)

if __name__ == "__main__":
    main()

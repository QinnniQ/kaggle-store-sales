from src.paths import ensure_dirs, DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, SUBMISSIONS_DIR

if __name__ == "__main__":
    ensure_dirs()
    print("[OK] Created folders:")
    for p in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, SUBMISSIONS_DIR]:
        print(" -", p)

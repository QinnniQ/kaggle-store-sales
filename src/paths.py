from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"

def ensure_dirs():
    for p in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, SUBMISSIONS_DIR]:
        p.mkdir(parents=True, exist_ok=True)

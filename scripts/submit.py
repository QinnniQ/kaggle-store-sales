from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path

COMPETITION = "store-sales-time-series-forecasting"
SUB_FILE = Path("outputs/submissions/submission_lgbm_baseline.csv")
MESSAGE = "LGBM baseline v1"

def main():
    if not SUB_FILE.exists():
        raise FileNotFoundError(f"Submission file not found: {SUB_FILE}")
    api = KaggleApi()
    api.authenticate()
    print("[OK] Kaggle authenticated")
    print(f"[...] Submitting {SUB_FILE} to {COMPETITION} with message: {MESSAGE}")
    api.competition_submit(file_name=str(SUB_FILE), competition=COMPETITION, message=MESSAGE)
    print("[OK] Submitted. Check your Kaggle Submissions tab for the score.")

if __name__ == "__main__":
    main()

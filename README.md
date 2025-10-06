# 🧮 Store Sales – Time Series Forecasting (Kaggle)

This repository contains a fully scripted **end-to-end time series forecasting pipeline** for the [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition by Corporación Favorita (Ecuador).  
It was built entirely in **VS Code** with a **Python 3.11 virtual environment**, following clean engineering practices and reproducible automation.

---

## 🚀 Project Overview

**Goal:**  
Predict daily unit sales for thousands of items sold across multiple stores using historical data and external regressors such as oil prices, holidays, and transactions.

**Highlights:**
- Modular, script-driven architecture (no notebooks)
- Automated Kaggle data download via API
- Feature engineering with lag/rolling windows, holidays, oil prices, and promotions
- LightGBM baseline with time-based validation
- RMSLE metric (aligned with Kaggle leaderboard)
- Command-line submission to Kaggle via API

---

## 🧠 Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| Core Language | Python 3.11 |
| Environment | `venv` (no Conda) |
| Modeling | LightGBM |
| Data | pandas, numpy, pyarrow |
| Evaluation | RMSLE |
| Automation | Kaggle API, joblib |
| IDE | Visual Studio Code |

---

## 🏗️ Project Structure

```
kaggle-store-sales/
│
├── data/
│   ├── raw/           ← original Kaggle CSVs (ignored)
│   └── processed/     ← cleaned / feature-engineered data (ignored)
│
├── scripts/
│   ├── make_folders.py
│   ├── download_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── submit.py
│
├── src/
│   ├── features.py    ← feature engineering logic
│   └── paths.py       ← path utilities
│
├── outputs/
│   └── submissions/   ← final Kaggle-ready CSVs (ignored)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Reproducible Workflow

Run each step from the project root (`C:\Users\<you>\Documents\Kaggle\kaggle-store-sales`):

```bash
# 1. Create all folder paths
python -m scripts.make_folders

# 2. Download Kaggle data
python -m scripts.download_data

# 3. Preprocess & feature engineer
python -m scripts.preprocess

# 4. Train LightGBM baseline
python -m scripts.train

# 5. Generate predictions
python -m scripts.predict

# 6. Submit to Kaggle
python -m scripts.submit
```

---

## 📊 Results (Baseline)

| Metric | Value | Notes |
|---------|-------|-------|
| Public RMSLE | **2.35768** | LightGBM baseline with minimal feature set |
| Private LB | TBD | improving via richer lag/rolling features |

Planned improvements:
- Extended lag & rolling statistics (56d, 91d)
- Holiday & promo interaction terms
- Native LightGBM categorical handling
- Time-series cross-validation
- Parameter tuning via Optuna

---

## 👨‍💻 About the Author

**Nicholai Gay** – *AI Engineer (IronHack Graduate)*  
Focused on building end-to-end machine learning pipelines, with experience in NLP, LangChain RAG agents, and MLOps workflows.

📍 Amsterdam, NL  
🔗 [LinkedIn](https://www.linkedin.com/in/nicholai-gay-201905148/) | [GitHub](https://github.com/QinnniQ) | [Kaggle Profile](https://www.kaggle.com/nicholaigay)

---

> 💡 *This project demonstrates applied MLOps thinking, reproducibility, and hands-on problem solving — key skills for real-world AI engineering.*

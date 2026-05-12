# fraud-shield-ml

Professional ML engineering project for detecting fraudulent credit card transactions.

## What’s included

- Clean sklearn baseline pipeline (preprocess + logistic regression)
- Train + evaluate modules
- FastAPI inference API (`/health`, `/model-info`, `/predict`)
- Streamlit UI for interactive predictions
- Pytest tests (preprocess + API contract)
- Dockerfile + docker-compose for containerized runs

## Project structure

```text
fraud-shield-ml/
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── app/
│   └── streamlit_app.py
├── notebooks/
│   └── eda.ipynb
├── tests/
│   ├── test_preprocess.py
│   └── test_api.py
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Configuration

All paths and tunables are driven by `config.py` via environment variables (see `config.get_settings()`).

Common environment variables:

- `FRAUD_SHIELD_RAW_DATA_PATH` (default: `data/creditcard.csv`)
- `FRAUD_SHIELD_MODEL_ARTIFACT_PATH` (default: `models/fraud_model.pkl`)
- `FRAUD_SHIELD_METADATA_PATH` (default: `models/metadata.json`)
- `FRAUD_SHIELD_METRICS_PATH` (default: `models/metrics.json`)
- `FRAUD_MODEL_VERSION` (optional; default `1.0.0` written into `metadata.json` at train time)
- `FRAUD_SHIELD_TARGET_COL` (default: `Class`)
- `FRAUD_SHIELD_TEST_SIZE` (default: `0.2`)
- `FRAUD_SHIELD_RANDOM_STATE` (default: `42`)
- `FRAUD_SHIELD_MAX_ITER` (default: `1000`)

## Setup (local)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

Make sure your CSV exists (by default `data/creditcard.csv`) and that it contains `target_col` (default: `Class`).

```bash
python src/train.py
```

Equivalent one-liner (from repo root, with `PYTHONPATH` set if needed):

```bash
python -c "from config import get_settings; from src.train import train_model; train_model(get_settings())"
```

`train_model` runs preprocessing, XGBoost training + tuning, **evaluation** (metrics JSON + plots), and saves `models/fraud_model.pkl`.

Standalone evaluation on the saved artifact:

```bash
python src/evaluate.py
```

## Run API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

`GET /model-info` reads `models/metadata.json` (written when you train).

`POST /predict` expects a **TransactionInput** JSON object: **`V1`–`V28`**, **`Amount`**, **`Time`** (all floats). Response: **`fraud_probability`**, **`is_fraud`**, **`risk_level`** (`LOW` / `MEDIUM` / `HIGH`), **`model_version`**.

## Run Streamlit UI

Start the API first (`uvicorn …`), then the dashboard (it calls `http://localhost:8000/predict` by default). Override with **`FRAUD_SHIELD_API_BASE`**.

```bash
streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
```

## Tests

```bash
pytest -q
```

## Docker

```bash
docker-compose up --build
```

API: `http://localhost:8000/health`, `http://localhost:8000/model-info`, `http://localhost:8000/predict`  
Streamlit: `http://localhost:8501`

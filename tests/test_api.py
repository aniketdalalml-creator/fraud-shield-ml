from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import api.main as api_main
from api.schemas import ModelInfoResponse, TransactionInput
from src.preprocess import FraudClassifierPipeline, Preprocessor, split_features_target


def _toy_numeric_for_preprocessor(*, n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n_fraud = max(3, n // 5)
    y = np.array([1] * n_fraud + [0] * (n - n_fraud))
    rng.shuffle(y)
    return pd.DataFrame(
        {
            "Time": rng.uniform(0, 200000, size=n),
            "Amount": rng.uniform(5, 300, size=n),
            "V1": rng.normal(size=n),
            "V2": rng.normal(size=n),
            "Class": y.astype(int),
        }
    )


def _make_artifact():
    df = _toy_numeric_for_preprocessor(n=60)
    X, y = split_features_target(df, target_col="Class")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    preprocessor = Preprocessor(random_state=0)
    X_tr, _, y_tr, _ = preprocessor.process_train_test(X_train, y_train, X_test, y_test)
    clf = LogisticRegression(max_iter=200, random_state=0)
    clf.fit(X_tr, y_tr)
    pipeline = FraudClassifierPipeline(preprocessor, clf)
    return {"pipeline": pipeline, "feature_columns": list(X.columns)}


def _full_transaction_dict(**overrides: float) -> dict:
    d = {f"V{i}": 0.0 for i in range(1, 29)}
    d["Amount"] = 100.0
    d["Time"] = 1000.0
    d.update(overrides)
    return d


@pytest.fixture
def mock_bundle(monkeypatch):
    artifact = _make_artifact()
    meta = ModelInfoResponse(
        model_version="test-1.0.0",
        training_date="2026-01-01T00:00:00+00:00",
        auc_roc=0.85,
    )

    def _load(path):
        return artifact

    def _meta(cfg):
        return meta

    monkeypatch.setattr(api_main, "load_model_bundle", _load)
    monkeypatch.setattr(api_main, "read_training_metadata", _meta)
    return artifact, meta


def test_health(mock_bundle) -> None:
    with TestClient(api_main.app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_health_model_missing(monkeypatch) -> None:
    def _missing(_path):
        raise FileNotFoundError("no")

    monkeypatch.setattr(api_main, "load_model_bundle", _missing)
    monkeypatch.setattr(
        api_main,
        "read_training_metadata",
        lambda cfg: ModelInfoResponse(model_version="x", training_date="t", auc_roc=0.5),
    )
    with TestClient(api_main.app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is False


def test_model_info(mock_bundle) -> None:
    with TestClient(api_main.app) as client:
        r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "test-1.0.0"
    assert body["auc_roc"] == 0.85


def test_model_info_not_found(monkeypatch) -> None:
    monkeypatch.setattr(api_main, "load_model_bundle", lambda p: _make_artifact())
    monkeypatch.setattr(api_main, "read_training_metadata", lambda cfg: None)
    with TestClient(api_main.app) as client:
        r = client.get("/model-info")
    assert r.status_code == 404


def test_predict_transaction(mock_bundle) -> None:
    payload = _full_transaction_dict(V1=0.5, V2=-0.2, Amount=50.0, Time=500.0)
    TransactionInput.model_validate(payload)

    with TestClient(api_main.app) as client:
        r = client.post("/predict", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert "fraud_probability" in out
    assert "is_fraud" in out
    assert out["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert out["model_version"] == "test-1.0.0"
    assert 0.0 <= out["fraud_probability"] <= 1.0


def test_predict_legacy_records_format(mock_bundle) -> None:
    payload = {"records": [_full_transaction_dict(V1=0.1, Amount=99.0, Time=1.0)]}
    with TestClient(api_main.app) as client:
        r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "fraud_probability" in r.json()


def test_predict_model_unavailable(monkeypatch) -> None:
    def _missing(_path):
        raise FileNotFoundError()

    monkeypatch.setattr(api_main, "load_model_bundle", _missing)
    monkeypatch.setattr(
        api_main,
        "read_training_metadata",
        lambda cfg: ModelInfoResponse(model_version="x", training_date="t", auc_roc=0.5),
    )
    body = _full_transaction_dict()
    with TestClient(api_main.app) as client:
        r = client.post("/predict", json=body)
    assert r.status_code == 503

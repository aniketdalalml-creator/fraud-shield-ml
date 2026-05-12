from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess import (
    Preprocessor,
    build_training_pipeline,
    load_preprocessor,
    save_preprocessor,
    split_features_target,
)


def _toy_fraud_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Amount": [10.0, 20.0, 15.0, 5.0],
            "V1": [0.5, -1.2, 0.3, 1.1],
            "Category": ["a", "b", "a", "b"],
            "Class": [0, 1, 0, 1],
        }
    )


def _toy_numeric_fraud_frame(*, n: int = 40, fraud_rate: float = 0.2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(2, int(round(n * fraud_rate)))
    n_ok = n - n_fraud
    y = np.array([1] * n_fraud + [0] * n_ok)
    rng.shuffle(y)
    return pd.DataFrame(
        {
            "Time": rng.uniform(0, 172800, size=n),
            "Amount": rng.uniform(1, 500, size=n),
            "V1": rng.normal(size=n),
            "V2": rng.normal(size=n),
            "Class": y.astype(int),
        }
    )


def test_split_features_target() -> None:
    df = _toy_fraud_dataframe()
    X, y = split_features_target(df, target_col="Class")

    assert "Class" not in X.columns
    assert "Amount" in X.columns
    assert "V1" in X.columns
    assert "Category" in X.columns
    assert len(X) == len(y) == 4


def test_training_pipeline_predict_proba() -> None:
    df = _toy_fraud_dataframe()
    X, y = split_features_target(df, target_col="Class")

    pipeline = build_training_pipeline(X, max_iter=50, random_state=0)
    pipeline.fit(X, y)

    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_preprocessor_process_train_test_and_smote() -> None:
    df = _toy_numeric_fraud_frame(n=60, fraud_rate=0.15, seed=1)
    X, y = split_features_target(df, target_col="Class")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    pre = Preprocessor(random_state=0)
    X_tr, X_te, y_tr, y_te = pre.process_train_test(X_train, y_train, X_test, y_test)

    assert X_tr.shape[1] == X_te.shape[1]
    assert X_tr.shape[0] >= len(y_train)
    assert X_te.shape[0] == len(y_te)
    assert len(y_tr) == X_tr.shape[0]


def test_save_load_preprocessor_roundtrip(tmp_path) -> None:
    df = _toy_numeric_fraud_frame(n=50, fraud_rate=0.2, seed=2)
    X, y = split_features_target(df, target_col="Class")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    pre = Preprocessor(random_state=42)
    pre.process_train_test(X_train, y_train, X_test, y_test)

    path = tmp_path / "pre.joblib"
    save_preprocessor(pre, path)
    loaded = load_preprocessor(path)

    Xt = loaded.transform(X_test)
    assert Xt.shape[0] == len(X_test)
    assert not np.isnan(Xt).any()

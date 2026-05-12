from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import DataLoader

_KAGGLE_FEATURES = ("Time", *(f"V{i}" for i in range(1, 29)), "Amount")
_TARGET = "Class"


def _minimal_kaggle_csv(path) -> None:
    row = {c: 0.0 for c in _KAGGLE_FEATURES}
    row[_TARGET] = 0
    pd.DataFrame([row]).to_csv(path, index=False)


def test_data_loader_loads_csv_from_data_folder(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "creditcard.csv"
    _minimal_kaggle_csv(csv_path)

    loader = DataLoader(project_root=tmp_path, random_state=0)
    df = loader.load()

    assert len(df) == 1
    assert list(df.columns) == list(_KAGGLE_FEATURES) + [_TARGET]


def test_data_loader_synthetic_fallback_calls_make_classification(tmp_path, monkeypatch) -> None:
    captured: dict = {}

    def fake_make_classification(**kwargs) -> tuple[np.ndarray, np.ndarray]:
        captured.update(kwargs)
        rng = np.random.default_rng(kwargs.get("random_state", 0))
        X = rng.standard_normal((500, 30))
        y = np.zeros(500, dtype=int)
        y[:1] = 1
        return X, y

    monkeypatch.setattr("src.data_loader.make_classification", fake_make_classification)

    loader = DataLoader(project_root=tmp_path, random_state=123)
    df = loader.load()

    assert captured["n_samples"] == 100_000
    assert captured["n_features"] == 30
    assert captured["weights"] == [0.998, 0.002]
    assert captured["random_state"] == 123

    assert len(df) == 500
    assert list(df.columns) == list(_KAGGLE_FEATURES) + [_TARGET]
    assert df[_TARGET].dtype == np.int64 or str(df[_TARGET].dtype) == "int64"

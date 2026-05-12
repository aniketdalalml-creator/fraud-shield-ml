from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

from config import Settings

logger = logging.getLogger(__name__)

# Kaggle Credit Card Fraud Detection: column order and names.
_KAGGLE_FEATURE_COLUMNS: tuple[str, ...] = (
    "Time",
    *(f"V{i}" for i in range(1, 29)),
    "Amount",
)
_KAGGLE_TARGET_COL = "Class"
_KAGGLE_FILENAME = "creditcard.csv"

_SYNTHETIC_N_SAMPLES = 100_000
_SYNTHETIC_N_FEATURES = 30
# list form required by scikit-learn >= 1.5 (internal `.copy()` on weights)
_SYNTHETIC_CLASS_WEIGHTS: list[float] = [0.998, 0.002]


class DataLoader:
    """Load the credit card fraud dataset from disk or synthesize a stand-in.

    This class first looks for the Kaggle *Credit Card Fraud Detection* CSV
    (``creditcard.csv``) under the project's ``data/`` directory. If the file is
    missing, it builds a reproducible synthetic tabular dataset with the same
    column layout (``Time``, ``V1``–``V28``, ``Amount``, ``Class``) using
    :func:`sklearn.datasets.make_classification`, including a highly imbalanced
    positive (fraud) rate similar to the real competition data.

    Parameters
    ----------
    project_root:
        Repository root; used to resolve ``data/creditcard.csv``.
    random_state:
        Seed passed to ``make_classification`` for reproducible synthetic data.
    csv_filename:
        Name of the CSV file inside ``data/`` (default matches Kaggle).
    """

    def __init__(
        self,
        *,
        project_root: Path,
        random_state: int = 42,
        csv_filename: str = _KAGGLE_FILENAME,
    ) -> None:
        self._project_root = Path(project_root).resolve()
        self._random_state = random_state
        self._csv_path = self._project_root / "data" / csv_filename

    @classmethod
    def from_settings(cls, cfg: Settings) -> DataLoader:
        """Build a loader using paths and random seed from application settings."""
        return cls(project_root=cfg.project_root, random_state=cfg.random_state)

    def load(self) -> pd.DataFrame:
        """Return a dataframe with Kaggle-style columns, from CSV or synthetic data."""
        if self._csv_path.exists():
            df = self._load_csv()
            logger.info("Loaded dataset from CSV: %s", self._csv_path)
        else:
            logger.warning(
                "CSV not found at %s; generating synthetic classification data.",
                self._csv_path,
            )
            df = self._generate_synthetic()

        self._log_dataset_summary(df)
        return df

    def _load_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self._csv_path)
        if df.empty:
            raise ValueError(f"CSV file is empty: {self._csv_path}")
        self._validate_kaggle_schema(df)
        return df

    def _validate_kaggle_schema(self, df: pd.DataFrame) -> None:
        expected = list(_KAGGLE_FEATURE_COLUMNS) + [_KAGGLE_TARGET_COL]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(
                "Loaded CSV is missing expected Kaggle-style columns: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )

    def _generate_synthetic(self) -> pd.DataFrame:
        X, y = make_classification(
            n_samples=_SYNTHETIC_N_SAMPLES,
            n_features=_SYNTHETIC_N_FEATURES,
            n_informative=22,
            n_redundant=5,
            n_clusters_per_class=3,
            n_classes=2,
            weights=_SYNTHETIC_CLASS_WEIGHTS,
            random_state=self._random_state,
            shuffle=True,
        )
        df = pd.DataFrame(X, columns=list(_KAGGLE_FEATURE_COLUMNS))
        df[_KAGGLE_TARGET_COL] = y.astype("int64")
        return df

    def _log_dataset_summary(self, df: pd.DataFrame) -> None:
        logger.info("Dataset shape: rows=%s, cols=%s", df.shape[0], df.shape[1])
        if _KAGGLE_TARGET_COL in df.columns:
            counts = df[_KAGGLE_TARGET_COL].value_counts().sort_index()
            dist = counts / len(df) * 100.0
            logger.info(
                "Class distribution: %s (counts); %s (percent of rows)",
                counts.to_dict(),
                dist.round(4).to_dict(),
            )
        missing = df.isna().sum()
        n_missing_cols = int((missing > 0).sum())
        total_missing = int(missing.sum())
        logger.info(
            "Missing values: total_cells=%s, columns_with_any_na=%s; per-column counts: %s",
            total_missing,
            n_missing_cols,
            missing[missing > 0].to_dict() if total_missing else {},
        )


def load_transactions_csv(path: Path) -> pd.DataFrame:
    """Load transaction data from a CSV file.

    Parameters
    ----------
    path:
        Path to the transactions CSV.
    """

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV file is empty: {path}")
    return df


def load_data(cfg: Settings) -> pd.DataFrame:
    """Load the fraud dataset from ``data/creditcard.csv`` or synthesize if absent."""
    df = DataLoader.from_settings(cfg).load()
    if cfg.target_col not in df.columns:
        raise ValueError(
            f"Target column {cfg.target_col!r} not found in dataset. "
            f"Columns: {list(df.columns)[:20]}..."
        )
    return df

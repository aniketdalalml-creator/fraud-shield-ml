from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


def split_features_target(
    df: pd.DataFrame,
    *,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a labeled dataframe into (X, y)."""

    if target_col not in df.columns:
        raise ValueError(f"target_col={target_col!r} not found in dataframe columns.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    if y.nunique() < 2:
        raise ValueError("Target column must contain at least two classes.")
    return X, y


def infer_numeric_and_categorical_columns(
    df: pd.DataFrame,
    *,
    excluded_cols: Iterable[str] = (),
) -> tuple[list[str], list[str]]:
    """Infer numeric vs categorical columns based on pandas dtypes."""

    excluded = set(excluded_cols)
    feature_cols = [c for c in df.columns if c not in excluded]

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in feature_cols:
        if is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def build_preprocessing_pipeline(df: pd.DataFrame) -> ColumnTransformer:
    """Create a preprocessing pipeline for numeric + categorical features."""

    numeric_cols, categorical_cols = infer_numeric_and_categorical_columns(df)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    if not transformers:
        raise ValueError("No feature columns found for preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_training_pipeline(
    X: pd.DataFrame,
    *,
    max_iter: int,
    random_state: int,
) -> Pipeline:
    """Build the end-to-end training pipeline (preprocess + classifier)."""

    preprocessor = build_preprocessing_pipeline(X)

    model = LogisticRegression(
        solver="liblinear",
        max_iter=max_iter,
        random_state=random_state,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


class Preprocessor:
    """Tabular preprocessing for fraud detection with train-only fitting rules.

    Training-time steps (see :meth:`process_train_test`):

    #. Drop duplicate rows (features + label alignment preserved on train).
    #. Drop rows where the fraction of missing values exceeds ``missing_row_threshold``,
       then median-impute remaining numeric NaNs (statistics **fit on train only**).
    #. Engineer ``Hour`` from ``Time``, ``Amount_log`` from ``Amount``, then drop
       ``Time`` and ``Amount``.
    #. :class:`~sklearn.preprocessing.RobustScaler` **fit on train**, transform train
       and test.
    #. :class:`~imblearn.over_sampling.SMOTE` on the **scaled training set only**.

    Inference uses :meth:`transform`, which applies median imputation (using
    statistics fit on train), feature engineering, and scaling **without**
    SMOTE, deduplication, or high-missing row drops—so batch row count matches
    the request (training still uses those cleaning steps).

    ``Time`` and ``Amount`` must be present in the input feature frame before
    feature engineering.
    """

    def __init__(
        self,
        *,
        random_state: int = 42,
        missing_row_threshold: float = 0.5,
    ) -> None:
        self.random_state = random_state
        self.missing_row_threshold = missing_row_threshold
        self._imputer: SimpleImputer | None = None
        self._scaler = RobustScaler()
        self._columns_before_fe_: list[str] | None = None
        self._columns_after_fe_: list[str] | None = None
        self._is_fitted = False

    def process_train_test(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the full training pipeline through SMOTE and return arrays.

        Parameters
        ----------
        X_train, y_train
            Training features and labels (same index).
        X_test, y_test
            Held-out features and labels.

        Returns
        -------
        X_train_resampled, X_test_scaled, y_train_resampled, y_test
            Resampled **scaled** training matrix, **scaled** test matrix, and label
            arrays (train labels resampled by SMOTE; test labels unchanged).
        """

        logger.info("Preprocessor.process_train_test: initial train shape=%s test shape=%s", X_train.shape, X_test.shape)

        Xt, yt = self._drop_duplicates_xy(X_train, y_train, split_name="train")
        Xv, yv = self._drop_duplicates_xy(X_test, y_test, split_name="test")

        Xt = self._drop_high_missing_rows(Xt, split_name="train")
        Xv = self._drop_high_missing_rows(Xv, split_name="test")
        yt = yt.loc[Xt.index]
        yv = yv.loc[Xv.index]

        self._imputer = SimpleImputer(strategy="median")
        Xv = Xv.reindex(columns=Xt.columns)
        Xt_imp = pd.DataFrame(
            self._imputer.fit_transform(Xt),
            columns=Xt.columns,
            index=Xt.index,
        )
        Xv_imp = pd.DataFrame(
            self._imputer.transform(Xv),
            columns=Xt.columns,
            index=Xv.index,
        )
        self._columns_before_fe_ = list(Xt_imp.columns)
        logger.info(
            "Median imputation fitted on train; train NaN count=%s test NaN count=%s",
            int(Xt_imp.isna().sum().sum()),
            int(Xv_imp.isna().sum().sum()),
        )

        Xt_fe = self._feature_engineering(Xt_imp, split_name="train")
        Xv_fe = self._feature_engineering(Xv_imp, split_name="test")
        self._columns_after_fe_ = list(Xt_fe.columns)

        Xtr = Xt_fe.to_numpy(dtype=np.float64, copy=False)
        Xte = Xv_fe.to_numpy(dtype=np.float64, copy=False)

        Xtr_scaled = self._scaler.fit_transform(Xtr)
        Xte_scaled = self._scaler.transform(Xte)
        self._is_fitted = True
        logger.info(
            "RobustScaler fit on train; scaled train shape=%s test shape=%s",
            Xtr_scaled.shape,
            Xte_scaled.shape,
        )

        ytr = yt.astype(int).to_numpy()
        yte = yv.astype(int).to_numpy()

        k_neighbors = self._smote_k_neighbors(ytr)
        smote = SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
        X_res, y_res = smote.fit_resample(Xtr_scaled, ytr)
        logger.info(
            "SMOTE on train only: shape %s -> %s; class counts %s -> %s",
            Xtr_scaled.shape,
            X_res.shape,
            dict(zip(*np.unique(ytr, return_counts=True))),
            dict(zip(*np.unique(y_res, return_counts=True))),
        )

        return X_res, Xte_scaled, y_res, yte

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply fitted preprocessing (no SMOTE) for inference or evaluation."""

        if not self._is_fitted or self._imputer is None or self._columns_before_fe_ is None:
            raise RuntimeError("Preprocessor must be fitted via process_train_test before transform.")

        Xa = align_dataframe_columns(X, self._columns_before_fe_)

        X_imp = pd.DataFrame(
            self._imputer.transform(Xa),
            columns=self._columns_before_fe_,
            index=Xa.index,
        )
        X_fe = self._feature_engineering(X_imp, split_name="inference")
        X_arr = X_fe.to_numpy(dtype=np.float64, copy=False)
        return self._scaler.transform(X_arr)

    def _drop_duplicates_xy(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        split_name: str,
    ) -> tuple[pd.DataFrame, pd.Series]:
        before = len(X)
        combo = X.copy()
        combo["_y_internal"] = y.values
        combo = combo.drop_duplicates()
        after = len(combo)
        logger.info("[%s] Drop duplicates: %s -> %s rows", split_name, before, after)
        y_out = pd.Series(combo["_y_internal"].values, index=combo.index, name=y.name)
        combo = combo.drop(columns=["_y_internal"])
        return combo, y_out

    def _drop_high_missing_rows(self, X: pd.DataFrame, *, split_name: str) -> pd.DataFrame:
        if X.empty or X.shape[1] == 0:
            return X
        n_cols = X.shape[1]
        ratio_missing = X.isna().sum(axis=1) / float(n_cols)
        keep = ratio_missing <= self.missing_row_threshold
        before, after = len(X), int(keep.sum())
        logger.info(
            "[%s] Drop rows with >%.0f%% missing features: %s -> %s rows",
            split_name,
            self.missing_row_threshold * 100,
            before,
            after,
        )
        return X.loc[keep].copy()

    def _feature_engineering(self, X: pd.DataFrame, *, split_name: str) -> pd.DataFrame:
        if "Time" not in X.columns or "Amount" not in X.columns:
            raise ValueError(
                "Feature engineering requires 'Time' and 'Amount' columns. "
                f"Columns present: {list(X.columns)}"
            )
        out = X.copy()
        out["Hour"] = (out["Time"].astype(float) % (3600 * 24)) / 3600.0
        # log1p is only defined for Amount >= -1 in ℝ; clip at 0 like non-negative transaction amounts.
        amt = np.maximum(out["Amount"].astype(float), 0.0)
        out["Amount_log"] = np.log1p(amt)
        out = out.drop(columns=["Time", "Amount"])
        logger.info(
            "[%s] Feature engineering: added Hour, Amount_log; dropped Time, Amount; shape=%s",
            split_name,
            out.shape,
        )
        return out

    @staticmethod
    def _smote_k_neighbors(y: np.ndarray) -> int:
        _, counts = np.unique(y, return_counts=True)
        minority = int(counts.min())
        if minority < 2:
            raise ValueError(
                "SMOTE requires at least 2 minority-class samples in training data; "
                f"got minority count={minority}."
            )
        return max(1, min(5, minority - 1))


class FraudClassifierPipeline:
    """Fitted :class:`Preprocessor` plus classifier, exposing ``predict`` / ``predict_proba``."""

    def __init__(self, preprocessor: Preprocessor, classifier: Any) -> None:
        self._preprocessor = preprocessor
        self._classifier = classifier

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self._preprocessor.transform(X)
        return self._classifier.predict(Xt)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self._preprocessor.transform(X)
        return self._classifier.predict_proba(Xt)


def save_preprocessor(preprocessor: Preprocessor, path: str | Path) -> None:
    """Persist a fitted :class:`Preprocessor` with joblib."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, p)
    logger.info("Saved preprocessor to %s", p.resolve())


def load_preprocessor(path: str | Path) -> Preprocessor:
    """Load a :class:`Preprocessor` saved by :func:`save_preprocessor`."""

    p = Path(path)
    obj = joblib.load(p)
    if not isinstance(obj, Preprocessor):
        raise TypeError(f"Expected Preprocessor at {p}, got {type(obj).__name__}")
    logger.info("Loaded preprocessor from %s", p.resolve())
    return obj


def align_dataframe_columns(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Align input dataframe columns to the training feature set.

    - Adds missing columns with NA values.
    - Drops any extra columns.
    - Returns columns in the exact training order.
    """

    aligned = df.copy()
    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = np.nan
    return aligned.reindex(columns=list(feature_columns))


def records_to_dataframe(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Convert JSON-like records to a dataframe for inference."""

    if not records:
        raise ValueError("No records provided.")
    return pd.DataFrame(list(records))

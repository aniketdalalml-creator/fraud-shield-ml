from __future__ import annotations

import inspect
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from config import Settings, get_settings
from src.data_loader import load_data
from src.evaluate import evaluate_model
from src.preprocess import FraudClassifierPipeline, Preprocessor, split_features_target

logger = logging.getLogger(__name__)


def _xgb_base_params(*, random_state: int = 42) -> dict[str, Any]:
    """Default XGBoost hyperparameters (SMOTE already balances classes)."""
    params: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "scale_pos_weight": 1,
        "eval_metric": "aucpr",
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
    }
    if "use_label_encoder" in inspect.signature(XGBClassifier).parameters:
        params["use_label_encoder"] = False
    return params


class ModelTrainer:
    """Train XGBoost on preprocessed arrays, then tune with RandomizedSearchCV.

    Expects ``X_train``, ``y_train`` **after** resampling (e.g. SMOTE), as
    ``numpy.ndarray`` inputs.

    Training flow:

    #. Fit a baseline :class:`~xgboost.XGBClassifier` with fixed hyperparameters.
    #. Run :class:`~sklearn.model_selection.RandomizedSearchCV` (20 iterations,
       5-fold stratified CV, ``scoring='roc_auc'``) and keep the best estimator.

    Attributes
    ----------
    best_estimator_
        Best model from randomized search.
    best_params_
        Hyperparameters of the best model.
    best_cv_score_
        Mean CV ROC-AUC of the best model.
    training_time_s_
        Wall-clock seconds for baseline fit + search.
    """

    def __init__(
        self,
        *,
        random_state: int = 42,
        search_n_iter: int = 20,
        cv_folds: int = 5,
    ) -> None:
        self.random_state = random_state
        self.search_n_iter = search_n_iter
        self.cv_folds = cv_folds
        self.best_estimator_: XGBClassifier | None = None
        self.best_params_: dict[str, Any] | None = None
        self.best_cv_score_: float | None = None
        self.training_time_s_: float | None = None
        self._baseline_fit_seconds: float | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> ModelTrainer:
        """Fit baseline XGBoost, then hyperparameter search; store best model."""

        if X_train.size == 0:
            raise ValueError("X_train is empty.")
        t_start = time.perf_counter()

        base_kw = _xgb_base_params(random_state=self.random_state)
        baseline = XGBClassifier(**base_kw)
        t0 = time.perf_counter()
        baseline.fit(X_train, y_train)
        self._baseline_fit_seconds = time.perf_counter() - t0
        logger.info("Baseline XGBoost fit finished in %.2f s.", self._baseline_fit_seconds)

        param_distributions: dict[str, Any] = {
            "n_estimators": randint(80, 320),
            "max_depth": randint(3, 11),
            "learning_rate": uniform(0.01, 0.19),
            "subsample": uniform(0.65, 0.34),
            "colsample_bytree": uniform(0.65, 0.34),
        }

        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        search = RandomizedSearchCV(
            XGBClassifier(**base_kw),
            param_distributions=param_distributions,
            n_iter=self.search_n_iter,
            cv=cv,
            scoring="roc_auc",
            random_state=self.random_state,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        search.fit(X_train, y_train)

        self.training_time_s_ = time.perf_counter() - t_start
        self.best_estimator_ = search.best_estimator_
        self.best_params_ = dict(search.best_params_)
        self.best_cv_score_ = float(search.best_score_)

        logger.info(
            "RandomizedSearchCV complete. Total training wall time: %.2f s | "
            "best CV ROC-AUC: %.6f | best params: %s",
            self.training_time_s_,
            self.best_cv_score_,
            self.best_params_,
        )
        return self


def _fraud_model_path(cfg: Settings) -> Path:
    """Resolved path for the primary fraud model bundle (``fraud_model.pkl``)."""
    return cfg.project_root / "models" / "fraud_model.pkl"


def train_model(cfg: Settings) -> dict:
    """Load data, preprocess, train XGBoost with tuning, evaluate, save ``fraud_model.pkl``."""

    df = load_data(cfg)
    X, y = split_features_target(df, target_col=cfg.target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    preprocessor = Preprocessor(random_state=cfg.random_state)
    X_train_res, _, y_train_res, _ = preprocessor.process_train_test(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    trainer = ModelTrainer(random_state=cfg.random_state, search_n_iter=20, cv_folds=5)
    trainer.fit(X_train_res, y_train_res)
    if trainer.best_estimator_ is None:
        raise RuntimeError("Training did not produce a best estimator.")

    artifact_eval = {
        "preprocessor": preprocessor,
        "model": trainer.best_estimator_,
        "feature_columns": list(X.columns),
        "target_col": cfg.target_col,
        "best_params": trainer.best_params_,
        "best_cv_roc_auc": trainer.best_cv_score_,
    }
    metrics = evaluate_model(
        cfg,
        artifact=artifact_eval,
        X_test=X_test,
        y_test=y_test,
    )

    pipeline = FraudClassifierPipeline(preprocessor, trainer.best_estimator_)
    artifact_save = {
        **artifact_eval,
        "pipeline": pipeline,
        "metrics": metrics,
    }

    out_path = _fraud_model_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact_save, out_path)
    logger.info("Saved model bundle to %s", out_path.resolve())

    metadata = {
        "model_version": os.environ.get("FRAUD_MODEL_VERSION", "1.0.0"),
        "training_date": datetime.now(timezone.utc).isoformat(),
        "auc_roc": float(metrics["auc_roc"]),
    }
    cfg.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Wrote training metadata to %s", cfg.metadata_path.resolve())

    cfg.model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.model_artifact_path.resolve() != out_path.resolve():
        joblib.dump(artifact_save, cfg.model_artifact_path)
        logger.info("Also saved copy to %s", cfg.model_artifact_path.resolve())

    return artifact_save


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = get_settings()
    train_model(cfg)


if __name__ == "__main__":
    import sys

    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    main()

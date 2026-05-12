from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from config import Settings, get_settings
from src.data_loader import load_data
from src.preprocess import align_dataframe_columns, split_features_target

logger = logging.getLogger(__name__)


def _ensure_json_serializable(metrics: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(metrics, default=str))


def _predict_proba_positive(artifact: dict[str, Any], X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (positive_class_proba, hard_predictions)."""
    feature_columns: list[str] = artifact["feature_columns"]
    Xa = align_dataframe_columns(X, feature_columns)

    if "pipeline" in artifact:
        pipeline = artifact["pipeline"]
        proba = pipeline.predict_proba(Xa)[:, 1]
        pred = pipeline.predict(Xa)
        return np.asarray(proba, dtype=float), np.asarray(pred, dtype=int)

    preprocessor = artifact["preprocessor"]
    model = artifact["model"]
    Xt = preprocessor.transform(Xa)
    proba = model.predict_proba(Xt)[:, 1]
    pred = model.predict(Xt)
    return np.asarray(proba, dtype=float), np.asarray(pred, dtype=int)


def evaluate_model(
    cfg: Settings,
    *,
    artifact: dict[str, Any] | None = None,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
) -> dict[str, Any]:
    """Evaluate the fraud model on a held-out set and persist metrics + plots.

    Computes AUC-ROC, AUC-PR (average precision), F1, precision, recall, and
    confusion matrix. Saves ``models/confusion_matrix.png`` and
    ``models/roc_curve.png`` under the project root.

    Parameters
    ----------
    cfg:
        Application settings (paths, target column, split params).
    artifact:
        In-memory bundle with ``preprocessor`` + ``model`` or ``pipeline``, and
        ``feature_columns``. If omitted, loads ``cfg.model_artifact_path``.
    X_test, y_test:
        Raw feature matrix and labels for evaluation. If omitted, reloads data
        and reproduces a stratified hold-out using ``cfg.test_size`` and
        ``cfg.random_state`` (same settings as training for a comparable split).
    """

    art = artifact if artifact is not None else joblib.load(cfg.model_artifact_path)

    if X_test is None or y_test is None:
        df = load_data(cfg)
        X, y = split_features_target(df, target_col=cfg.target_col)
        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=y,
        )

    y_true = y_test.astype(int).to_numpy()
    probs, preds = _predict_proba_positive(art, X_test)

    auc_roc = float(roc_auc_score(y_true, probs))
    auc_pr = float(average_precision_score(y_true, probs))
    f1 = float(f1_score(y_true, preds))
    precision = float(precision_score(y_true, preds, zero_division=0))
    recall = float(recall_score(y_true, preds, zero_division=0))
    cm = confusion_matrix(y_true, preds)
    cm_list = cm.tolist()

    metrics: dict[str, Any] = {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm_list,
        "n_test": int(len(y_true)),
    }

    models_dir = cfg.project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    cm_path = models_dir / "confusion_matrix.png"
    roc_path = models_dir / "roc_curve.png"

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax_cm,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    ax_cm.set_title("Confusion matrix")
    fig_cm.tight_layout()
    fig_cm.savefig(cm_path, dpi=150)
    plt.close(fig_cm)
    logger.info("Saved confusion matrix plot to %s", cm_path.resolve())

    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, probs, ax=ax_roc, name="Model")
    ax_roc.set_title("ROC curve")
    fig_roc.tight_layout()
    fig_roc.savefig(roc_path, dpi=150)
    plt.close(fig_roc)
    logger.info("Saved ROC curve plot to %s", roc_path.resolve())

    logger.info(
        "Evaluation metrics | AUC-ROC=%.6f AUC-PR=%.6f F1=%.6f Precision=%.6f Recall=%.6f "
        "| confusion_matrix=%s | n_test=%s",
        auc_roc,
        auc_pr,
        f1,
        precision,
        recall,
        cm_list,
        metrics["n_test"],
    )

    cfg.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.metrics_path.write_text(json.dumps(_ensure_json_serializable(metrics), indent=2))

    return metrics


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = get_settings()
    evaluate_model(cfg)


if __name__ == "__main__":
    import sys

    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    main()

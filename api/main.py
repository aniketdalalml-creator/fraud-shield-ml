# Example curl commands (run from repo root with API on localhost:8000):
#
#   curl -s http://127.0.0.1:8000/health
#   curl -s http://127.0.0.1:8000/model-info
#   curl -s -X POST http://127.0.0.1:8000/predict \
#     -H "Content-Type: application/json" \
#     -d '{"V1":-1.36,"V2":-0.07,"V3":2.54,"V4":1.0,"V5":-0.55,"V6":0.04,"V7":0.52,"V8":0.44,"V9":-0.63,
#          "V10":-0.08,"V11":-0.09,"V12":-0.08,"V13":0.09,"V14":-0.06,"V15":-0.05,"V16":-0.06,"V17":0.03,
#          "V18":0.01,"V19":0.02,"V20":0.0,"V21":0.0,"V22":0.0,"V23":0.0,"V24":0.0,"V25":0.0,"V26":0.0,
#          "V27":0.0,"V28":0.0,"Amount":149.62,"Time":0.0}'
#
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import anyio
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from api.schemas import ModelInfoResponse, PredictRequestBody, PredictionOutput
from config import Settings, get_settings
from src.preprocess import FraudClassifierPipeline, align_dataframe_columns

logger = logging.getLogger(__name__)


def load_model_bundle(path: Path) -> dict[str, Any]:
    """Load ``fraud_model.pkl`` (or compatible joblib bundle) from disk.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the payload is not a valid bundle.
    """

    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError("Model artifact must be a dict.")
    if "feature_columns" not in artifact:
        raise ValueError("Model artifact missing feature_columns.")
    if "pipeline" in artifact:
        return artifact
    if "model" in artifact and "preprocessor" in artifact:
        return {
            **artifact,
            "pipeline": FraudClassifierPipeline(artifact["preprocessor"], artifact["model"]),
        }
    raise ValueError("Model artifact missing pipeline or model/preprocessor.")


def read_training_metadata(cfg: Settings) -> ModelInfoResponse | None:
    """Parse ``metadata.json`` written at training time. Returns ``None`` if missing."""

    p = cfg.metadata_path
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return ModelInfoResponse(
            model_version=str(data["model_version"]),
            training_date=str(data["training_date"]),
            auc_roc=float(data["auc_roc"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Invalid metadata.json at %s: %s", p, exc)
        return None


def _risk_level(probability: float) -> Literal["LOW", "MEDIUM", "HIGH"]:
    if probability < 0.3:
        return "LOW"
    if probability <= 0.7:
        return "MEDIUM"
    return "HIGH"


def run_single_prediction(
    artifact: dict[str, Any],
    tx: BaseModel,
    *,
    model_version: str,
) -> PredictionOutput:
    """Run preprocess + classifier for one validated transaction row."""

    feature_columns: list[str] = artifact["feature_columns"]
    pipeline = artifact["pipeline"]
    row = tx.model_dump()
    df = align_dataframe_columns(pd.DataFrame([row]), feature_columns)

    try:
        proba_arr = pipeline.predict_proba(df)
        fraud_probability = float(min(1.0, max(0.0, proba_arr[0, 1])))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    is_fraud = bool(fraud_probability >= 0.5)
    return PredictionOutput(
        fraud_probability=fraud_probability,
        is_fraud=is_fraud,
        risk_level=_risk_level(fraud_probability),
        model_version=model_version,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    app.state.model_info: ModelInfoResponse | None = await anyio.to_thread.run_sync(
        read_training_metadata,
        cfg,
    )
    app.state.artifact = None
    try:
        app.state.artifact = await anyio.to_thread.run_sync(load_model_bundle, cfg.model_artifact_path)
        logger.info("Loaded model bundle from %s", cfg.model_artifact_path.resolve())
    except FileNotFoundError:
        logger.warning("Model bundle not found at startup: %s", cfg.model_artifact_path)
    except Exception:
        logger.exception("Failed to load model bundle at startup")
    yield
    app.state.artifact = None
    app.state.model_info = None


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log UTC timestamp, path, elapsed ms, and response status."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        t0 = time.perf_counter()
        ts = datetime.now(timezone.utc).isoformat()
        path = request.url.path
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.exception("%s %s %.2fms (error)", ts, path, elapsed_ms)
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("%s %s %.2fms status=%s", ts, path, elapsed_ms, response.status_code)
        return response


app = FastAPI(title="fraud-shield-ml", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)


@app.get("/health")
def health(request: Request) -> dict[str, bool | str]:
    loaded = getattr(request.app.state, "artifact", None) is not None
    return {"status": "ok", "model_loaded": loaded}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    meta: ModelInfoResponse | None = getattr(request.app.state, "model_info", None)
    if meta is None:
        raise HTTPException(
            status_code=404,
            detail="metadata.json not found or invalid. Train the model to generate it.",
        )
    return meta


@app.post("/predict", response_model=PredictionOutput)
def predict(request: Request, body: PredictRequestBody) -> PredictionOutput:
    artifact = getattr(request.app.state, "artifact", None)
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Ensure models/fraud_model.pkl exists and restart the API.",
        )
    meta: ModelInfoResponse | None = getattr(request.app.state, "model_info", None)
    version = meta.model_version if meta is not None else "unknown"
    try:
        return run_single_prediction(artifact, body.root, model_version=version)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in /predict")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

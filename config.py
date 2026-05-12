from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Settings:
    """Central place for configuration.

    All project paths and tunable parameters should be sourced from here (directly
    or via environment variables), to avoid hardcoded values scattered across the codebase.
    """

    project_root: Path

    # Data
    raw_data_path: Path

    # Artifacts
    model_artifact_path: Path
    metadata_path: Path
    metrics_path: Path

    # Preprocessing / training
    target_col: str
    test_size: float
    random_state: int
    max_iter: int

    # API
    api_host: str
    api_port: int

    # Streamlit
    streamlit_host: str
    streamlit_port: int


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key)
    return Path(raw) if raw else default


def get_settings() -> Settings:
    """Build Settings from defaults overridden by environment variables."""

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    models_dir = project_root / "models"

    return Settings(
        project_root=project_root,
        raw_data_path=_env_path(
            "FRAUD_SHIELD_RAW_DATA_PATH",
            data_dir / "creditcard.csv",
        ),
        model_artifact_path=_env_path(
            "FRAUD_SHIELD_MODEL_ARTIFACT_PATH",
            models_dir / "fraud_model.pkl",
        ),
        metadata_path=_env_path(
            "FRAUD_SHIELD_METADATA_PATH",
            models_dir / "metadata.json",
        ),
        metrics_path=_env_path("FRAUD_SHIELD_METRICS_PATH", models_dir / "metrics.json"),
        target_col=_env("FRAUD_SHIELD_TARGET_COL", "Class"),
        test_size=float(_env("FRAUD_SHIELD_TEST_SIZE", "0.2")),
        random_state=int(_env("FRAUD_SHIELD_RANDOM_STATE", "42")),
        max_iter=int(_env("FRAUD_SHIELD_MAX_ITER", "1000")),
        api_host=_env("FRAUD_SHIELD_API_HOST", "0.0.0.0"),
        api_port=int(_env("FRAUD_SHIELD_API_PORT", "8000")),
        streamlit_host=_env("FRAUD_SHIELD_STREAMLIT_HOST", "0.0.0.0"),
        streamlit_port=int(_env("FRAUD_SHIELD_STREAMLIT_PORT", "8501")),
    )


from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, create_model, model_validator


def _example_v(i: int) -> float:
    return round(((-1) ** i) * 0.15 * i, 4)


def _build_transaction_input_model() -> type[BaseModel]:
    """Build ``TransactionInput`` with V1–V28, ``Amount``, ``Time`` and per-field examples."""
    fields: dict[str, tuple[type, Field]] = {}
    for i in range(1, 29):
        fields[f"V{i}"] = (
            float,
            Field(..., description=f"Anonymized feature V{i}.", examples=[_example_v(i)]),
        )
    fields["Amount"] = (float, Field(..., description="Transaction amount.", examples=[149.62]))
    fields["Time"] = (float, Field(..., description="Seconds since first transaction in dataset.", examples=[0.0]))
    return create_model(
        "TransactionInput",
        __config__=ConfigDict(extra="forbid"),
        **fields,  # type: ignore[arg-type]
    )


TransactionInput = _build_transaction_input_model()


class PredictRequestBody(RootModel[TransactionInput]):  # type: ignore[valid-type]
    """Request body for ``POST /predict``.

    Accepts either a flat :class:`TransactionInput` JSON object **or** the legacy
    shape ``{"records": [<one transaction dict>]}`` so older clients and mixed
    deployments keep working.
    """

    @model_validator(mode="before")
    @classmethod
    def _unwrap_legacy_records(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("records"), list):
            recs = data["records"]
            if len(recs) != 1:
                raise ValueError("Legacy `records` format requires exactly one transaction object.")
            return recs[0]
        return data


class PredictionOutput(BaseModel):
    """Single-transaction fraud scoring response."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "fraud_probability": 0.42,
                "is_fraud": False,
                "risk_level": "MEDIUM",
                "model_version": "1.0.0",
            }
        }
    )

    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Estimated P(fraud).")
    is_fraud: bool = Field(..., description="Binary decision (typically threshold 0.5 on fraud_probability).")
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ...,
        description='Bucket from fraud_probability: <30% LOW, 30–70% MEDIUM, >70% HIGH.',
    )
    model_version: str = Field(..., description="Release label from training metadata.")


class ModelInfoResponse(BaseModel):
    """Training-time metadata surfaced for operators."""

    model_version: str
    training_date: str
    auc_roc: float

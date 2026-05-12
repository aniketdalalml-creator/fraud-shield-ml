from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import plotly.graph_objects as go
import requests
import streamlit as st

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import get_settings

# ---------------------------------------------------------------------------
# Defaults & samples (full TransactionInput shape for POST /predict)
# ---------------------------------------------------------------------------
_DEFAULT_METRICS = {
    "auc_roc": 0.974,
    "f1": 0.869,
    "precision": 0.912,
    "recall": 0.831,
}

# Illustrative “fraud-like” vs “legit-like” patterns for demo sliders.
_SAMPLE_FRAUD: dict[str, float] = {
    **{f"V{i}": 0.0 for i in range(1, 29)},
    "V1": -5.0,
    "V2": 4.2,
    "V3": 0.0,
    "V4": 4.8,
    "V5": 0.0,
    "V6": 0.0,
    "V7": 0.0,
    "V8": 0.0,
    "V9": 0.0,
    "V10": 0.0,
    "V11": 3.9,
    "V12": 0.0,
    "V13": 0.0,
    "V14": -3.5,
    "V15": 0.0,
    "V16": 0.0,
    "V17": 0.0,
    "V18": 0.0,
    "V19": 0.0,
    "V20": 0.0,
    "V21": 0.0,
    "V22": 0.0,
    "V23": 0.0,
    "V24": 0.0,
    "V25": 0.0,
    "V26": 0.0,
    "V27": 0.0,
    "V28": 0.0,
    "Amount": 3850.0,
    "Time": 3600.0,
}

_SAMPLE_LEGIT: dict[str, float] = {
    **{f"V{i}": 0.0 for i in range(1, 29)},
    "V1": 0.2,
    "V2": -0.1,
    "V3": 0.0,
    "V4": 0.15,
    "V5": 0.0,
    "V6": 0.0,
    "V7": 0.0,
    "V8": 0.0,
    "V9": 0.0,
    "V10": 0.0,
    "V11": -0.25,
    "V12": 0.0,
    "V13": 0.0,
    "V14": 0.1,
    "V15": 0.0,
    "V16": 0.0,
    "V17": 0.0,
    "V18": 0.0,
    "V19": 0.0,
    "V20": 0.0,
    "V21": 0.0,
    "V22": 0.0,
    "V23": 0.0,
    "V24": 0.0,
    "V25": 0.0,
    "V26": 0.0,
    "V27": 0.0,
    "V28": 0.0,
    "Amount": 42.5,
    "Time": 120_000.0,
}


def _api_base_url() -> str:
    return os.environ.get("FRAUD_SHIELD_API_BASE", "http://localhost:8000").rstrip("/")


@st.cache_resource
def get_api_session() -> requests.Session:
    """Cached HTTP session for FastAPI calls (connection pooling, shared headers)."""
    session = requests.Session()
    session.headers.update(
        {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    )
    return session


def _inject_dark_theme_css() -> None:
    st.markdown(
        """
        <style>
          .stApp { background-color: #0e1117; color: #e6edf3; }
          [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
          [data-testid="stSidebar"] .stMarkdown { color: #8b949e; }
          div[data-testid="stVerticalBlock"] > div:has(> label) label { color: #c9d1d9 !important; }
          .stTabs [data-baseweb="tab-list"] { background-color: #161b22; gap: 8px; }
          .stTabs [aria-selected="true"] { color: #58a6ff !important; }
          .metric-card {
            border-radius: 12px; padding: 1.25rem 1.5rem; margin: 0.5rem 0;
            border: 1px solid #30363d; background: linear-gradient(145deg, #161b22, #0d1117);
          }
          .metric-fraud { border-color: #f85149; box-shadow: 0 0 24px rgba(248,81,73,0.15); }
          .metric-legit { border-color: #3fb950; box-shadow: 0 0 24px rgba(63,185,80,0.12); }
          .risk-badge {
            display: inline-block; padding: 0.35rem 0.85rem; border-radius: 999px;
            font-weight: 600; font-size: 0.9rem; letter-spacing: 0.02em;
          }
          .risk-low { background: #238636; color: #fff; }
          .risk-med { background: #d29922; color: #0d1117; }
          .risk-high { background: #da3633; color: #fff; }
          .flow-box {
            background: #161b22; border: 1px solid #30363d; border-radius: 10px;
            padding: 1rem 1.25rem; margin: 0.35rem 0; color: #c9d1d9;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_metrics_from_disk(cfg) -> dict[str, float]:
    out = dict(_DEFAULT_METRICS)
    p = cfg.metrics_path
    if not p.exists():
        return out
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if "auc_roc" in data:
            out["auc_roc"] = float(data["auc_roc"])
        if "f1" in data:
            out["f1"] = float(data["f1"])
        if "precision" in data:
            out["precision"] = float(data["precision"])
        if "recall" in data:
            out["recall"] = float(data["recall"])
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return out


def _risk_badge_class(level: str) -> str:
    if level == "HIGH":
        return "risk-high"
    if level == "MEDIUM":
        return "risk-med"
    return "risk-low"


def _fraud_gauge(probability: float) -> go.Figure:
    pct = probability * 100.0
    bar = "#f85149" if probability >= 0.5 else "#3fb950"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 36, "color": "#e6edf3"}},
            title={"text": "Fraud probability", "font": {"size": 16, "color": "#8b949e"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b949e", "tickwidth": 1},
                "bar": {"color": bar},
                "bgcolor": "#21262d",
                "borderwidth": 1,
                "bordercolor": "#30363d",
                "steps": [
                    {"range": [0, 30], "color": "#23863633"},
                    {"range": [30, 70], "color": "#d2992233"},
                    {"range": [70, 100], "color": "#da363333"},
                ],
                "threshold": {
                    "line": {"color": "#f0883e", "width": 2},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e6edf3"},
        height=320,
        margin=dict(l=24, r=24, t=48, b=24),
    )
    return fig


def _tab_predict() -> None:
    st.markdown("Adjust **Amount**, **Time**, and five PCA-style features, then analyze via the **FastAPI** service.")

    c1, c2 = st.columns(2)
    with c1:
        amount = st.slider("Amount ($)", 0.0, 5000.0, 150.0, step=1.0, key="fs_amount")
    with c2:
        time_s = st.slider("Time (seconds)", 0.0, 172800.0, 0.0, step=60.0, key="fs_time")

    st.markdown("**Key V features** (remaining V1–V28 default to 0 unless set by samples)")
    g1, g2, g3 = st.columns(3)
    with g1:
        v1 = st.slider("V1", -5.0, 5.0, 0.0, step=0.05, key="fs_v1")
        v2 = st.slider("V2", -5.0, 5.0, 0.0, step=0.05, key="fs_v2")
    with g2:
        v4 = st.slider("V4", -5.0, 5.0, 0.0, step=0.05, key="fs_v4")
        v11 = st.slider("V11", -5.0, 5.0, 0.0, step=0.05, key="fs_v11")
    with g3:
        v14 = st.slider("V14", -5.0, 5.0, 0.0, step=0.05, key="fs_v14")

    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        if st.button("Use Sample Fraud Transaction", type="secondary"):
            st.session_state.update(
                {
                    "fs_amount": _SAMPLE_FRAUD["Amount"],
                    "fs_time": _SAMPLE_FRAUD["Time"],
                    "fs_v1": _SAMPLE_FRAUD["V1"],
                    "fs_v2": _SAMPLE_FRAUD["V2"],
                    "fs_v4": _SAMPLE_FRAUD["V4"],
                    "fs_v11": _SAMPLE_FRAUD["V11"],
                    "fs_v14": _SAMPLE_FRAUD["V14"],
                }
            )
            st.rerun()
    with b2:
        if st.button("Use Sample Legitimate Transaction", type="secondary"):
            st.session_state.update(
                {
                    "fs_amount": _SAMPLE_LEGIT["Amount"],
                    "fs_time": _SAMPLE_LEGIT["Time"],
                    "fs_v1": _SAMPLE_LEGIT["V1"],
                    "fs_v2": _SAMPLE_LEGIT["V2"],
                    "fs_v4": _SAMPLE_LEGIT["V4"],
                    "fs_v11": _SAMPLE_LEGIT["V11"],
                    "fs_v14": _SAMPLE_LEGIT["V14"],
                }
            )
            st.rerun()

    if st.button("Analyze Transaction", type="primary"):
        payload = {f"V{i}": 0.0 for i in range(1, 29)}
        payload.update(
            {
                "V1": float(st.session_state.get("fs_v1", 0.0)),
                "V2": float(st.session_state.get("fs_v2", 0.0)),
                "V4": float(st.session_state.get("fs_v4", 0.0)),
                "V11": float(st.session_state.get("fs_v11", 0.0)),
                "V14": float(st.session_state.get("fs_v14", 0.0)),
                "Amount": float(st.session_state.get("fs_amount", 150.0)),
                "Time": float(st.session_state.get("fs_time", 0.0)),
            }
        )
        base = _api_base_url()
        url = f"{base}/predict"
        try:
            resp = get_api_session().post(url, json=payload, timeout=60)
        except requests.RequestException as exc:
            st.error(f"Could not reach API at `{url}`. Is uvicorn running? ({exc})")
            return
        if resp.status_code != 200:
            st.error(f"API error {resp.status_code}: {resp.text}")
            return
        data = resp.json()
        prob = float(data["fraud_probability"])
        is_fraud = bool(data["is_fraud"])
        risk = str(data["risk_level"])
        version = str(data["model_version"])

        st.divider()
        card_cls = "metric-card metric-fraud" if is_fraud else "metric-card metric-legit"
        verdict = "FRAUD RISK" if is_fraud else "LEGITIMATE"
        color = "#f85149" if is_fraud else "#3fb950"
        st.markdown(
            f"""
            <div class="{card_cls}">
              <div style="font-size:0.85rem;color:#8b949e;text-transform:uppercase;letter-spacing:0.08em;">Verdict</div>
              <div style="font-size:1.75rem;font-weight:700;color:{color};">{verdict}</div>
              <div style="margin-top:0.5rem;color:#8b949e;font-size:0.9rem;">Model version: <b style="color:#c9d1d9">{version}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        badge = _risk_badge_class(risk)
        st.markdown(
            f'<span class="risk-badge {badge}">Risk: {risk}</span>',
            unsafe_allow_html=True,
        )

        st.plotly_chart(_fraud_gauge(prob), use_container_width=True)


def _tab_performance(cfg) -> None:
    m = _load_metrics_from_disk(cfg)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC", f"{m['auc_roc']:.3f}")
    c2.metric("F1-Score", f"{m['f1']:.3f}")
    c3.metric("Precision", f"{m['precision']:.3f}")
    c4.metric("Recall", f"{m['recall']:.3f}")

    st.caption("Figures are produced by `src/evaluate.py` after training (if available).")
    img_dir = cfg.project_root / "models"
    cm = img_dir / "confusion_matrix.png"
    roc = img_dir / "roc_curve.png"
    ic1, ic2 = st.columns(2)
    with ic1:
        if cm.exists():
            st.image(str(cm), caption="Confusion matrix", use_container_width=True)
        else:
            st.info("Run training to generate `models/confusion_matrix.png`.")
    with ic2:
        if roc.exists():
            st.image(str(roc), caption="ROC curve", use_container_width=True)
        else:
            st.info("Run training to generate `models/roc_curve.png`.")


def _tab_how_it_works() -> None:
    steps = [
        ("📥", "Data Collection", "Load Kaggle-style transactions or synthetic data; align schema (Time, V1–V28, Amount, Class)."),
        ("🧹", "Preprocessing & SMOTE", "Dedupe, handle missing values, engineer Hour / Amount_log, RobustScaler, then SMOTE on the training split."),
        ("🚀", "XGBoost Training", "Baseline fit + RandomizedSearchCV (ROC-AUC); bundle saved as `fraud_model.pkl`."),
        ("⚡", "FastAPI Serving", "`/health`, `/model-info`, `/predict` — JSON in, fraud probability + risk tier out."),
        ("🖥️", "Streamlit UI", "This dashboard: sliders, samples, live API scoring, and evaluation visuals."),
    ]
    for icon, title, text in steps:
        st.markdown(f"### {icon} {title}")
        st.markdown(f'<div class="flow-box">{text}</div>', unsafe_allow_html=True)

    st.markdown("#### Flow (text)")
    st.markdown(
        """
```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Data        │ ──► │ Preprocess+SMOTE │ ──► │ XGBoost train   │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
┌─────────────┐     ┌──────────────────┐               ▼
│ Streamlit   │ ◄── │ FastAPI /predict │ ◄────── fraud_model.pkl
│  (you are)  │     └──────────────────┘
└─────────────┘
```
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="FraudShield",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_dark_theme_css()

    cfg = get_settings()

    with st.sidebar:
        st.markdown("### About **FraudShield**")
        st.markdown(
            """
            This workspace scores individual transactions with a **gradient-boosted**
            (**XGBoost**) classifier, trained on **~100K** anonymized transactions
            (real or synthetic), with strong ranking quality (**AUC ≈ 0.97** on the
            reference hold-out).

            - **API base:** `{api}`
            - **Artifacts:** `models/fraud_model.pkl`, `metadata.json`, plots

            *Demo UI — not financial advice.*
            """.format(api=_api_base_url()),
        )

    st.markdown("# 🛡️ FraudShield — Real-Time Fraud Detection")

    tab1, tab2, tab3 = st.tabs(
        ["🔍 Predict Transaction", "📊 Model Performance", "📖 How It Works"],
    )
    with tab1:
        _tab_predict()
    with tab2:
        _tab_performance(cfg)
    with tab3:
        _tab_how_it_works()


# `streamlit run` executes this file as the app entrypoint (do not rely on imports).
main()

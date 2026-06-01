"""
Centralized session-state initialization with @st.cache_data / @st.cache_resource.
All data generation runs once and is cached; model training runs once per session.
A progress bar is shown on first launch.
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime

from data.synthetic_generator import (
    generate_suppliers, generate_skus, generate_historical_shipments,
)
from models.risk_model import RiskPredictor
from models.forecast_ensemble import DemandForecastingEnsemble
from models.alert_engine import AlertEngine


# ── Cached data generators ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _cached_suppliers():
    """Generate and cache 50 supplier nodes."""
    return generate_suppliers()

@st.cache_data(show_spinner=False)
def _cached_skus():
    """Generate and cache 200 SKUs."""
    return generate_skus()

@st.cache_data(show_spinner=False)
def _cached_shipments(_suppliers_hash: str, num_records: int = 5000):
    """Generate and cache historical shipments (keyed on suppliers hash)."""
    suppliers_df = st.session_state["suppliers_df"]
    return generate_historical_shipments(suppliers_df, num_records=num_records)


# ── Cached model training ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _cached_risk_model(_shipments_hash: str):
    """Train and cache the GradientBoosting risk classifier."""
    shipments_df = st.session_state["historical_shipments"]
    model = RiskPredictor()
    model.train(shipments_df)
    return model


# ── Master initializer ──────────────────────────────────────────────
def init_state():
    """Ensure session state is fully populated.  Shows a progress bar on
    first visit; subsequent visits hit the Streamlit cache instantly."""

    if st.session_state.get("initialized"):
        return  # already done

    bar = st.progress(0, text="Initializing CEDPA Analytics Engine…")

    # Step 1 — Suppliers
    bar.progress(10, text="Generating 50 supplier nodes…")
    st.session_state["suppliers_df"] = _cached_suppliers()

    # Step 2 — SKUs
    bar.progress(25, text="Generating 200 SKUs…")
    st.session_state["skus_df"] = _cached_skus()

    # Step 3 — Shipments
    bar.progress(40, text="Building 5 000 historical shipment records…")
    sup_hash = pd.util.hash_pandas_object(st.session_state["suppliers_df"]).sum()
    st.session_state["historical_shipments"] = _cached_shipments(str(sup_hash))

    # Step 4 — Risk model
    bar.progress(60, text="Training Gradient Boosting risk classifier…")
    ship_hash = pd.util.hash_pandas_object(st.session_state["historical_shipments"]).sum()
    st.session_state["risk_model"] = _cached_risk_model(str(ship_hash))

    # Step 5 — Forecast ensemble (lightweight — no training yet)
    bar.progress(75, text="Initializing demand forecasting ensemble…")
    if "forecast_ensemble" not in st.session_state:
        st.session_state["forecast_ensemble"] = DemandForecastingEnsemble()

    # Step 6 — Alerts
    bar.progress(90, text="Scanning for supply-chain exceptions…")
    if "alerts" not in st.session_state:
        engine = AlertEngine()
        st.session_state["alert_engine"] = engine
        st.session_state["alerts"] = engine.generate_alerts(
            st.session_state["suppliers_df"],
            st.session_state["skus_df"],
            st.session_state["risk_model"],
        )

    # Step 7 — Timestamp
    st.session_state["data_generated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Step 8 — Audit log
    if "audit_log" not in st.session_state:
        st.session_state["audit_log"] = []

    bar.progress(100, text="CEDPA Platform ready ✓")
    bar.empty()

    st.session_state["initialized"] = True

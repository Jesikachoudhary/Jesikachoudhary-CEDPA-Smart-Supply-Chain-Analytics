"""
Page 11 — Real Data Loader
Upload real supply chain data, map it to CEDPA's 6 required features,
and retrain the Gradient Boosting risk classifier.

Tabs
----
1. Universal Smart Uploader  — any CSV / Excel
2. Olist Brazil Converter     — orders + items + sellers CSVs
3. Walmart Sales Converter    — train.csv (Store/Dept/Date/Weekly_Sales)
4. SCMS Shipment Converter    — USAID delivery history dataset
5. Custom Company Data        — template generator + status card
"""

import os
import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.theme import inject_css, render_sidebar_brand, get_colors, ACCENT, SUCCESS, DANGER, WARNING
from utils.state import init_state
from models.risk_model import RiskPredictor

# DataCo converter (bundled in data/ folder)
try:
    from data.dataco_converter import convert as _dataco_convert, DATACO_PATH
    DATACO_AVAILABLE = os.path.exists(DATACO_PATH)
except Exception:
    DATACO_AVAILABLE = False

# ─── Page bootstrap ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CEDPA — Real Data Loader",
    page_icon="📂",
    layout="wide",
)
inject_css()
init_state()
render_sidebar_brand()

# ─── Constants ─────────────────────────────────────────────────────────────────

CEDPA_FEATURES = [
    "lead_time_variance",
    "supplier_reliability",
    "geo_risk_index",
    "inventory_buffer",
    "shipment_delay_history",
    "disruption_risk",
]

FEATURE_LABELS = {
    "lead_time_variance":    "Lead Time Variance  (float, days²)",
    "supplier_reliability":  "Supplier Reliability  (float 0.0 – 1.0, on-time rate)",
    "geo_risk_index":        "Geographic Risk Index  (float 0.0 – 1.0)",
    "inventory_buffer":      "Inventory Buffer  (float −0.5 – 1.0, stock / safety ratio)",
    "shipment_delay_history":"Shipment Delay History  (float 0 – 10 scale)",
    "disruption_risk":       "Disruption Risk  (int 0 = safe, 1 = disrupted)",
}

FEATURE_RANGES = {
    "lead_time_variance":    (0.5,  7.0),
    "supplier_reliability":  (0.72, 0.98),
    "geo_risk_index":        (0.10, 0.45),
    "inventory_buffer":      (-0.10, 0.80),
    "shipment_delay_history":(0.5,  8.5),
}

# Auto-suggest keywords for column matching
KEYWORD_MAP: dict[str, list[str]] = {
    "lead_time_variance": [
        "lead_time_var", "lt_variance", "ltv", "lead_var",
        "delivery_var", "lead_time", "lt_std", "delay_var",
    ],
    "supplier_reliability": [
        "reliability", "on_time", "otd", "on_time_pct",
        "on_time_rate", "performance", "delivery_rate", "fill_rate",
    ],
    "geo_risk_index": [
        "geo_risk", "country_risk", "region_risk", "geo",
        "location_risk", "country_score", "geography",
    ],
    "inventory_buffer": [
        "inventory_buffer", "inv_buffer", "stock_buffer",
        "safety_buffer", "buffer", "inv_ratio",
    ],
    "shipment_delay_history": [
        "delay_hist", "delay_history", "avg_delay",
        "shipment_delay", "delay_index", "delay_avg",
    ],
    "disruption_risk": [
        "disruption", "disruption_risk", "risk", "label",
        "target", "disrupted", "incident", "failure", "is_disrupted",
    ],
}

NO_MAP = "── Not in my data ──"

# Country → geographic risk lookup (30 countries)
COUNTRY_RISK: dict[str, float] = {
    "usa": 0.18, "united states": 0.18, "us": 0.18, "u.s.": 0.18,
    "canada": 0.13, "mexico": 0.30,
    "germany": 0.11, "uk": 0.13, "united kingdom": 0.13,
    "france": 0.14, "netherlands": 0.12, "spain": 0.16,
    "italy": 0.18, "poland": 0.17, "switzerland": 0.10,
    "sweden": 0.11, "denmark": 0.11, "norway": 0.12,
    "china": 0.22, "japan": 0.15, "south korea": 0.16, "korea": 0.16,
    "india": 0.25, "indonesia": 0.28, "vietnam": 0.28,
    "thailand": 0.22, "malaysia": 0.20, "bangladesh": 0.38,
    "pakistan": 0.40, "philippines": 0.30,
    "singapore": 0.11, "australia": 0.10, "new zealand": 0.10,
    "brazil": 0.32, "argentina": 0.35, "colombia": 0.35,
    "peru": 0.37, "chile": 0.25,
    "nigeria": 0.48, "kenya": 0.38, "ethiopia": 0.42,
    "ghana": 0.36, "south africa": 0.30, "egypt": 0.35,
    "tanzania": 0.40, "rwanda": 0.36,
    "russia": 0.38, "ukraine": 0.45, "turkey": 0.28,
}

# Brazilian state → geographic risk
BRAZIL_STATE_RISK: dict[str, float] = {
    "SP": 0.18, "RJ": 0.22, "MG": 0.20, "BA": 0.28, "RS": 0.20,
    "PR": 0.19, "PE": 0.30, "CE": 0.30, "PA": 0.38, "AM": 0.40,
    "SC": 0.18, "GO": 0.22, "ES": 0.21, "PB": 0.30, "RN": 0.29,
    "MA": 0.35, "MT": 0.27, "MS": 0.25, "PI": 0.33, "AL": 0.32,
    "SE": 0.31, "TO": 0.30, "RO": 0.32, "AC": 0.38, "AP": 0.36,
    "RR": 0.38, "DF": 0.15,
}


# ─── Helper functions ──────────────────────────────────────────────────────────

def auto_suggest(columns: list[str], feature: str) -> str | None:
    """Return the best-matching column for a CEDPA feature (keyword scoring)."""
    keywords = KEYWORD_MAP.get(feature, [])
    col_lower = [c.lower() for c in columns]

    # 1. Exact keyword match
    for kw in keywords:
        for i, cl in enumerate(col_lower):
            if kw == cl:
                return columns[i]

    # 2. Keyword is substring of column name
    for kw in keywords:
        for i, cl in enumerate(col_lower):
            if kw in cl:
                return columns[i]

    # 3. Column name is substring of keyword
    for kw in keywords:
        for i, cl in enumerate(col_lower):
            if cl in kw and len(cl) > 3:
                return columns[i]

    return None


def compute_disruption_risk(df: pd.DataFrame) -> pd.Series:
    """Compute the binary disruption label from the CEDPA sigmoid formula."""
    score = (
        (1.0 - df["supplier_reliability"]) * 4.5
        + df["geo_risk_index"] * 3.5
        + (df["lead_time_variance"] / 8.0) * 2.8
        - df["inventory_buffer"] * 2.0
        + (df["shipment_delay_history"] / 10.0) * 3.2
    )
    prob = 1.0 / (1.0 + np.exp(-(score - 4.5)))
    return (prob > 0.5).astype(int)


def show_metric_cards(df: pd.DataFrame) -> None:
    """Render 4 glassmorphic KPI cards for a mapped dataframe."""
    n_rows  = len(df)
    n_cols  = len(df.columns)
    n_disrp = int(df["disruption_risk"].sum()) if "disruption_risk" in df.columns else 0
    missing = df.isnull().sum().sum()
    miss_pc = missing / max(n_rows * n_cols, 1) * 100

    for col, label, val, color in zip(
        st.columns(4),
        ["Mapped Rows", "Feature Columns", "Disruptions (label=1)", "Missing Values"],
        [f"{n_rows:,}", f"{n_cols}", f"{n_disrp:,}", f"{miss_pc:.1f}%"],
        [ACCENT, "#8B5CF6", DANGER, WARNING],
    ):
        with col:
            st.markdown(f"""
<div class="glass-card" style="border-top:4px solid {color};text-align:center;padding:16px">
  <div style="font-size:.75rem;color:var(--muted);text-transform:uppercase;font-weight:700">{label}</div>
  <div style="font-size:2rem;font-weight:700;color:{color};
       font-family:'Outfit',sans-serif;margin-top:4px">{val}</div>
</div>""", unsafe_allow_html=True)


def show_feature_histograms(df: pd.DataFrame) -> None:
    """Render compact Plotly histograms for the 5 non-label feature columns."""
    c = get_colors()
    feats = [f for f in CEDPA_FEATURES[:-1] if f in df.columns]
    if not feats:
        return

    for col, feat in zip(st.columns(len(feats)), feats):
        with col:
            fig = go.Figure(go.Histogram(
                x=df[feat].dropna(), nbinsx=25,
                marker=dict(color=ACCENT, opacity=0.85),
            ))
            fig.update_layout(
                title=dict(
                    text=feat.replace("_", " ").title(),
                    font=dict(size=10, color=c["text"]),
                ),
                height=160,
                margin=dict(l=2, r=2, t=28, b=2),
                paper_bgcolor=c["plot_bg"],
                plot_bgcolor=c["plot_bg"],
                showlegend=False,
                xaxis=dict(tickfont=dict(size=7, color=c["muted"]),
                           gridcolor=c["grid"], showgrid=True),
                yaxis=dict(tickfont=dict(size=7, color=c["muted"]),
                           gridcolor=c["grid"], showgrid=True),
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})


def validate_and_retrain(df: pd.DataFrame, source_name: str) -> bool:
    """
    Validate the 6-column dataframe, retrain the GradientBoosting risk model,
    regenerate alerts, and update all relevant session state keys.
    Returns True on success.
    """
    errors: list[str] = []

    # ── Checks ────────────────────────────────────────────────────────
    if len(df) < 50:
        errors.append(f"Need ≥ 50 rows to train. Got **{len(df)}**.")

    for f in CEDPA_FEATURES:
        if f not in df.columns:
            errors.append(f"Missing required column: **{f}**")

    if "supplier_reliability" in df.columns:
        r = df["supplier_reliability"]
        if r.min() < 0 or r.max() > 1:
            errors.append(
                "**supplier_reliability** must be in [0, 1]. "
                f"Found range [{r.min():.3f}, {r.max():.3f}]."
            )

    if "disruption_risk" in df.columns:
        classes = set(df["disruption_risk"].unique())
        if not {0, 1}.issubset(classes):
            errors.append(
                "**disruption_risk** must contain both 0 and 1. "
                f"Found only: {classes}"
            )

    if errors:
        for e in errors:
            st.error(e)
        return False

    # ── Retrain ───────────────────────────────────────────────────────
    prog = st.progress(0, text="Preparing data…")
    try:
        clean = df[CEDPA_FEATURES].dropna().reset_index(drop=True)

        prog.progress(20, text=f"Training on {len(clean):,} records…")
        model = RiskPredictor()
        model.train(clean)

        prog.progress(60, text="Regenerating supply chain alerts…")
        engine = st.session_state.get("alert_engine")
        if engine is None:
            from models.alert_engine import AlertEngine
            engine = AlertEngine()
            st.session_state["alert_engine"] = engine

        alerts = engine.generate_alerts(
            st.session_state["suppliers_df"],
            st.session_state["skus_df"],
            model,
        )

        prog.progress(85, text="Updating platform state…")
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.update({
            "risk_model":           model,
            "historical_shipments": clean,
            "alerts":               alerts,
            "using_real_data":      True,
            "real_data_source":     source_name,
            "data_generated_at":    now_str,
        })
        st.session_state.setdefault("audit_log", []).append({
            "timestamp": now_str,
            "action":    "model_retrained",
            "detail": (
                f"Source: {source_name} | Rows: {len(clean):,} | "
                f"Accuracy: {model.metrics['accuracy']*100:.1f}% | "
                f"ROC-AUC: {model.metrics['roc_auc']:.3f}"
            ),
        })

        prog.progress(100, text="Done ✓")
        prog.empty()
        return True

    except Exception as exc:
        prog.empty()
        st.error(f"Training failed: {exc}")
        return False


def _download_csv_button(df: pd.DataFrame, filename: str, label: str = "📥 Download Mapped CSV"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def _retrain_section(df: pd.DataFrame, source_name: str, key_suffix: str) -> None:
    """Shared retrain + download strip used by all converter tabs."""
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        _download_csv_button(df, f"cedpa_{key_suffix}.csv")
    with c2:
        if st.button(
            "🔁 Retrain CEDPA Model",
            type="primary",
            key=f"retrain_{key_suffix}",
            use_container_width=True,
        ):
            if validate_and_retrain(df, source_name):
                st.balloons()
                acc = st.session_state["risk_model"].metrics["accuracy"] * 100
                st.success(
                    f"✅ Model retrained on **{len(df):,} rows** from {source_name}!  "
                    f"New accuracy: **{acc:.1f}%**"
                )


def _show_data_source_status() -> None:
    """Render the current data source status card with reset button."""
    using_real = st.session_state.get("using_real_data", False)
    source     = st.session_state.get("real_data_source", "Synthetic 50-node simulation")
    model      = st.session_state.get("risk_model")
    acc_str    = f"{model.metrics['accuracy']*100:.1f}%" if model else "—"
    hist       = st.session_state.get("historical_shipments")
    row_count  = len(hist) if hist is not None else 0
    label      = "Real Data" if using_real else "Synthetic Data"
    color      = SUCCESS if using_real else "#8B5CF6"
    icon       = "📊" if using_real else "🔬"

    st.markdown(f"""
<div class="glass-card" style="border-left:4px solid {color}">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <div style="font-size:.75rem;color:var(--muted);text-transform:uppercase;font-weight:700">
        Current Data Source</div>
      <div style="font-size:1.2rem;font-weight:700;color:{color};margin-top:2px">{label}</div>
      <div style="font-size:.85rem;color:var(--text);margin-top:6px;line-height:1.6">
        Source: <b>{source}</b><br/>
        Training rows: <b>{row_count:,}</b>&nbsp;|&nbsp;Model accuracy: <b>{acc_str}</b>
      </div>
    </div>
    <div style="font-size:3rem;opacity:.7">{icon}</div>
  </div>
</div>""", unsafe_allow_html=True)

    if using_real:
        if st.button("🔄 Reset to Synthetic Data", key="reset_synthetic"):
            _reset_to_synthetic()


def _reset_to_synthetic() -> None:
    """Re-generate synthetic shipments and retrain the model from scratch."""
    from data.synthetic_generator import generate_historical_shipments

    prog = st.progress(0, text="Restoring synthetic supply chain data…")
    suppliers_df = st.session_state["suppliers_df"]
    skus_df      = st.session_state["skus_df"]

    prog.progress(30, text="Generating 5,000 synthetic shipment records…")
    hist = generate_historical_shipments(suppliers_df, num_records=5000)

    prog.progress(60, text="Retraining Gradient Boosting on synthetic data…")
    model = RiskPredictor()
    model.train(hist)

    prog.progress(85, text="Regenerating alert queue…")
    engine = st.session_state.get("alert_engine")
    if engine is None:
        from models.alert_engine import AlertEngine
        engine = AlertEngine()
        st.session_state["alert_engine"] = engine

    alerts = engine.generate_alerts(suppliers_df, skus_df, model)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state.update({
        "historical_shipments": hist,
        "risk_model":           model,
        "alerts":               alerts,
        "using_real_data":      False,
        "real_data_source":     "Synthetic 50-node simulation",
        "data_generated_at":    now_str,
    })
    st.session_state.setdefault("audit_log", []).append({
        "timestamp": now_str,
        "action":    "reset_to_synthetic",
        "detail":    f"Retrained on {len(hist):,} synthetic rows.",
    })

    prog.progress(100, text="Synthetic data restored ✓")
    prog.empty()
    st.success("✅ Reset complete. Model retrained on 5,000 synthetic shipments.")
    st.rerun()


# ─── Page header ───────────────────────────────────────────────────────────────

st.markdown(
    '<div class="page-header">'
    '<h1>Real Data Loader</h1>'
    '<p>Upload real supply chain data to replace the synthetic simulation and retrain the CEDPA risk classifier.</p>'
    '</div>',
    unsafe_allow_html=True,
)

using_real = st.session_state.get("using_real_data", False)
if using_real:
    src = st.session_state.get("real_data_source", "")
    acc = st.session_state["risk_model"].metrics["accuracy"] * 100
    st.success(
        f"✅ **Currently using real data** — Source: {src} | "
        f"Model accuracy: {acc:.1f}%"
    )
else:
    st.info("ℹ️ Platform is running on **synthetic** data. Upload a real dataset below to retrain.")

st.markdown("---")

# ─── Tabs ──────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════
# DATACO AUTO-LOADER  (shown only when file exists in data/ folder)
# ══════════════════════════════════════════════════════════════════════
if DATACO_AVAILABLE:
    st.markdown("""
<div class="glass-card" style="border-left:4px solid #10B981;">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <div style="font-size:.75rem;color:var(--muted);text-transform:uppercase;
           font-weight:700;letter-spacing:.05em">Dataset detected in data/ folder</div>
      <div style="font-size:1.15rem;font-weight:700;color:#10B981;margin-top:2px">
        📊 DataCoSupplyChainDataset.csv &nbsp;·&nbsp;
        <span style="color:var(--text);font-size:.9rem">180,519 rows &nbsp;·&nbsp; 53 columns</span>
      </div>
      <div style="font-size:.82rem;color:var(--muted);margin-top:4px">
        All 6 CEDPA features auto-mapped — no manual column matching needed.
      </div>
    </div>
    <div style="font-size:2.8rem;opacity:.7">🗄️</div>
  </div>
</div>""", unsafe_allow_html=True)

    dc_col1, dc_col2, dc_col3 = st.columns([2, 2, 3])

    with dc_col1:
        load_all = st.button(
            "⚡ Load All 180K Rows & Retrain",
            type="primary",
            use_container_width=True,
            key="dc_load_all",
        )

    with dc_col2:
        load_sample = st.button(
            "🔬 Load 10K Sample & Retrain",
            use_container_width=True,
            key="dc_load_sample",
        )

    with dc_col3:
        st.caption(
            "💡 **Full load** gives maximum accuracy (~55% disruption class balance).  "
            "**Sample** is faster — ideal for testing."
        )

    sample_n = None
    trigger  = False
    if load_all:
        sample_n, trigger = None, True
    elif load_sample:
        sample_n, trigger = 10_000, True

    if trigger:
        with st.spinner("Converting DataCo columns → CEDPA features…"):
            try:
                dc_df = _dataco_convert(DATACO_PATH, sample_n=sample_n)
                st.session_state["dataco_df"] = dc_df
            except Exception as exc:
                st.error(f"Conversion failed: {exc}")
                dc_df = None

        if dc_df is not None:
            st.success(
                f"✅ Converted **{len(dc_df):,} rows** from DataCo dataset. "
                "Preview and column stats below."
            )

    if "dataco_df" in st.session_state:
        dc_df = st.session_state["dataco_df"]

        with st.expander("📄 Mapped data preview (first 20 rows)", expanded=False):
            st.dataframe(dc_df.head(20), use_container_width=True)

        # ── Column mapping explanation ─────────────────────────────
        with st.expander("🗺️ How DataCo columns were mapped to CEDPA features", expanded=False):
            st.markdown("""
| CEDPA Feature | DataCo Source Column | Transformation |
|---|---|---|
| `lead_time_variance` | `Days for shipping (real)` − `Days for shipment (scheduled)` | Squared deviation, clipped [0.2, 9.0] |
| `supplier_reliability` | `Days for shipment (scheduled)` / `Days for shipping (real)` | Ratio clipped [0.55, 1.0] |
| `geo_risk_index` | `Order Country` | 60-country Spanish + English risk lookup table |
| `inventory_buffer` | `Order Item Profit Ratio` | Min-max scaled −2.75→+0.5 to [−0.5, 0.8] |
| `shipment_delay_history` | `Days for shipping (real)` − `Days for shipment (scheduled)` | Delay days (≥0) scaled to 0–10 |
| `disruption_risk` | `Late_delivery_risk` | Already 0/1 — no transformation needed ✅ |
""")

        # ── Metrics ────────────────────────────────────────────────
        show_metric_cards(dc_df)

        # ── Distributions ──────────────────────────────────────────
        st.markdown("#### Feature Distributions")
        show_feature_histograms(dc_df)

        # ── Retrain ────────────────────────────────────────────────
        _retrain_section(dc_df, "DataCo Supply Chain Dataset", "dataco")

    st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Universal Uploader",
    "🇧🇷 Olist Brazil",
    "🛒 Walmart Sales",
    "✈️ SCMS Shipments",
    "🏢 Custom Company",
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — UNIVERSAL SMART UPLOADER
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Universal Smart Column Mapper")
    st.markdown(
        "Upload any CSV or Excel file. The system auto-suggests column mappings "
        "using keyword matching. Override any suggestion, then choose how to "
        "derive features that are not present in your data."
    )

    uploaded = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        key="t1_upload",
    )

    if uploaded is not None:
        # ── Clear cached result when a new file is uploaded ────────────
        if st.session_state.get("t1_prev_file") != uploaded.name:
            st.session_state.pop("t1_mapped", None)
            st.session_state["t1_prev_file"] = uploaded.name

        # ── Load file ──────────────────────────────────────────────────
        try:
            if uploaded.name.lower().endswith((".xlsx", ".xls")):
                xl      = pd.ExcelFile(uploaded)
                sheet   = st.selectbox("Select worksheet", xl.sheet_names, key="t1_sheet")
                raw_df  = pd.read_excel(uploaded, sheet_name=sheet)
            else:
                raw_df  = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read file: {exc}")
            raw_df = None

        if raw_df is not None:
            all_cols      = list(raw_df.columns)
            cols_with_none = [NO_MAP] + all_cols
            numeric_cols  = raw_df.select_dtypes(include="number").columns.tolist()
            str_cols      = raw_df.select_dtypes(include="object").columns.tolist()

            st.success(f"Loaded **{len(raw_df):,} rows × {len(all_cols)} columns**")
            with st.expander("👁 Preview raw data (first 10 rows)", expanded=False):
                st.dataframe(raw_df.head(10), use_container_width=True)

            # ── Column mapping UI ──────────────────────────────────────
            st.markdown("---")
            st.markdown("#### Column Mapping")
            st.markdown(
                "For each of the 6 required CEDPA features, select the matching "
                "column from your data. Auto-suggestions are pre-selected. "
                "If a feature is missing, choose how to derive it."
            )

            for feat in CEDPA_FEATURES:
                suggestion  = auto_suggest(all_cols, feat)
                default_idx = cols_with_none.index(suggestion) if suggestion else 0

                mc1, mc2 = st.columns([3, 4])
                with mc1:
                    selected = st.selectbox(
                        f"**{FEATURE_LABELS[feat]}**",
                        cols_with_none,
                        index=default_idx,
                        key=f"t1_map_{feat}",
                    )

                with mc2:
                    if selected == NO_MAP:
                        lo, hi = FEATURE_RANGES.get(feat, (0.0, 1.0))
                        rand_label = f"Use random values ({lo} – {hi})"

                        if feat == "lead_time_variance":
                            method = st.radio(
                                "Derive lead_time_variance from:",
                                [rand_label, "Scale a lead-time column to 0.5 – 7.0"],
                                horizontal=True, key=f"t1_dm_{feat}",
                            )
                            if "Scale" in method and numeric_cols:
                                st.selectbox(
                                    "Lead-time column", numeric_cols,
                                    key=f"t1_dc_{feat}",
                                )

                        elif feat == "supplier_reliability":
                            method = st.radio(
                                "Derive supplier_reliability from:",
                                [rand_label, "Normalize a 0 – 100 reliability column"],
                                horizontal=True, key=f"t1_dm_{feat}",
                            )
                            if "Normalize" in method and numeric_cols:
                                st.selectbox(
                                    "Reliability column (any scale)", numeric_cols,
                                    key=f"t1_dc_{feat}",
                                )

                        elif feat == "geo_risk_index":
                            method = st.radio(
                                "Derive geo_risk_index from:",
                                [rand_label, "Map a country / region column → risk scores"],
                                horizontal=True, key=f"t1_dm_{feat}",
                            )
                            if "Map" in method:
                                country_pool = str_cols or all_cols
                                st.selectbox(
                                    "Country column", country_pool,
                                    key=f"t1_dc_{feat}",
                                )

                        elif feat == "inventory_buffer":
                            method = st.radio(
                                "Derive inventory_buffer from:",
                                [rand_label,
                                 "Compute (current_stock − safety_stock) / safety_stock"],
                                horizontal=True, key=f"t1_dm_{feat}",
                            )
                            if "Compute" in method and len(numeric_cols) >= 2:
                                bc1, bc2 = st.columns(2)
                                with bc1:
                                    st.selectbox(
                                        "Current stock column", numeric_cols,
                                        key=f"t1_dstock_{feat}",
                                    )
                                with bc2:
                                    st.selectbox(
                                        "Safety stock column", numeric_cols,
                                        key=f"t1_dsafety_{feat}",
                                    )

                        elif feat == "shipment_delay_history":
                            method = st.radio(
                                "Derive shipment_delay_history from:",
                                [rand_label, "Scale any delay column to 0 – 10"],
                                horizontal=True, key=f"t1_dm_{feat}",
                            )
                            if "Scale" in method and numeric_cols:
                                st.selectbox(
                                    "Delay column", numeric_cols,
                                    key=f"t1_dc_{feat}",
                                )

                        elif feat == "disruption_risk":
                            method = st.radio(
                                "Derive disruption_risk from:",
                                ["Compute from CEDPA formula (recommended)",
                                 "Use an existing binary (0 / 1) column"],
                                horizontal=True, key=f"t1_dm_{feat}",
                            )
                            if "existing" in method.lower():
                                st.selectbox(
                                    "Binary column", all_cols,
                                    key=f"t1_dc_{feat}",
                                )

            # ── Apply mapping ──────────────────────────────────────────
            st.markdown("---")
            if st.button("✅ Apply Mapping & Preview", type="primary", key="t1_apply"):
                rng_m = np.random.default_rng(42)
                n     = len(raw_df)
                result: dict[str, np.ndarray] = {}

                for feat in CEDPA_FEATURES[:-1]:
                    sel = st.session_state.get(f"t1_map_{feat}", NO_MAP)
                    lo, hi = FEATURE_RANGES[feat]

                    if sel != NO_MAP:
                        # ── Direct mapping ─────────────────────────
                        vals = pd.to_numeric(raw_df[sel], errors="coerce")
                        if feat == "supplier_reliability" and vals.max() > 1.5:
                            vals = vals / vals.max()
                        clips = {
                            "supplier_reliability":  (0.0, 1.0),
                            "inventory_buffer":      (-0.5, 1.0),
                            "shipment_delay_history":(0.0, 10.0),
                            "lead_time_variance":    (0.0, 50.0),
                            "geo_risk_index":        (0.0, 1.0),
                        }
                        if feat in clips:
                            vals = vals.clip(*clips[feat])
                        fallback = rng_m.uniform(lo, hi, n)
                        result[feat] = vals.fillna(pd.Series(fallback)).values

                    else:
                        # ── Derivation ─────────────────────────────
                        method = st.session_state.get(f"t1_dm_{feat}", "")

                        if feat == "lead_time_variance":
                            col = st.session_state.get(f"t1_dc_{feat}")
                            if "Scale" in method and col:
                                v = pd.to_numeric(raw_df[col], errors="coerce")
                                mx = v.max()
                                scaled = (v / mx * 6.5 + 0.5).clip(0.5, 9.0) if mx > 0 else v
                                result[feat] = scaled.fillna(rng_m.uniform(lo, hi)).values
                            else:
                                result[feat] = rng_m.uniform(lo, hi, n)

                        elif feat == "supplier_reliability":
                            col = st.session_state.get(f"t1_dc_{feat}")
                            if "Normalize" in method and col:
                                v = pd.to_numeric(raw_df[col], errors="coerce")
                                mx = v.max()
                                norm = (v / mx).clip(0.0, 1.0) if mx > 0 else v
                                result[feat] = norm.fillna(rng_m.uniform(lo, hi)).values
                            else:
                                result[feat] = rng_m.uniform(lo, hi, n)

                        elif feat == "geo_risk_index":
                            col = st.session_state.get(f"t1_dc_{feat}")
                            if "Map" in method and col:
                                mapped_vals = (
                                    raw_df[col].astype(str).str.strip().str.lower()
                                    .map(COUNTRY_RISK)
                                    .fillna(0.25)
                                )
                                result[feat] = mapped_vals.values
                            else:
                                result[feat] = rng_m.uniform(lo, hi, n)

                        elif feat == "inventory_buffer":
                            stock_col  = st.session_state.get(f"t1_dstock_{feat}")
                            safety_col = st.session_state.get(f"t1_dsafety_{feat}")
                            if "Compute" in method and stock_col and safety_col:
                                curr   = pd.to_numeric(raw_df[stock_col], errors="coerce")
                                safety = pd.to_numeric(raw_df[safety_col], errors="coerce").replace(0, 1)
                                buf    = ((curr - safety) / safety).clip(-0.5, 0.8)
                                result[feat] = buf.fillna(rng_m.uniform(lo, hi)).values
                            else:
                                result[feat] = rng_m.uniform(lo, hi, n)

                        elif feat == "shipment_delay_history":
                            col = st.session_state.get(f"t1_dc_{feat}")
                            if "Scale" in method and col:
                                v = pd.to_numeric(raw_df[col], errors="coerce")
                                mx = v.abs().max()
                                scaled = (v.clip(lower=0) / mx * 10).clip(0, 10) if mx > 0 else v
                                result[feat] = scaled.fillna(rng_m.uniform(lo, hi)).values
                            else:
                                result[feat] = rng_m.uniform(lo, hi, n)

                # Build temp dataframe to compute disruption label
                temp = pd.DataFrame(result)

                # Disruption risk column
                sel_dr = st.session_state.get("t1_map_disruption_risk", NO_MAP)
                if sel_dr != NO_MAP:
                    temp["disruption_risk"] = pd.to_numeric(
                        raw_df[sel_dr], errors="coerce"
                    ).fillna(0).astype(int).values
                else:
                    dr_method = st.session_state.get("t1_dm_disruption_risk", "")
                    col = st.session_state.get("t1_dc_disruption_risk")
                    if "existing" in dr_method.lower() and col:
                        temp["disruption_risk"] = pd.to_numeric(
                            raw_df[col], errors="coerce"
                        ).fillna(0).astype(int).values
                    else:
                        temp["disruption_risk"] = compute_disruption_risk(temp).values

                st.session_state["t1_mapped"] = temp
                st.success(f"Mapping applied — **{len(temp):,} rows** ready for retraining.")

            # ── Show results ───────────────────────────────────────────
            if "t1_mapped" in st.session_state:
                mapped = st.session_state["t1_mapped"]

                st.markdown("#### Mapped Data Preview")
                st.dataframe(mapped.head(20), use_container_width=True)

                st.markdown("#### Data Quality Metrics")
                show_metric_cards(mapped)

                st.markdown("#### Feature Distributions")
                show_feature_histograms(mapped)

                _retrain_section(mapped, f"Universal Upload: {uploaded.name}", "tab1")
    else:
        st.info(
            "Upload a CSV or Excel file above to begin. "
            "Keyword-based auto-mapping works best when your column names "
            "contain terms like *reliability*, *delay*, *lead_time*, *on_time*, *geo_risk*, etc."
        )


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — OLIST BRAZIL CONVERTER
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Olist Brazil E-Commerce Converter")
    st.markdown(
        "Upload three files from the "
        "[Olist public dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). "
        "The converter joins them, aggregates per seller, and derives all 6 CEDPA features."
    )

    with st.expander("📋 Expected columns per file", expanded=False):
        st.markdown("""
| File | Key columns needed |
|---|---|
| `olist_orders_dataset.csv` | order_id · order_purchase_timestamp · order_delivered_customer_date · order_estimated_delivery_date |
| `olist_order_items_dataset.csv` | order_id · seller_id |
| `olist_sellers_dataset.csv` | seller_id · seller_state |
""")

    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        orders_file = st.file_uploader(
            "olist_orders_dataset.csv", type=["csv"], key="t2_orders"
        )
    with oc2:
        items_file = st.file_uploader(
            "olist_order_items_dataset.csv", type=["csv"], key="t2_items"
        )
    with oc3:
        sellers_file = st.file_uploader(
            "olist_sellers_dataset.csv", type=["csv"], key="t2_sellers"
        )

    if orders_file and items_file and sellers_file:
        try:
            with st.spinner("Joining and transforming Olist datasets…"):
                orders  = pd.read_csv(orders_file)
                items   = pd.read_csv(items_file)
                sellers = pd.read_csv(sellers_file)

                # Merge items ← orders ← sellers
                merged = (
                    items[["order_id", "seller_id"]]
                    .merge(
                        orders[[
                            "order_id",
                            "order_purchase_timestamp",
                            "order_delivered_customer_date",
                            "order_estimated_delivery_date",
                        ]],
                        on="order_id",
                        how="left",
                    )
                    .merge(
                        sellers[["seller_id", "seller_state"]],
                        on="seller_id",
                        how="left",
                    )
                )

                # Parse dates
                for col in [
                    "order_purchase_timestamp",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                ]:
                    merged[col] = pd.to_datetime(merged[col], errors="coerce")

                merged["lead_time_days"] = (
                    merged["order_delivered_customer_date"]
                    - merged["order_purchase_timestamp"]
                ).dt.days
                merged["on_time"] = (
                    merged["order_delivered_customer_date"]
                    <= merged["order_estimated_delivery_date"]
                ).astype(float)

                # Aggregate per seller
                agg = (
                    merged.groupby("seller_id")
                    .agg(
                        lead_time_std =("lead_time_days",  "std"),
                        on_time_rate  =("on_time",          "mean"),
                        order_count   =("order_id",         "count"),
                        seller_state  =("seller_state",     "first"),
                    )
                    .reset_index()
                )

                max_orders = max(agg["order_count"].max(), 1)

                agg["lead_time_variance"]    = (agg["lead_time_std"].fillna(1.0) ** 2).clip(0.2, 9.0)
                agg["supplier_reliability"]  = agg["on_time_rate"].fillna(0.80).clip(0.0, 1.0)
                agg["geo_risk_index"]        = (
                    agg["seller_state"]
                    .map(BRAZIL_STATE_RISK)
                    .fillna(0.28)
                )
                agg["inventory_buffer"]      = (
                    (agg["order_count"] / max_orders - 0.5)
                    .clip(-0.5, 0.8)
                )
                agg["shipment_delay_history"] = (
                    (1 - agg["supplier_reliability"]) * 10
                ).clip(0.0, 10.0)

                agg["disruption_risk"] = compute_disruption_risk(agg).values
                result_df = agg[CEDPA_FEATURES].copy()

            st.success(f"Converted **{len(result_df):,} sellers** → CEDPA feature set.")

            with st.expander("📄 Mapped data preview", expanded=True):
                st.dataframe(result_df.head(20), use_container_width=True)

            show_metric_cards(result_df)
            st.markdown("#### Feature Distributions")
            show_feature_histograms(result_df)
            _retrain_section(result_df, "Olist Brazil Dataset", "tab2")

        except Exception as exc:
            st.error(f"Processing failed: {exc}")
    else:
        st.info("Upload all three Olist CSV files to proceed.")


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — WALMART SALES CONVERTER
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Walmart Store Sales Converter")
    st.markdown(
        "Upload **train.csv** from the "
        "[Walmart Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting) "
        "Kaggle dataset. The converter aggregates per Store × Dept combination."
    )

    with st.expander("📋 Expected columns", expanded=False):
        st.markdown(
            "| Column | Type |\n|---|---|\n"
            "| Store | int |\n| Dept | int |\n| Date | date |\n"
            "| Weekly_Sales | float |\n| IsHoliday | bool |"
        )

    walmart_file = st.file_uploader(
        "train.csv (Store / Dept / Date / Weekly_Sales / IsHoliday)",
        type=["csv"],
        key="t3_walmart",
    )

    if walmart_file:
        try:
            with st.spinner("Aggregating Walmart store data…"):
                train = pd.read_csv(walmart_file)

                # Validate required columns
                needed = {"Store", "Dept", "Weekly_Sales", "IsHoliday"}
                missing_cols = needed - set(train.columns)
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                    st.stop()

                agg = (
                    train.groupby(["Store", "Dept"])
                    .agg(
                        mean_sales    =("Weekly_Sales", "mean"),
                        std_sales     =("Weekly_Sales", "std"),
                        median_sales  =("Weekly_Sales", "median"),
                        neg_weeks     =("Weekly_Sales", lambda x: (x < 0).sum()),
                        total_weeks   =("Weekly_Sales", "count"),
                        holiday_weeks =("IsHoliday",   "sum"),
                    )
                    .reset_index()
                )

                max_store = max(agg["Store"].max(), 1)
                std_safe  = agg["std_sales"].replace(0, 1.0)
                cv        = (std_safe / (agg["mean_sales"].abs() + 1e-9)).clip(0, 5)
                h_ratio   = (agg["holiday_weeks"] / agg["total_weeks"]).clip(0, 1)

                agg["lead_time_variance"]    = (cv ** 2 * 5).clip(0.2, 9.0)
                agg["supplier_reliability"]  = (
                    1 - (agg["neg_weeks"] / agg["total_weeks"])
                ).clip(0.6, 0.99)
                agg["geo_risk_index"]        = (
                    agg["Store"] / max_store * 0.30 + 0.10
                ).clip(0.10, 0.40)
                agg["inventory_buffer"]      = (
                    (agg["mean_sales"] - agg["median_sales"]) / std_safe
                ).clip(-0.5, 0.8)
                agg["shipment_delay_history"] = (cv * 6 + h_ratio * 2).clip(0.0, 10.0)
                agg["disruption_risk"]        = compute_disruption_risk(agg).values

                result_df = agg[CEDPA_FEATURES].copy()

            st.success(
                f"Converted **{len(result_df):,} Store × Dept combinations** → CEDPA feature set."
            )

            with st.expander("📄 Mapped data preview", expanded=True):
                st.dataframe(result_df.head(20), use_container_width=True)

            show_metric_cards(result_df)
            st.markdown("#### Feature Distributions")
            show_feature_histograms(result_df)
            _retrain_section(result_df, "Walmart Store Sales Dataset", "tab3")

        except Exception as exc:
            st.error(f"Processing failed: {exc}")
    else:
        st.info("Upload Walmart's **train.csv** to proceed.")


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — SUPPLY CHAIN SHIPMENT CONVERTER (SCMS / USAID)
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### USAID SCMS Delivery History Converter")
    st.markdown(
        "Upload **SCMS_Delivery_History_Dataset.csv** from the "
        "[Supply Chain Shipment Pricing Dataset](https://www.kaggle.com/datasets/divyeshardeshana/supply-chain-shipment-pricing-data). "
        "The converter aggregates per vendor / manufacturing site."
    )

    with st.expander("📋 Key columns auto-detected", expanded=False):
        st.markdown("""
The converter automatically detects these columns (case-insensitive):

| Role | Column name examples |
|---|---|
| Grouping key | `Vendor`, `Manufacturing Site` |
| Scheduled date | `Scheduled Delivery Date`, `PO Sent to Vendor Date` |
| Actual date | `Delivered to Client Date`, `Delivery Recorded Date` |
| Country | `Country` |
""")

    scms_file = st.file_uploader(
        "SCMS_Delivery_History_Dataset.csv",
        type=["csv"],
        key="t4_scms",
    )

    if scms_file:
        try:
            with st.spinner("Parsing SCMS dataset…"):
                scms = pd.read_csv(scms_file, encoding="latin-1")
                scms.columns = scms.columns.str.strip()

                # ── Auto-detect columns (case-insensitive) ────────────
                def _find_col(df: pd.DataFrame, *fragments: str) -> str | None:
                    for frag in fragments:
                        for col in df.columns:
                            if frag.lower() in col.lower():
                                return col
                    return None

                vendor_col  = _find_col(scms, "vendor", "manufacturing site")
                sched_col   = _find_col(scms, "scheduled delivery", "po sent")
                actual_col  = _find_col(scms, "delivered to client", "delivery recorded")
                country_col = _find_col(scms, "country")

                missing_required = [
                    name for name, col in [
                        ("Vendor / Manufacturing Site", vendor_col),
                        ("Scheduled Delivery Date", sched_col),
                        ("Delivered to Client Date", actual_col),
                    ]
                    if col is None
                ]
                if missing_required:
                    st.error(
                        "Could not auto-detect required columns: "
                        + ", ".join(missing_required)
                        + ". Check column names in the dataset."
                    )
                    st.stop()

                # ── Date parsing and delay computation ────────────────
                scms[sched_col]  = pd.to_datetime(scms[sched_col],  errors="coerce")
                scms[actual_col] = pd.to_datetime(scms[actual_col], errors="coerce")
                scms["delay_days"] = (scms[actual_col] - scms[sched_col]).dt.days
                scms["on_time"]    = (scms["delay_days"] <= 0).astype(int)

                # ── Aggregate per vendor ───────────────────────────────
                grp_dict = {
                    "delay_mean": (  "delay_days", "mean"),
                    "delay_std":  (  "delay_days", "std"),
                    "on_time_rate":(  "on_time",   "mean"),
                    "order_count": (  "on_time",   "count"),
                }
                if country_col:
                    grp_dict["country"] = (country_col, "first")

                agg = scms.groupby(vendor_col).agg(**grp_dict).reset_index()

                median_orders = max(agg["order_count"].median(), 1)
                max_delay     = max(agg["delay_mean"].clip(lower=0).max(), 1)

                agg["lead_time_variance"]    = (agg["delay_std"].fillna(1.0) ** 2).clip(0.2, 9.0)
                agg["supplier_reliability"]  = agg["on_time_rate"].fillna(0.80).clip(0.0, 1.0)

                if country_col and "country" in agg.columns:
                    agg["geo_risk_index"] = (
                        agg["country"].astype(str).str.strip().str.lower()
                        .map(COUNTRY_RISK)
                        .fillna(0.25)
                    )
                else:
                    agg["geo_risk_index"] = 0.25

                agg["inventory_buffer"] = (
                    (agg["order_count"] - median_orders) / (median_orders + 1)
                ).clip(-0.5, 0.8)
                agg["shipment_delay_history"] = (
                    agg["delay_mean"].clip(lower=0) / max_delay * 10
                ).clip(0.0, 10.0)
                agg["disruption_risk"] = compute_disruption_risk(agg).values

                result_df = agg[CEDPA_FEATURES].copy()

            st.success(f"Converted **{len(result_df):,} vendors** → CEDPA feature set.")

            with st.expander("📄 Mapped data preview", expanded=True):
                st.dataframe(result_df.head(20), use_container_width=True)

            show_metric_cards(result_df)
            st.markdown("#### Feature Distributions")
            show_feature_histograms(result_df)
            _retrain_section(result_df, "USAID SCMS Delivery History", "tab4")

        except Exception as exc:
            st.error(f"Processing failed: {exc}")
    else:
        st.info("Upload **SCMS_Delivery_History_Dataset.csv** to proceed.")


# ══════════════════════════════════════════════════════════════════════
# TAB 5 — CUSTOM COMPANY DATA
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Custom Company Data")
    st.markdown(
        "Tell us what data you have, generate a tailored template CSV, "
        "fill it in, then upload it back to retrain the platform."
    )

    # ── Section A: What data do you have? ─────────────────────────────
    st.markdown("#### 1 · What supply chain data do you have?")

    ca1, ca2 = st.columns(2)
    with ca1:
        has_dates     = st.checkbox("Order / delivery dates",           key="t5_dates")
        has_suppliers = st.checkbox("Supplier / vendor IDs or names",   key="t5_suppliers")
        has_country   = st.checkbox("Supplier country / city / region", key="t5_country")
    with ca2:
        has_inventory  = st.checkbox("Inventory levels or stock quantities",    key="t5_inventory")
        has_otd        = st.checkbox("On-time delivery flags or rates",         key="t5_otd")
        has_disruption = st.checkbox("Disruption / incident flags",             key="t5_disruption")

    # ── Section B: Generate template ──────────────────────────────────
    st.markdown("---")
    st.markdown("#### 2 · Generate a Tailored Template CSV")

    if st.button("📋 Generate Template", key="t5_gen_template"):
        raw_cols: list[str] = []
        if has_suppliers:
            raw_cols += ["supplier_id", "supplier_name"]
        if has_dates:
            raw_cols += ["order_date", "delivery_date", "scheduled_delivery_date"]
        if has_country:
            raw_cols += ["country", "region", "city"]
        if has_inventory:
            raw_cols += ["current_stock", "safety_stock_level"]
        if has_otd:
            raw_cols += ["on_time_delivery_flag", "on_time_pct"]
        if has_disruption:
            raw_cols += ["incident_flag"]

        if not raw_cols:
            raw_cols = ["supplier_id", "country", "on_time_pct", "current_stock",
                        "safety_stock_level", "avg_delay_days", "incident_flag"]

        # Append CEDPA columns (as empty reference columns)
        cedpa_ref = [f for f in CEDPA_FEATURES if f not in raw_cols]
        all_template_cols = raw_cols + cedpa_ref

        sample_row = {c: "FILL_IN" for c in raw_cols}
        sample_row.update({c: "(auto-computed or fill in)" for c in cedpa_ref})
        hint_row   = {c: "..." for c in raw_cols}
        hint_row.update({c: "see CEDPA docs" for c in cedpa_ref})

        tpl_df = pd.DataFrame([sample_row, hint_row])

        st.session_state["t5_template"] = tpl_df
        st.success(
            f"Template generated with **{len(raw_cols)} raw columns** + "
            f"**{len(cedpa_ref)} CEDPA reference columns**."
        )
        st.dataframe(tpl_df, use_container_width=True)

    if "t5_template" in st.session_state:
        _download_csv_button(
            st.session_state["t5_template"],
            "cedpa_custom_template.csv",
            "📥 Download Template CSV",
        )

    # ── Section C: Upload filled template ─────────────────────────────
    st.markdown("---")
    st.markdown("#### 3 · Upload Your Filled Template")

    filled_file = st.file_uploader(
        "Upload filled template (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        key="t5_filled",
    )

    if filled_file:
        try:
            if filled_file.name.lower().endswith((".xlsx", ".xls")):
                filled_df = pd.read_excel(filled_file)
            else:
                filled_df = pd.read_csv(filled_file)

            st.success(f"Loaded **{len(filled_df):,} rows × {len(filled_df.columns)} columns**")

            # Feature column check
            st.markdown("#### Feature Column Check")
            all_present = True
            chk_col1, chk_col2 = st.columns(2)
            for i, feat in enumerate(CEDPA_FEATURES):
                present = feat in filled_df.columns
                if not present:
                    all_present = False
                icon  = "✅" if present else "❌"
                color = SUCCESS if present else DANGER
                (chk_col1 if i < 3 else chk_col2).markdown(
                    f"<span style='color:{color};font-size:1rem;font-weight:700'>{icon}</span> "
                    f"**{feat}** — <span style='color:var(--muted);font-size:.85rem'>"
                    f"{FEATURE_LABELS[feat]}</span>",
                    unsafe_allow_html=True,
                )

            st.markdown("")
            if all_present:
                st.markdown("All 6 CEDPA features found. Ready to retrain.")
                _retrain_section(filled_df, f"Custom Template: {filled_file.name}", "tab5")
            else:
                st.warning(
                    "Some features are missing. Switch to the **📁 Universal Uploader** tab "
                    "to map and derive any missing columns."
                )
                if st.button("→ Go to Universal Uploader", key="t5_goto_tab1"):
                    st.info("Select the '📁 Universal Uploader' tab above.")

        except Exception as exc:
            st.error(f"Could not read file: {exc}")
    else:
        st.info("Upload your filled template CSV or Excel file to continue.")

    # ── Section D: Data source status card ────────────────────────────
    st.markdown("---")
    st.markdown("#### Current Data Source Status")
    _show_data_source_status()

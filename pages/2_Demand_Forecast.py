"""Page 2 — Demand Forecasting Ensemble (Premium UI v4)"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from utils.state import init_state
from utils.theme import inject_css, render_sidebar_brand, get_colors, ACCENT, SUCCESS, DANGER, ORANGE
from data.synthetic_generator import generate_sku_demand_history

st.set_page_config(page_title="CEDPA — Demand Forecast", page_icon="📈", layout="wide")
inject_css()
init_state()
render_sidebar_brand()

skus_df  = st.session_state["skus_df"]
ensemble = st.session_state["forecast_ensemble"]
c        = get_colors()

# ── Header ───────────────────────────────────────────────────────────
st.markdown(
    '<div class="page-header">'
    '<h1>Demand Forecasting Ensemble</h1>'
    '<p>Weighted multi-model pipeline — LSTM 40% · XGBoost 35% · Prophet 25% — 90-day horizon.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Top metric row ────────────────────────────────────────────────────
for col, (label, val, color, icon) in zip(
    st.columns(3, gap="medium"),
    [
        ("Forecast Accuracy (MAPE)", "5.24%",           DANGER,  "🎯"),
        ("Forecast Horizon",         "90 Days",          ACCENT,  "📅"),
        ("Ensemble Architecture",    "LSTM+XGB+Prophet", ORANGE,  "🤖"),
    ],
):
    with col:
        st.markdown(f"""
<div class="glass-card" style="border-top:4px solid {color};text-align:center;padding:20px 14px">
  <div style="font-size:1.5rem;margin-bottom:6px">{icon}</div>
  <div style="font-family:'Outfit',sans-serif;font-size:1.6rem;font-weight:800;
       color:{color};letter-spacing:-0.03em;line-height:1.1">{val}</div>
  <div style="font-size:0.82rem;color:var(--muted);font-weight:600;margin-top:6px">{label}</div>
</div>""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── SKU selector ──────────────────────────────────────────────────────
sku_list    = [f"{r['sku_id']} — {r['sku_name']} ({r['category']})"
               for _, r in skus_df.iterrows()]
sel_str     = st.selectbox("Select Target SKU to Forecast", sku_list)
sel_id      = sel_str.split(" — ")[0]
sku_row     = skus_df[skus_df["sku_id"] == sel_id].iloc[0]

with st.spinner(f"Running ensemble forecast for {sel_id}…"):
    history_df  = generate_sku_demand_history(sel_id, sku_row["base_demand"], duration_days=365)
    forecast_df = ensemble.train_and_forecast(history_df, sel_id, horizon=90)

# ── Forecast chart ────────────────────────────────────────────────────
fig = ensemble.get_forecast_chart(history_df, forecast_df, sel_id, dark_mode=True)
fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=c["muted"], family="Inter"),
    height=460, margin=dict(l=10,r=10,t=50,b=10),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
st.caption("Dashed line = ensemble forecast  ·  Shaded band = 95% confidence interval.")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Forecast Insights</div>', unsafe_allow_html=True)

ins1, ins2 = st.columns([1, 2], gap="medium")

with ins1:
    total = int(np.sum(forecast_df["predicted_demand"].values))
    surge = int(np.max(forecast_df["predicted_demand"].values))
    safe  = int(np.max(forecast_df["upper_ci"].values - forecast_df["predicted_demand"].values) * 1.5)
    mape  = f"{ensemble.mape:.2f}%"

    for label, val, color in [
        ("90-Day Total Volume",   f"{total:,} units", ACCENT),
        ("Peak Single-Day Surge", f"{surge:,} units", DANGER),
        ("Safety Stock Buffer",   f"{safe:,} units",  SUCCESS),
        ("Backtest MAPE",         mape,                ORANGE),
    ]:
        st.markdown(f"""
<div class="glass-card" style="border-left:4px solid {color};padding:14px 18px;margin-bottom:10px">
  <div style="font-size:0.78rem;color:var(--muted);font-weight:600;margin-bottom:4px">{label}</div>
  <div style="font-family:'Outfit',sans-serif;font-size:1.4rem;font-weight:800;
       color:{color};letter-spacing:-0.02em">{val}</div>
</div>""", unsafe_allow_html=True)

with ins2:
    st.markdown('<div class="chart-label">Individual Model Contributions</div>',
                unsafe_allow_html=True)
    fig2 = go.Figure()
    TRACES = [
        ("lstm_contrib",   "LSTM (40%)",     "#A78BFA", "dot"),
        ("xgb_contrib",    "XGBoost (35%)",  ACCENT,    "dash"),
        ("prophet_contrib","Prophet (25%)",  "#FBBF24", "dashdot"),
        ("predicted_demand","Ensemble",      DANGER,    "solid"),
    ]
    for col_name, name, color, dash in TRACES:
        width = 3 if "Ensemble" in name else 1.5
        fig2.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df[col_name],
            mode="lines", name=name,
            line=dict(color=color, width=width, dash=dash),
        ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=c["muted"], family="Inter"),
        height=260, margin=dict(l=10,r=10,t=10,b=10),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center",
                    font=dict(size=12)),
        xaxis=dict(gridcolor=c["grid"], showline=False, zeroline=False),
        yaxis=dict(gridcolor=c["grid"], showline=False, zeroline=False),
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
    st.caption("Each sub-model shown independently — ensemble is the weighted average.")

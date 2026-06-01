"""Page 4 — What-If Scenario Simulator (Premium UI v4)"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from utils.state import init_state
from utils.theme import (
    inject_css, render_sidebar_brand, get_colors,
    ACCENT, SUCCESS, DANGER, ORANGE,
)

st.set_page_config(page_title="CEDPA — Scenario Simulator", page_icon="⚙️", layout="wide")
inject_css()
init_state()
render_sidebar_brand()
c = get_colors()

# ── Header ───────────────────────────────────────────────────────────
st.markdown(
    '<div class="page-header">'
    '<h1>What-If Scenario Simulator</h1>'
    '<p>Adjust procurement parameters and see instant impact on cost and disruption risk.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Cost model functions ──────────────────────────────────────────────
def holding_cost(ss, lt):
    return 100_000.0 * (1.0 + ss * 2.2) * (1.0 + lt / 20.0)

def stockout_cost(rel, ss):
    return 200_000.0 * np.exp((0.85 - rel) * 4.5) * np.exp(-ss * 6.0)

def disruption_prob(rel, lt):
    z = (0.85 - rel) * 6.0 + (lt - 12.0) * 0.15
    return 1.0 / (1.0 + np.exp(-z - 0.92))

# Baseline constants
BASE_REL, BASE_LT, BASE_SS = 0.85, 12, 0.20

sim_col, result_col = st.columns([1, 2], gap="large")

with sim_col:
    st.markdown('<div class="section-heading" style="font-size:1.1rem">'
                'Simulation Controls</div>', unsafe_allow_html=True)

    sim_rel = st.slider("Supplier Reliability (%)", 30, 100,
                        int(BASE_REL * 100), 1) / 100.0
    sim_lt  = st.slider("Transit Lead Time (Days)", 1, 30, BASE_LT)
    sim_ss  = st.slider("Safety Stock Buffer (%)",  0, 50,
                        int(BASE_SS * 100), 1) / 100.0

    st.markdown(f"""
<div class="glass-card" style="border-left:4px solid {ACCENT};
     padding:14px 16px;margin-top:8px">
  <div style="font-size:0.82rem;color:var(--muted);line-height:1.65">
    💡 <b style="color:{ACCENT}">Tip:</b> Increasing safety stock raises
    holding costs but protects against stockouts caused by poor supplier reliability.
  </div>
</div>""", unsafe_allow_html=True)

with result_col:
    st.markdown('<div class="section-heading" style="font-size:1.1rem">'
                'Before vs. After Comparison</div>', unsafe_allow_html=True)

    base_h  = holding_cost(BASE_SS, BASE_LT)
    base_s  = stockout_cost(BASE_REL, BASE_SS)
    base_t  = base_h + base_s
    base_r  = disruption_prob(BASE_REL, BASE_LT)

    sim_h   = holding_cost(sim_ss, sim_lt)
    sim_s   = stockout_cost(sim_rel, sim_ss)
    sim_t   = sim_h + sim_s
    sim_r   = disruption_prob(sim_rel, sim_lt)

    net     = sim_t - base_t
    sc      = SUCCESS if net <= 0 else DANGER

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown(f"""
<div class="glass-card" style="border-top:4px solid var(--muted);padding:18px">
  <div style="font-size:0.78rem;color:var(--muted);font-weight:700;
       text-transform:uppercase;letter-spacing:0.06em;margin-bottom:12px">
    Baseline Network
  </div>
  <div style="margin-bottom:10px">
    <div style="font-size:0.78rem;color:var(--muted)">Disruption Risk</div>
    <div style="font-size:1.4rem;font-weight:800;color:var(--text);
         font-family:'Outfit',sans-serif">{base_r*100:.1f}%</div>
  </div>
  <div style="margin-bottom:10px">
    <div style="font-size:0.78rem;color:var(--muted)">Holding Overhead</div>
    <div style="font-size:1.1rem;font-weight:700;color:var(--text)">${base_h:,.0f}</div>
  </div>
  <div style="margin-bottom:10px">
    <div style="font-size:0.78rem;color:var(--muted)">Stockout Penalties</div>
    <div style="font-size:1.1rem;font-weight:700;color:var(--text)">${base_s:,.0f}</div>
  </div>
  <div style="border-top:1px dashed rgba(255,255,255,0.08);padding-top:10px;margin-top:6px">
    <div style="font-size:0.78rem;color:#38BDF8;font-weight:700">Net Operating Cost</div>
    <div style="font-size:1.5rem;font-weight:800;color:#38BDF8;
         font-family:'Outfit',sans-serif">${base_t:,.0f}</div>
  </div>
</div>""", unsafe_allow_html=True)

    with c2:
        sim_risk_color = SUCCESS if sim_r <= base_r else DANGER
        st.markdown(f"""
<div class="glass-card" style="border-top:4px solid {sc};padding:18px">
  <div style="font-size:0.78rem;color:{sc};font-weight:700;
       text-transform:uppercase;letter-spacing:0.06em;margin-bottom:12px">
    Simulated State
  </div>
  <div style="margin-bottom:10px">
    <div style="font-size:0.78rem;color:var(--muted)">Disruption Risk</div>
    <div style="font-size:1.4rem;font-weight:800;font-family:'Outfit',sans-serif;
         color:{sim_risk_color}">{sim_r*100:.1f}%</div>
  </div>
  <div style="margin-bottom:10px">
    <div style="font-size:0.78rem;color:var(--muted)">Holding Overhead</div>
    <div style="font-size:1.1rem;font-weight:700;color:var(--text)">${sim_h:,.0f}</div>
  </div>
  <div style="margin-bottom:10px">
    <div style="font-size:0.78rem;color:var(--muted)">Stockout Penalties</div>
    <div style="font-size:1.1rem;font-weight:700;color:var(--text)">${sim_s:,.0f}</div>
  </div>
  <div style="border-top:1px dashed rgba(255,255,255,0.08);padding-top:10px;margin-top:6px">
    <div style="font-size:0.78rem;color:{sc};font-weight:700">Net Operating Cost</div>
    <div style="font-size:1.5rem;font-weight:800;color:{sc};
         font-family:'Outfit',sans-serif">${sim_t:,.0f}</div>
  </div>
</div>""", unsafe_allow_html=True)

# ── Result banner ─────────────────────────────────────────────────────
if net < 0:
    st.success(f"✅ **Optimised:** This configuration saves **${abs(net):,.0f} / month** vs baseline.")
else:
    st.warning(f"⚠️ **Higher Cost:** This configuration adds **${net:,.0f} / month** to operating expenses.")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Financial Simulation Matrix</div>',
            unsafe_allow_html=True)

bar_col, note_col = st.columns([3, 1], gap="medium")
with bar_col:
    cats = ["Holding Cost", "Stockout Cost", "Net Operating Cost"]
    fig  = go.Figure(data=[
        go.Bar(name="Baseline",   x=cats, y=[base_h, base_s, base_t],
               marker_color="#475569"),
        go.Bar(name="Simulated",  x=cats, y=[sim_h,  sim_s,  sim_t],
               marker_color=ACCENT),
    ])
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=c["muted"], family="Inter"),
        margin=dict(l=10,r=10,t=10,b=10), height=320,
        xaxis=dict(gridcolor=c["grid"]),
        yaxis=dict(gridcolor=c["grid"], title="Cost ($ / month)"),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center",
                    font=dict(size=12)),
        hoverlabel=dict(bgcolor="#162036", bordercolor=ACCENT,
                        font=dict(color=c["text"])),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    st.caption("Grouped bars compare Baseline vs Simulated across all cost categories.")

with note_col:
    risk_delta = sim_r - base_r
    rd_color   = SUCCESS if risk_delta < 0 else DANGER
    rd_arrow   = "▼" if risk_delta < 0 else "▲"

    st.markdown(f"""
<div class="glass-card" style="border-left:5px solid {ORANGE};padding:16px 18px">
  <div style="font-size:0.82rem;color:var(--muted);font-weight:600;margin-bottom:12px">
    Risk Delta</div>
  <div style="font-family:'Outfit',sans-serif;font-size:2rem;font-weight:800;
       color:{rd_color};letter-spacing:-0.03em">
    {rd_arrow} {abs(risk_delta)*100:.1f}%
  </div>
  <div style="font-size:0.78rem;color:var(--muted);margin-top:6px">
    vs. baseline disruption probability
  </div>
</div>

<div class="glass-card" style="border-left:5px solid {sc};padding:16px 18px;margin-top:0">
  <div style="font-size:0.82rem;color:var(--muted);font-weight:600;margin-bottom:12px">
    Cost Impact</div>
  <div style="font-family:'Outfit',sans-serif;font-size:1.6rem;font-weight:800;
       color:{sc};letter-spacing:-0.03em">
    {"−" if net<0 else "+"} ${abs(net):,.0f}
  </div>
  <div style="font-size:0.78rem;color:var(--muted);margin-top:6px">per month vs baseline</div>
</div>""", unsafe_allow_html=True)

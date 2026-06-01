"""
Page 8 — Advanced Analytics
Monte Carlo simulation, EOQ calculator, Isolation Forest anomaly detection,
Multi-SKU forecast comparison, and ABC analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime

from utils.state import init_state
from utils.theme import inject_css, render_sidebar_brand, get_colors, ACCENT, DANGER, WARNING, SUCCESS, ORANGE

from data.synthetic_generator import generate_sku_demand_history

# Safe import
ISOLATION_FOREST_AVAILABLE = False
try:
    from sklearn.ensemble import IsolationForest
    ISOLATION_FOREST_AVAILABLE = True
except Exception:
    pass

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(page_title="CEDPA — Advanced Analytics", page_icon="🧪", layout="wide")
inject_css()
init_state()
render_sidebar_brand()
c = get_colors()

suppliers_df = st.session_state["suppliers_df"]
skus_df      = st.session_state["skus_df"]
ensemble     = st.session_state["forecast_ensemble"]

st.markdown(
    '<div class="page-header">'
    '<h1>Advanced Analytics Suite</h1>'
    '<p>Monte Carlo simulation · EOQ calculator · Anomaly detection · Multi-SKU comparison · ABC analysis.</p>'
    '</div>',
    unsafe_allow_html=True,
)

tab_mc, tab_eoq, tab_anom, tab_multi, tab_abc = st.tabs([
    "🎲 Monte Carlo", "📐 EOQ Calculator", "🔍 Anomaly Detection",
    "📊 Multi-SKU Comparison", "🏷️ ABC Analysis"
])

# ═══ TAB 1: MONTE CARLO SIMULATION ═════════════════════════════════
with tab_mc:
    st.markdown("### Monte Carlo Cost Distribution Simulation")
    st.markdown("Run 1 000 iterations varying supplier reliability, lead time, and safety stock to model the full distribution of operating costs.")

    mc1, mc2 = st.columns([1, 2])

    with mc1:
        n_iters = st.slider("Iterations", 100, 5000, 1000, step=100)
        mc_rel_range = st.slider("Reliability range (%)", 50, 100, (65, 95))
        mc_lt_range = st.slider("Lead time range (days)", 3, 25, (5, 18))
        mc_ss_range = st.slider("Safety stock range (%)", 5, 45, (10, 35))

    with mc2:
        if st.button("▶ Run Monte Carlo Simulation", use_container_width=True):
            rng = np.random.default_rng(42)
            costs = []

            bar = st.progress(0, "Simulating…")
            for i in range(n_iters):
                rel = rng.uniform(mc_rel_range[0]/100, mc_rel_range[1]/100)
                lt  = rng.uniform(mc_lt_range[0], mc_lt_range[1])
                ss  = rng.uniform(mc_ss_range[0]/100, mc_ss_range[1]/100)

                hold = 100_000 * (1 + ss*2.2) * (1 + lt/20)
                stock = 200_000 * np.exp((0.85 - rel)*4.5) * np.exp(-ss*6)
                costs.append(hold + stock)

                if i % (n_iters // 10) == 0:
                    bar.progress(int(i / n_iters * 100))

            bar.progress(100, "Complete ✓"); bar.empty()

            costs = np.array(costs)
            p10, p50, p90 = np.percentile(costs, [10, 50, 90])

            fig = go.Figure()
            fig.add_trace(go.Histogram(x=costs, nbinsx=50, marker_color=ACCENT, opacity=0.7, name="Cost Distribution"))
            for pval, pname, col in [(p10, "P10", SUCCESS), (p50, "P50", WARNING), (p90, "P90", DANGER)]:
                fig.add_vline(x=pval, line_dash="dash", line_color=col, annotation_text=f"{pname}: ${pval:,.0f}")

            fig.update_layout(
                title="Monte Carlo Cost Distribution",
                xaxis_title="Total Operating Cost ($)", yaxis_title="Frequency",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=c["muted"]), height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            mc_c1, mc_c2, mc_c3 = st.columns(3)
            for col, label, val, color in [(mc_c1, "P10 (Optimistic)", p10, SUCCESS),
                                            (mc_c2, "P50 (Median)", p50, WARNING),
                                            (mc_c3, "P90 (Pessimistic)", p90, DANGER)]:
                with col:
                    st.markdown(f"""
<div class="glass-card" style="border-top:3px solid {color};text-align:center">
  <div class="kpi-title">{label}</div>
  <div class="kpi-value">${val:,.0f}</div>
</div>""", unsafe_allow_html=True)

# ═══ TAB 2: EOQ CALCULATOR ═════════════════════════════════════════
with tab_eoq:
    st.markdown("### Economic Order Quantity (EOQ) Calculator")
    st.markdown("The classic EOQ formula minimizes total inventory cost: **EOQ = √(2DS / H)**")

    sku_list = [f"{r['sku_id']} - {r['sku_name']}" for _, r in skus_df.iterrows()]
    sel_sku = st.selectbox("Select SKU", sku_list, key="eoq_sku")
    sku_id = sel_sku.split(" - ")[0]
    sku_row = skus_df[skus_df["sku_id"] == sku_id].iloc[0]

    eq1, eq2 = st.columns(2)
    with eq1:
        annual_demand = st.number_input("Annual Demand (D)", value=int(sku_row["base_demand"] * 365), step=100)
        ordering_cost = st.number_input("Ordering Cost per Order ($S)", value=150.0, step=10.0)
    with eq2:
        holding_cost = st.number_input("Annual Holding Cost per Unit ($H)",
                                       value=round(sku_row["carrying_cost_daily"] * 365, 2), step=1.0)

    if holding_cost > 0 and annual_demand > 0:
        eoq = np.sqrt(2 * annual_demand * ordering_cost / holding_cost)
        orders_per_year = annual_demand / eoq
        total_cost = (annual_demand / eoq) * ordering_cost + (eoq / 2) * holding_cost

        r1, r2, r3 = st.columns(3)
        results = [
            ("Optimal Order Qty (EOQ)", f"{eoq:,.0f} units", ACCENT),
            ("Orders per Year", f"{orders_per_year:.1f}", ORANGE),
            ("Total Annual Cost", f"${total_cost:,.2f}", SUCCESS),
        ]
        for col, (label, val, color) in zip([r1, r2, r3], results):
            with col:
                st.markdown(f"""
<div class="glass-card" style="border-top:3px solid {color};text-align:center">
  <div class="kpi-title">{label}</div>
  <div class="kpi-value">{val}</div>
</div>""", unsafe_allow_html=True)

        # Cost curve
        st.markdown("#### Total Cost Curve")
        q_range = np.linspace(max(1, eoq * 0.1), eoq * 3, 200)
        tc = (annual_demand / q_range) * ordering_cost + (q_range / 2) * holding_cost

        fig_tc = go.Figure()
        fig_tc.add_trace(go.Scatter(x=q_range, y=tc, mode="lines",
                                     line=dict(color=ACCENT, width=2), name="Total Cost"))
        fig_tc.add_vline(x=eoq, line_dash="dash", line_color=SUCCESS,
                         annotation_text=f"EOQ = {eoq:.0f}")
        fig_tc.update_layout(
            xaxis_title="Order Quantity", yaxis_title="Total Annual Cost ($)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=c["muted"]), height=300, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_tc, use_container_width=True)

        st.latex(r"EOQ = \sqrt{\frac{2 \times D \times S}{H}} = \sqrt{\frac{2 \times "
                 + str(int(annual_demand)) + r" \times " + str(int(ordering_cost))
                 + r"}{" + f"{holding_cost:.2f}" + r"}} = " + f"{eoq:.0f}")

# ═══ TAB 3: ANOMALY DETECTION ══════════════════════════════════════
with tab_anom:
    st.markdown("### Isolation Forest Anomaly Detection")
    st.markdown("Detect anomalous demand patterns in historical shipment data.")

    sku_list2 = [f"{r['sku_id']} - {r['sku_name']}" for _, r in skus_df.head(50).iterrows()]
    sel_sku2 = st.selectbox("Select SKU", sku_list2, key="anom_sku")
    sku_id2 = sel_sku2.split(" - ")[0]
    sku_row2 = skus_df[skus_df["sku_id"] == sku_id2].iloc[0]

    history = generate_sku_demand_history(sku_id2, sku_row2["base_demand"], duration_days=365)

    contamination = st.slider("Contamination (% anomalies)", 1, 15, 5) / 100

    if ISOLATION_FOREST_AVAILABLE:
        X = history[["demand"]].values
        iso = IsolationForest(contamination=contamination, random_state=42)
        history["anomaly"] = iso.fit_predict(X)
        anomalies = history[history["anomaly"] == -1]

        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(x=history["date"], y=history["demand"],
                                    mode="lines", line=dict(color=ACCENT, width=1.5), name="Demand"))
        fig_a.add_trace(go.Scatter(x=anomalies["date"], y=anomalies["demand"],
                                    mode="markers", marker=dict(color=DANGER, size=8, symbol="x"),
                                    name=f"Anomalies ({len(anomalies)})"))
        fig_a.update_layout(
            title=f"Anomaly Detection — {sku_id2}",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=c["muted"]), height=400, margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_a, use_container_width=True)
        st.info(f"Detected **{len(anomalies)} anomalous days** out of 365 ({len(anomalies)/365*100:.1f}%)")
    else:
        st.error("scikit-learn is required for Isolation Forest. Run `pip install scikit-learn`.")

# ═══ TAB 4: MULTI-SKU COMPARISON ═══════════════════════════════════
with tab_multi:
    st.markdown("### Multi-SKU Forecast Comparison")
    st.markdown("Select 2–5 SKUs and overlay their 90-day forecasts on one chart.")

    sku_options = [f"{r['sku_id']} - {r['sku_name']}" for _, r in skus_df.head(30).iterrows()]
    selected_skus = st.multiselect("Select SKUs (2–5)", sku_options, default=sku_options[:3], max_selections=5)

    if len(selected_skus) >= 2:
        fig_multi = go.Figure()
        colors = [ACCENT, DANGER, SUCCESS, ORANGE, WARNING]

        for i, sku_str in enumerate(selected_skus):
            sid = sku_str.split(" - ")[0]
            srow = skus_df[skus_df["sku_id"] == sid].iloc[0]
            hist = generate_sku_demand_history(sid, srow["base_demand"], duration_days=365)
            fcast = ensemble.train_and_forecast(hist, sid, horizon=90)

            fig_multi.add_trace(go.Scatter(
                x=fcast["date"], y=fcast["predicted_demand"],
                mode="lines", name=sid, line=dict(color=colors[i % len(colors)], width=2)
            ))

        fig_multi.update_layout(
            title="90-Day Forecast Overlay",
            xaxis_title="Date", yaxis_title="Predicted Demand (units)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=c["muted"]), height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_multi, use_container_width=True)
    else:
        st.warning("Please select at least 2 SKUs to compare.")

# ═══ TAB 5: ABC ANALYSIS ═══════════════════════════════════════════
with tab_abc:
    st.markdown("### ABC Inventory Classification")
    st.markdown("Classify SKUs into A/B/C tiers by annual demand value (price × demand).")

    abc_df = skus_df.copy()
    abc_df["annual_value"] = abc_df["unit_price"] * abc_df["base_demand"] * 365
    abc_df = abc_df.sort_values("annual_value", ascending=False).reset_index(drop=True)
    abc_df["cumulative_pct"] = abc_df["annual_value"].cumsum() / abc_df["annual_value"].sum() * 100

    abc_df["tier"] = "C"
    abc_df.loc[abc_df["cumulative_pct"] <= 80, "tier"] = "A"
    abc_df.loc[(abc_df["cumulative_pct"] > 80) & (abc_df["cumulative_pct"] <= 95), "tier"] = "B"

    tier_counts = abc_df["tier"].value_counts()

    # Summary cards
    tc1, tc2, tc3 = st.columns(3)
    for col, tier, color in [(tc1, "A", DANGER), (tc2, "B", WARNING), (tc3, "C", SUCCESS)]:
        cnt = tier_counts.get(tier, 0)
        pct = abc_df[abc_df["tier"] == tier]["annual_value"].sum() / abc_df["annual_value"].sum() * 100
        with col:
            st.markdown(f"""
<div class="glass-card" style="border-top:3px solid {color};text-align:center">
  <div class="kpi-title">Tier {tier} ({cnt} SKUs)</div>
  <div class="kpi-value">{pct:.1f}%</div>
  <div class="metric-label-sub">of total annual value</div>
</div>""", unsafe_allow_html=True)

    # Pareto chart
    fig_abc = go.Figure()
    tier_colors = abc_df["tier"].map({"A": DANGER, "B": WARNING, "C": SUCCESS})
    fig_abc.add_trace(go.Bar(x=abc_df["sku_id"], y=abc_df["annual_value"],
                              marker_color=tier_colors.tolist(), name="Annual Value"))
    fig_abc.add_trace(go.Scatter(x=abc_df["sku_id"], y=abc_df["cumulative_pct"],
                                  mode="lines", name="Cumulative %", yaxis="y2",
                                  line=dict(color=c["text"], width=2)))
    fig_abc.update_layout(
        yaxis=dict(title="Annual Value ($)", gridcolor=c["grid"]),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
        xaxis=dict(showticklabels=False, title="SKUs (sorted by value)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=c["muted"]), height=400, margin=dict(l=20, r=40, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        barmode="overlay"
    )
    st.plotly_chart(fig_abc, use_container_width=True)

    # Top items table
    st.markdown("#### Top 10 Tier-A SKUs")
    display_df = abc_df[abc_df["tier"] == "A"].head(10)[
        ["sku_id", "sku_name", "category", "unit_price", "base_demand", "annual_value"]
    ].copy()
    display_df["annual_value"] = display_df["annual_value"].apply(lambda x: f"${x:,.0f}")
    display_df["unit_price"] = display_df["unit_price"].apply(lambda x: f"${x:,.2f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

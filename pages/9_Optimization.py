"""
Page 9 — LP Procurement Optimization
Linear programming optimizer using PuLP to minimize total supply chain cost
subject to demand, capacity, and reliability constraints.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.state import init_state
from utils.theme import inject_css, render_sidebar_brand, get_colors, ACCENT, DANGER, WARNING, SUCCESS

# Safe import
PULP_AVAILABLE = False
try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    pass

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(page_title="CEDPA — Optimization", page_icon="🎯", layout="wide")
inject_css()
init_state()
render_sidebar_brand()
c = get_colors()

suppliers_df = st.session_state["suppliers_df"]
skus_df      = st.session_state["skus_df"]

st.markdown(
    '<div class="page-header">'
    '<h1>Procurement Optimization Engine</h1>'
    '<p>Linear programming (PuLP) — minimize total supply chain cost subject to demand, capacity, and reliability constraints.</p>'
    '</div>',
    unsafe_allow_html=True,
)

if not PULP_AVAILABLE:
    st.error("PuLP is not installed. Run `pip install pulp` to enable the LP optimizer.")
    st.info("The optimizer uses PuLP to solve a linear program minimizing total procurement + holding + risk cost.")
    st.stop()

# ── Problem setup ───────────────────────────────────────────────────
st.markdown("### Problem Definition")
st.markdown("""
The optimizer allocates procurement volumes across suppliers to **minimize total cost** subject to:
- **Demand constraint**: Total allocation must meet required demand
- **Capacity constraint**: Each supplier has a maximum monthly capacity
- **Reliability constraint**: Minimum average reliability across the allocation
""")

# Use top 10 suppliers for tractability
top_sups = suppliers_df.head(10).copy()

# Generate synthetic cost and capacity data
rng = np.random.default_rng(123)
top_sups["unit_cost"]   = rng.uniform(8.0, 25.0, len(top_sups)).round(2)
top_sups["capacity"]    = rng.integers(500, 3000, len(top_sups))
top_sups["risk_penalty"] = ((1 - top_sups["reliability"]) * 10).round(2)

# ── User controls ──────────────────────────────────────────────────
ctrl1, ctrl2 = st.columns(2)
with ctrl1:
    total_demand = st.number_input("Total Monthly Demand (units)", value=5000, step=100, min_value=100)
    min_reliability = st.slider("Min Average Reliability (%)", 70, 98, 82) / 100.0
with ctrl2:
    st.markdown("#### Supplier Cost Table")
    display_sups = top_sups[["supplier_id", "city", "unit_cost", "capacity", "reliability"]].copy()
    display_sups["reliability"] = (display_sups["reliability"] * 100).round(1).astype(str) + "%"
    st.dataframe(display_sups, use_container_width=True, hide_index=True)

# ── Solve LP ────────────────────────────────────────────────────────
if st.button("▶ Run LP Optimization", use_container_width=True):
    with st.spinner("Solving linear program…"):
        # Decision variables: x[i] = units allocated to supplier i
        prob = pulp.LpProblem("CEDPA_Procurement", pulp.LpMinimize)

        x = {}
        for _, sup in top_sups.iterrows():
            x[sup["supplier_id"]] = pulp.LpVariable(
                f"x_{sup['supplier_id']}", lowBound=0, upBound=sup["capacity"], cat="Continuous"
            )

        # Objective: minimize total cost = unit_cost * x + risk_penalty * x
        prob += pulp.lpSum(
            (sup["unit_cost"] + sup["risk_penalty"]) * x[sup["supplier_id"]]
            for _, sup in top_sups.iterrows()
        ), "Total_Cost"

        # Constraint 1: meet total demand
        prob += pulp.lpSum(x[sid] for sid in x) >= total_demand, "Demand_Met"

        # Constraint 2: reliability weighted average >= min threshold
        # sum(rel_i * x_i) / sum(x_i) >= min_rel  →  sum((rel_i - min_rel) * x_i) >= 0
        prob += pulp.lpSum(
            (sup["reliability"] - min_reliability) * x[sup["supplier_id"]]
            for _, sup in top_sups.iterrows()
        ) >= 0, "Min_Reliability"

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]

    if status == "Optimal":
        st.success(f"✅ Optimization Status: **{status}** — Optimal solution found!")

        # Extract results
        results = []
        for _, sup in top_sups.iterrows():
            alloc = x[sup["supplier_id"]].varValue or 0
            cost = alloc * (sup["unit_cost"] + sup["risk_penalty"])
            results.append({
                "Supplier": sup["supplier_id"],
                "City": sup["city"],
                "Allocated": int(alloc),
                "Unit Cost": f"${sup['unit_cost']:.2f}",
                "Risk Penalty": f"${sup['risk_penalty']:.2f}",
                "Total Cost": f"${cost:,.2f}",
                "Reliability": f"{sup['reliability']*100:.1f}%",
            })

        results_df = pd.DataFrame(results)
        optimal_cost = pulp.value(prob.objective)

        # Baseline cost (naive equal split)
        equal_alloc = total_demand / len(top_sups)
        baseline_cost = sum(
            equal_alloc * (sup["unit_cost"] + sup["risk_penalty"])
            for _, sup in top_sups.iterrows()
        )
        savings = baseline_cost - optimal_cost
        savings_pct = (savings / baseline_cost) * 100

        # KPI cards
        k1, k2, k3 = st.columns(3)
        for col, label, val, color in [
            (k1, "LP-Optimized Cost", f"${optimal_cost:,.0f}", SUCCESS),
            (k2, "Baseline Cost (Equal Split)", f"${baseline_cost:,.0f}", WARNING),
            (k3, "Cost Savings", f"${savings:,.0f} ({savings_pct:.1f}%)", ACCENT),
        ]:
            with col:
                st.markdown(f"""
<div class="glass-card" style="border-top:3px solid {color};text-align:center">
  <div class="kpi-title">{label}</div>
  <div class="kpi-value">{val}</div>
</div>""", unsafe_allow_html=True)

        # Allocation table
        st.markdown("### Optimal Supplier Allocation")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Pie chart
        active = results_df[results_df["Allocated"] > 0]
        if not active.empty:
            fig_pie = go.Figure(go.Pie(
                labels=active["Supplier"] + " (" + active["City"] + ")",
                values=active["Allocated"].astype(int),
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Set2[:len(active)])
            ))
            fig_pie.update_layout(
                title="Allocation Distribution",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=c["muted"]), height=350, margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(font=dict(size=10))
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Sensitivity analysis ────────────────────────────────────
        st.markdown("### 📉 Sensitivity Analysis")
        st.markdown("How does the optimal cost change as the minimum reliability constraint tightens?")

        rel_values = np.arange(0.70, 0.96, 0.02)
        opt_costs = []

        for r_min in rel_values:
            p = pulp.LpProblem("Sensitivity", pulp.LpMinimize)
            xv = {sup["supplier_id"]: pulp.LpVariable(f"s_{sup['supplier_id']}",
                  lowBound=0, upBound=sup["capacity"]) for _, sup in top_sups.iterrows()}
            p += pulp.lpSum((sup["unit_cost"] + sup["risk_penalty"]) * xv[sup["supplier_id"]]
                            for _, sup in top_sups.iterrows())
            p += pulp.lpSum(xv[sid] for sid in xv) >= total_demand
            p += pulp.lpSum((sup["reliability"] - r_min) * xv[sup["supplier_id"]]
                            for _, sup in top_sups.iterrows()) >= 0
            p.solve(pulp.PULP_CBC_CMD(msg=0))

            if pulp.LpStatus[p.status] == "Optimal":
                opt_costs.append(pulp.value(p.objective))
            else:
                opt_costs.append(None)

        fig_sens = go.Figure(go.Scatter(
            x=[f"{r*100:.0f}%" for r in rel_values],
            y=opt_costs, mode="lines+markers",
            line=dict(color=ACCENT, width=2),
            marker=dict(size=6)
        ))
        fig_sens.update_layout(
            xaxis_title="Min Reliability Constraint",
            yaxis_title="Optimal Cost ($)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=c["muted"]), height=300, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_sens, use_container_width=True)
        st.caption("As the reliability constraint tightens, the optimizer must favor more expensive but reliable suppliers, increasing cost.")

    else:
        st.error(f"Optimization failed with status: **{status}**. Try relaxing constraints.")

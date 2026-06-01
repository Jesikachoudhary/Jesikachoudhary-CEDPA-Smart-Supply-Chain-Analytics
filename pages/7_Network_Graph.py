"""
Page 7 — Supplier Dependency Network Graph
Interactive force-directed graph showing supplier-SKU dependencies
using NetworkX for layout and Plotly for rendering.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.state import init_state
from utils.theme import inject_css, render_sidebar_brand, get_colors, ACCENT, DANGER, WARNING, SUCCESS

# Safe import
NX_AVAILABLE = False
try:
    import networkx as nx
    NX_AVAILABLE = True
except Exception:
    pass

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(page_title="CEDPA — Network Graph", page_icon="🕸️", layout="wide")
inject_css()
init_state()
render_sidebar_brand()
c = get_colors()

suppliers_df = st.session_state["suppliers_df"]
skus_df      = st.session_state["skus_df"]
risk_model   = st.session_state["risk_model"]

st.markdown(
    '<div class="page-header">'
    '<h1>Supplier Dependency Network Graph</h1>'
    '<p>Force-directed layout showing SKU-Supplier linkages and single points of failure.</p>'
    '</div>',
    unsafe_allow_html=True,
)

if not NX_AVAILABLE:
    st.error("NetworkX is not installed. Run `pip install networkx` to enable network graph visualization.")
    st.stop()

# ── Build graph ─────────────────────────────────────────────────────
rng = np.random.default_rng(777)
G = nx.Graph()

# Add supplier nodes
for _, sup in suppliers_df.iterrows():
    risk_score = min(max(sup["geo_risk"] + sup["reliability"] * 0.1, 0.05), 0.95)
    G.add_node(sup["supplier_id"], node_type="supplier", label=f"{sup['supplier_id']}\n{sup['city']}",
               city=sup["city"], reliability=sup["reliability"], geo_risk=sup["geo_risk"],
               risk_score=risk_score)

# Add SKU nodes and edges (each SKU depends on 2-4 suppliers)
for _, sku in skus_df.iterrows():
    G.add_node(sku["sku_id"], node_type="sku", label=sku["sku_name"],
               category=sku["category"], base_demand=sku["base_demand"])
    num_suppliers = int(rng.integers(2, 5))
    linked_sups = suppliers_df.sample(n=num_suppliers, random_state=rng)
    for _, linked_sup in linked_sups.iterrows():
        G.add_edge(sku["sku_id"], linked_sup["supplier_id"],
                   weight=float(rng.uniform(0.3, 1.0)))

# ── Layout ──────────────────────────────────────────────────────────
# Use spring layout for force-directed positioning
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

# ── Build Plotly traces ─────────────────────────────────────────────
# Edge traces
edge_x, edge_y = [], []
for u, v in G.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y, mode="lines",
    line=dict(width=0.3, color="rgba(148,163,184,0.15)"),
    hoverinfo="none"
)

# Supplier node trace
sup_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "supplier"]
sup_x = [pos[n][0] for n in sup_nodes]
sup_y = [pos[n][1] for n in sup_nodes]
sup_degrees = [G.degree(n) for n in sup_nodes]
sup_risks = [G.nodes[n].get("risk_score", 0.5) for n in sup_nodes]

# Color by risk: green → yellow → red
sup_colors = []
for r in sup_risks:
    if r > 0.65:
        sup_colors.append(DANGER)
    elif r > 0.40:
        sup_colors.append(WARNING)
    else:
        sup_colors.append(SUCCESS)

sup_text = [f"<b>{n}</b><br>{G.nodes[n].get('city','')}<br>"
            f"Dependencies: {G.degree(n)}<br>"
            f"Reliability: {G.nodes[n].get('reliability',0)*100:.1f}%<br>"
            f"Risk Score: {G.nodes[n].get('risk_score',0)*100:.1f}%"
            for n in sup_nodes]

sup_trace = go.Scatter(
    x=sup_x, y=sup_y, mode="markers",
    marker=dict(size=[max(6, d * 0.8) for d in sup_degrees],
                color=sup_colors, line=dict(width=1, color="rgba(255,255,255,0.3)")),
    text=sup_text, hoverinfo="text", name="Suppliers"
)

# SKU node trace (smaller, blue)
sku_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "sku"]
# Only show a sample for readability
sku_sample = sku_nodes[:50]
sku_x = [pos[n][0] for n in sku_sample]
sku_y = [pos[n][1] for n in sku_sample]

sku_text = [f"<b>{n}</b><br>Category: {G.nodes[n].get('category','')}<br>"
            f"Base Demand: {G.nodes[n].get('base_demand',0):.0f}"
            for n in sku_sample]

sku_trace = go.Scatter(
    x=sku_x, y=sku_y, mode="markers",
    marker=dict(size=4, color=ACCENT, opacity=0.5),
    text=sku_text, hoverinfo="text", name="SKUs"
)

# ── Plot ────────────────────────────────────────────────────────────
fig = go.Figure(data=[edge_trace, sku_trace, sup_trace])
fig.update_layout(
    showlegend=True,
    legend=dict(font=dict(color=c["text"])),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(visible=False), yaxis=dict(visible=False),
    margin=dict(l=0, r=0, t=0, b=0), height=600,
    font=dict(color=c["muted"])
)

st.plotly_chart(fig, use_container_width=True)

# ── Single Points of Failure analysis ───────────────────────────────
st.markdown("### 🔴 Single Points of Failure — Most-Connected Supplier Nodes")
st.markdown("Suppliers with the highest number of SKU dependencies represent critical bottlenecks.")

spof_data = []
for n in sup_nodes:
    d = G.nodes[n]
    spof_data.append({
        "Supplier ID": n,
        "City": d.get("city", ""),
        "Dependencies": G.degree(n),
        "Reliability": f"{d.get('reliability', 0)*100:.1f}%",
        "Risk Score": f"{d.get('risk_score', 0)*100:.1f}%",
    })

spof_df = pd.DataFrame(spof_data).sort_values("Dependencies", ascending=False).head(10)
st.dataframe(spof_df, use_container_width=True, hide_index=True)

# ── Node detail inspector ───────────────────────────────────────────
st.markdown("### 🔍 Node Inspector")
all_nodes = [f"{n} ({G.nodes[n].get('node_type', 'unknown')})" for n in list(sup_nodes[:50]) + sku_sample[:20]]
selected = st.selectbox("Select a node to inspect", all_nodes)
sel_id = selected.split(" (")[0]

if sel_id in G.nodes:
    nd = G.nodes[sel_id]
    neighbors = list(G.neighbors(sel_id))
    st.markdown(f"""
<div class="glass-card">
  <div style="font-size:1.1rem;font-weight:700;color:var(--text);margin-bottom:8px">{sel_id}</div>
  <div style="font-size:.85rem;color:var(--muted)">
    Type: <b>{nd.get('node_type','—').title()}</b><br/>
    Connections: <b>{len(neighbors)}</b><br/>
    {'City: <b>' + nd.get('city','') + '</b><br/>' if nd.get('city') else ''}
    {'Category: <b>' + nd.get('category','') + '</b><br/>' if nd.get('category') else ''}
  </div>
  <div style="margin-top:8px;font-size:.8rem;color:var(--muted)">Connected to: {', '.join(neighbors[:15])}</div>
</div>""", unsafe_allow_html=True)

# Export
st.download_button("📥 Export Graph Data (CSV)", data=spof_df.to_csv(index=False),
                   file_name="CEDPA_Network_SPOF.csv", mime="text/csv")

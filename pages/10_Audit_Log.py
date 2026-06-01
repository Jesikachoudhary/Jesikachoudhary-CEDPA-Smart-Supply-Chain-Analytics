"""
Page 10 — Audit Log & System Monitor
Tracks all user actions: acknowledged alerts, scenario runs, exports.
"""
import streamlit as st
import pandas as pd
import datetime

from utils.state import init_state
from utils.theme import inject_css, render_sidebar_brand, get_colors

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(page_title="CEDPA — Audit Log", page_icon="📋", layout="wide")
inject_css()
init_state()
render_sidebar_brand()
c = get_colors()

st.markdown(
    '<div class="page-header">'
    '<h1>Audit Log & System Monitor</h1>'
    '<p>Complete trail of all user actions — acknowledged alerts, scenario runs, model retrains, and exports.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ═══ SYSTEM HEALTH ══════════════════════════════════════════════════
st.markdown("### System Health Monitor")

risk_model = st.session_state.get("risk_model")
alerts = st.session_state.get("alerts", [])
ts = st.session_state.get("data_generated_at", "—")

h1, h2, h3, h4 = st.columns(4)
health = [
    ("GBoost Accuracy", f"{risk_model.metrics['accuracy']*100:.2f}%" if risk_model else "—", "#10B981"),
    ("F1 Score", f"{risk_model.metrics['f1_score']*100:.1f}%" if risk_model else "—", "#38BDF8"),
    ("Active Alerts", str(len(alerts)), "#F59E0B"),
    ("Data Freshness", ts, "#8B5CF6"),
]
for col, (label, val, color) in zip([h1, h2, h3, h4], health):
    with col:
        st.markdown(f"""
<div class="glass-card" style="border-top:3px solid {color};text-align:center">
  <div class="kpi-title">{label}</div>
  <div style="font-size:1.4rem;font-weight:700;color:{color};font-family:'Outfit',sans-serif">{val}</div>
</div>""", unsafe_allow_html=True)

# ═══ AUDIT LOG TABLE ═══════════════════════════════════════════════
st.markdown("### User Activity Log")

audit = st.session_state.get("audit_log", [])

if not audit:
    st.info("No actions recorded yet. Acknowledge alerts, run scenarios, or export reports to populate the log.")
else:
    log_df = pd.DataFrame(audit)
    # Format for display
    if "ts" in log_df.columns:
        log_df["ts"] = pd.to_datetime(log_df["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    log_df.columns = [c.replace("_", " ").title() for c in log_df.columns]
    st.dataframe(log_df.sort_index(ascending=False), use_container_width=True, hide_index=True)

    # Export
    st.download_button("📥 Export Audit Log CSV",
                       data=log_df.to_csv(index=False),
                       file_name=f"CEDPA_Audit_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv")

    # Clear log
    if st.button("🗑️ Clear Audit Log"):
        st.session_state["audit_log"] = []
        st.rerun()

# ═══ ACTIVITY SUMMARY ═════════════════════════════════════════════
if audit:
    st.markdown("### Activity Summary")
    action_counts = pd.Series([a.get("action", "unknown") for a in audit]).value_counts()

    import plotly.express as px
    fig = px.bar(x=action_counts.index, y=action_counts.values,
                 color_discrete_sequence=["#38BDF8"],
                 labels={"x": "Action Type", "y": "Count"})
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=c["muted"]), height=250, margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

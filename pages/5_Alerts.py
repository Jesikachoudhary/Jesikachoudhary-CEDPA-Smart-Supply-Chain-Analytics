"""
Page 5 — Live Exception & Mitigation Feed
Fixed: break → continue in filter loop so all three filters work correctly.
Added: email-preview expanders for Critical alerts, audit-log tracking.
"""
import streamlit as st
import time

from utils.state import init_state
from utils.theme import inject_css, render_sidebar_brand

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(page_title="CEDPA — Alerts", page_icon="🚨", layout="wide")
inject_css()
init_state()
render_sidebar_brand()

alerts = st.session_state["alerts"]

# ── Title ───────────────────────────────────────────────────────────
st.markdown(
    '<div class="page-header">'
    '<h1>Live Exception & Mitigation Feed</h1>'
    '<p>Real-time priority-queued logistics alerts with automated resolution guidelines.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Metrics header ──────────────────────────────────────────────────
crit_count = len([a for a in alerts if a["priority"] == "Critical"])
warn_count = len([a for a in alerts if a["priority"] == "Warning"])
info_count = len([a for a in alerts if a["priority"] == "Info"])

c1, c2, c3 = st.columns(3)
for col, label, count, color in [
    (c1, "Critical Exceptions", crit_count, "#EF4444"),
    (c2, "Warning Notifications", warn_count, "#F59E0B"),
    (c3, "Standard Info Tags", info_count, "#38BDF8"),
]:
    with col:
        st.markdown(f"""
<div class="glass-card" style="border-left:4px solid {color};
     background:rgba({','.join(str(int(color.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.05);
     text-align:center;padding:15px">
  <span style="font-size:.75rem;color:{color};font-weight:700;text-transform:uppercase">{label}</span>
  <div style="font-size:2.5rem;font-family:'Outfit',sans-serif;font-weight:700;color:{color};margin-top:2px">
    {count} Active</div>
</div>""", unsafe_allow_html=True)

# ── Filters ─────────────────────────────────────────────────────────
st.markdown("#### Filter Dashboard Exceptions")
fc1, fc2, fc3 = st.columns(3)
with fc1:
    sel_priority = st.selectbox("Priority Level", ["All Priorities", "Critical", "Warning", "Info"])
with fc2:
    sel_category = st.selectbox("Threat Category",
        ["All Categories", "Disruption Risk", "Supplier Health", "Stockout Danger", "Inventory Reorder"])
with fc3:
    search_query = st.text_input("Search (SKU, City, Supplier, ID)", "").strip().lower()

# ── FIXED filter loop: 'continue' instead of 'break' ───────────────
filtered_alerts = []
for alt in alerts:
    # A. Priority filter
    if sel_priority != "All Priorities" and alt["priority"] != sel_priority:
        continue                       # ← was 'break' — skipped everything
    # B. Category filter
    if sel_category != "All Categories" and alt["category"] != sel_category:
        continue                       # ← was 'break'
    # C. Text search filter
    if search_query:
        match_str = " ".join([
            alt.get("id", ""), alt.get("sku_affected", ""),
            alt.get("supplier_name", ""), alt.get("city", ""),
            alt.get("recommendation", ""),
        ]).lower()
        if search_query not in match_str:
            continue                   # ← was 'break'
    filtered_alerts.append(alt)

# ── Alert timeline ──────────────────────────────────────────────────
st.markdown("#### Operational Logs Timeline")

if not filtered_alerts:
    st.info("No active exceptions match the selected filtering bounds.")
else:
    for idx, alt in enumerate(filtered_alerts):
        p_color = {"Critical": "#EF4444", "Warning": "#F59E0B"}.get(alt["priority"], "#38BDF8")
        bg_tint = "rgba(239,68,68,0.02)" if alt["priority"] == "Critical" else "rgba(30,41,59,0.2)"

        card_col, btn_col = st.columns([6, 1])

        with card_col:
            st.markdown(f"""
<div style="background:{bg_tint};border:1px solid rgba(255,255,255,0.06);
     border-left:5px solid {p_color};border-radius:8px;padding:16px 20px;margin-bottom:12px">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="font-weight:700;color:var(--text);font-size:1rem">{alt['title']}</span>
    <span style="font-family:monospace;background:rgba(255,255,255,0.06);padding:2px 6px;
          border-radius:4px;font-size:.75rem;color:var(--muted)">{alt['id']}</span>
  </div>
  <div style="color:var(--muted);font-size:.78rem;margin:5px 0">
    Category: <b>{alt['category']}</b> | Location: <b>{alt['city']} Hub</b> |
    SKU: <b>{alt['sku_affected']}</b> | Risk: <b>{alt['risk_score']*100:.1f}%</b>
  </div>
  <div style="color:var(--text);font-size:.85rem;font-weight:500;border-top:1px solid rgba(255,255,255,0.04);
       padding-top:6px;margin-top:6px">
    💡 <b>Action:</b> {alt['recommendation']}
  </div>
</div>""", unsafe_allow_html=True)

        with btn_col:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            if st.button("Acknowledge ✔", key=f"ack_{alt['id']}_{idx}", use_container_width=True):
                st.session_state["alerts"] = [a for a in st.session_state["alerts"] if a["id"] != alt["id"]]
                st.session_state.setdefault("audit_log", []).append(
                    {"action": "alert_acknowledged", "id": alt["id"], "ts": __import__("datetime").datetime.now().isoformat()})
                st.success(f"Alert {alt['id']} acknowledged.")
                time.sleep(0.3)
                st.rerun()

        # ── Mock email preview for Critical alerts ──────────
        if alt["priority"] == "Critical":
            with st.expander(f"📧 Mock Email Notification — {alt['id']}", expanded=False):
                st.markdown(f"""
**From:** cedpa-alerts@supply-chain.io  
**To:** ops-manager@company.com  
**Subject:** 🔴 CRITICAL — {alt['title']}

---

**Alert ID:** {alt['id']}  
**Location:** {alt['city']} Hub  
**Affected SKU:** {alt['sku_affected']}  
**Risk Probability:** {alt['risk_score']*100:.1f}%  

**Recommended Action:**  
{alt['recommendation']}

---
*This is a simulated notification generated by the CEDPA Alert Engine.*
""")

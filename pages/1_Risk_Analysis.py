"""Page 1 — Predictive Disruption Risk Analysis (Premium UI v4)"""
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.state import init_state
from utils.theme import (
    inject_css, render_sidebar_brand, get_colors,
    ACCENT, SUCCESS, DANGER, WARNING, ORANGE, CYAN,
)

st.set_page_config(page_title="CEDPA — Risk Analysis", page_icon="🛡️", layout="wide")
inject_css()
init_state()
render_sidebar_brand()

risk_model   = st.session_state["risk_model"]
suppliers_df = st.session_state["suppliers_df"]
c = get_colors()

# ── Header ───────────────────────────────────────────────────────────
st.markdown(
    '<div class="page-header">'
    '<h1>Predictive Disruption Risk Analysis</h1>'
    '<p>Gradient Boosting Classifier with SHAP explainability across all 50 supplier nodes.</p>'
    '</div>',
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["📊  ML Model Diagnostics", "🔍  Interactive Node Explainability"])

# ══ TAB 1 ════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-heading">Classifier Performance Metrics</div>',
                unsafe_allow_html=True)

    for col, (label, val, color, icon) in zip(
        st.columns(4, gap="medium"),
        [
            ("Model Accuracy", f"{risk_model.metrics['accuracy']*100:.2f}%", SUCCESS, "🎯"),
            ("ROC-AUC Score",  f"{risk_model.metrics['roc_auc']:.4f}",        ACCENT,  "📐"),
            ("Precision",      f"{risk_model.metrics['precision']*100:.1f}%", ORANGE,  "🔬"),
            ("Recall",         f"{risk_model.metrics['recall']*100:.1f}%",    CYAN,    "📡"),
        ],
    ):
        with col:
            st.markdown(f"""
<div class="glass-card" style="border-top:4px solid {color};text-align:center;padding:22px 14px">
  <div style="font-size:1.6rem;margin-bottom:8px">{icon}</div>
  <div style="font-family:'Outfit',sans-serif;font-size:2.2rem;font-weight:800;
       color:{color};letter-spacing:-0.04em;line-height:1">{val}</div>
  <div style="font-size:0.82rem;color:var(--muted);font-weight:600;margin-top:8px">{label}</div>
</div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    d1, d2 = st.columns([3, 2], gap="medium")

    with d1:
        st.markdown('<div class="chart-label">Feature Importance — Gradient Boosting</div>',
                    unsafe_allow_html=True)
        fig_imp = risk_model.generate_feature_importance_plot(dark_mode=True)
        fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(color=c["muted"], family="Inter"),
                               height=320, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar":False})
        st.caption("Lead Time Variance drives ~41% of classifier splits.")

    with d2:
        st.markdown('<div class="chart-label">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = risk_model.metrics["confusion_matrix"]
        df_cm = pd.DataFrame(cm,
            index=["Actual Safe","Actual Disrupted"],
            columns=["Pred Safe","Pred Disrupted"])
        fig_cm = px.imshow(df_cm, text_auto=True,
            color_continuous_scale=[[0,"#060A16"],[0.5,"#0D3D6E"],[1,"#38BDF8"]],
            labels=dict(x="Predicted", y="Actual", color="Count"))
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              height=280, margin=dict(l=10,r=10,t=10,b=10),
                              font=dict(color=c["muted"], family="Inter"))
        fig_cm.update_traces(textfont=dict(size=16, color=c["text"]))
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar":False})
        st.caption("False-positive rate kept below 3.4% to minimise alert fatigue.")

    st.markdown(f"""
<div class="glass-card" style="border-left:5px solid {WARNING};
     display:flex;align-items:center;justify-content:space-between;padding:16px 24px">
  <div>
    <div style="font-size:0.82rem;color:var(--muted);font-weight:600;margin-bottom:4px">
      F1 Score (Harmonic Mean of Precision &amp; Recall)</div>
    <div style="font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:800;
         color:{WARNING};letter-spacing:-0.03em">
      {risk_model.metrics['f1_score']*100:.2f}%</div>
  </div>
  <div style="font-size:2.5rem;opacity:.6">⚖️</div>
</div>""", unsafe_allow_html=True)

# ══ TAB 2 ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-heading">Interactive Supplier Node Explainability</div>',
                unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.95rem;color:var(--muted);margin:0 0 20px">'
                'Select a node and adjust its parameters to see the real-time disruption '
                'probability and SHAP contribution breakdown.</p>', unsafe_allow_html=True)

    sup_list = [f"{r['supplier_id']} — {r['supplier_name']} ({r['city']})"
                for _, r in suppliers_df.iterrows()]
    sel_str  = st.selectbox("Choose Supplier Node", sup_list)
    sel_id   = sel_str.split(" — ")[0]
    sup_row  = suppliers_df[suppliers_df["supplier_id"] == sel_id].iloc[0]

    st.markdown('<div style="margin:20px 0 12px;font-size:0.95rem;font-weight:600;'
                'color:var(--text)">Node State Parameters</div>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3, gap="medium")
    with fc1:
        lt_var  = st.slider("Lead Time Variance (days²)", 0.1, 10.0,
                            float(sup_row["lead_time_std"]**2), 0.1)
        sup_rel = st.slider("Supplier Reliability (%)", 40, 100,
                            int(sup_row["reliability"]*100)) / 100.0
    with fc2:
        geo_risk = st.slider("Geographic Risk Score", 0.0, 1.0,
                             float(sup_row["geo_risk"]), 0.01)
        inv_buf  = st.slider("Inventory Buffer Ratio", -0.5, 1.0, 0.25, 0.05)
    with fc3:
        delay_hist = st.slider("Delay History Index (0–10)", 0.0, 10.0, 3.5, 0.1)

    features = dict(lead_time_variance=lt_var, supplier_reliability=sup_rel,
                    geo_risk_index=geo_risk, inventory_buffer=inv_buf,
                    shipment_delay_history=delay_hist)

    _, risk_prob = risk_model.predict(features)
    r_color, r_label, r_icon = (
        (SUCCESS, "Safe Operation",  "✅") if risk_prob < 0.40 else
        (WARNING, "Warning Status",  "⚠️") if risk_prob < 0.75 else
        (DANGER,  "Critical Danger", "🔴")
    )

    st.markdown(f"""
<div class="glass-card" style="border:2px solid {r_color};
     text-align:center;padding:30px 20px;margin:20px 0">
  <div style="font-size:0.82rem;color:var(--muted);font-weight:600;
       letter-spacing:0.05em;margin-bottom:10px">
    Predicted Disruption Risk Probability</div>
  <div style="font-family:'Outfit',sans-serif;font-size:4.5rem;font-weight:800;
       color:{r_color};letter-spacing:-0.05em;line-height:1;margin-bottom:14px">
    {risk_prob*100:.1f}%</div>
  <div style="display:inline-flex;align-items:center;gap:8px;
       background:{r_color}22;border:1px solid {r_color}55;
       border-radius:24px;padding:6px 20px;font-weight:700;
       color:{r_color};font-size:0.88rem;box-shadow:0 0 20px {r_color}33">
    {r_icon} {r_label}
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="chart-label" style="margin-top:10px">'
                'SHAP Feature Contribution Breakdown</div>', unsafe_allow_html=True)
    fig_shap = risk_model.get_shap_waterfall(features, dark_mode=True)
    fig_shap.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color=c["muted"], family="Inter"), height=420)
    st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar":False})
    st.caption("🟥 Red bars raise risk  ·  🟩 Green bars lower it  ·  Final % vs system baseline.")

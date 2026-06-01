"""Page 3 — Geographic Intelligence Map (Premium UI v4)"""
import io
import numpy as np
import streamlit as st

from utils.state import init_state
from utils.theme import inject_css, render_sidebar_brand, SUCCESS, DANGER, WARNING

FOLIUM_AVAILABLE = False
try:
    import folium
    from folium.plugins import HeatMap
    FOLIUM_AVAILABLE = True
except Exception:
    pass

st.set_page_config(page_title="CEDPA — Geographic Map", page_icon="🌐", layout="wide")
inject_css()
init_state()
render_sidebar_brand()

suppliers_df = st.session_state["suppliers_df"]
skus_df      = st.session_state["skus_df"]

st.markdown(
    '<div class="page-header">'
    '<h1>Geographic Intelligence Map</h1>'
    '<p>Global logistics node monitoring across 15 cities with disruption risk markers.</p>'
    '</div>',
    unsafe_allow_html=True,
)

if not FOLIUM_AVAILABLE:
    st.error("Folium not installed. Run: pip install folium")
    st.map(suppliers_df)
else:
    m   = folium.Map(location=[20.0, 15.0], zoom_start=2,
                     tiles="CartoDB dark_matter", control_scale=True)
    rng = np.random.default_rng(303)
    heat_data = []

    for _, s in suppliers_df.iterrows():
        rs  = min(max(s["geo_risk"] + s["reliability"] * 0.05, 0.05), 0.95)
        heat_data.append([s["latitude"], s["longitude"],
                          float(rng.uniform(100, 1500)) / 1000.0])

        if rs > 0.65:
            mc, bg, rec = "red",    "#EF4444", "Divert shipments immediately."
        elif rs > 0.35:
            mc, bg, rec = "orange", "#F59E0B", "Monitor lead times, boost safety stock 15%."
        else:
            mc, bg, rec = "green",  "#10B981", "Maintain standard operations."

        top_skus = ", ".join(skus_df.sample(n=3, random_state=rng)["sku_id"].values)
        popup_html = f"""
<div style="font-family:'Segoe UI',Arial,sans-serif;width:230px;font-size:12px;color:#1E293B">
  <h4 style="margin:0 0 8px;color:#1E3A8A;font-size:14px">{s['city']} Node</h4>
  <div style="display:flex;justify-content:space-between;margin-bottom:6px">
    <span style="font-weight:600">Supplier ID</span>
    <code style="background:var(--text);padding:1px 5px;border-radius:4px">{s['supplier_id']}</code>
  </div>
  <div style="display:flex;justify-content:space-between;margin-bottom:8px">
    <span style="font-weight:600">Disruption Risk</span>
    <span style="background:{bg};color:#fff;padding:2px 8px;border-radius:10px;
          font-weight:700;font-size:11px">{rs*100:.1f}%</span>
  </div>
  <div style="margin-bottom:8px">
    <span style="font-weight:600">Top SKUs:</span><br>
    <span style="color:#475569">{top_skus}</span>
  </div>
  <div style="border-top:1px solid var(--text);padding-top:6px">
    <span style="font-weight:700;color:#0F766E">Recommendation:</span><br>
    <span style="color:#334155;font-size:11px">{rec}</span>
  </div>
</div>"""
        folium.Marker(
            location=[s["latitude"], s["longitude"]],
            popup=folium.Popup(folium.IFrame(popup_html, width=250, height=185), max_width=270),
            icon=folium.Icon(color=mc, icon="info-sign"),
            tooltip=f"{s['city']} — Click for details",
        ).add_to(m)

    HeatMap(heat_data, radius=18, blur=15, min_opacity=0.35).add_to(m)

    mc1, ic1 = st.columns([4, 1], gap="medium")
    with mc1:
        st.components.v1.html(m._repr_html_(), height=530, scrolling=False)
        st.caption("Hover nodes to inspect hubs · Overlay = shipment volume density.")
    with ic1:
        st.markdown('<div class="section-heading" style="font-size:1.05rem">'
                    'Zone Legend</div>', unsafe_allow_html=True)
        for color, label, desc in [
            (DANGER,  "🔴 High Risk", "Coastal terminals with severe weather / congestion."),
            (WARNING, "🟡 Moderate",  "Nodes in customs ERP transition."),
            (SUCCESS, "🟢 Safe",      "Inland corridors with high schedule consistency."),
        ]:
            st.markdown(f"""
<div class="glass-card" style="border-left:4px solid {color};padding:12px 14px;margin-bottom:8px">
  <div style="font-weight:700;color:var(--text);font-size:0.88rem;margin-bottom:3px">{label}</div>
  <div style="font-size:0.78rem;color:var(--muted);line-height:1.5">{desc}</div>
</div>""", unsafe_allow_html=True)

        try:
            html_buf = io.BytesIO()
            m.save(html_buf, close_file=False)
            st.download_button("📥 Export Map", data=html_buf.getvalue(),
                               file_name="CEDPA_Map.html", mime="text/html",
                               use_container_width=True)
        except Exception as e:
            st.error(str(e))

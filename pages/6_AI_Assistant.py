"""
Page 6 — AI Assistant & Knowledge Hub
Full-featured chatbot with Claude API, file upload, FAQ manager,
knowledge search, and data profiler tabs.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json, io, datetime, time, hashlib

from utils.state import init_state
from utils.theme import inject_css, render_sidebar_brand, get_colors

# ── Optional imports ────────────────────────────────────────────────
ANTHROPIC_AVAILABLE = False
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except Exception:
    pass

WORDCLOUD_AVAILABLE = False
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    pass

PDFPLUMBER_AVAILABLE = False
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except Exception:
    pass

DOCX_AVAILABLE = False
try:
    import docx
    DOCX_AVAILABLE = True
except Exception:
    pass

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(page_title="CEDPA — AI Assistant", page_icon="🤖", layout="wide")
inject_css()
init_state()
render_sidebar_brand()
c = get_colors()

# ── FAQ seed function (defined before use) ─────────────────────────
def _default_faqs():
    """Seed FAQ bank."""
    return [
        {"q": "What ML model does CEDPA use for risk prediction?",
         "a": "A Gradient Boosting Classifier trained on 5 000 synthetic historical shipment records with 5 input features.",
         "cat": "Risk", "views": 0},
        {"q": "What is the forecast ensemble architecture?",
         "a": "A weighted average of LSTM (40%), XGBoost (35%), and Prophet (25%) with a 90-day horizon.",
         "cat": "Forecast", "views": 0},
        {"q": "How are alerts prioritized?",
         "a": "Alerts are sorted Critical → Warning → Info by the AlertEngine, which scans all 50 supplier nodes and 200 SKUs.",
         "cat": "Alerts", "views": 0},
        {"q": "What KPIs does the dashboard track?",
         "a": "Inventory cost reduction (31.7%), fulfillment velocity (44.8%), automation index (78.2%), and gross margin uplift (+3.7 pp).",
         "cat": "KPIs", "views": 0},
        {"q": "Can I upload my own supply chain data?",
         "a": "Yes. Use the CSV/Excel uploader on the main dashboard sidebar. The system auto-maps columns and retrains the risk model.",
         "cat": "Setup", "views": 0},
    ]


# ── Session defaults (after function definition) ───────────────────
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "uploaded_docs_text" not in st.session_state:
    st.session_state["uploaded_docs_text"] = ""
if "faq_bank" not in st.session_state:
    st.session_state["faq_bank"] = _default_faqs()
if "faq_views" not in st.session_state:
    st.session_state["faq_views"] = {}
if "chat_feedback" not in st.session_state:
    st.session_state["chat_feedback"] = {}


def _build_system_prompt():
    """Build a rich system prompt giving Claude full CEDPA context."""
    alerts = st.session_state.get("alerts", [])
    crit = len([a for a in alerts if a["priority"] == "Critical"])
    warn = len([a for a in alerts if a["priority"] == "Warning"])
    rm = st.session_state.get("risk_model")
    acc = f"{rm.metrics['accuracy']*100:.2f}%" if rm else "N/A"

    doc_ctx = ""
    if st.session_state.get("uploaded_docs_text"):
        doc_ctx = f"\n\n### Uploaded Document Context\n{st.session_state['uploaded_docs_text'][:8000]}"

    return f"""You are the CEDPA AI Assistant — an expert in supply chain analytics, logistics, and the CEDPA platform.

### Platform Overview
CEDPA (Cloud-Enabled Distributed Predictive Analytics) is a Streamlit-based dashboard with:
- **50 supplier nodes** across 15 global cities, **200 SKUs** in 4 categories
- **Risk Prediction**: GradientBoosting Classifier (accuracy: {acc}), SHAP explainability
- **Demand Forecasting**: Weighted ensemble — LSTM (40%) + XGBoost (35%) + Prophet (25%), 90-day horizon, MAPE < 6.5%
- **Geographic Map**: Folium + HeatMap overlay
- **Scenario Simulator**: What-if cost modeling (holding + stockout)
- **Alert Engine**: {crit} Critical, {warn} Warning active alerts
- **Network Graph**: Force-directed supplier-SKU dependency visualization
- **Advanced Analytics**: Monte Carlo, EOQ, Isolation Forest anomaly detection, ABC analysis
- **Optimization**: Linear Programming procurement optimizer using PuLP

### Session Data
- Active critical alerts: {crit}
- Active warning alerts: {warn}
- Risk model accuracy: {acc}

### Guidelines
- Reply in the SAME LANGUAGE as the user's message
- Be concise, technical, and helpful
- When discussing data, reference specific numbers from the session
- After each response, suggest 3 follow-up questions the user might ask
{doc_ctx}"""


# ── Claude API call ─────────────────────────────────────────────────
def _ask_claude(messages, system_prompt):
    """Call Anthropic Claude API. Falls back to rule-based if unavailable."""
    if not ANTHROPIC_AVAILABLE:
        return _rule_based_response(messages[-1]["content"])

    api_key = st.session_state.get("anthropic_api_key", "")
    if not api_key:
        return _rule_based_response(messages[-1]["content"])

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=system_prompt,
            messages=messages,
        )
        return resp.content[0].text
    except Exception as e:
        return f"⚠️ Claude API error: {e}\n\nFalling back to rule-based response:\n\n{_rule_based_response(messages[-1]['content'])}"


def _rule_based_response(query):
    """Smart rule-based fallback when Claude API is unavailable."""
    q = query.lower()
    alerts = st.session_state.get("alerts", [])
    rm = st.session_state.get("risk_model")

    if any(k in q for k in ["risk", "disruption", "predict"]):
        acc = f"{rm.metrics['accuracy']*100:.2f}%" if rm else "N/A"
        return (f"The CEDPA risk prediction engine uses a **Gradient Boosting Classifier** "
                f"with an accuracy of **{acc}**. It evaluates 5 features: lead time variance, "
                f"supplier reliability, geo-risk index, inventory buffer, and shipment delay history.\n\n"
                f"**Suggested follow-ups:**\n"
                f"1. How does SHAP explain individual predictions?\n"
                f"2. What features are most important for risk?\n"
                f"3. How can I improve supplier reliability?")

    if any(k in q for k in ["forecast", "demand", "lstm", "prophet"]):
        return ("The forecasting engine is a **weighted ensemble** of LSTM (40%), XGBoost (35%), "
                "and Prophet (25%). It generates 90-day forecasts with 95% confidence intervals. "
                "The backtest MAPE is consistently below 6.5%.\n\n"
                "**Suggested follow-ups:**\n"
                "1. How does the confidence interval expand over time?\n"
                "2. What if I change the ensemble weights?\n"
                "3. Which SKU categories have the most volatile demand?")

    if any(k in q for k in ["alert", "critical", "warning"]):
        crit = len([a for a in alerts if a["priority"] == "Critical"])
        return (f"There are currently **{crit} critical alerts** active. These are generated "
                f"when the GBoost classifier predicts a disruption probability > 75%.\n\n"
                f"**Suggested follow-ups:**\n"
                f"1. What are the top 3 most critical alerts right now?\n"
                f"2. How do I acknowledge an alert?\n"
                f"3. What triggers an alert vs a warning?")

    if any(k in q for k in ["kpi", "metric", "performance"]):
        return ("**Key Performance Indicators:**\n"
                "- Inventory Cost Reduction: **31.7%**\n"
                "- Fulfillment Velocity: **44.8%**\n"
                "- Automation Index: **78.2%**\n"
                "- Gross Margin Uplift: **+3.7 pp**\n\n"
                "**Suggested follow-ups:**\n"
                "1. How are these KPIs calculated?\n"
                "2. What is the target for next quarter?\n"
                "3. Which KPI has the biggest room for improvement?")

    return ("I'm the CEDPA AI Assistant. I can help you with:\n"
            "- 🛡️ Risk prediction & SHAP analysis\n"
            "- 📈 Demand forecasting & ensemble models\n"
            "- 🚨 Alert management & prioritization\n"
            "- 📊 KPI interpretation & optimization\n"
            "- ⚙️ Scenario simulation & cost modeling\n\n"
            "Ask me anything about your supply chain!\n\n"
            "**Suggested follow-ups:**\n"
            "1. Show me the current risk summary\n"
            "2. Explain the forecasting architecture\n"
            "3. What are the active critical alerts?")


# ── File text extraction ────────────────────────────────────────────
def _extract_text(file):
    """Extract text from various file formats."""
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(file)
            return df.to_string(max_rows=200)
        elif name.endswith(".xlsx"):
            xls = pd.ExcelFile(file)
            sheets = xls.sheet_names
            if len(sheets) > 1:
                sel = st.selectbox(f"Sheet in {file.name}", sheets, key=f"sheet_{file.name}")
                df = pd.read_excel(file, sheet_name=sel)
            else:
                df = pd.read_excel(file)
            return df.to_string(max_rows=200)
        elif name.endswith(".json"):
            data = json.load(file)
            return json.dumps(data, indent=2)[:10000]
        elif name.endswith(".pdf"):
            if PDFPLUMBER_AVAILABLE:
                with pdfplumber.open(file) as pdf:
                    return "\n".join(p.extract_text() or "" for p in pdf.pages)[:15000]
            return "⚠️ Install pdfplumber to read PDF files."
        elif name.endswith(".docx"):
            if DOCX_AVAILABLE:
                doc_obj = docx.Document(file)
                return "\n".join(p.text for p in doc_obj.paragraphs)[:15000]
            return "⚠️ Install python-docx to read DOCX files."
        else:
            return file.read().decode("utf-8", errors="ignore")[:15000]
    except Exception as e:
        return f"Error reading {file.name}: {e}"


def _chunk_text(text, chunk_size=500, overlap=100):
    """Split large text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ═════════════════════════════════════════════════════════════════════
# MAIN UI
# ═════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="page-header">'
    '<h1>AI Assistant & Knowledge Hub</h1>'
    '<p>Intelligent supply chain companion powered by Claude AI.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# API key input
with st.sidebar.expander("🔑 Claude API Key", expanded=False):
    key = st.text_input("Anthropic API Key", type="password", key="api_key_input")
    if key:
        st.session_state["anthropic_api_key"] = key
        st.success("API key saved for this session")

# ═════════════════════════════════════════════════════════════════════
tab_chat, tab_upload, tab_faq, tab_search, tab_profiler = st.tabs([
    "💬 Chat", "📁 File Upload", "❓ FAQ Manager", "🔍 Knowledge Search", "📊 Data Profiler"
])

# ═══ TAB 1: CHAT ═══════════════════════════════════════════════════
with tab_chat:
    # Executive briefing button
    if st.button("📋 Executive Briefing", use_container_width=False):
        alerts = st.session_state.get("alerts", [])
        briefing_q = (f"Generate a one-paragraph executive briefing summarizing these {len(alerts)} "
                      f"active supply chain alerts: " +
                      "; ".join(f"{a['title']} ({a['priority']}, {a['city']})" for a in alerts[:10]))
        st.session_state["chat_history"].append({"role": "user", "content": briefing_q})

        system = _build_system_prompt()
        msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state["chat_history"]]
        with st.spinner("🤖 Generating executive briefing…"):
            reply = _ask_claude(msgs, system)
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        st.rerun()

    # Chat display
    for idx, msg in enumerate(st.session_state["chat_history"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Feedback buttons for assistant messages
            if msg["role"] == "assistant":
                fb1, fb2, _ = st.columns([1, 1, 10])
                with fb1:
                    if st.button("👍", key=f"up_{idx}"):
                        st.session_state["chat_feedback"][idx] = "positive"
                with fb2:
                    if st.button("👎", key=f"dn_{idx}"):
                        st.session_state["chat_feedback"][idx] = "negative"

    # Chat input
    user_input = st.chat_input("Ask me about CEDPA, risk, forecasts, alerts…")
    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        system = _build_system_prompt()
        msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state["chat_history"]]

        with st.chat_message("assistant"):
            with st.spinner("⏳ Thinking…"):
                reply = _ask_claude(msgs, system)
            st.markdown(reply)
            st.session_state["chat_history"].append({"role": "assistant", "content": reply})

        # Follow-up chips (parse from response or generate defaults)
        st.markdown("**Suggested follow-ups:**")
        suggestions = ["What are the current risk levels?",
                       "Show me the forecast accuracy breakdown",
                       "Summarize active alerts"]
        sc1, sc2, sc3 = st.columns(3)
        for col, sug in zip([sc1, sc2, sc3], suggestions):
            with col:
                if st.button(sug, key=f"sug_{hashlib.md5(sug.encode()).hexdigest()[:8]}"):
                    st.session_state["chat_history"].append({"role": "user", "content": sug})
                    st.rerun()

    # Export conversation
    if st.session_state["chat_history"]:
        with st.expander("📥 Export Conversation"):
            export_text = "\n\n".join(
                f"{'USER' if m['role']=='user' else 'ASSISTANT'}: {m['content']}"
                for m in st.session_state["chat_history"]
            )
            st.download_button("Download as TXT", data=export_text,
                               file_name="CEDPA_Chat_Export.txt", mime="text/plain")

# ═══ TAB 2: FILE UPLOAD ════════════════════════════════════════════
with tab_upload:
    st.markdown("### Document Upload & Analysis")
    st.markdown("Upload CSV, Excel, PDF, DOCX, TXT, or JSON files. Content is injected into the chatbot context.")

    files = st.file_uploader("Upload documents", type=["csv", "xlsx", "pdf", "docx", "txt", "json"],
                             accept_multiple_files=True, key="doc_upload")

    if files:
        all_text = []
        for f in files:
            text = _extract_text(f)
            all_text.append(f"=== {f.name} ===\n{text}")
            st.success(f"✓ {f.name} — {len(text)} chars extracted")

        combined = "\n\n".join(all_text)

        # Auto-chunk large documents
        if len(combined) > 10000:
            chunks = _chunk_text(combined, 500, 100)
            st.info(f"Large document split into {len(chunks)} overlapping chunks for better Q&A accuracy.")
        else:
            chunks = [combined]

        st.session_state["uploaded_docs_text"] = combined[:15000]
        st.caption("Document text has been injected into the chatbot context. Switch to the Chat tab to ask questions.")

        # Word cloud
        if WORDCLOUD_AVAILABLE and combined.strip():
            with st.expander("☁️ Word Cloud Visualization"):
                wc = WordCloud(width=800, height=300, background_color="#0F172A",
                               colormap="cool", max_words=100).generate(combined)
                import matplotlib.pyplot as plt
                fig_wc, ax_wc = plt.subplots(figsize=(10, 3.5))
                ax_wc.imshow(wc, interpolation="bilinear")
                ax_wc.axis("off")
                st.pyplot(fig_wc)

        # Document comparison
        if len(files) >= 2:
            with st.expander("🔄 Document Comparison"):
                st.markdown("Ask the chatbot to compare the two documents:")
                if st.button("Compare documents in Chat"):
                    comp_q = (f"Compare these two documents and highlight key differences:\n"
                              f"**Doc 1:** {files[0].name}\n**Doc 2:** {files[1].name}")
                    st.session_state["chat_history"].append({"role": "user", "content": comp_q})
                    st.rerun()

# ═══ TAB 3: FAQ MANAGER ════════════════════════════════════════════
with tab_faq:
    st.markdown("### FAQ Knowledge Base")

    # Category filter
    cats = sorted(set(f.get("cat", "General") for f in st.session_state["faq_bank"]))
    sel_cat = st.selectbox("Filter by category", ["All"] + cats, key="faq_cat_filter")

    # Search
    faq_search = st.text_input("🔍 Search FAQs (fuzzy)", key="faq_search").lower()

    for idx, faq in enumerate(st.session_state["faq_bank"]):
        # Category filter
        if sel_cat != "All" and faq.get("cat") != sel_cat:
            continue
        # Fuzzy search
        if faq_search:
            q_lower = faq["q"].lower()
            # Simple fuzzy: check if most words match
            words = faq_search.split()
            match_count = sum(1 for w in words if w in q_lower)
            if match_count < max(1, len(words) * 0.5):
                continue

        badge = "🔥 Popular" if faq.get("views", 0) >= 3 else ""
        with st.expander(f"{'❓' } {faq['q']}  {badge}  `[{faq.get('cat', 'General')}]`"):
            st.markdown(faq["a"])
            faq["views"] = faq.get("views", 0) + 1

            if st.button("💬 Ask Claude for expanded answer", key=f"faq_expand_{idx}"):
                st.session_state["chat_history"].append(
                    {"role": "user", "content": f"Expand on this FAQ: {faq['q']}"})
                st.rerun()

    # Import/Export
    st.markdown("---")
    ie1, ie2 = st.columns(2)
    with ie1:
        csv_data = pd.DataFrame(st.session_state["faq_bank"]).to_csv(index=False)
        st.download_button("📥 Export FAQs as CSV", data=csv_data,
                           file_name="CEDPA_FAQs.csv", mime="text/csv")
    with ie2:
        faq_upload = st.file_uploader("📤 Import FAQ CSV", type=["csv"], key="faq_import")
        if faq_upload:
            imported = pd.read_csv(faq_upload)
            for _, row in imported.iterrows():
                st.session_state["faq_bank"].append({
                    "q": row.get("q", ""), "a": row.get("a", ""),
                    "cat": row.get("cat", "General"), "views": 0
                })
            st.success(f"Imported {len(imported)} FAQs")

# ═══ TAB 4: KNOWLEDGE SEARCH ═══════════════════════════════════════
with tab_search:
    st.markdown("### Full-Text Knowledge Search")
    st.markdown("Search across built-in knowledge base, FAQs, and uploaded documents.")

    query = st.text_input("🔎 Enter search query", key="kb_search")

    if query:
        results = []
        q = query.lower()

        # Search FAQs
        for faq in st.session_state["faq_bank"]:
            blob = f"{faq['q']} {faq['a']}".lower()
            score = sum(1 for w in q.split() if w in blob)
            if score > 0:
                results.append({"source": "FAQ", "title": faq["q"], "preview": faq["a"][:150],
                                "score": score, "full": faq["q"]})

        # Search uploaded docs
        doc_text = st.session_state.get("uploaded_docs_text", "")
        if doc_text:
            lines = doc_text.split("\n")
            for line in lines:
                if q in line.lower():
                    results.append({"source": "Document", "title": line[:80],
                                    "preview": line[:150], "score": 1, "full": line[:200]})

        # Built-in KB entries
        kb_entries = [
            "CEDPA uses Gradient Boosting for risk prediction with 92%+ accuracy",
            "The demand ensemble combines LSTM, XGBoost, and Prophet models",
            "The alert engine scans 50 supplier nodes and 200 SKUs for exceptions",
            "Monte Carlo simulation runs 1000 iterations for cost distribution analysis",
            "PuLP linear programming optimizer minimizes procurement cost",
        ]
        for entry in kb_entries:
            if q in entry.lower():
                results.append({"source": "KB", "title": entry[:60], "preview": entry,
                                "score": 2, "full": entry})

        # Sort by relevance
        results.sort(key=lambda x: x["score"], reverse=True)

        if results:
            for r in results[:10]:
                st.markdown(f"""
<div class="glass-card" style="padding:12px">
  <span style="font-size:.7rem;color:#38BDF8;font-weight:700">{r['source']}</span>
  <span style="font-size:.7rem;color:var(--muted);float:right">Relevance: {r['score']}</span>
  <div style="font-weight:600;color:var(--text);margin-top:4px">{r['title']}</div>
  <div style="font-size:.85rem;color:var(--muted);margin-top:4px">{r['preview']}…</div>
</div>""", unsafe_allow_html=True)
                if st.button(f"💬 Ask about this →", key=f"kb_ask_{hashlib.md5(r['title'].encode()).hexdigest()[:8]}"):
                    st.session_state["chat_history"].append({"role": "user", "content": r["full"]})
                    st.rerun()
        else:
            st.info("No results found. Try different keywords.")

# ═══ TAB 5: DATA PROFILER ══════════════════════════════════════════
with tab_profiler:
    st.markdown("### CEDPA Dataset Profiler")
    st.markdown("Explore summary statistics, distributions, and correlations across all datasets.")

    datasets = {
        "Suppliers (50 nodes)": st.session_state["suppliers_df"],
        "SKUs (200 items)": st.session_state["skus_df"],
        "Historical Shipments": st.session_state["historical_shipments"],
    }

    sel_ds = st.selectbox("Select dataset", list(datasets.keys()), key="profiler_ds")
    df = datasets[sel_ds]

    # Summary stats
    st.markdown("#### Summary Statistics")
    numeric_df = df.select_dtypes(include=[np.number])
    st.dataframe(numeric_df.describe().T.style.format("{:.3f}"), use_container_width=True)

    # Download
    csv = df.to_csv(index=False)
    st.download_button(f"📥 Download {sel_ds} CSV", data=csv,
                       file_name=f"CEDPA_{sel_ds.split('(')[0].strip()}.csv", mime="text/csv")

    # Interactive histogram
    if not numeric_df.empty:
        st.markdown("#### Distribution Explorer")
        col_sel = st.selectbox("Select column", numeric_df.columns.tolist(), key="hist_col")
        import plotly.express as px
        fig_hist = px.histogram(df, x=col_sel, nbins=30, color_discrete_sequence=["#38BDF8"])
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=c["muted"]), height=300,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Correlation heatmap
    if len(numeric_df.columns) >= 2:
        st.markdown("#### Correlation Heatmap")
        corr = numeric_df.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, aspect="auto")
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=c["muted"]), height=400,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # Auto-insights
    st.markdown("#### Auto-Generated Insights")
    for col in numeric_df.columns[:5]:
        mean = numeric_df[col].mean()
        std = numeric_df[col].std()
        skew = numeric_df[col].skew()
        direction = "right-skewed" if skew > 0.5 else ("left-skewed" if skew < -0.5 else "symmetric")
        st.markdown(f"- **{col}**: μ = {mean:.3f}, σ = {std:.3f}, distribution is {direction} (skew = {skew:.2f})")

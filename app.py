"""
app.py
======
Main entry point for the DecisionIQ Streamlit app.

Design decision: app.py stays thin — it's just page routing + layout.
All business logic lives in modules/ so it stays testable outside
Streamlit and so each module can be reused by the PDF report generator
without duplicating code.

v2 changes (enterprise polish pass):
- Enterprise dark-sidebar / white-content theme via injected CSS
- Landing page + data-gating: no analytics page is usable until the
  user uploads a CSV or clicks "Use Demo Dataset"
- "Ask AI" renamed to "AI Strategy Consultant", rebuilt as a chat-style
  Question -> SQL -> Result -> Summary -> Recommendations flow
- KPI cards upgraded with icon + value + trend arrow + interpretation
- Decision Engine findings rendered as bordered cards with a
  severity/impact/recommendation/priority layout

NOTE: a few fields referenced below (month-over-month deltas for trend
arrows, and Business Impact / Estimated Benefit / Priority on Decision
Engine findings) are currently derived here with light placeholder logic
because analytics.py / decision_engine.py don't emit them yet. Those two
modules are next in line — once they return real trend deltas and
impact/benefit/priority fields, swap the placeholder logic below for the
real values (marked with `# TODO(module-upgrade)` comments).
"""

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env so os.environ.get("GEMINI_API_KEY") works in modules/ai.py

from modules.data import ingest_pipeline, load_from_sqlite, REQUIRED_COLUMNS
from modules import analytics
from modules import decision_engine
from modules import forecasting
from modules import ai
from modules import reports

st.set_page_config(
    page_title="DecisionIQ | Pharma Decision Intelligence",
    page_icon="💊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Enterprise theme (dark sidebar / white content / blue-green-orange-red)
# ---------------------------------------------------------------------------
PRIMARY = "#2563EB"    # blue
SUCCESS = "#16A34A"    # green
WARNING = "#F59E0B"    # orange
CRITICAL = "#DC2626"   # red
INK = "#0F172A"        # dark sidebar
CARD_BG = "#FFFFFF"
MUTED = "#64748B"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #F8FAFC; }}
    section[data-testid="stSidebar"] {{
        background-color: {INK};
    }}
    section[data-testid="stSidebar"] * {{
        color: #E2E8F0 !important;
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        font-size: 0.95rem;
    }}
    h1, h2, h3 {{
        font-family: "Segoe UI", system-ui, sans-serif;
        color: {INK};
    }}
    .kpi-card {{
        background: {CARD_BG};
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 3px rgba(15,23,42,0.08), 0 1px 2px rgba(15,23,42,0.04);
        border: 1px solid #E2E8F0;
        margin-bottom: 0.9rem;
    }}
    .kpi-icon {{ font-size: 1.4rem; }}
    .kpi-value {{
        font-size: 1.6rem; font-weight: 700; color: {INK};
        display: flex; align-items: baseline; flex-wrap: wrap; gap: 0.45rem;
    }}
    .kpi-label {{ font-size: 0.8rem; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.03em; }}
    .kpi-trend-up {{ color: {SUCCESS}; font-weight: 600; font-size: 0.85rem; }}
    .kpi-trend-down {{ color: {CRITICAL}; font-weight: 600; font-size: 0.85rem; }}
    .kpi-interpretation {{ font-size: 0.82rem; color: {MUTED}; margin-top: 0.3rem; }}

    .finding-card {{
        background: {CARD_BG};
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        border-left: 6px solid {MUTED};
        box-shadow: 0 1px 3px rgba(15,23,42,0.08);
        margin-bottom: 1rem;
    }}
    .finding-card.high {{ border-left-color: {CRITICAL}; }}
    .finding-card.medium {{ border-left-color: {WARNING}; }}
    .finding-card.low {{ border-left-color: {SUCCESS}; }}
    .badge {{
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-right: 0.4rem;
        text-transform: uppercase;
    }}
    .badge-high {{ background: #FEE2E2; color: {CRITICAL}; }}
    .badge-medium {{ background: #FEF3C7; color: {WARNING}; }}
    .badge-low {{ background: #DCFCE7; color: {SUCCESS}; }}
    .badge-priority {{ background: #DBEAFE; color: {PRIMARY}; }}

    .landing-hero {{
        text-align: center;
        padding: 3rem 1rem 2rem 1rem;
    }}
    .landing-title {{ font-size: 2.4rem; font-weight: 800; color: {INK}; margin-bottom: 0.2rem; }}
    .landing-subtitle {{ font-size: 1.1rem; color: {PRIMARY}; font-weight: 600; margin-bottom: 1.2rem; }}
    .landing-tagline {{ font-size: 1rem; color: {MUTED}; max-width: 640px; margin: 0 auto 2rem auto; }}
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_TEMPLATE = "plotly_white"  # shared baseline; analytics.py should standardize on this


def kpi_card(col, icon, label, value, trend=None, interpretation=None):
    """Render one enterprise-style KPI card into a given st.columns() slot.

    trend: optional float, e.g. +4.2 or -1.8 (percent). None hides the arrow.
    """
    trend_html = ""
    if trend is not None:
        arrow = "▲" if trend >= 0 else "▼"
        cls = "kpi-trend-up" if trend >= 0 else "kpi-trend-down"
        trend_html = f'<span class="{cls}">{arrow} {abs(trend):.1f}%</span>'

    interp_html = f'<div class="kpi-interpretation">{interpretation}</div>' if interpretation else ""

    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value} {trend_html}</div>
            {interp_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def severity_class(sev):
    return {"High": "high", "Medium": "medium", "Low": "low"}.get(sev, "low")


def severity_badge(sev):
    return {
        "High": '<span class="badge badge-high">High</span>',
        "Medium": '<span class="badge badge-medium">Medium</span>',
        "Low": '<span class="badge badge-low">Low</span>',
    }.get(sev, "")


# ---------------------------------------------------------------------------
# Session state: data-gating
# ---------------------------------------------------------------------------
if "data_loaded" not in st.session_state:
    # Reflect reality: if the warehouse already has rows (e.g. from a prior
    # run of this session or a persisted sqlite file), treat data as loaded.
    st.session_state.data_loaded = not load_from_sqlite().empty


def load_demo_dataset():
    """Load the bundled sample CSV through the same ingest pipeline as a
    real upload, so validation/cleaning logic is exercised identically."""
    demo_path = os.path.join("data", "sample_sales.csv")
    if not os.path.exists(demo_path):
        st.error(
            "Demo dataset not found at data/sample_sales.csv. "
            "Run `python data/generate_sample_data.py` first."
        )
        return False
    raw_df = pd.read_csv(demo_path)
    result = ingest_pipeline(raw_df)
    if not result["success"]:
        st.error("Demo dataset failed validation — check data/generate_sample_data.py output.")
        return False
    st.session_state.data_loaded = True
    return True


# ---------------------------------------------------------------------------
# Landing page (shown until data is loaded)
# ---------------------------------------------------------------------------
if not st.session_state.data_loaded:
    st.markdown(
        """
        <div class="landing-hero">
            <div class="landing-title">💊 DecisionIQ</div>
            <div class="landing-subtitle">AI-Powered Decision Intelligence Platform</div>
            <div class="landing-tagline">
                Helping pharmaceutical companies transform commercial sales data
                into strategic business decisions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    lcol1, lcol2 = st.columns(2)

    with lcol1:
        with st.container(border=True):
            st.markdown("### 📂 Upload CSV")
            st.caption(f"Required columns: `{', '.join(REQUIRED_COLUMNS)}`")
            uploaded_file = st.file_uploader("Upload sales CSV", type=["csv"], key="landing_uploader")
            if uploaded_file is not None:
                raw_df = pd.read_csv(uploaded_file)
                result = ingest_pipeline(raw_df)
                if not result["success"]:
                    st.error("Validation failed — missing required columns:")
                    st.write(result["validation"]["missing_columns"])
                else:
                    st.session_state.data_loaded = True
                    st.rerun()

    with lcol2:
        with st.container(border=True):
            st.markdown("### 🚀 Use Demo Dataset")
            st.caption("Instantly explore DecisionIQ with realistic sample pharma sales data.")
            if st.button("Load Demo Dataset", type="primary", use_container_width=True):
                if load_demo_dataset():
                    st.rerun()

    st.stop()

# ---------------------------------------------------------------------------
# Sidebar navigation (only reachable once data is loaded)
# ---------------------------------------------------------------------------
st.sidebar.title("💊 DecisionIQ")
st.sidebar.caption("AI-Powered Decision Intelligence for Pharma Sales")

page = st.sidebar.radio(
    "Navigate",
    [
        "📁 Data Upload",
        "📊 Executive Dashboard",
        "🧠 Decision Engine",
        "📈 Forecasting",
        "🤖 AI Strategy Consultant",
        "📄 Reports",
    ],
)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset / Load Different Dataset"):
    st.session_state.data_loaded = False
    st.session_state.pop("last_ai_summary", None)
    st.session_state.pop("last_ai_result", None)
    st.session_state.pop("ai_chat_history", None)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Built by Asmita Roy · Portfolio Project")

# ---------------------------------------------------------------------------
# Page: Data Upload
# ---------------------------------------------------------------------------
if page == "📁 Data Upload":
    st.title("📁 Data Ingestion")
    st.write(
        "Upload a pharma sales CSV to populate the analytics warehouse. "
        "The file must contain the required columns listed below."
    )

    with st.expander("Required columns"):
        st.code(", ".join(REQUIRED_COLUMNS))

    uploaded_file = st.file_uploader("Upload sales CSV", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        result = ingest_pipeline(raw_df)

        if not result["success"]:
            st.error("Validation failed — missing required columns:")
            st.write(result["validation"]["missing_columns"])
        else:
            v = result["validation"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows Uploaded", v["row_count"])
            col2.metric("Rows Ingested", result["rows_ingested"])
            col3.metric("Duplicates Removed", v["duplicate_rows"])
            col4.metric("Rows Dropped", result["rows_dropped"])

            if v["missing_values"]:
                st.warning(f"Missing values found and handled: {v['missing_values']}")

            st.success("Data successfully cleaned and loaded into the SQLite warehouse.")
            st.dataframe(raw_df.head(10))

    st.markdown("---")
    st.caption(
        "No file to test with? Run `python data/generate_sample_data.py` locally, "
        "then upload `data/sample_sales.csv` here, or use the demo dataset from the landing page."
    )

# ---------------------------------------------------------------------------
# Page: Executive Dashboard
# ---------------------------------------------------------------------------
elif page == "📊 Executive Dashboard":
    st.title("📊 Executive Dashboard")

    df = load_from_sqlite()

    if df.empty:
        st.warning("No dataset loaded. Please upload a CSV or use the demo dataset.")
        st.stop()

    kpis = analytics.compute_kpis(df)

    # --- KPI cards: icon + value + trend + interpretation ---
    # TODO(module-upgrade): analytics.compute_kpis() should eventually return
    # explicit month-over-month deltas per KPI. Until then, monthly_growth /
    # quarterly_growth (which it already computes) are reused as the trend
    # signal for the KPIs they most directly relate to.
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    kpi_card(r1c1, "💰", "Total Revenue", f"${kpis['total_revenue']:,.0f}",
             trend=kpis.get("monthly_growth"),
             interpretation="Cumulative revenue across all uploaded transactions.")
    kpi_card(r1c2, "📈", "Total Profit", f"${kpis['total_profit']:,.0f}",
             interpretation="Revenue minus cost of goods across the full dataset.")
    kpi_card(r1c3, "🎯", "Profit Margin", f"{kpis['profit_margin']:.1f}%",
             interpretation="Healthy pharma margins typically run 15–25%.")
    kpi_card(r1c4, "🧾", "Total Orders", f"{kpis['total_orders']:,}",
             interpretation="Total number of transactions recorded.")

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    kpi_card(r2c1, "💵", "Avg Order Value", f"${kpis['avg_order_value']:,.0f}")
    kpi_card(r2c2, "🌍", "Best Region", kpis["best_region"],
             interpretation="Highest revenue-generating region.")
    kpi_card(r2c3, "🧪", "Top Therapy Area", kpis["top_therapy"])
    kpi_card(r2c4, "💊", "Top Drug", kpis["top_drug"])

    r3c1, r3c2, r3c3 = st.columns(3)
    kpi_card(r3c1, "⚠️", "Worst Region", kpis["worst_region"],
             interpretation="Candidate for the Decision Engine to investigate.")
    kpi_card(r3c2, "📆", "Monthly Growth", f"{kpis['monthly_growth']:.1f}%",
             interpretation="Latest full month vs. the month before it.")
    kpi_card(r3c3, "📊", "Quarterly Growth", f"{kpis['quarterly_growth']:.1f}%",
             interpretation="Latest quarter vs. the quarter before it.")

    st.markdown("---")

    # --- Charts, 2 per row ---
    # TODO(module-upgrade): analytics.py chart functions should apply a
    # shared enterprise Plotly template (see PLOTLY_TEMPLATE above) for
    # visual consistency across every chart in the app.
    c1, c2 = st.columns(2)
    c1.plotly_chart(analytics.revenue_trend_chart(df), use_container_width=True)
    c2.plotly_chart(analytics.profit_trend_chart(df), use_container_width=True)

    c3, c4 = st.columns(2)
    c3.plotly_chart(analytics.regional_revenue_chart(df), use_container_width=True)
    c4.plotly_chart(analytics.therapy_revenue_chart(df), use_container_width=True)

    c5, c6 = st.columns(2)
    c5.plotly_chart(analytics.top_drugs_chart(df), use_container_width=True)
    c6.plotly_chart(analytics.inventory_chart(df), use_container_width=True)

    c7, c8 = st.columns(2)
    c7.plotly_chart(analytics.discount_chart(df), use_container_width=True)
    c8.plotly_chart(analytics.customer_segmentation_chart(df), use_container_width=True)

    st.plotly_chart(analytics.profit_heatmap(df), use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Decision Analytics Engine
# ---------------------------------------------------------------------------
elif page == "🧠 Decision Engine":
    st.title("🧠 Decision Analytics Engine")
    st.write(
        "Automatically detected business issues, ranked by severity, "
        "each with a concrete recommended action."
    )

    df = load_from_sqlite()

    if df.empty:
        st.warning("No dataset loaded. Please upload a CSV or use the demo dataset.")
        st.stop()

    issues = decision_engine.run_decision_engine(df)

    if not issues:
        st.success("No significant issues detected in the current dataset.")
    else:
        high = [i for i in issues if i["severity"] == "High"]
        med = [i for i in issues if i["severity"] == "Medium"]
        low = [i for i in issues if i["severity"] == "Low"]

        c1, c2, c3 = st.columns(3)
        c1.metric("High Severity", len(high))
        c2.metric("Medium Severity", len(med))
        c3.metric("Low Severity", len(low))

        st.markdown("---")

        # TODO(module-upgrade): decision_engine.run_decision_engine() should
        # be extended to return `business_impact`, `estimated_benefit`, and
        # `priority` per finding directly. Until then, priority is derived
        # from severity order (position within its severity tier) and
        # business_impact/estimated_benefit fall back to the existing
        # finding/recommendation text so the card layout is future-proof.
        for idx, issue in enumerate(issues, start=1):
            sev = issue["severity"]
            impact = issue.get("business_impact", issue["finding"])
            benefit = issue.get("estimated_benefit", "Not yet quantified — see recommendation")
            priority = issue.get("priority", f"P{idx}")

            st.markdown(
                f"""
                <div class="finding-card {severity_class(sev)}">
                    <div>{severity_badge(sev)}<span class="badge badge-priority">{priority}</span>
                        <strong>{issue['category']}</strong></div>
                    <div style="margin-top:0.6rem;"><strong>Business Impact:</strong> {impact}</div>
                    <div style="margin-top:0.3rem;"><strong>Recommendation:</strong> {issue['recommendation']}</div>
                    <div style="margin-top:0.3rem;"><strong>Estimated Benefit:</strong> {benefit}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Page: Predictive Analytics (Forecasting)
# ---------------------------------------------------------------------------
elif page == "📈 Forecasting":
    st.title("📈 Predictive Analytics")
    st.write(
        "A Random Forest Regressor trained on monthly revenue history, "
        "forecasting next month's expected revenue."
    )

    df = load_from_sqlite()

    if df.empty:
        st.warning("No dataset loaded. Please upload a CSV or use the demo dataset.")
        st.stop()

    result = forecasting.train_forecast_model(df)

    if not result["success"]:
        st.warning(result["reason"])
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Next Month Forecast",
        f"${result['next_month_forecast']:,.0f}",
        help=f"{result['confidence_level']}% confidence interval: "
             f"${result['next_month_forecast_low']:,.0f} – ${result['next_month_forecast_high']:,.0f}",
    )
    c2.metric("RMSE", f"${result['rmse']:,.0f}")
    c3.metric("MAE", f"${result['mae']:,.0f}")
    c4.metric("R² Score", f"{result['r2']:.3f}")
    st.caption(
        f"📊 {result['confidence_level']}% confidence interval for next month: "
        f"**${result['next_month_forecast_low']:,.0f} – ${result['next_month_forecast_high']:,.0f}**"
    )

    with st.expander("💡 Why Random Forest?"):
        st.write(
            "Random Forest was chosen over a single decision tree or linear "
            "regression because it averages many decorrelated trees trained on "
            "bootstrapped samples, which reduces overfitting on a relatively "
            "small monthly-revenue history while still capturing non-linear "
            "seasonality effects that a linear model would miss. It also "
            "exposes feature importances directly, which supports the "
            "'Business Interpretation' section below — useful for explaining "
            "the forecast to a non-technical stakeholder."
        )

    st.caption(
        "Model trained on a chronological 80/20 split (not random) to avoid "
        "leaking future information into training — standard practice for time series."
    )

    st.markdown("---")

    import plotly.graph_objects as go
    comp = result["comparison"]
    fig = go.Figure()

    # Shaded confidence band (drawn first so the Actual/Predicted lines
    # render on top of it). Upper bound traced left-to-right, lower bound
    # traced right-to-left, with fill='toself' closing the shape between
    # them — the standard Plotly pattern for a confidence band.
    fig.add_trace(go.Scatter(
        x=pd.concat([comp["Date"], comp["Date"][::-1]]),
        y=pd.concat([comp["Upper"], comp["Lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(37,99,235,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name=f"{result['confidence_level']}% CI",
    ))
    fig.add_trace(go.Scatter(x=comp["Date"], y=comp["Actual"], mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=comp["Date"], y=comp["Predicted"], mode="lines+markers", name="Predicted"))

    fig.update_layout(title="Actual vs Predicted Revenue (Test Period)", template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Feature Importance")
        for feat, imp in sorted(result["feature_importances"].items(), key=lambda x: -x[1]):
            st.write(f"**{feat}**: {imp:.1%}")
    with col_b:
        st.markdown("#### Model Summary")
        st.write(f"- **Algorithm:** Random Forest Regressor")
        st.write(f"- **R² Score:** {result['r2']:.3f}")
        st.write(f"- **RMSE:** ${result['rmse']:,.0f}")
        st.write(f"- **MAE:** ${result['mae']:,.0f}")
        st.markdown("#### Business Interpretation")
        top_feat = max(result["feature_importances"], key=result["feature_importances"].get)
        st.write(
            f"`{top_feat}` is the strongest driver of next month's forecast. "
            "Use this forecast alongside the Decision Engine findings to "
            "prioritize where commercial teams should focus next quarter."
        )

# ---------------------------------------------------------------------------
# Page: AI Strategy Consultant (Gemini Text-to-SQL, chat-style)
# ---------------------------------------------------------------------------
elif page == "🤖 AI Strategy Consultant":
    st.title("🤖 AI Strategy Consultant")
    st.write(
        "Ask a business question in plain English. The consultant converts "
        "it into SQL, runs it against the warehouse, and returns an "
        "executive summary with strategic recommendations."
    )

    df = load_from_sqlite()
    if df.empty:
        st.warning("No dataset loaded. Please upload a CSV or use the demo dataset.")
        st.stop()

    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []

    example_questions = [
        "Show revenue by therapy area",
        "Which region generated the highest profit?",
        "Top 10 drugs by revenue",
        "Monthly sales trend",
    ]
    st.caption("Example questions: " + " · ".join(f"*{q}*" for q in example_questions))

    # Replay chat history
    for turn in st.session_state.ai_chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.markdown("**Generated SQL**")
            st.code(turn["sql"], language="sql")
            if turn.get("data") is not None:
                st.markdown("**SQL Result**")
                st.dataframe(turn["data"], use_container_width=True)
            if turn.get("summary"):
                st.markdown("**Executive Summary & Strategic Recommendations**")
                st.markdown(turn["summary"])

    question = st.chat_input("Ask a business question about your pharma sales data...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Generating SQL and querying the warehouse..."):
                result = ai.ask_question(question)

            turn = {"question": question, "sql": result.get("sql", ""), "data": None, "summary": None}

            if not result["success"]:
                st.error(result["error"])
                if result.get("sql"):
                    st.markdown("**Generated SQL**")
                    st.code(result["sql"], language="sql")
            else:
                st.markdown("**Generated SQL**")
                st.code(result["sql"], language="sql")
                st.markdown("**SQL Result**")
                st.dataframe(result["data"], use_container_width=True)
                turn["data"] = result["data"]

                with st.spinner("Consulting AI business analyst..."):
                    try:
                        insight_text = ai.generate_executive_insights(question, result["data"])
                        st.markdown("**Executive Summary & Strategic Recommendations**")
                        st.markdown(insight_text)
                        turn["summary"] = insight_text
                        st.session_state["last_ai_summary"] = insight_text
                    except Exception as e:
                        st.error(f"Could not generate insights: {e}")

            st.session_state.ai_chat_history.append(turn)

# ---------------------------------------------------------------------------
# Page: Executive PDF Report
# ---------------------------------------------------------------------------
elif page == "📄 Reports":
    st.title("📄 Executive Report")
    st.write(
        "Generate a downloadable PDF combining KPIs, the revenue trend, "
        "the forecast, Decision Engine findings, and AI Strategy "
        "recommendations."
    )

    df = load_from_sqlite()
    if df.empty:
        st.warning("No dataset loaded. Please upload a CSV or use the demo dataset.")
        st.stop()

    include_ai = st.checkbox(
        "Include the most recent AI Strategy Consultant summary "
        "(generate one on the 'AI Strategy Consultant' page first)",
        value=bool(st.session_state.get("last_ai_summary")),
    )

    if st.button("Generate PDF Report", type="primary"):
        with st.spinner("Building report..."):
            ai_summary = st.session_state.get("last_ai_summary") if include_ai else None
            pdf_bytes = reports.generate_executive_pdf(df, ai_summary=ai_summary)
        st.session_state["last_pdf_bytes"] = pdf_bytes
        st.success("Report generated.")

    # TODO(module-upgrade): true PDF page-image preview belongs in
    # reports.py (e.g. render page 1 as a PNG via pdf2image and return it
    # alongside the PDF bytes). Until that helper exists, we show a
    # lightweight in-app preview of the report's text content here.
    if st.session_state.get("last_pdf_bytes"):
        with st.expander("📖 Preview report contents", expanded=True):
            try:
                import io
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(st.session_state["last_pdf_bytes"]))
                for i, page_obj in enumerate(reader.pages, start=1):
                    st.markdown(f"**Page {i}**")
                    st.text(page_obj.extract_text())
            except Exception:
                st.caption("Preview unavailable — download the PDF below to view it.")

        st.download_button(
            label="⬇️ Download Executive Report (PDF)",
            data=st.session_state["last_pdf_bytes"],
            file_name="DecisionIQ_Executive_Report.pdf",
            mime="application/pdf",
        )

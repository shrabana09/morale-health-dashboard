"""
dashboard.py
-------------
Streamlit frontend for the Agentic Sentiment Analysis & Morale Health Index System.

Upload a CSV or XLSX (text/clean_text, sentiment, emotion[, mhi_score]) -> it's sent
to the FastAPI backend's /analyze endpoint, which stores it, computes any missing
mhi_score values, aggregates stats, and calls the Groq agent for a summary. This
page then renders everything: metric cards, a gauge, donut charts, a trend line
across past uploads, the raw data table, and the AI insights panel.
"""

import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Morale Health Index Dashboard",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling — gradient header + rounded metric cards
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .mhi-header {
        background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 50%, #0EA5E9 100%);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.35);
    }
    .mhi-header h1 {
        color: white;
        margin: 0;
        font-size: 2.1rem;
        font-weight: 800;
    }
    .mhi-header p {
        color: rgba(255,255,255,0.88);
        margin: 0.35rem 0 0 0;
        font-size: 1.02rem;
    }
    .metric-card {
        background: #1E293B;
        border: 1px solid rgba(148,163,184,0.15);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }
    .metric-card .label {
        color: #94A3B8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-card .value {
        color: #F1F5F9;
        font-size: 1.9rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }
    .insight-panel {
        background: #1E293B;
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        border: 1px solid rgba(148,163,184,0.15);
        margin-top: 1rem;
    }
    .insight-panel h3 {
        margin-top: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="mhi-header">
        <h1>📊 Morale Health Index Dashboard</h1>
        <p>Agentic Sentiment & Emotion Analysis — powered by Groq (LLaMA)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a CSV or XLSX file (columns: text/clean_text, sentiment, emotion — mhi_score optional)",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is not None:
    if st.button("Analyze", type="primary"):
        with st.spinner("Uploading, scoring, and generating AI insights..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                )
            }
            try:
                resp = requests.post(f"{API_BASE_URL}/analyze", files=files, timeout=180)
                resp.raise_for_status()
                st.session_state["last_result"] = resp.json()
                st.session_state["last_batch_id"] = resp.json()["batch_id"]
                st.success(f"Analyzed {resp.json()['row_count']:,} rows.")
            except requests.exceptions.RequestException as exc:
                st.error(f"Could not reach the backend at {API_BASE_URL}. Is main.py running? ({exc})")

st.divider()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
result = st.session_state.get("last_result")

if result is None:
    st.info("Upload a file above and click **Analyze** to see the dashboard.")
    st.stop()

stats = result["aggregated_stats"]
insight = result["insight"]
batch_id = result["batch_id"]

# --- Metric cards ---
col1, col2, col3, col4 = st.columns(4)
for col, label, value in zip(
    [col1, col2, col3, col4],
    ["Total Comments", "Average MHI", "Positive %", "Negative %"],
    [
        f"{stats['total_rows']:,}",
        f"{stats['avg_mhi_score']:.1f}",
        f"{stats['sentiment_percentages'].get('positive', 0):.1f}%",
        f"{stats['sentiment_percentages'].get('negative', 0):.1f}%",
    ],
):
    with col:
        st.markdown(
            f"""<div class="metric-card"><div class="label">{label}</div>
            <div class="value">{value}</div></div>""",
            unsafe_allow_html=True,
        )

st.write("")

# --- Gauge + Donuts ---
gauge_col, donut1_col, donut2_col = st.columns([1.1, 1, 1])

with gauge_col:
    mhi = stats["avg_mhi_score"]
    gauge_color = "#EF4444" if mhi < 40 else "#F59E0B" if mhi < 65 else "#22C55E"
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=mhi,
            title={"text": "Morale Health Index"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 40], "color": "#450A0A"},
                    {"range": [40, 65], "color": "#451A03"},
                    {"range": [65, 100], "color": "#052E16"},
                ],
            },
        )
    )
    fig_gauge.update_layout(
        height=320, margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F1F5F9"),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with donut1_col:
    sent_df = pd.DataFrame(
        {"sentiment": list(stats["sentiment_counts"].keys()), "count": list(stats["sentiment_counts"].values())}
    )
    fig_sent = px.pie(
        sent_df, names="sentiment", values="count", hole=0.55, title="Sentiment Distribution",
        color="sentiment",
        color_discrete_map={"positive": "#22C55E", "neutral": "#94A3B8", "negative": "#EF4444"},
    )
    fig_sent.update_layout(
        height=320, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F1F5F9"), legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_sent, use_container_width=True)

with donut2_col:
    emo_df = pd.DataFrame(
        {"emotion": list(stats["emotion_counts"].keys()), "count": list(stats["emotion_counts"].values())}
    )
    fig_emo = px.pie(
        emo_df, names="emotion", values="count", hole=0.55, title="Emotion Distribution",
    )
    fig_emo.update_layout(
        height=320, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F1F5F9"), legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_emo, use_container_width=True)

# --- MHI trend across batches ---
st.subheader("MHI Trend Across Uploads")
try:
    batches_resp = requests.get(f"{API_BASE_URL}/batches", timeout=30)
    batches_resp.raise_for_status()
    batches = batches_resp.json()
    if batches:
        trend_df = pd.DataFrame(batches)
        trend_df["created_at"] = pd.to_datetime(trend_df["created_at"])
        fig_trend = px.line(
            trend_df, x="created_at", y="avg_mhi_score", markers=True,
            labels={"created_at": "Upload Time", "avg_mhi_score": "Avg MHI Score"},
        )
        fig_trend.update_traces(line_color="#7C3AED", marker=dict(size=9, color="#0EA5E9"))
        fig_trend.update_layout(
            height=320, margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#F1F5F9"),
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.caption("No prior batches yet.")
except requests.exceptions.RequestException:
    st.caption("Could not load trend data from the backend.")

# --- Raw data table ---
st.subheader("Raw Data")
try:
    entries_resp = requests.get(f"{API_BASE_URL}/entries", params={"batch_id": batch_id, "limit": 2000}, timeout=30)
    entries_resp.raise_for_status()
    entries_df = pd.DataFrame(entries_resp.json())
    if not entries_df.empty:
        st.dataframe(
            entries_df[["text", "sentiment", "emotion", "mhi_score"]],
            use_container_width=True, height=350,
        )
        st.caption(f"Showing up to 2,000 of {stats['total_rows']:,} rows for this batch.")
except requests.exceptions.RequestException:
    st.caption("Could not load raw entries from the backend.")

# --- AI Insights panel ---
st.markdown('<div class="insight-panel">', unsafe_allow_html=True)
st.markdown("### 🤖 AI Insights (Groq / LLaMA summary of the aggregated data)")
st.write(insight.get("summary", ""))

ic1, ic2, ic3 = st.columns(3)
with ic1:
    st.markdown("**Key Themes**")
    for t in insight.get("key_themes", []):
        st.markdown(f"- {t}")
with ic2:
    st.markdown("**Anomalies**")
    for a in insight.get("anomalies", []):
        st.markdown(f"- {a}")
with ic3:
    st.markdown("**Recommendations**")
    for r in insight.get("recommendations", []):
        st.markdown(f"- {r}")
st.markdown("</div>", unsafe_allow_html=True)

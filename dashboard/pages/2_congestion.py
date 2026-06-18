import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.api import get_detections

st.title(" Congestion Overview")

if "job_id" not in st.session_state:
    st.warning("Upload a video on the Analyse page first.")
    st.stop()

job_id = st.session_state["job_id"]
COLOR_MAP = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}

with st.spinner("Loading..."):
    detections = get_detections(job_id)

if not detections:
    st.error("No detection data found.")
    st.stop()

df = pd.DataFrame(detections)

# Metric cards
dist = df["congestion_level"].value_counts().to_dict()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total frames",      len(df))
c2.metric("🟢 Low frames",    dist.get("Low", 0))
c3.metric("🟡 Medium frames", dist.get("Medium", 0))
c4.metric("🔴 High frames",   dist.get("High", 0))

st.markdown("---")

# Vehicle count over time
fig = go.Figure()
for level, color in COLOR_MAP.items():
    mask = df["congestion_level"] == level
    fig.add_trace(go.Scatter(
        x=df.loc[mask, "timestamp_sec"],
        y=df.loc[mask, "vehicle_count"],
        mode="markers",
        marker=dict(color=color, size=4),
        name=level
    ))
fig.add_trace(go.Scatter(
    x=df["timestamp_sec"],
    y=df["vehicle_count"],
    mode="lines",
    line=dict(color="#6366f1", width=2),
    name="Count",
    showlegend=False
))
fig.update_layout(
    title="Vehicle count over time",
    xaxis_title="Time (seconds)",
    yaxis_title="Vehicle count",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig, use_container_width=True)

# Pie chart
col1, col2 = st.columns(2)
with col1:
    fig_pie = px.pie(
        names=list(dist.keys()),
        values=list(dist.values()),
        color=list(dist.keys()),
        color_discrete_map=COLOR_MAP,
        hole=0.45,
        title="Level distribution"
    )
    fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.markdown("### Peak moments")
    peak_row  = df.loc[df["vehicle_count"].idxmax()]
    quiet_row = df.loc[df["vehicle_count"].idxmin()]
    st.markdown(f"""
| Moment | Time | Vehicles | Level |
|--------|------|----------|-------|
| 🔴 Peak | {peak_row['timestamp_sec']:.1f}s | {int(peak_row['vehicle_count'])} | {peak_row['congestion_level']} |
| 🟢 Quiet | {quiet_row['timestamp_sec']:.1f}s | {int(quiet_row['vehicle_count'])} | {quiet_row['congestion_level']} |
    """)

with st.expander("View raw data"):
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇ Download CSV", df.to_csv(index=False),
                       f"detections_{job_id[:8]}.csv", "text/csv")
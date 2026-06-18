import streamlit as st
import pandas as pd
import plotly.express as px
from utils.api import get_detections, get_violations, get_summary

st.title(" Analytics")

if "job_id" not in st.session_state:
    st.warning("Upload a video on the Analyse page first.")
    st.stop()

job_id = st.session_state["job_id"]
COLOR_MAP = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}

df_det = pd.DataFrame(get_detections(job_id))
df_vio = pd.DataFrame(get_violations(job_id))
summary = get_summary()

# Congestion heatmap
st.markdown("### Congestion intensity over time")
df_det["time_bucket"]  = (df_det["timestamp_sec"] // 10 * 10).astype(int)
df_det["level_score"]  = df_det["congestion_level"].map({"Low":1,"Medium":2,"High":3})
heatmap = df_det.groupby("time_bucket")["level_score"].mean().reset_index()
fig_heat = px.bar(heatmap, x="time_bucket", y="level_score",
                  color="level_score",
                  color_continuous_scale=["#22c55e","#f59e0b","#ef4444"],
                  range_color=[1,3])
fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)",
                       yaxis=dict(tickvals=[1,2,3], ticktext=["Low","Medium","High"]))
st.plotly_chart(fig_heat, use_container_width=True)

col1, col2 = st.columns(2)

# Violation donut
with col1:
    st.markdown("### Violation breakdown")
    if not df_vio.empty:
        vc = df_vio["violation_type"].value_counts().reset_index()
        vc.columns = ["Type","Count"]
        fig_d = px.pie(vc, names="Type", values="Count", hole=0.55)
        fig_d.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_d, use_container_width=True)
    else:
        st.info("No violations detected.")

# Confidence distribution
with col2:
    st.markdown("### Detection confidence")
    if not df_vio.empty:
        fig_c = px.histogram(df_vio, x="confidence", nbins=20,
                             color="violation_type")
        fig_c.add_vline(x=0.5, line_dash="dash", annotation_text="Threshold")
        fig_c.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.info("No violations to show.")

# Bengaluru map
st.markdown("### Bengaluru congestion map")
junctions = pd.DataFrame([
    {"junction":"Silk Board",         "lat":12.9177,"lon":77.6225,"level":"High",   "vehicles":28},
    {"junction":"Hebbal Flyover",     "lat":13.0358,"lon":77.5970,"level":"Medium", "vehicles":11},
    {"junction":"Marathahalli",       "lat":12.9591,"lon":77.6974,"level":"High",   "vehicles":22},
    {"junction":"KR Circle",          "lat":12.9767,"lon":77.5713,"level":"Low",    "vehicles":4},
    {"junction":"Whitefield Signal",  "lat":12.9698,"lon":77.7499,"level":"Medium", "vehicles":14},
    {"junction":"Jayanagar 4th Block","lat":12.9250,"lon":77.5938,"level":"Low",    "vehicles":3},
])
if not df_det.empty:
    junctions.loc[0,"level"]    = df_det["congestion_level"].mode()[0]
    junctions.loc[0,"vehicles"] = int(df_det["vehicle_count"].mean())

fig_map = px.scatter_mapbox(
    junctions, lat="lat", lon="lon",
    color="level", size="vehicles",
    color_discrete_map=COLOR_MAP,
    hover_name="junction",
    size_max=28, zoom=11,
    center={"lat":12.9716,"lon":77.5946},
    mapbox_style="open-street-map"
)
fig_map.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=500,
                       margin=dict(l=0,r=0,t=0,b=0))
st.plotly_chart(fig_map, use_container_width=True)
import streamlit as st

st.set_page_config(
    page_title="TrafficLens AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🚦 TrafficLens AI")
st.sidebar.caption("Bengaluru CCTV Intelligence")
st.sidebar.markdown("---")

if "job_id" in st.session_state:
    st.sidebar.success(f"Active job: `{st.session_state.job_id[:8]}...`")
else:
    st.sidebar.info("No video analysed yet. Go to **Analyse** to upload one.")

st.title("🚦 TrafficLens AI")
st.markdown("AI-powered traffic intelligence from Bengaluru CCTV footage.")
st.markdown("---")

from utils.api import get_summary
try:
    summary = get_summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Videos Processed",  summary["total_videos_processed"])
    c2.metric("Total Violations",   summary["total_violations"])
    c3.metric("Peak Congestion",    summary["peak_congestion_level"])
    c4.metric("Avg Vehicle Count",  summary["avg_vehicle_count"])
except Exception:
    st.warning("FastAPI server must be running on port 8000.")
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from utils.api import upload_video, get_job_status

st.title(" Analyse Video")
st.markdown("Upload a traffic video and run the full AI pipeline.")

uploaded = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded:
    st.video(uploaded)
    run = st.button(" Analyse", type="primary")

    if run:
        with st.spinner("Uploading..."):
            response = upload_video(uploaded.read(), uploaded.name)
            st.session_state["job_id"]     = response["job_id"]
            st.session_state["job_status"] = "queued"
            st.session_state["progress"]   = 5

# Only auto-refresh when actively processing — stops once complete/failed
job_status = st.session_state.get("job_status", "idle")
job_active = job_status not in ("complete", "failed", "idle")

if job_active:
    # Stop refreshing as soon as job finishes
    st_autorefresh(interval=4000, limit=100, key="poll_refresh")

if "job_id" in st.session_state and job_status not in ("idle",):
    job_id = st.session_state["job_id"]

    try:
        result     = get_job_status(job_id)
        status     = result["status"]
        st.session_state["job_status"] = status

        if status == "complete":
            st.progress(100, text="Done!")
            st.success(" Analysis complete! Go to Congestion or Violations in the sidebar.")

        elif status == "failed":
            st.error("Processing failed. Check Celery worker terminal.")

        else:
            new_progress = min(st.session_state.get("progress", 5) + 6, 88)
            st.session_state["progress"] = new_progress
            st.progress(new_progress, text=f"Running YOLO pipeline... ({status})")
            st.info(f"Job ID: `{job_id[:8]}...`")

    except Exception as e:
        st.warning(f"Waiting for server... ({str(e)[:60]})")

elif job_status == "complete":
    st.success(
        f" Last job `{st.session_state['job_id'][:8]}...` completed. "
        "Check Congestion or Violations in the sidebar."
    )
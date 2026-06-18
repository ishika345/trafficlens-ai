import streamlit as st
import time
from utils.api import upload_video, get_job_status

st.title(" Analyse Video")
st.markdown("Upload a traffic video and run the full AI pipeline.")

uploaded = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov"]
)

if uploaded:
    st.video(uploaded)

    run = st.button(" Analyse", type="primary")

    if run:
        with st.spinner("Uploading..."):
            response = upload_video(uploaded.read(), uploaded.name)
            job_id = response["job_id"]
            st.session_state["job_id"] = job_id

        st.info(f"Job queued: `{job_id}`")

        progress_bar = st.progress(0, text="Processing video...")
        fake_progress = 0

        while True:
            result = get_job_status(job_id)
            status = result["status"]

            if status == "complete":
                progress_bar.progress(100, text="Done!")
                st.success(" Analysis complete! Go to Congestion or Violations in the sidebar.")
                break
            elif status == "failed":
                st.error(" Processing failed. Check Celery worker terminal.")
                break
            else:
                fake_progress = min(fake_progress + 8, 90)
                progress_bar.progress(fake_progress, text="Running YOLO pipeline...")
                time.sleep(3)
                st.rerun()
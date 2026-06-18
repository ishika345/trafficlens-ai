import streamlit as st
import pandas as pd
from utils.api import get_violations

st.title(" Violations Log")

if "job_id" not in st.session_state:
    st.warning("Upload a video on the Analyse page first.")
    st.stop()

job_id = st.session_state["job_id"]

col1, col2 = st.columns(2)
with col1:
    v_type = st.selectbox("Violation type", ["All", "no_helmet", "signal_jump", "wrong_way"])
with col2:
    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.5, 0.05)

filter_type = None if v_type == "All" else v_type

with st.spinner("Loading violations..."):
    violations = get_violations(job_id, violation_type=filter_type)

if not violations:
    st.success("No violations found.")
    st.stop()

df = pd.DataFrame(violations)
df = df[df["confidence"] >= min_conf].reset_index(drop=True)

type_counts = df["violation_type"].value_counts().to_dict()
cols = st.columns(len(type_counts) + 1)
cols[0].metric("Total", len(df))
labels = {"no_helmet": " No helmet", "signal_jump": "🚦 Signal jump", "wrong_way": " Wrong way"}
for i, (vtype, count) in enumerate(type_counts.items()):
    cols[i+1].metric(labels.get(vtype, vtype), count)

st.markdown("---")

display_df = df[["frame_number", "timestamp_sec", "violation_type", "confidence"]].copy()
display_df["confidence"]    = display_df["confidence"].map("{:.1%}".format)
display_df["timestamp_sec"] = display_df["timestamp_sec"].map("{:.1f}s".format)
display_df.columns = ["Frame", "Time", "Type", "Confidence"]
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.download_button("⬇ Download CSV", df.to_csv(index=False),
                   f"violations_{job_id[:8]}.csv", "text/csv")
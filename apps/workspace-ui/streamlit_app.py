from __future__ import annotations

import streamlit as st

# IMPORTANT: st.set_page_config() must be called FIRST, before any other st.* calls
# This is the single location for page config across the entire multi-page app
st.set_page_config(page_title="Screenalytics Workspace", layout="wide")

import ui_helpers as helpers

cfg = helpers.init_page("Screenalytics Workspace")
st.title("Screenalytics Workspace")
st.caption("Launch the upload helper or jump into any workspace page. Configure API access via the sidebar.")

nav_cols = st.columns(2)
with nav_cols[0]:
    st.page_link("pages/0_Upload_Video.py", label="Upload Video", icon="⏫")
with nav_cols[1]:
    st.page_link("pages/1_Episodes.py", label="Episodes Browser", icon="🎞️")

more_cols = st.columns(2)
with more_cols[0]:
    st.page_link("pages/2_Episode_Detail.py", label="Episode Detail", icon="📺")
with more_cols[1]:
    st.page_link("pages/4_Cast.py", label="Cast Management", icon="🎭")

st.divider()
st.caption("Need more tools?")
misc_cols = st.columns(2)
with misc_cols[0]:
    st.page_link("pages/3_Faces_Review.py", label="Faces Review", icon="👁️")
with misc_cols[1]:
    st.page_link("pages/4_Screentime.py", label="Screentime & Health", icon="⏱️")

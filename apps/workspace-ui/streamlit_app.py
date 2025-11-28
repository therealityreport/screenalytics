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
    if st.button("Upload Video", use_container_width=True):
        helpers.try_switch_page("pages/0_Upload_Video.py")
with nav_cols[1]:
    if st.button("Episodes Browser", use_container_width=True):
        helpers.try_switch_page("pages/1_Episodes.py")

more_cols = st.columns(2)
with more_cols[0]:
    if st.button("Episode Detail", use_container_width=True):
        helpers.try_switch_page("pages/2_Episode_Detail.py")
with more_cols[1]:
    if st.button("Cast Management", use_container_width=True):
        helpers.try_switch_page("pages/4_Cast.py")

st.divider()
st.caption("Need more tools?")
misc_cols = st.columns(2)
with misc_cols[0]:
    if st.button("Faces Review", use_container_width=True):
        helpers.try_switch_page("pages/3_Faces_Review.py")
with misc_cols[1]:
    if st.button("Screentime & Health", use_container_width=True):
        helpers.try_switch_page("pages/4_Screentime.py")

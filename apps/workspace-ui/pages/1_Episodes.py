from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import requests
import streamlit as st

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402

cfg = helpers.init_page("Episodes")
st.title("Episodes Browser")

try:
    episodes_payload = helpers.api_get("/episodes")
except requests.RequestException as exc:
    st.error(helpers.describe_error(f"{cfg['api_base']}/episodes", exc))
    st.stop()

episodes = episodes_payload.get("episodes", [])
if not episodes:
    st.info("No episodes yet. Use Upload & Run first.")
    st.stop()

search_query = st.text_input("Search", "", help="Filter by ep_id or show.").strip().lower()
filtered = [
    ep
    for ep in episodes
    if not search_query
    or search_query in ep["ep_id"].lower()
    or search_query in (ep["show_slug"] or "").lower()
]
if not filtered:
    st.warning("No episodes match that filter.")
    st.stop()

helpers.ds(filtered)

option_lookup = {ep["ep_id"]: ep for ep in filtered}
selected_ep_id = st.selectbox(
    "Episode",
    list(option_lookup.keys()),
    format_func=lambda eid: f"{eid} ({option_lookup[eid]['show_slug']})",
)

col1, col2 = st.columns(2)
with col1:
    if st.button("Open details", use_container_width=True):
        helpers.set_ep_id(selected_ep_id)
        helpers.try_switch_page("pages/2_Episode_Detail.py")
with col2:
    st.write("")

st.subheader("Quick detect/track")
col_stride, col_fps, col_stub, col_device = st.columns(4)
with col_stride:
    stride_value = st.number_input("Stride", min_value=1, max_value=50, value=helpers.DEFAULT_STRIDE, step=1)
with col_fps:
    fps_value = st.number_input("FPS", min_value=0.0, max_value=120.0, value=0.0, step=1.0)
with col_stub:
    stub_toggle = st.checkbox("Use stub", value=False)
default_device_label = helpers.device_default_label()
with col_device:
    device_choice = st.selectbox(
        "Device",
        helpers.DEVICE_LABELS,
        index=helpers.device_label_index(default_device_label),
    )
device_value = helpers.DEVICE_VALUE_MAP[device_choice]

if st.button("Run detect/track", use_container_width=True):
    job_payload: dict[str, Any] = {
        "ep_id": selected_ep_id,
        "stub": bool(stub_toggle),
        "stride": int(stride_value),
        "device": device_value,
    }
    if fps_value > 0:
        job_payload["fps"] = fps_value
    mode_label = "stub (no ML)" if stub_toggle else "YOLOv8 + ByteTrack"
    with st.spinner(f"Running detect/track ({mode_label})…"):
        try:
            resp = helpers.api_post("/jobs/detect_track", job_payload)
        except requests.RequestException as exc:
            st.error(helpers.describe_error(f"{cfg['api_base']}/jobs/detect_track", exc))
        else:
            st.success(
                f"detections: {resp['detections_count']}, tracks: {resp['tracks_count']}"
            )

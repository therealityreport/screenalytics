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


def _api_post_json(path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    base = st.session_state.get("api_base")
    if not base:
        st.error("API base URL missing; run init_page().")
        return None
    try:
        resp = requests.post(f"{base}{path}", json=payload, timeout=90)
        resp.raise_for_status()
        return resp.json() if resp.content else {}
    except requests.RequestException as exc:
        st.error(helpers.describe_error(f"{base}{path}", exc))
        return None


def _reset_delete_state() -> None:
    st.session_state.pop("episodes_delete_target", None)


def _reset_purge_state() -> None:
    st.session_state.pop("episodes_purge_open", None)


def _show_single_delete(ep_id: str) -> None:
    st.markdown("#### Delete this episode")
    if st.button("Delete episode", key=f"episodes_delete_btn_{ep_id}"):
        st.session_state["episodes_delete_target"] = ep_id
    target = st.session_state.get("episodes_delete_target")
    if target != ep_id:
        st.caption("Removes the EpisodeStore entry plus local video/manifests/frames for this episode.")
        return
    with st.container(border=True):
        st.warning(
            f"You are about to delete `{ep_id}`. This removes the EpisodeStore record, manifests, frames/crops, and identities."
        )
        delete_s3 = st.checkbox("Also delete S3 artifacts (frames/crops/manifests)", value=False)
        cols = st.columns(2)
        with cols[0]:
            if st.button("Confirm delete", type="primary", key=f"confirm_delete_{ep_id}"):
                payload = {"include_s3": delete_s3}
                resp = _api_post_json(f"/episodes/{ep_id}/delete", payload)
                if resp is not None:
                    deleted = resp.get("deleted", {})
                    st.success(
                        f"Deleted {ep_id}: local dirs removed={deleted.get('local_dirs', 0)}, "
                        f"S3 objects deleted={deleted.get('s3_objects', 0)}."
                    )
                    _reset_delete_state()
                    st.rerun()
        with cols[1]:
            if st.button("Cancel", key=f"cancel_delete_{ep_id}"):
                _reset_delete_state()
                st.rerun()


def _show_purge_section() -> None:
    st.markdown("#### Delete ALL episodes & data")
    if st.button("Delete ALL episodes & data", key="episodes_purge_btn"):
        st.session_state["episodes_purge_open"] = True
    if not st.session_state.get("episodes_purge_open"):
        st.caption("Danger zone: wipes every tracked episode.")
        return
    with st.container(border=True):
        st.error(
            "This action deletes every EpisodeStore entry plus optional local/S3 artifacts. "
            "Type DELETE ALL to confirm."
        )
        delete_s3 = st.checkbox(
            "Also delete S3 artifacts (frames/crops/manifests) for every episode",
            value=False,
            key="purge_include_s3",
        )
        confirm_text = st.text_input("Type DELETE ALL to confirm", key="purge_confirm")
        cols = st.columns(2)
        with cols[0]:
            if st.button("Confirm purge", type="primary", key="purge_confirm_btn"):
                payload = {
                    "confirm": confirm_text.strip(),
                    "include_s3": delete_s3,
                }
                resp = _api_post_json("/episodes/delete_all", payload)
                if resp is not None:
                    totals = resp.get("deleted", {})
                    st.success(
                        "Purged episodes: "
                        f"{resp.get('count', 0)} removed · local dirs cleared={totals.get('local_dirs', 0)} · "
                        f"S3 objects deleted={totals.get('s3_objects', 0)}."
                    )
                    _reset_purge_state()
                    st.rerun()
        with cols[1]:
            if st.button("Cancel purge", key="purge_cancel_btn"):
                _reset_purge_state()
                st.rerun()

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
col_stride, col_fps, col_stub = st.columns(3)
with col_stride:
    stride_value = st.number_input("Stride", min_value=1, max_value=50, value=helpers.DEFAULT_STRIDE, step=1)
with col_fps:
    fps_value = st.number_input("FPS", min_value=0.0, max_value=120.0, value=0.0, step=1.0)
with col_stub:
    stub_toggle = st.checkbox("Use stub", value=False)
st.caption(
    f"Detector: {helpers.LABEL.get(helpers.DEFAULT_DETECTOR, helpers.DEFAULT_DETECTOR)} · "
    f"Tracker: {helpers.LABEL.get(helpers.DEFAULT_TRACKER, helpers.DEFAULT_TRACKER)} · "
    "Device: auto"
)

if st.button("Run detect/track", use_container_width=True):
    job_payload = helpers.default_detect_track_payload(
        selected_ep_id,
        stub=bool(stub_toggle),
        stride=int(stride_value),
        det_thresh=helpers.DEFAULT_DET_THRESH,
    )
    if fps_value > 0:
        job_payload["fps"] = fps_value
    mode_label = (
        f"{helpers.LABEL.get(helpers.DEFAULT_DETECTOR, helpers.DEFAULT_DETECTOR)} + "
        f"{helpers.LABEL.get(helpers.DEFAULT_TRACKER, helpers.DEFAULT_TRACKER)}"
    )
    with st.spinner(f"Running detect/track ({mode_label})…"):
        try:
            resp = helpers.api_post("/jobs/detect_track", job_payload)
        except requests.RequestException as exc:
            st.error(helpers.describe_error(f"{cfg['api_base']}/jobs/detect_track", exc))
        else:
            st.success(
                f"detections: {resp['detections_count']}, tracks: {resp['tracks_count']}"
            )

st.divider()
_show_single_delete(selected_ep_id)
_show_purge_section()

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

import requests
import streamlit as st

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402

from py_screenalytics.artifacts import get_path  # noqa: E402

cfg = helpers.init_page("Episode Detail")
st.title("Episode Detail")




def _prompt_for_episode() -> None:
    try:
        payload = helpers.api_get("/episodes")
    except requests.RequestException as exc:
        st.error(helpers.describe_error(f"{cfg['api_base']}/episodes", exc))
        st.stop()
    episodes = payload.get("episodes", [])
    if not episodes:
        st.info("No episodes yet.")
        st.stop()
    option_lookup = {ep["ep_id"]: ep for ep in episodes}
    selection = st.selectbox(
        "Episode",
        list(option_lookup.keys()),
        format_func=lambda eid: f"{eid} ({option_lookup[eid]['show_slug']})",
    )
    if st.button("Load episode", use_container_width=True):
        helpers.set_ep_id(selection)
    st.stop()


ep_id = helpers.get_ep_id()
if not ep_id:
    _prompt_for_episode()

try:
    details = helpers.api_get(f"/episodes/{ep_id}")
except requests.RequestException as exc:
    st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/{ep_id}", exc))
    st.stop()

st.subheader(f"Episode `{ep_id}`")
st.write(
    f"Show `{details['show_slug']}` · Season {details['season_number']} Episode {details['episode_number']}"
)
st.write(
    f"S3 v2 → `{details['s3']['v2_key']}` (exists={details['s3']['v2_exists']})"
)
st.write(
    f"S3 v1 → `{details['s3']['v1_key']}` (exists={details['s3']['v1_exists']})"
)
if not details["s3"]["v2_exists"] and details["s3"]["v1_exists"]:
    st.warning("Legacy v1 object detected; mirroring will use it until the v2 path is populated.")
st.write(
    f"Local → {helpers.link_local(details['local']['path'])} (exists={details['local']['exists']})"
)

col_hydrate, col_detect = st.columns(2)
with col_hydrate:
    if st.button("Mirror from S3", use_container_width=True):
        mirror_path = f"/episodes/{ep_id}/mirror"
        try:
            resp = helpers.api_post(mirror_path)
        except requests.RequestException as exc:
            st.error(helpers.describe_error(f"{cfg['api_base']}{mirror_path}", exc))
        else:
            st.success(
                f"Local: {helpers.link_local(resp['local_video_path'])} | size {helpers.human_size(resp.get('bytes'))}"
            )
default_device_label = helpers.device_default_label()
with col_detect:
    stride_value = st.number_input("Stride", min_value=1, max_value=50, value=helpers.DEFAULT_STRIDE, step=1)
    fps_value = st.number_input("FPS", min_value=0.0, max_value=120.0, value=0.0, step=1.0)
    stub_toggle = st.checkbox("Use stub (fast, no ML)", value=False)
    device_choice = st.selectbox(
        "Device",
        helpers.DEVICE_LABELS,
        index=helpers.device_label_index(default_device_label),
    )
    device_value = helpers.DEVICE_VALUE_MAP[device_choice]
    save_frames = st.checkbox("Save frames to S3", value=False)
    save_crops = st.checkbox("Save face crops to S3", value=False)
    jpeg_quality = st.number_input("JPEG quality", min_value=50, max_value=100, value=85, step=5)
    if st.button("Run detect/track", use_container_width=True):
        job_payload: Dict[str, Any] = {
            "ep_id": ep_id,
            "stub": bool(stub_toggle),
            "stride": int(stride_value),
            "device": device_value,
            "save_frames": bool(save_frames),
            "save_crops": bool(save_crops),
            "jpeg_quality": int(jpeg_quality),
        }
        if fps_value > 0:
            job_payload["fps"] = fps_value
        mode_label = "stub (no ML)" if stub_toggle else "YOLOv8 + ByteTrack"
        with st.spinner(f"Running detect/track ({mode_label})…"):
            summary, error_message = helpers.run_job_with_progress(
                ep_id,
                "/jobs/detect_track",
                job_payload,
                requested_device=device_value,
                async_endpoint="/jobs/detect_track_async",
            )
        if error_message:
            st.error(error_message)
        else:
            normalized = helpers.normalize_summary(ep_id, summary)
            detections = normalized.get("detections")
            tracks = normalized.get("tracks")
            frames_exported = normalized.get("frames_exported")
            crops_exported = normalized.get("crops_exported")
            details_line = [
                f"detections: {detections:,}" if isinstance(detections, int) else "detections: ?",
                f"tracks: {tracks:,}" if isinstance(tracks, int) else "tracks: ?",
            ]
            if frames_exported:
                details_line.append(f"frames exported: {frames_exported:,}")
            if crops_exported:
                details_line.append(f"crops exported: {crops_exported:,}")
            st.success("Completed · " + " · ".join(details_line))
            prefixes = helpers.episode_artifact_prefixes(ep_id)
            if prefixes:
                st.markdown("**S3 artifact prefixes**")
                st.code(prefixes["frames"])
                st.code(prefixes["crops"])
                st.code(prefixes["manifests"])
            action_col1, action_col2 = st.columns(2)
            with action_col1:
                st.button(
                    "Open Faces Review",
                    use_container_width=True,
                    on_click=lambda: helpers.try_switch_page("pages/3_Faces_Review.py"),
                )
            with action_col2:
                st.button(
                    "Open Screentime",
                    use_container_width=True,
                    on_click=lambda: helpers.try_switch_page("pages/4_Screentime.py"),
                )

col_faces, col_cluster, col_screen = st.columns(3)
with col_faces:
    st.markdown("### Faces Harvest")
    faces_stub = st.checkbox("Use stub (fast)", value=True, key="faces_stub_detail")
    faces_device_choice = st.selectbox(
        "Device",
        helpers.DEVICE_LABELS,
        index=helpers.device_label_index(default_device_label),
        key="faces_device_choice",
    )
    faces_device_value = helpers.DEVICE_VALUE_MAP[faces_device_choice]
    faces_save_crops = st.checkbox("Save face crops to S3", value=True, key="faces_save_crops_detail")
    faces_jpeg_quality = st.number_input(
        "JPEG quality",
        min_value=50,
        max_value=100,
        value=85,
        step=5,
        key="faces_jpeg_quality_detail",
    )
    if st.button("Run Faces Harvest", use_container_width=True):
        payload = {
            "ep_id": ep_id,
            "stub": bool(faces_stub),
            "device": faces_device_value,
            "save_crops": bool(faces_save_crops),
            "jpeg_quality": int(faces_jpeg_quality),
        }
        with st.spinner("Running faces harvest…"):
            summary, error_message = helpers.run_job_with_progress(
                ep_id,
                "/jobs/faces_embed",
                payload,
                requested_device=faces_device_value,
                async_endpoint="/jobs/faces_embed_async",
            )
        if error_message:
            st.error(error_message)
        else:
            normalized = helpers.normalize_summary(ep_id, summary)
            faces_count = normalized.get("faces")
            crops_exported = normalized.get("crops_exported")
            details = []
            if isinstance(faces_count, int):
                details.append(f"faces: {faces_count:,}")
            if crops_exported:
                details.append(f"crops exported: {crops_exported:,}")
            st.success("Faces harvest complete" + (" · " + ", ".join(details) if details else ""))
            prefixes = normalized.get("artifacts", {}).get("s3_prefixes") or helpers.episode_artifact_prefixes(ep_id)
            if prefixes:
                st.markdown("**S3 prefixes**")
                st.code(prefixes.get("crops"))
                st.code(prefixes.get("manifests"))
            st.button(
                "Open Faces Review",
                use_container_width=True,
                on_click=lambda: helpers.try_switch_page("pages/3_Faces_Review.py"),
            )
with col_cluster:
    st.markdown("### Cluster Identities")
    cluster_stub = st.checkbox("Use stub (fast)", value=True, key="cluster_stub_detail")
    cluster_device_choice = st.selectbox(
        "Device",
        helpers.DEVICE_LABELS,
        index=helpers.device_label_index(default_device_label),
        key="cluster_device_choice",
    )
    cluster_device_value = helpers.DEVICE_VALUE_MAP[cluster_device_choice]
    if st.button("Run Cluster", use_container_width=True):
        payload = {
            "ep_id": ep_id,
            "stub": bool(cluster_stub),
            "device": cluster_device_value,
        }
        with st.spinner("Clustering faces…"):
            summary, error_message = helpers.run_job_with_progress(
                ep_id,
                "/jobs/cluster",
                payload,
                requested_device=cluster_device_value,
                async_endpoint="/jobs/cluster_async",
            )
        if error_message:
            st.error(error_message)
        else:
            normalized = helpers.normalize_summary(ep_id, summary)
            identities_count = normalized.get("identities")
            faces_count = normalized.get("faces")
            details = []
            if isinstance(identities_count, int):
                details.append(f"identities: {identities_count:,}")
            if isinstance(faces_count, int):
                details.append(f"faces: {faces_count:,}")
            st.success("Clustering complete" + (" · " + ", ".join(details) if details else ""))
            prefixes = normalized.get("artifacts", {}).get("s3_prefixes") or helpers.episode_artifact_prefixes(ep_id)
            if prefixes:
                st.markdown("**Manifests prefix**")
                st.code(prefixes.get("manifests"))
            st.button(
                "Go to Facebank",
                use_container_width=True,
                on_click=lambda: helpers.try_switch_page("pages/3_Faces_Review.py"),
            )
with col_screen:
    if st.button("Compute screentime", use_container_width=True):
        try:
            resp = helpers.api_post("/jobs/screentime", {"ep_id": ep_id})
        except requests.RequestException as exc:
            st.error(helpers.describe_error(f"{cfg['api_base']}/jobs/screentime", exc))
        else:
            st.success(resp.get("status", "screentime job queued"))

st.subheader("Artifacts")
st.write(f"Video → {helpers.link_local(get_path(ep_id, 'video'))}")
st.write(f"Detections → {helpers.link_local(get_path(ep_id, 'detections'))}")
st.write(f"Tracks → {helpers.link_local(get_path(ep_id, 'tracks'))}")
prefixes = helpers.episode_artifact_prefixes(ep_id)
if prefixes:
    st.write(f"S3 Frames → `{prefixes['frames']}`")
    st.write(f"S3 Crops → `{prefixes['crops']}`")
    st.write(f"S3 Manifests → `{prefixes['manifests']}`")
manifests_dir = helpers.DATA_ROOT / "manifests" / ep_id
st.write(f"Faces → {helpers.link_local(manifests_dir / 'faces.jsonl')}")
st.write(f"Identities → {helpers.link_local(manifests_dir / 'identities.json')}")
analytics_dir = helpers.DATA_ROOT / "analytics" / ep_id
st.write(f"Screentime (json) → {helpers.link_local(analytics_dir / 'screentime.json')}")
st.write(f"Screentime (csv) → {helpers.link_local(analytics_dir / 'screentime.csv')}")

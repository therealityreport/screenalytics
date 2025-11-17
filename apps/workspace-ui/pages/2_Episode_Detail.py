from __future__ import annotations

import sys
import time
from datetime import datetime
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

SCREENTIME_JOB_KEY = "episode_detail_screentime_job"


def _load_job_defaults(ep_id: str, job_type: str) -> Dict[str, Any]:
    try:
        resp = helpers.api_get(f"/jobs?ep_id={ep_id}&job_type={job_type}&limit=1")
    except requests.RequestException:
        return {}
    jobs = resp.get("jobs") or []
    if not jobs:
        return {}
    requested = jobs[0].get("requested")
    return requested if isinstance(requested, dict) else {}


def _format_timestamp(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return value
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

cfg = helpers.init_page("Episode Detail")
st.title("Episode Detail")

if "detector" in st.session_state:
    del st.session_state["detector"]
if "tracker" in st.session_state:
    del st.session_state["tracker"]



def _handle_missing_episode(ep_id: str) -> None:
    st.warning("Episode not tracked yet.")
    parsed = helpers.parse_ep_id(ep_id)
    if not parsed:
        st.info("Unable to parse show/season/episode. Use the S3 browser to create it.")
        st.stop()
    payload = {
        "ep_id": ep_id,
        "show_slug": str(parsed["show"]).lower(),
        "season": int(parsed["season"]),
        "episode": int(parsed["episode"]),
    }
    if st.button("Create episode in store", key="episode_detail_create"):
        try:
            helpers.api_post("/episodes/upsert_by_id", payload)
        except requests.RequestException as exc:
            st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/upsert_by_id", exc))
        else:
            st.success("Episode tracked. Reloading…")
            helpers.set_ep_id(ep_id)
            st.rerun()
    st.stop()



def _prompt_for_episode() -> None:
    st.subheader("Select Episode from S3")

    # Fetch shows from S3
    try:
        shows_payload = helpers.api_get("/episodes/s3_shows")
    except requests.RequestException as exc:
        st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/s3_shows", exc))
        st.stop()

    shows = shows_payload.get("shows", [])
    if not shows:
        st.info("No shows found in S3. Upload an episode first.")
        st.stop()

    # Show dropdown
    show_options = {show["show"]: show for show in shows}
    selected_show = st.selectbox(
        "Show",
        list(show_options.keys()),
        format_func=lambda s: f"{s} ({show_options[s]['episode_count']} episodes)",
        key="episode_detail_show_select",
    )

    if not selected_show:
        st.stop()

    # Fetch episodes for selected show
    try:
        episodes_payload = helpers.api_get(f"/episodes/s3_shows/{selected_show}/episodes")
    except requests.RequestException as exc:
        st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/s3_shows/{selected_show}/episodes", exc))
        st.stop()

    episodes = episodes_payload.get("episodes", [])
    if not episodes:
        st.warning(f"No episodes found for show '{selected_show}'")
        st.stop()

    # Episode dropdown
    episode_options = {ep["ep_id"]: ep for ep in episodes}
    selected_ep_id = st.selectbox(
        "Episode",
        list(episode_options.keys()),
        format_func=lambda eid: f"S{episode_options[eid]['season']:02d}E{episode_options[eid]['episode']:02d} ({eid}) {'✓' if episode_options[eid]['exists_in_store'] else '⚠'}",
        key="episode_detail_ep_select",
    )

    if not selected_ep_id:
        st.stop()

    selected_episode = episode_options[selected_ep_id]

    # Show episode info
    st.caption(f"S3 key: `{selected_episode['key']}`")
    if selected_episode['exists_in_store']:
        st.caption("✓ Tracked in episode store")
    else:
        st.warning("⚠ Not tracked in episode store yet. Click 'Load Episode' to create it.")

    if st.button("Load Episode", use_container_width=True, type="primary"):
        # If not in store, create it first
        if not selected_episode['exists_in_store']:
            parsed = helpers.parse_ep_id(selected_ep_id)
            if parsed:
                payload = {
                    "ep_id": selected_ep_id,
                    "show_slug": str(parsed["show"]).lower(),
                    "season": int(parsed["season"]),
                    "episode": int(parsed["episode"]),
                }
                try:
                    helpers.api_post("/episodes/upsert_by_id", payload)
                    st.success(f"Episode `{selected_ep_id}` created in store.")
                except requests.RequestException as exc:
                    st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/upsert_by_id", exc))
                    st.stop()

        helpers.set_ep_id(selected_ep_id)
        st.rerun()

    st.stop()


def _format_phase_status(label: str, status: Dict[str, Any], count_key: str) -> str:
    status_value = str(status.get("status") or "missing").lower()
    if status_value == "success":
        count_val = status.get(count_key)
        parts = [f"{label}: Complete"]
        if isinstance(count_val, int):
            parts.append(f"({count_val:,})")
        finished = _format_timestamp(status.get("finished_at"))
        if finished:
            parts.append(f"• finished {finished}")
        return " ".join(parts)
    if status_value == "missing":
        base = f"{label}: Not started"
    else:
        base = f"{label}: {status_value.title()}"
    finished = _format_timestamp(status.get("finished_at"))
    if finished:
        base += f" • last run {finished}"
    if status.get("error"):
        base += f" • {status['error']}"
    return base


ep_id = helpers.get_ep_id()
if not ep_id:
    _prompt_for_episode()
ep_id = ep_id.strip()
canonical_ep_id = ep_id.lower()
if canonical_ep_id != ep_id:
    helpers.set_ep_id(canonical_ep_id)
    st.experimental_rerun()
ep_id = canonical_ep_id

try:
    details = helpers.api_get(f"/episodes/{ep_id}")
except requests.HTTPError as exc:
    if exc.response is not None and exc.response.status_code == 404:
        _handle_missing_episode(ep_id)
    st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/{ep_id}", exc))
    st.stop()
except requests.RequestException as exc:
    st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/{ep_id}", exc))
    st.stop()

status_payload = helpers.get_episode_status(ep_id)
if status_payload is None:
    detect_phase_status: Dict[str, Any] = {}
    faces_phase_status: Dict[str, Any] = {"status": "unknown"}
    cluster_phase_status: Dict[str, Any] = {"status": "unknown"}
else:
    detect_phase_status = status_payload.get("detect_track") or {}
    faces_phase_status = status_payload.get("faces_embed") or {}
    cluster_phase_status = status_payload.get("cluster") or {}

prefixes = helpers.episode_artifact_prefixes(ep_id)
bucket_name = cfg.get("bucket")
tracks_path = get_path(ep_id, "tracks")
manifests_dir = get_path(ep_id, "detections").parent
faces_path = manifests_dir / "faces.jsonl"
identities_path = manifests_dir / "identities.json"
detect_job_defaults = _load_job_defaults(ep_id, "detect_track")
local_video_exists = bool(details["local"].get("exists"))

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
if prefixes:
    st.caption(
        "S3 artifacts → "
        f"Frames {helpers.s3_uri(prefixes['frames'], bucket_name)} | "
        f"Crops {helpers.s3_uri(prefixes['crops'], bucket_name)} | "
        f"Manifests {helpers.s3_uri(prefixes['manifests'], bucket_name)}"
    )
if tracks_path.exists():
    st.caption(f"Latest detector: {helpers.tracks_detector_label(ep_id)}")
    st.caption(f"Latest tracker: {helpers.tracks_tracker_label(ep_id)}")

# Add pipeline state indicators
if status_payload:
    st.divider()
    header_cols = st.columns([3, 1])
    with header_cols[0]:
        st.subheader("Pipeline Status")
    with header_cols[1]:
        if st.button("Refresh status", key="episode_status_refresh", use_container_width=True):
            st.experimental_rerun()
    st.caption(f"Status refreshed at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    col1, col2, col3 = st.columns(3)

    scenes_ready = bool(status_payload.get("scenes_ready"))
    tracks_ready_status = bool(status_payload.get("tracks_ready"))
    faces_harvested_status = bool(status_payload.get("faces_harvested"))

    detect_status_value = str(detect_phase_status.get("status") or "missing").lower()
    faces_status_value = str(faces_phase_status.get("status") or "missing").lower()
    cluster_status_value = str(cluster_phase_status.get("status") or "missing").lower()

    with col1:
        if detect_status_value == "success":
            st.success("✅ **Detect/Track**: Complete")
            det = detect_phase_status.get("detector") or "--"
            trk = detect_phase_status.get("tracker") or "--"
            st.caption(f"{det} + {trk}")
            detections = detect_phase_status.get("detections")
            tracks = detect_phase_status.get("tracks")
            st.caption(f"{(detections or 0):,} detections, {(tracks or 0):,} tracks")
        elif detect_status_value == "partial":
            st.warning("⚠️ **Detect/Track**: Detections present but tracks missing")
            st.caption("Rerun detect/track to rebuild tracks.")
        elif detect_status_value == "missing":
            st.info("⏳ **Detect/Track**: Not started")
            st.caption("Run detect/track first.")
        else:
            st.error(f"⚠️ **Detect/Track**: {detect_status_value.title()}")
            if detect_phase_status.get("error"):
                st.caption(detect_phase_status["error"])
        finished = _format_timestamp(detect_phase_status.get("finished_at"))
        if finished:
            st.caption(f"Last run: {finished}")

    with col2:
        if faces_status_value == "success":
            st.success("✅ **Faces Harvest**: Complete")
            face_count = faces_phase_status.get("faces") or 0
            st.caption(f"{face_count:,} faces embedded")
        elif faces_status_value not in {"missing", "unknown"}:
            st.warning(f"⚠️ **Faces Harvest**: {faces_status_value.title()}")
            if faces_phase_status.get("error"):
                st.caption(faces_phase_status["error"])
        elif tracks_ready_status:
            st.info("⏳ **Faces Harvest**: Ready to run")
            st.caption("Click 'Run Faces Harvest' below.")
        else:
            st.info("⏳ **Faces Harvest**: Waiting for tracks")
            st.caption("Complete detect/track first.")
        finished = _format_timestamp(faces_phase_status.get("finished_at"))
        if finished:
            st.caption(f"Last run: {finished}")

    with col3:
        identities_count = cluster_phase_status.get("identities") or 0
        if cluster_status_value == "success" and identities_count > 0:
            st.success("✅ **Cluster**: Complete")
            st.caption(f"{identities_count:,} identities found")
        elif cluster_status_value not in {"missing", "unknown"}:
            st.warning(f"⚠️ **Cluster**: {cluster_status_value.title()}")
            if cluster_phase_status.get("error"):
                st.caption(cluster_phase_status["error"])
        elif faces_harvested_status:
            st.info("⏳ **Cluster**: Ready to run")
            st.caption("Click 'Run Cluster' below.")
        else:
            st.info("⏳ **Cluster**: Waiting for faces")
            st.caption("Complete faces harvest first.")
        finished = _format_timestamp(cluster_phase_status.get("finished_at"))
        if finished:
            st.caption(f"Last run: {finished}")

    st.divider()

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
    st.markdown("### Detect/Track Faces")

    # Show current detector/tracker configuration prominently
    st.info(
        f"**Configuration**: This will run **full face detection + tracking** on every frame.\n\n"
        f"- **Face Detector**: {helpers.LABEL.get(helpers.DEFAULT_DETECTOR, helpers.DEFAULT_DETECTOR)} (RetinaFace)\n"
        f"- **Tracker**: {helpers.LABEL.get(helpers.DEFAULT_TRACKER, helpers.DEFAULT_TRACKER)} (ByteTrack)\n"
        f"- **Stride**: {helpers.DEFAULT_STRIDE} (every frame for 24fps episodes)\n"
        f"- **Device**: auto (falls back to CPU)\n\n"
        f"This will generate `detections.jsonl` and `tracks.jsonl` for the episode."
    )

    stride_default = int(detect_job_defaults.get("stride") or helpers.DEFAULT_STRIDE)
    fps_default = float(detect_job_defaults.get("fps") or 0.0)
    save_frames_default = detect_job_defaults.get("save_frames")
    if save_frames_default is None:
        save_frames_default = True
    save_crops_default = detect_job_defaults.get("save_crops")
    if save_crops_default is None:
        save_crops_default = True
    jpeg_quality_default = int(detect_job_defaults.get("jpeg_quality") or 85)
    max_gap_default = int(detect_job_defaults.get("max_gap") or helpers.DEFAULT_MAX_GAP)
    scene_threshold_default = float(detect_job_defaults.get("scene_threshold") or helpers.SCENE_THRESHOLD_DEFAULT)
    scene_min_len_default = int(detect_job_defaults.get("scene_min_len") or helpers.SCENE_MIN_LEN_DEFAULT)
    scene_warmup_default = int(detect_job_defaults.get("scene_warmup_dets") or helpers.SCENE_WARMUP_DETS_DEFAULT)
    if "scene_detector_choice" not in st.session_state and detect_job_defaults.get("scene_detector"):
        st.session_state["scene_detector_choice"] = detect_job_defaults["scene_detector"]

    stride_value = st.number_input("Stride", min_value=1, max_value=50, value=stride_default, step=1)
    fps_value = st.number_input("FPS", min_value=0.0, max_value=120.0, value=fps_default, step=1.0)
    save_frames = st.checkbox("Save frames to S3", value=bool(save_frames_default))
    save_crops = st.checkbox("Save face crops to S3", value=bool(save_crops_default))
    jpeg_quality = st.number_input("JPEG quality", min_value=50, max_value=100, value=jpeg_quality_default, step=5)
    max_gap_value = st.number_input("Max gap (frames)", min_value=1, max_value=240, value=max_gap_default, step=1)
    with st.expander("Advanced detect/track", expanded=False):
        scene_detector_label = st.selectbox(
            "Scene detector",
            helpers.SCENE_DETECTOR_LABELS,
            index=helpers.scene_detector_label_index(st.session_state.get("scene_detector_choice")),
            help="PySceneDetect uses content detection for accurate hard cuts; switch to the internal HSV histogram fallback or disable if PySceneDetect is unavailable.",
            key="scene_detector_select",
        )
        scene_detector_value = helpers.scene_detector_value_from_label(scene_detector_label)
        st.session_state["scene_detector_choice"] = scene_detector_value
        scene_threshold_value = st.number_input(
            "Scene cut threshold",
            min_value=0.0,
            value=scene_threshold_default,
            step=0.05,
            help="PySceneDetect defaults to 27.0 (ContentDetector threshold). The histogram fallback still expects 0-2 deltas.",
        )
        scene_min_len_value = st.number_input(
            "Minimum frames between cuts",
            min_value=1,
            max_value=600,
            value=scene_min_len_default,
            step=1,
        )
        scene_warmup_value = st.number_input(
            "Warmup detections after cut",
            min_value=0,
            max_value=25,
            value=scene_warmup_default,
            step=1,
            help="Force full detection on the first N frames after each cut",
        )
    run_disabled = False
    if not local_video_exists:
        st.warning("Mirror the episode locally before running detect/track.")
        run_disabled = True
    run_label = "Run detect/track"
    if st.button(run_label, use_container_width=True, disabled=run_disabled):
        job_payload = helpers.default_detect_track_payload(
            ep_id,
            stride=int(stride_value),
            det_thresh=helpers.DEFAULT_DET_THRESH,
        )
        job_payload.update(
            {
                "save_frames": bool(save_frames),
                "save_crops": bool(save_crops),
                "jpeg_quality": int(jpeg_quality),
                "max_gap": int(max_gap_value),
                "scene_detector": scene_detector_value,
                "scene_threshold": float(scene_threshold_value),
                "scene_min_len": int(scene_min_len_value),
                "scene_warmup_dets": int(scene_warmup_value),
            }
        )
        if fps_value > 0:
            job_payload["fps"] = fps_value
        mode_label = (
            f"{helpers.LABEL.get(helpers.DEFAULT_DETECTOR, helpers.DEFAULT_DETECTOR)} + "
            f"{helpers.LABEL.get(helpers.DEFAULT_TRACKER, helpers.DEFAULT_TRACKER)}"
        )
        with st.spinner(f"Running detect/track ({mode_label})…"):
            summary, error_message = helpers.run_job_with_progress(
                ep_id,
                "/jobs/detect_track",
                job_payload,
                requested_device=helpers.DEFAULT_DEVICE,
                async_endpoint="/jobs/detect_track_async",
                requested_detector=helpers.DEFAULT_DETECTOR,
                requested_tracker=helpers.DEFAULT_TRACKER,
            )
        if error_message:
            if "RetinaFace weights missing or could not initialize" in error_message:
                st.error(error_message)
                st.caption("Run `python scripts/fetch_models.py` then retry.")
            else:
                st.error(error_message)
        else:
            normalized = helpers.normalize_summary(ep_id, summary)
            detections = helpers.coerce_int(normalized.get("detections"))
            tracks = helpers.coerce_int(normalized.get("tracks"))
            frames_exported = helpers.coerce_int(normalized.get("frames_exported"))
            crops_exported = helpers.coerce_int(normalized.get("crops_exported"))
            detector_summary = normalized.get("detector")
            tracker_summary = normalized.get("tracker")
            detector_is_scene = (
                isinstance(detector_summary, str) and detector_summary in helpers.SCENE_DETECTOR_LABEL_MAP
            )
            has_detections = detections is not None and detections > 0
            has_tracks = tracks is not None and tracks > 0
            issue_messages: list[str] = []
            if detector_is_scene:
                detector_label = helpers.SCENE_DETECTOR_LABEL_MAP.get(detector_summary, detector_summary)
                issue_messages.append(
                    f"Pipeline stopped after scene detection ({detector_label}); detect/track never ran."
                )
            if not has_detections or not has_tracks:
                det_label = helpers.format_count(detections) or "0"
                track_label = helpers.format_count(tracks) or "0"
                issue_messages.append(
                    f"No detections/tracks were created (detections={det_label}, tracks={track_label})."
                )
            if issue_messages:
                st.error(
                    " ".join(issue_messages) + " Please rerun **Detect/Track Faces** to generate the manifests."
                )
            else:
                details_line = [
                    f"detections: {helpers.format_count(detections)}" if detections is not None else "detections: ?",
                    f"tracks: {helpers.format_count(tracks)}" if tracks is not None else "tracks: ?",
                ]
                if frames_exported:
                    details_line.append(f"frames exported: {helpers.format_count(frames_exported)}")
                if crops_exported:
                    details_line.append(f"crops exported: {helpers.format_count(crops_exported)}")
                if detector_summary:
                    details_line.append(f"detector: {helpers.detector_label_from_value(detector_summary)}")
                if tracker_summary:
                    details_line.append(f"tracker: {helpers.tracker_label_from_value(tracker_summary)}")
                st.success("Completed · " + " · ".join(details_line))
                metrics = normalized.get("metrics") or {}
                if metrics:
                    st.markdown("**Track quality**")
                    stat_cols = st.columns(3)
                    stat_cols[0].metric("tracks born", metrics.get("tracks_born", 0))
                    stat_cols[1].metric("tracks lost", metrics.get("tracks_lost", 0))
                    stat_cols[2].metric("ID switches", metrics.get("id_switches", 0))
                    longest = metrics.get("longest_tracks") or []
                    if longest:
                        for entry in longest:
                            label = f"Track {entry.get('track_id')} · {entry.get('frame_count', 0)} frames"
                            if entry.get("frame_count", 0) >= 500:
                                st.warning(label)
                            else:
                                st.caption(label)
                s3_prefixes = normalized.get("artifacts", {}).get("s3_prefixes") or prefixes
                if s3_prefixes:
                    st.markdown("**S3 artifact prefixes**")
                    st.code(helpers.s3_uri(s3_prefixes.get("frames"), bucket_name))
                    st.code(helpers.s3_uri(s3_prefixes.get("crops"), bucket_name))
                    st.code(helpers.s3_uri(s3_prefixes.get("manifests"), bucket_name))
                scene_badge = helpers.scene_cuts_badge_text(normalized)
                if scene_badge:
                    st.caption(f"🎬 {scene_badge}")
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

tracks_ready = bool(status_payload and status_payload.get("tracks_ready"))
faces_ready = bool(status_payload and status_payload.get("faces_harvested"))
detector_face_only = helpers.detector_is_face_only(ep_id)

col_faces, col_cluster, col_screen = st.columns(3)
with col_faces:
    st.markdown("### Faces Harvest")
    st.caption(_format_phase_status("Faces Harvest", faces_phase_status, "faces"))

    # Add pipeline state indicator
    detect_track_info = detect_phase_status
    if detect_track_info:
        detector_name = detect_track_info.get("detector")
        tracker_name = detect_track_info.get("tracker")
        if detector_name and tracker_name:
            st.caption(f"📊 Current pipeline: {detector_name} + {tracker_name}")

    faces_device_choice = st.selectbox(
        "Device",
        helpers.DEVICE_LABELS,
        index=helpers.device_label_index(default_device_label),
        key="faces_device_choice",
    )
    faces_device_value = helpers.DEVICE_VALUE_MAP[faces_device_choice]
    faces_save_frames = st.checkbox("Save sampled frames", value=True, key="faces_save_frames_detail")
    faces_save_crops = st.checkbox("Save face crops to S3", value=True, key="faces_save_crops_detail")
    faces_jpeg_quality = st.number_input(
        "JPEG quality",
        min_value=50,
        max_value=100,
        value=85,
        step=5,
        key="faces_jpeg_quality_detail",
    )
    if faces_device_value == "mps":
        st.caption(
            "ArcFace embeddings run on CPU when MPS is selected. Tracker/crop export still uses the requested device."
        )

    # Improved messaging for when Harvest Faces is disabled
    if not local_video_exists:
        st.warning("Mirror the episode locally before running Faces Harvest.")
    elif not tracks_ready:
        st.warning(
            "**Harvest Faces is unavailable**: Face detection/tracking has not run yet.\n\n"
            "Run **Detect/Track Faces** first to generate `tracks.jsonl` for this episode. "
            "The detect/track job must complete successfully with RetinaFace + ByteTrack before you can harvest faces."
        )
        if detect_track_info and detect_track_info.get("detector") == "pyscenedetect":
            st.error(
                "⚠️ **Scene detection only**: Your last run only executed scene detection (PySceneDetect), "
                "not full face detection + tracking. Please run **Detect/Track Faces** again to generate tracks."
            )
    elif not detector_face_only:
        st.warning("Current tracks were generated with a legacy detector. Rerun detect/track with RetinaFace or YOLOv8-face.")

    faces_disabled = (not tracks_ready) or (not detector_face_only) or (not local_video_exists)
    if st.button("Run Faces Harvest", use_container_width=True, disabled=faces_disabled):
        payload = {
            "ep_id": ep_id,
            "device": faces_device_value,
            "save_frames": bool(faces_save_frames),
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
                requested_detector=helpers.tracks_detector_value(ep_id),
                requested_tracker=helpers.tracks_tracker_value(ep_id),
            )
        if error_message:
            if "tracks.jsonl" in error_message.lower():
                st.error("Run detect/track first.")
            else:
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
            s3_prefixes = normalized.get("artifacts", {}).get("s3_prefixes") or prefixes
            if s3_prefixes:
                st.markdown("**S3 prefixes**")
                st.code(helpers.s3_uri(s3_prefixes.get("crops"), bucket_name))
                st.code(helpers.s3_uri(s3_prefixes.get("manifests"), bucket_name))
            st.button(
                "Open Faces Review",
                use_container_width=True,
                on_click=lambda: helpers.try_switch_page("pages/3_Faces_Review.py"),
            )
with col_cluster:
    st.markdown("### Cluster Identities")
    st.caption(_format_phase_status("Cluster Identities", cluster_phase_status, "identities"))
    cluster_device_choice = st.selectbox(
        "Device",
        helpers.DEVICE_LABELS,
        index=helpers.device_label_index(default_device_label),
        key="cluster_device_choice",
    )
    cluster_device_value = helpers.DEVICE_VALUE_MAP[cluster_device_choice]
    if not local_video_exists:
        st.warning("Mirror the episode locally before clustering identities.")
    elif not faces_ready:
        st.caption("Run faces harvest first.")
    elif not detector_face_only:
        st.warning("Current tracks were generated with a legacy detector. Rerun detect/track first.")
    cluster_disabled = (not faces_ready) or (not detector_face_only) or (not local_video_exists)
    if st.button("Run Cluster", use_container_width=True, disabled=cluster_disabled):
        payload = {
            "ep_id": ep_id,
            "device": cluster_device_value,
        }
        with st.spinner("Clustering faces…"):
            summary, error_message = helpers.run_job_with_progress(
                ep_id,
                "/jobs/cluster",
                payload,
                requested_device=cluster_device_value,
                async_endpoint="/jobs/cluster_async",
                requested_detector=helpers.tracks_detector_value(ep_id),
                requested_tracker=helpers.tracks_tracker_value(ep_id),
            )
        if error_message:
            if "faces.jsonl" in error_message.lower():
                st.error("Run faces harvest first.")
            else:
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
            s3_prefixes = normalized.get("artifacts", {}).get("s3_prefixes") or prefixes
            if s3_prefixes:
                st.markdown("**Manifests prefix**")
                st.code(helpers.s3_uri(s3_prefixes.get("manifests"), bucket_name))
            st.button(
                "Go to Facebank",
                use_container_width=True,
                on_click=lambda: helpers.try_switch_page("pages/3_Faces_Review.py"),
            )
with col_screen:
    st.markdown("### Screentime")
    screentime_disabled = not local_video_exists
    if screentime_disabled:
        st.warning("Mirror the episode locally before computing screentime.")
    if st.button("Compute screentime", use_container_width=True, disabled=screentime_disabled):
        with st.spinner("Starting screentime analysis…"):
            try:
                resp = helpers.api_post("/jobs/screen_time/analyze", {"ep_id": ep_id})
            except requests.RequestException as exc:
                st.error(helpers.describe_error(f"{cfg['api_base']}/jobs/screen_time/analyze", exc))
            else:
                job_id = resp.get("job_id")
                if job_id:
                    st.session_state[SCREENTIME_JOB_KEY] = job_id
                st.success("Screen time job queued.")
                st.experimental_rerun()

    screentime_job_id = st.session_state.get(SCREENTIME_JOB_KEY)
    if screentime_job_id:
        try:
            job_progress_resp = helpers.api_get(f"/jobs/{screentime_job_id}/progress")
        except requests.RequestException as exc:
            st.warning(helpers.describe_error(f"{cfg['api_base']}/jobs/{screentime_job_id}/progress", exc))
        else:
            job_state = job_progress_resp.get("state")
            progress_data = job_progress_resp.get("progress") or {}
            if job_state == "running":
                st.info(f"Screentime job {screentime_job_id[:12]}… is running")
                frames_done = progress_data.get("frames_done", 0)
                frames_total = max(int(progress_data.get("frames_total") or 1), 1)
                st.progress(min(frames_done / frames_total, 1.0))
                st.caption(f"Frames {frames_done:,} / {frames_total:,}")
                time.sleep(2)
                st.experimental_rerun()
            elif job_state == "succeeded":
                st.success("Screentime analysis complete.")
                st.caption(f"JSON → {helpers.link_local(helpers.DATA_ROOT / 'analytics' / ep_id / 'screentime.json')}")
                st.caption(f"CSV → {helpers.link_local(helpers.DATA_ROOT / 'analytics' / ep_id / 'screentime.csv')}")
                if st.button("Dismiss screentime status", key="dismiss_screentime_job_success"):
                    st.session_state.pop(SCREENTIME_JOB_KEY, None)
                    st.experimental_rerun()
            elif job_state == "failed":
                st.error(f"Screentime job failed: {job_progress_resp.get('error') or 'unknown error'}")
                if st.button("Dismiss screentime status", key="dismiss_screentime_job_failed"):
                    st.session_state.pop(SCREENTIME_JOB_KEY, None)
                    st.experimental_rerun()
            else:
                st.info(f"Screentime job status: {job_state or 'unknown'}")
                if st.button("Dismiss screentime status", key="dismiss_screentime_job_other"):
                    st.session_state.pop(SCREENTIME_JOB_KEY, None)
                    st.experimental_rerun()

st.subheader("Artifacts")


def _render_artifact_entry(label: str, local_path: Path, key_suffix: str, s3_key: str | None = None) -> None:
    st.write(f"{label} → {helpers.link_local(local_path)}")
    if not s3_key:
        return
    uri_col, button_col = st.columns([4, 1])
    uri_col.code(helpers.s3_uri(s3_key, bucket_name))
    if button_col.button("Presign", key=f"presign_{key_suffix}"):
        try:
            presign_resp = helpers.api_get("/files/presign", params={"key": s3_key})
        except requests.RequestException as exc:
            st.error(helpers.describe_error(f"{cfg['api_base']}/files/presign", exc))
        else:
            url_value = presign_resp.get("url")
            if url_value:
                st.code(url_value)
                ttl_val = presign_resp.get("expires_in")
                if ttl_val:
                    st.caption(f"Expires in {ttl_val}s")
            else:
                st.warning("Presign unavailable for this key.")


manifests_prefix = (prefixes or {}).get("manifests") if prefixes else None
_render_artifact_entry(
    "Video",
    get_path(ep_id, "video"),
    "video",
    details["s3"]["v2_key"] or details["s3"]["v1_key"],
)
detections_key = f"{manifests_prefix}detections.jsonl" if manifests_prefix else None
tracks_key = f"{manifests_prefix}tracks.jsonl" if manifests_prefix else None
faces_key = f"{manifests_prefix}faces.jsonl" if manifests_prefix else None
identities_key = f"{manifests_prefix}identities.json" if manifests_prefix else None
_render_artifact_entry("Detections", get_path(ep_id, "detections"), "detections", detections_key)
_render_artifact_entry("Tracks", get_path(ep_id, "tracks"), "tracks", tracks_key)
_render_artifact_entry("Faces", faces_path, "faces", faces_key)
_render_artifact_entry("Identities", identities_path, "identities", identities_key)
analytics_dir = helpers.DATA_ROOT / "analytics" / ep_id
_render_artifact_entry("Screentime (json)", analytics_dir / "screentime.json", "screentime_json")
_render_artifact_entry("Screentime (csv)", analytics_dir / "screentime.csv", "screentime_csv")

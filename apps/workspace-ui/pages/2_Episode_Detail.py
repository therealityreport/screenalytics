from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import requests
import streamlit as st

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402

from py_screenalytics.artifacts import get_path  # noqa: E402

SCREENTIME_JOB_KEY = "episode_detail_screentime_job"
FRAME_JPEG_SIZE_EST_BYTES = 220_000
CROP_JPEG_SIZE_EST_BYTES = 40_000
AVG_FACES_PER_FRAME = 1.5


def _load_job_defaults(
    ep_id: str, job_type: str
) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    try:
        resp = helpers.api_get(f"/jobs?ep_id={ep_id}&job_type={job_type}&limit=1")
    except requests.RequestException:
        return {}, None
    jobs = resp.get("jobs") or []
    if not jobs:
        return {}, None
    job = jobs[0]
    requested = job.get("requested")
    if isinstance(requested, dict):
        return dict(requested), job
    return {}, job


def _format_timestamp(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return value
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _choose_value(*candidates: Any, fallback: str) -> str:
    for candidate in candidates:
        if isinstance(candidate, str):
            cleaned = candidate.strip()
            if cleaned:
                return cleaned.lower()
    return fallback


def _count_manifest_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return None


def _manifest_has_rows(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    return True
    except OSError:
        return False
    return False


def _detect_track_manifests_ready(detections_path: Path, tracks_path: Path) -> dict:
    detections_ready = _manifest_has_rows(detections_path)
    tracks_ready = _manifest_has_rows(tracks_path)
    return {
        "detections_ready": detections_ready,
        "tracks_ready": tracks_ready,
        "manifest_ready": tracks_ready,
        "tracks_only_fallback": tracks_ready and not detections_ready,
    }


def _compute_detect_track_effective_status(
    detect_status: Dict[str, Any],
    *,
    manifest_ready: bool,
    tracks_ready_flag: bool,
    job_state: str | None = None,
) -> tuple[str, bool, bool]:
    normalized_job_state = str(job_state or "").strip().lower()
    if normalized_job_state == "running":
        return "running", False, False
    normalized_status = str(detect_status.get("status") or "missing").strip().lower()
    if not normalized_status:
        normalized_status = "missing"
    manifest_tracks_ready = bool(manifest_ready)
    if normalized_status == "success":
        if manifest_tracks_ready:
            return "success", True, False
        return "stale", False, False
    if manifest_tracks_ready:
        return "success", True, True
    return normalized_status, False, False


def _estimated_sampled_frames(meta: Dict[str, Any] | None, stride: int) -> int | None:
    if not meta:
        return None
    frames_val = meta.get("frames") if isinstance(meta, dict) else None
    fps_detected = meta.get("fps_detected") if isinstance(meta, dict) else None
    duration_sec = meta.get("duration_sec") if isinstance(meta, dict) else None
    frames = None
    try:
        if frames_val is not None:
            frames = float(frames_val)
        elif duration_sec and (fps_detected or fps_detected == 0):
            frames = float(duration_sec) * float(fps_detected or 0)
        elif duration_sec:
            frames = float(duration_sec) * 24.0
    except (TypeError, ValueError):
        frames = None
    if not frames or frames <= 0:
        return None
    stride_val = max(int(stride or 1), 1)
    return max(int(frames // stride_val), 0)


cfg = helpers.init_page("Episode Detail")
st.title("Episode Detail")
flash_message = st.session_state.pop("episode_detail_flash", None)
if flash_message:
    st.success(flash_message)

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
            st.error(
                helpers.describe_error(f"{cfg['api_base']}/episodes/upsert_by_id", exc)
            )
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
        episodes_payload = helpers.api_get(
            f"/episodes/s3_shows/{selected_show}/episodes"
        )
    except requests.RequestException as exc:
        st.error(
            helpers.describe_error(
                f"{cfg['api_base']}/episodes/s3_shows/{selected_show}/episodes", exc
            )
        )
        st.stop()

    episodes = episodes_payload.get("episodes", [])
    if not episodes:
        st.warning(f"No episodes found for show '{selected_show}'")
        st.stop()

    tracked_eps = [ep for ep in episodes if ep.get("exists_in_store")]
    orphan_eps = [ep for ep in episodes if not ep.get("exists_in_store")]
    show_orphans = True
    if orphan_eps:
        default_show = not bool(tracked_eps)
        show_orphans = st.checkbox(
            f"Show {len(orphan_eps)} orphan uploads (⚠)",
            value=default_show,
            key="episode_detail_show_orphans",
            help="Orphan uploads are raw S3 videos that were removed from the EpisodeStore. "
            "Delete them permanently with `python tools/prune_orphan_episodes.py --apply`.",
        )
        if not show_orphans and tracked_eps:
            st.info(
                f"Hiding {len(orphan_eps)} orphan uploads. Run `python tools/prune_orphan_episodes.py --apply` to remove them."
            )
    filtered_episodes = [
        ep for ep in episodes if show_orphans or ep.get("exists_in_store")
    ]
    if not filtered_episodes:
        st.warning(
            "No tracked episodes available. Upload a video or enable orphan view above."
        )
        st.stop()

    # Episode dropdown
    episode_options = {ep["ep_id"]: ep for ep in filtered_episodes}
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
    if selected_episode["exists_in_store"]:
        st.caption("✓ Tracked in episode store")
    else:
        st.warning(
            "⚠ Not tracked in episode store yet. Click 'Load Episode' to create it."
        )

    if st.button("Load Episode", use_container_width=True, type="primary"):
        # If not in store, create it first
        if not selected_episode["exists_in_store"]:
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
                    st.error(
                        helpers.describe_error(
                            f"{cfg['api_base']}/episodes/upsert_by_id", exc
                        )
                    )
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


def _ensure_local_artifacts(ep_id: str, details: Dict[str, Any]) -> bool:
    local_block = details.setdefault("local", {})
    video_path = get_path(ep_id, "video")
    if video_path.exists():
        local_block["path"] = str(video_path)
        local_block["exists"] = True
        return True
    s3_meta = details.get("s3") or {}
    if not (s3_meta.get("v2_exists") or s3_meta.get("v1_exists")):
        st.error(
            "Episode is not mirrored in S3; mirror/upload the video before running this job."
        )
        return False
    mirror_path = f"/episodes/{ep_id}/mirror"
    with st.spinner("Mirroring artifacts from S3…"):
        try:
            resp = helpers.api_post(mirror_path)
        except requests.RequestException as exc:
            st.error(helpers.describe_error(f"{cfg['api_base']}{mirror_path}", exc))
            return False
        st.success(
            f"Mirrored to {helpers.link_local(resp['local_video_path'])} "
            f"({helpers.human_size(resp.get('bytes'))})"
        )
        local_block["path"] = resp.get("local_video_path") or str(video_path)
        local_block["exists"] = True
        return True


def _launch_detect_job(
    local_exists: bool,
    ep_id: str,
    details: Dict[str, Any],
    job_payload: Dict[str, Any],
    device_value: str,
    detector_value: str,
    tracker_value: str,
    mode_label: str,
    device_label: str,
):
    current_local = local_exists
    if not current_local:
        if not _ensure_local_artifacts(ep_id, details):
            return current_local, None, "mirror_failed"
        current_local = True
    with st.spinner(f"Running detect/track ({mode_label} on {device_label})…"):
        summary, error_message = helpers.run_job_with_progress(
            ep_id,
            "/jobs/detect_track",
            job_payload,
            requested_device=device_value,
            async_endpoint="/jobs/detect_track_async",
            requested_detector=detector_value,
            requested_tracker=tracker_value,
        )
    return current_local, summary, error_message


ep_id = helpers.get_ep_id()
if not ep_id:
    _prompt_for_episode()
ep_id = ep_id.strip()
canonical_ep_id = ep_id.lower()
if canonical_ep_id != ep_id:
    helpers.set_ep_id(canonical_ep_id)
    st.rerun()
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
detections_path = get_path(ep_id, "detections")
manifests_dir = detections_path.parent
faces_path = manifests_dir / "faces.jsonl"
identities_path = manifests_dir / "identities.json"
detect_job_defaults, detect_job_record = _load_job_defaults(ep_id, "detect_track")
faces_job_defaults, _ = _load_job_defaults(ep_id, "faces_embed")
cluster_job_defaults, _ = _load_job_defaults(ep_id, "cluster")
local_video_exists = bool(details["local"].get("exists"))
video_meta_key = f"episode_detail_video_meta::{ep_id}"
video_meta = st.session_state.get(video_meta_key)
if local_video_exists:
    if video_meta is None:
        try:
            video_meta = helpers.api_get(f"/episodes/{ep_id}/video_meta")
        except requests.RequestException:
            video_meta = None
        else:
            st.session_state[video_meta_key] = video_meta
else:
    st.session_state.pop(video_meta_key, None)


st.subheader(f"Episode `{ep_id}`")
st.write(
    f"Show `{details['show_slug']}` · Season {details['season_number']} Episode {details['episode_number']}"
)
st.write(f"S3 v2 → `{details['s3']['v2_key']}` (exists={details['s3']['v2_exists']})")
st.write(f"S3 v1 → `{details['s3']['v1_key']}` (exists={details['s3']['v1_exists']})")
if not details["s3"]["v2_exists"] and details["s3"]["v1_exists"]:
    st.warning(
        "Legacy v1 object detected; mirroring will use it until the v2 path is populated."
    )
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

manifest_state = _detect_track_manifests_ready(detections_path, tracks_path)

# Get status values from API
faces_status_value = str(faces_phase_status.get("status") or "missing").lower()
cluster_status_value = str(cluster_phase_status.get("status") or "missing").lower()
tracks_ready_flag = bool((status_payload or {}).get("tracks_ready"))
detect_job_state = (detect_job_record or {}).get("state")
detect_status_value, tracks_ready, using_manifest_fallback = (
    _compute_detect_track_effective_status(
        detect_phase_status,
        manifest_ready=manifest_state["manifest_ready"],
        tracks_ready_flag=tracks_ready_flag,
        job_state=detect_job_state,
    )
)

# Other status values
faces_ready_state = faces_status_value == "success"
faces_count_value = helpers.coerce_int(faces_phase_status.get("faces"))
identities_count_value = helpers.coerce_int(cluster_phase_status.get("identities"))
faces_manifest_count = _count_manifest_rows(faces_path)
if not faces_ready_state and faces_manifest_count is not None:
    faces_ready_state = True
    if faces_count_value is None:
        faces_count_value = faces_manifest_count

# Add pipeline state indicators
if status_payload:
    st.divider()

    header_cols = st.columns([3, 1])
    with header_cols[0]:
        st.subheader("Pipeline Status")
    with header_cols[1]:
        if st.button(
            "Refresh status", key="episode_status_refresh", use_container_width=True
        ):
            st.rerun()
    st.caption(
        f"Status refreshed at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    col1, col2, col3 = st.columns(3)

    with col1:
        if detect_status_value == "success":
            st.success("✅ **Detect/Track**: Complete")
            det = detect_phase_status.get("detector") or "--"
            trk = detect_phase_status.get("tracker") or "--"
            st.caption(f"{det} + {trk}")
            detections = detect_phase_status.get("detections")
            tracks = detect_phase_status.get("tracks")
            st.caption(f"{(detections or 0):,} detections, {(tracks or 0):,} tracks")
            ratio_value = helpers.coerce_float(
                detect_phase_status.get("track_to_detection_ratio")
                or detect_phase_status.get("track_ratio")
            )
            if ratio_value is not None:
                st.caption(f"Tracks / detections: {ratio_value:.2f}")
                if ratio_value < 0.1:
                    st.caption(
                        "⚠️ Track-to-detection ratio < 0.10. Consider lowering ByteTrack thresholds or rerunning detect/track."
                    )
            # Show manifest-fallback caption when status was inferred from manifests
            if using_manifest_fallback:
                st.caption(
                    "ℹ️ _Detect/Track completion inferred from manifests (status API missing/stale)._"
                )
        elif detect_status_value == "running":
            st.info("⏳ **Detect/Track**: Running")
            if detect_job_record and detect_job_record.get("started_at"):
                st.caption(f"Started at {detect_job_record['started_at']}")
            st.caption("Live progress appears in the log panel below.")
        elif detect_status_value == "stale":
            st.warning("⚠️ **Detect/Track**: Status stale (manifests missing)")
            st.caption(
                "Rerun Detect/Track Faces to rebuild detections/tracks for this episode."
            )
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
        if faces_ready_state:
            st.success("✅ **Faces Harvest**: Complete")
            face_count_label = helpers.format_count(faces_count_value) or "0"
            st.caption(f"Faces: {face_count_label} (harvest completed)")
        elif faces_status_value not in {"missing", "unknown"}:
            st.warning(f"⚠️ **Faces Harvest**: {faces_status_value.title()}")
            if faces_phase_status.get("error"):
                st.caption(faces_phase_status["error"])
        elif tracks_ready:
            st.info("⏳ **Faces Harvest**: Ready to run")
            st.caption("Click 'Run Faces Harvest' below.")
        else:
            st.info("⏳ **Faces Harvest**: Waiting for tracks")
            st.caption("Complete detect/track first.")
        finished = _format_timestamp(faces_phase_status.get("finished_at"))
        if finished:
            st.caption(f"Last run: {finished}")

    with col3:
        identities_label = helpers.format_count(identities_count_value) or "0"
        if cluster_status_value == "success":
            st.success("✅ **Cluster**: Complete")
            st.caption(f"Identities: {identities_label}")
        elif cluster_status_value not in {"missing", "unknown"}:
            st.warning(f"⚠️ **Cluster**: {cluster_status_value.title()}")
            if cluster_phase_status.get("error"):
                st.caption(cluster_phase_status["error"])
        elif faces_ready_state:
            if (faces_count_value or 0) == 0:
                st.info("ℹ️ **Cluster**: No faces to cluster")
                st.caption("Faces harvest finished with 0 faces → expect 0 identities.")
            else:
                st.info("⏳ **Cluster**: Ready to run")
                st.caption("Click 'Run Cluster' below.")
        else:
            st.info("⏳ **Cluster**: Waiting for faces")
            st.caption("Complete faces harvest first.")
        finished = _format_timestamp(cluster_phase_status.get("finished_at"))
        if finished:
            st.caption(f"Last run: {finished}")

    st.divider()

detector_override = st.session_state.pop("episode_detail_detector_override", None)
tracker_override = st.session_state.pop("episode_detail_tracker_override", None)
device_override = st.session_state.pop("episode_detail_device_override", None)
autorun_detect = st.session_state.pop("episode_detail_detect_autorun_flag", False)

detect_detector_value = _choose_value(
    detector_override,
    detect_job_defaults.get("detector"),
    detect_phase_status.get("detector"),
    helpers.tracks_detector_value(ep_id),
    fallback=helpers.DEFAULT_DETECTOR,
)
detect_tracker_value = _choose_value(
    tracker_override,
    detect_job_defaults.get("tracker"),
    detect_phase_status.get("tracker"),
    helpers.tracks_tracker_value(ep_id),
    fallback=helpers.DEFAULT_TRACKER,
)
detect_device_default_value = _choose_value(
    device_override,
    detect_job_defaults.get("device"),
    detect_job_defaults.get("requested_device"),
    detect_phase_status.get("device"),
    detect_phase_status.get("requested_device"),
    fallback=helpers.DEFAULT_DEVICE,
)
detect_device_label_default = helpers.device_label_from_value(
    detect_device_default_value
)
detect_detector_label = helpers.detector_label_from_value(detect_detector_value)
detect_tracker_label = helpers.tracker_label_from_value(detect_tracker_value)

faces_device_default_value = _choose_value(
    faces_job_defaults.get("device"),
    faces_job_defaults.get("requested_device"),
    faces_job_defaults.get("embed_device"),
    faces_phase_status.get("device"),
    faces_phase_status.get("requested_device"),
    faces_phase_status.get("embed_device"),
    fallback=detect_device_default_value,
)
faces_device_label_default = helpers.device_label_from_value(faces_device_default_value)
faces_save_frames_default = faces_job_defaults.get("save_frames")
if faces_save_frames_default is None:
    faces_save_frames_default = True
faces_save_crops_default = faces_job_defaults.get("save_crops")
if faces_save_crops_default is None:
    faces_save_crops_default = True
faces_jpeg_quality_default = (
    helpers.coerce_int(faces_job_defaults.get("jpeg_quality")) or 85
)

cluster_device_default_value = _choose_value(
    cluster_job_defaults.get("device"),
    cluster_phase_status.get("device"),
    cluster_phase_status.get("requested_device"),
    fallback=faces_device_default_value,
)
cluster_device_label_default = helpers.device_label_from_value(
    cluster_device_default_value
)
cluster_thresh_default_raw = (
    cluster_job_defaults.get("cluster_thresh")
    or cluster_phase_status.get("cluster_thresh")
    or helpers.DEFAULT_CLUSTER_SIMILARITY
)
try:
    cluster_thresh_default = float(cluster_thresh_default_raw)
except (TypeError, ValueError):
    cluster_thresh_default = helpers.DEFAULT_CLUSTER_SIMILARITY
cluster_thresh_default = min(max(cluster_thresh_default, 0.4), 0.9)
min_cluster_size_default = helpers.coerce_int(
    cluster_job_defaults.get("min_cluster_size")
)
if min_cluster_size_default is None:
    min_cluster_size_default = helpers.coerce_int(
        cluster_phase_status.get("min_cluster_size")
    )
if min_cluster_size_default is None:
    min_cluster_size_default = 2

col_hydrate, col_detect = st.columns(2)
with col_hydrate:
    if st.button("Mirror from S3", use_container_width=True):
        mirror_path = f"/episodes/{ep_id}/mirror"
        try:
            resp = helpers.api_post(mirror_path)
        except requests.RequestException as exc:
            st.error(helpers.describe_error(f"{cfg['api_base']}{mirror_path}", exc))
        else:
            st.session_state["episode_detail_flash"] = (
                f"Mirrored → {helpers.link_local(resp['local_video_path'])} | size {helpers.human_size(resp.get('bytes'))}"
            )
            st.rerun()
with col_detect:
    st.markdown("### Detect/Track Faces")

    stride_default = int(detect_job_defaults.get("stride") or helpers.DEFAULT_STRIDE)
    fps_default = float(detect_job_defaults.get("fps") or 0.0)
    det_thresh_default = float(
        detect_job_defaults.get("det_thresh") or helpers.DEFAULT_DET_THRESH
    )
    save_frames_default = detect_job_defaults.get("save_frames")
    if save_frames_default is None:
        save_frames_default = True
    save_crops_default = detect_job_defaults.get("save_crops")
    if save_crops_default is None:
        save_crops_default = True
    jpeg_quality_default = int(detect_job_defaults.get("jpeg_quality") or 85)
    max_gap_default = int(detect_job_defaults.get("max_gap") or helpers.DEFAULT_MAX_GAP)
    scene_threshold_default = float(
        detect_job_defaults.get("scene_threshold") or helpers.SCENE_THRESHOLD_DEFAULT
    )
    scene_min_len_default = int(
        detect_job_defaults.get("scene_min_len") or helpers.SCENE_MIN_LEN_DEFAULT
    )
    scene_warmup_default = int(
        detect_job_defaults.get("scene_warmup_dets")
        or helpers.SCENE_WARMUP_DETS_DEFAULT
    )
    if "scene_detector_choice" not in st.session_state and detect_job_defaults.get(
        "scene_detector"
    ):
        st.session_state["scene_detector_choice"] = detect_job_defaults[
            "scene_detector"
        ]

    stride_hint = (
        "every frame" if stride_default == 1 else f"every {stride_default}th frame"
    )
    st.info(
        f"**Configuration**: This will run **full face detection + tracking** on sampled frames.\n\n"
        f"- **Face Detector**: {detect_detector_label}\n"
        f"- **Tracker**: {detect_tracker_label}\n"
        f"- **Stride**: {stride_default} ({stride_hint})\n"
        f"- **Device**: {detect_device_label_default}\n\n"
        f"This job exports `detections.jsonl` and `tracks.jsonl` plus optional frames/crops."
    )

    stride_value = st.number_input(
        "Stride", min_value=1, max_value=50, value=stride_default, step=1
    )
    fps_value = st.number_input(
        "FPS", min_value=0.0, max_value=120.0, value=fps_default, step=1.0
    )
    # Automatically save to S3
    save_frames = True
    save_crops = True
    jpeg_quality = st.number_input(
        "JPEG quality", min_value=50, max_value=100, value=jpeg_quality_default, step=5
    )

    session_prefix = f"episode_detail_detect::{ep_id}"
    max_gap_key = f"{session_prefix}::max_gap"
    max_gap_seed = int(st.session_state.get(max_gap_key, max_gap_default))
    max_gap_value = st.number_input(
        "Max gap (frames)", min_value=1, max_value=240, value=max_gap_seed, step=1
    )
    st.session_state[max_gap_key] = int(max_gap_value)

    det_thresh_key = f"{session_prefix}::det_thresh"
    det_thresh_seed = float(st.session_state.get(det_thresh_key, det_thresh_default))
    det_thresh_value = st.slider(
        "Detection threshold",
        min_value=0.1,
        max_value=0.9,
        value=float(det_thresh_seed),
        step=0.01,
        help="Lower thresholds increase recall but may introduce more false positives.",
    )
    st.session_state[det_thresh_key] = float(det_thresh_value)

    track_high_default = helpers.coerce_float(
        detect_job_defaults.get("track_high_thresh")
    )
    if track_high_default is None:
        track_high_default = helpers.coerce_float(
            detect_phase_status.get("track_high_thresh")
        )
    if track_high_default is None:
        track_high_default = helpers.TRACK_HIGH_THRESH_DEFAULT
    track_new_default = helpers.coerce_float(
        detect_job_defaults.get("new_track_thresh")
    )
    if track_new_default is None:
        track_new_default = helpers.coerce_float(
            detect_phase_status.get("new_track_thresh")
        )
    if track_new_default is None:
        track_new_default = helpers.TRACK_NEW_THRESH_DEFAULT
    track_high_value: float | None = track_high_default
    track_new_value: float | None = track_new_default

    with st.expander("Advanced detect/track", expanded=False):
        scene_detector_label = st.selectbox(
            "Scene detector",
            helpers.SCENE_DETECTOR_LABELS,
            index=helpers.scene_detector_label_index(
                st.session_state.get("scene_detector_choice")
            ),
            help="PySceneDetect uses content detection for accurate hard cuts; switch to HSV fallback or disable if unavailable.",
            key="scene_detector_select",
        )
        scene_detector_value = helpers.scene_detector_value_from_label(
            scene_detector_label
        )
        st.session_state["scene_detector_choice"] = scene_detector_value

        scene_thresh_key = f"{session_prefix}::scene_threshold"
        scene_thresh_seed = float(
            st.session_state.get(scene_thresh_key, scene_threshold_default)
        )
        scene_threshold_value = st.number_input(
            "Scene cut threshold",
            min_value=0.0,
            value=scene_thresh_seed,
            step=0.05,
            help="PySceneDetect defaults to 27.0 (ContentDetector threshold). HSV fallback expects 0-2 deltas.",
        )
        st.session_state[scene_thresh_key] = float(scene_threshold_value)

        scene_min_key = f"{session_prefix}::scene_min_len"
        scene_min_seed = int(st.session_state.get(scene_min_key, scene_min_len_default))
        scene_min_len_value = st.number_input(
            "Minimum frames between cuts",
            min_value=1,
            max_value=600,
            value=scene_min_seed,
            step=1,
        )
        st.session_state[scene_min_key] = int(scene_min_len_value)

        scene_warmup_key = f"{session_prefix}::scene_warmup"
        scene_warmup_seed = int(
            st.session_state.get(scene_warmup_key, scene_warmup_default)
        )
        scene_warmup_value = st.number_input(
            "Warmup detections after cut",
            min_value=0,
            max_value=25,
            value=scene_warmup_seed,
            step=1,
            help="Force full detection on the first N frames after each cut",
        )
        st.session_state[scene_warmup_key] = int(scene_warmup_value)

        if detect_tracker_value == "bytetrack":
            st.markdown("#### Advanced tracking")
            track_high_session_key = f"{session_prefix}::track_high_thresh"
            track_high_seed = float(
                st.session_state.get(track_high_session_key, track_high_default)
            )
            track_high_value = st.slider(
                "track_high_thresh",
                min_value=0.30,
                max_value=0.95,
                value=float(track_high_seed),
                step=0.01,
                help="High-confidence gate for extending existing ByteTrack tracks.",
            )
            st.session_state[track_high_session_key] = float(track_high_value)
            track_new_session_key = f"{session_prefix}::new_track_thresh"
            track_new_seed = float(
                st.session_state.get(track_new_session_key, track_new_default)
            )
            track_new_value = st.slider(
                "new_track_thresh",
                min_value=0.30,
                max_value=0.95,
                value=float(track_new_seed),
                step=0.01,
                help="Minimum score required to spawn a new ByteTrack track.",
            )
            st.session_state[track_new_session_key] = float(track_new_value)
            st.caption(
                "Lower thresholds increase recall but may introduce more false tracks; higher thresholds are stricter."
            )

    detect_device_choice = st.selectbox(
        "Device",
        helpers.DEVICE_LABELS,
        index=helpers.device_label_index(detect_device_label_default),
        key="detect_device_choice",
    )
    detect_device_value = helpers.DEVICE_VALUE_MAP[detect_device_choice]
    detect_device_label = helpers.device_label_from_value(detect_device_value)

    if detect_tracker_value != "bytetrack":
        track_high_value = None
        track_new_value = None

    sampled_frames_est = _estimated_sampled_frames(video_meta, stride_value)
    if save_frames and sampled_frames_est:
        quality_factor = max(min(jpeg_quality / 85.0, 2.0), 0.5)
        est_frame_bytes = int(
            sampled_frames_est * FRAME_JPEG_SIZE_EST_BYTES * quality_factor
        )
        st.caption(
            f"Frames: ≈{helpers.human_size(est_frame_bytes)} for {sampled_frames_est:,} sampled frames (estimate)."
        )
    if save_crops:
        estimated_faces = helpers.coerce_int(detect_phase_status.get("detections"))
        if estimated_faces is None and sampled_frames_est:
            estimated_faces = int(sampled_frames_est * AVG_FACES_PER_FRAME)
        if estimated_faces:
            est_crop_bytes = int(estimated_faces * CROP_JPEG_SIZE_EST_BYTES)
            st.caption(
                f"Crops: ≈{helpers.human_size(est_crop_bytes)} for approximately {estimated_faces:,} faces."
            )

    job_payload = helpers.default_detect_track_payload(
        ep_id,
        stride=int(stride_value),
        det_thresh=float(det_thresh_value),
        device=detect_device_value,
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
    job_payload["detector"] = detect_detector_value
    job_payload["tracker"] = detect_tracker_value
    if (
        detect_tracker_value == "bytetrack"
        and track_high_value is not None
        and track_new_value is not None
    ):
        job_payload["track_high_thresh"] = float(track_high_value)
        job_payload["new_track_thresh"] = float(track_new_value)
    if fps_value > 0:
        job_payload["fps"] = fps_value
    mode_label = f"{detect_detector_label} + {detect_tracker_label}"

    def _process_detect_result(
        summary: Dict[str, Any] | None, error_message: str | None
    ) -> None:
        if error_message:
            if error_message == "mirror_failed":
                return
            if "RetinaFace weights missing or could not initialize" in error_message:
                st.error(error_message)
                st.caption("Run `python scripts/fetch_models.py` then retry.")
            else:
                st.error(error_message)
            return
        if not summary:
            return
        normalized = helpers.normalize_summary(ep_id, summary)
        detections = helpers.coerce_int(normalized.get("detections"))
        tracks = helpers.coerce_int(normalized.get("tracks"))
        frames_exported = helpers.coerce_int(normalized.get("frames_exported"))
        crops_exported = helpers.coerce_int(normalized.get("crops_exported"))
        detector_summary = normalized.get("detector")
        tracker_summary = normalized.get("tracker")
        track_ratio_value = helpers.coerce_float(
            normalized.get("track_to_detection_ratio") or normalized.get("track_ratio")
        )
        detector_is_scene = (
            isinstance(detector_summary, str)
            and detector_summary in helpers.SCENE_DETECTOR_LABEL_MAP
        )
        has_detections = detections is not None and detections > 0
        has_tracks = tracks is not None and tracks > 0
        issue_messages: list[str] = []
        if detector_is_scene:
            detector_label = helpers.SCENE_DETECTOR_LABEL_MAP.get(
                detector_summary, detector_summary
            )
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
                " ".join(issue_messages)
                + " Please rerun **Detect/Track Faces** to generate the manifests."
            )
            return
        if track_ratio_value is not None and track_ratio_value < 0.1:
            st.warning(
                "⚠️ Track-to-detection ratio is below 0.10. Consider lowering ByteTrack thresholds or inspecting the episode."
            )
        details_line = [
            (
                f"detections: {helpers.format_count(detections)}"
                if detections is not None
                else "detections: ?"
            ),
            (
                f"tracks: {helpers.format_count(tracks)}"
                if tracks is not None
                else "tracks: ?"
            ),
        ]
        if track_ratio_value is not None:
            details_line.append(f"tracks/detections: {track_ratio_value:.2f}")
        if frames_exported:
            details_line.append(
                f"frames exported: {helpers.format_count(frames_exported)}"
            )
        if crops_exported:
            details_line.append(
                f"crops exported: {helpers.format_count(crops_exported)}"
            )
        if detector_summary:
            details_line.append(
                f"detector: {helpers.detector_label_from_value(detector_summary)}"
            )
        if tracker_summary:
            details_line.append(
                f"tracker: {helpers.tracker_label_from_value(tracker_summary)}"
            )
        st.session_state["episode_detail_flash"] = (
            "Detect/track complete · " + " · ".join(details_line)
        )
        st.rerun()

    if autorun_detect:
        local_video_exists, summary, error_message = _launch_detect_job(
            local_video_exists,
            ep_id,
            details,
            job_payload,
            detect_device_value,
            detect_detector_value,
            detect_tracker_value,
            mode_label,
            detect_device_label,
        )
        _process_detect_result(summary, error_message)

    if not local_video_exists:
        st.info(
            "Local mirror missing; Detect/Track will mirror automatically before starting."
        )

    run_label = "Run detect/track"
    if st.button(run_label, use_container_width=True):
        local_video_exists, summary, error_message = _launch_detect_job(
            local_video_exists,
            ep_id,
            details,
            job_payload,
            detect_device_value,
            detect_detector_value,
            detect_tracker_value,
            mode_label,
            detect_device_label,
        )
        _process_detect_result(summary, error_message)

faces_ready = faces_ready_state
detector_manifest_value = helpers.tracks_detector_value(ep_id)
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
            st.caption(
                f"📊 Current pipeline: {helpers.detector_label_from_value(detector_name)} + "
                f"{helpers.tracker_label_from_value(tracker_name)}"
            )

    faces_device_choice = st.selectbox(
        "Device",
        helpers.DEVICE_LABELS,
        index=helpers.device_label_index(faces_device_label_default),
        key="faces_device_choice",
    )
    faces_device_value = helpers.DEVICE_VALUE_MAP[faces_device_choice]
    faces_save_frames = st.checkbox(
        "Save sampled frames (auto-enabled)",
        value=True,
        disabled=True,
        key="faces_save_frames_detail",
    )
    faces_save_crops = st.checkbox(
        "Save face crops to S3",
        value=bool(faces_save_crops_default),
        key="faces_save_crops_detail",
    )
    faces_thumb_size_default = int(faces_job_defaults.get("thumb_size") or 256)
    faces_jpeg_quality = st.number_input(
        "JPEG quality",
        min_value=50,
        max_value=100,
        value=int(faces_jpeg_quality_default),
        step=5,
        key="faces_jpeg_quality_detail",
    )
    faces_thumb_size = st.number_input(
        "Thumbnail size",
        min_value=64,
        max_value=512,
        value=faces_thumb_size_default,
        step=32,
        key="faces_thumb_size_detail",
    )
    if faces_device_value == "mps":
        st.caption(
            "ArcFace embeddings run on CPU when MPS is selected. Tracker/crop export still uses the requested device."
        )

    # Improved messaging for when Harvest Faces is disabled
    if not local_video_exists:
        st.info(
            "Local mirror missing; video will be mirrored from S3 automatically when Faces Harvest starts."
        )
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
        if detector_manifest_value is None:
            st.warning(
                "Unable to determine which detector produced the current tracks. Rerun Detect/Track Faces "
                "with RetinaFace + ByteTrack before harvesting."
            )
        else:
            st.warning(
                f"Current tracks were generated with unsupported detector "
                f"{helpers.detector_label_from_value(detector_manifest_value)}. Rerun Detect/Track Faces "
                "with RetinaFace + ByteTrack before harvesting."
            )
        if st.button(
            "Rerun Detect/Track (RetinaFace + ByteTrack)",
            key="faces_rerun_detect",
            use_container_width=True,
        ):
            st.session_state["episode_detail_detector_override"] = (
                helpers.DEFAULT_DETECTOR
            )
            st.session_state["episode_detail_tracker_override"] = (
                helpers.DEFAULT_TRACKER
            )
            st.session_state["episode_detail_device_override"] = helpers.DEFAULT_DEVICE
            st.session_state["episode_detail_detect_autorun_flag"] = True
            st.session_state["episode_detail_flash"] = (
                "Starting Detect/Track with RetinaFace + ByteTrack…"
            )
            st.rerun()

    faces_disabled = (not tracks_ready) or (not detector_face_only)
    if st.button(
        "Run Faces Harvest", use_container_width=True, disabled=faces_disabled
    ):
        can_run_faces = True
        if not local_video_exists:
            can_run_faces = _ensure_local_artifacts(ep_id, details)
            if can_run_faces:
                local_video_exists = True
        if can_run_faces:
            payload = {
                "ep_id": ep_id,
                "device": faces_device_value,
                "save_frames": bool(faces_save_frames),
                "save_crops": bool(faces_save_crops),
                "jpeg_quality": int(faces_jpeg_quality),
                "thumb_size": int(faces_thumb_size),
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
                details.append(f"thumb size: {int(faces_thumb_size)}px")
                flash_msg = "Faces harvest complete" + (
                    " · " + ", ".join(details) if details else ""
                )
                st.session_state["episode_detail_flash"] = flash_msg
                st.rerun()
with col_cluster:
    st.markdown("### Cluster Identities")
    st.caption(
        _format_phase_status("Cluster Identities", cluster_phase_status, "identities")
    )
    cluster_device_choice = st.selectbox(
        "Device",
        helpers.DEVICE_LABELS,
        index=helpers.device_label_index(cluster_device_label_default),
        key="cluster_device_choice",
    )
    cluster_device_value = helpers.DEVICE_VALUE_MAP[cluster_device_choice]
    cluster_thresh_value = st.slider(
        "Cluster similarity threshold",
        min_value=0.4,
        max_value=0.9,
        value=float(cluster_thresh_default),
        step=0.01,
        help="Higher thresholds require tighter ArcFace similarity between faces to form a cluster.",
    )
    min_cluster_size_value = st.number_input(
        "Minimum tracks per identity",
        min_value=1,
        max_value=50,
        value=int(min_cluster_size_default),
        step=1,
        help="Clusters smaller than this are discarded as noise.",
    )
    if not local_video_exists:
        st.info(
            "Local mirror missing; artifacts will be mirrored automatically when clustering starts."
        )
    elif not faces_ready:
        st.caption("Run faces harvest first.")
    elif (faces_count_value or 0) == 0:
        st.info(
            "Faces harvest completed with 0 faces → clustering will immediately finish with 0 identities."
        )
    elif not detector_face_only:
        st.warning(
            "Current tracks were generated with a legacy detector. Rerun detect/track first."
        )
    cluster_disabled = (not faces_ready) or (not detector_face_only)
    if st.button("Run Cluster", use_container_width=True, disabled=cluster_disabled):
        can_run_cluster = True
        if not local_video_exists:
            can_run_cluster = _ensure_local_artifacts(ep_id, details)
            if can_run_cluster:
                local_video_exists = True
        if can_run_cluster:
            payload = {
                "ep_id": ep_id,
                "device": cluster_device_value,
                "cluster_thresh": float(cluster_thresh_value),
                "min_cluster_size": int(min_cluster_size_value),
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
                flash_msg = (
                    f"Clustered (thresh {cluster_thresh_value:.2f}, min {int(min_cluster_size_value)})"
                    + (" · " + ", ".join(details) if details else "")
                )
                st.session_state["episode_detail_flash"] = flash_msg
                st.rerun()
with col_screen:
    st.markdown("### Screentime")
    screentime_disabled = False
    if not local_video_exists:
        st.info(
            "Local mirror missing; video will be mirrored automatically when screentime starts."
        )
    if st.button(
        "Compute screentime", use_container_width=True, disabled=screentime_disabled
    ):
        can_run_screen = True
        if not local_video_exists:
            can_run_screen = _ensure_local_artifacts(ep_id, details)
            if can_run_screen:
                local_video_exists = True
        if can_run_screen:
            with st.spinner("Starting screentime analysis…"):
                try:
                    resp = helpers.api_post(
                        "/jobs/screen_time/analyze", {"ep_id": ep_id}
                    )
                except requests.RequestException as exc:
                    st.error(
                        helpers.describe_error(
                            f"{cfg['api_base']}/jobs/screen_time/analyze", exc
                        )
                    )
                else:
                    job_id = resp.get("job_id")
                    if job_id:
                        st.session_state[SCREENTIME_JOB_KEY] = job_id
                    st.success("Screen time job queued.")
                    st.rerun()

    screentime_job_id = st.session_state.get(SCREENTIME_JOB_KEY)
    if screentime_job_id:
        try:
            job_progress_resp = helpers.api_get(f"/jobs/{screentime_job_id}/progress")
        except requests.RequestException as exc:
            st.warning(
                helpers.describe_error(
                    f"{cfg['api_base']}/jobs/{screentime_job_id}/progress", exc
                )
            )
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
                st.rerun()
            elif job_state == "succeeded":
                st.success("Screentime analysis complete.")
                st.caption(
                    f"JSON → {helpers.link_local(helpers.DATA_ROOT / 'analytics' / ep_id / 'screentime.json')}"
                )
                st.caption(
                    f"CSV → {helpers.link_local(helpers.DATA_ROOT / 'analytics' / ep_id / 'screentime.csv')}"
                )
                if st.button(
                    "Dismiss screentime status", key="dismiss_screentime_job_success"
                ):
                    st.session_state.pop(SCREENTIME_JOB_KEY, None)
                    st.rerun()
            elif job_state == "failed":
                st.error(
                    f"Screentime job failed: {job_progress_resp.get('error') or 'unknown error'}"
                )
                if st.button(
                    "Dismiss screentime status", key="dismiss_screentime_job_failed"
                ):
                    st.session_state.pop(SCREENTIME_JOB_KEY, None)
                    st.rerun()
            else:
                st.info(f"Screentime job status: {job_state or 'unknown'}")
                if st.button(
                    "Dismiss screentime status", key="dismiss_screentime_job_other"
                ):
                    st.session_state.pop(SCREENTIME_JOB_KEY, None)
                    st.rerun()

st.subheader("Artifacts")


def _render_artifact_entry(
    label: str, local_path: Path, key_suffix: str, s3_key: str | None = None
) -> None:
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
_render_artifact_entry(
    "Detections", get_path(ep_id, "detections"), "detections", detections_key
)
_render_artifact_entry("Tracks", get_path(ep_id, "tracks"), "tracks", tracks_key)
_render_artifact_entry("Faces", faces_path, "faces", faces_key)
_render_artifact_entry("Identities", identities_path, "identities", identities_key)
analytics_dir = helpers.DATA_ROOT / "analytics" / ep_id
_render_artifact_entry(
    "Screentime (json)", analytics_dir / "screentime.json", "screentime_json"
)
_render_artifact_entry(
    "Screentime (csv)", analytics_dir / "screentime.csv", "screentime_csv"
)

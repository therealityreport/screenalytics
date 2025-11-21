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


def _load_job_defaults(ep_id: str, job_type: str) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
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


def _format_runtime(runtime_sec: Any) -> str | None:
    try:
        total = float(runtime_sec)
    except (TypeError, ValueError):
        return None
    if total < 0:
        return None
    seconds = int(round(total))
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 3600:
        minutes, rem = divmod(seconds, 60)
        return f"{minutes}m {rem:02d}s"
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    return f"{hours}h {minutes:02d}m"


def _choose_value(*candidates: Any, fallback: str) -> str:
    for candidate in candidates:
        if isinstance(candidate, str):
            cleaned = candidate.strip()
            if cleaned:
                return cleaned.lower()
    return fallback


def _resolved_device_label(label: str | None) -> str:
    normalized = label if label in helpers.DEVICE_LABELS else None
    if not normalized or normalized == "Auto":
        return helpers._guess_device_label()
    return normalized


def _detect_setting_key(ep_id: str, field: str) -> str:
    return f"episode_detail_detect::{ep_id}::{field}"


def _job_activity_key(ep_id: str) -> str:
    return f"{ep_id}::job_active"


def _detect_job_state_key(ep_id: str) -> str:
    return f"{ep_id}::detect_job_running"


def _set_job_active(ep_id: str, active: bool) -> None:
    st.session_state[_job_activity_key(ep_id)] = bool(active)


def _job_active(ep_id: str) -> bool:
    return bool(st.session_state.get(_job_activity_key(ep_id), False))


def _status_cache_key(ep_id: str) -> str:
    return f"{ep_id}::status_payload"


def _status_timestamp_key(ep_id: str) -> str:
    return f"{ep_id}::status_fetched_at"


def _status_fetch_token_key(ep_id: str) -> str:
    return f"{ep_id}::status_fetch_token"


def _status_force_refresh_key(ep_id: str) -> str:
    return f"{ep_id}::status_force_refresh"


def _refresh_click_key(ep_id: str) -> str:
    return f"{ep_id}::status_refresh_clicked_at"


def _trigger_safe_detect_rerun(ep_id: str, message: str) -> None:
    st.session_state["episode_detail_detector_override"] = helpers.DEFAULT_DETECTOR
    st.session_state["episode_detail_tracker_override"] = helpers.DEFAULT_TRACKER
    st.session_state["episode_detail_device_override"] = helpers.DEFAULT_DEVICE
    st.session_state["episode_detail_detect_autorun_flag"] = True
    st.session_state["episode_detail_flash"] = message
    st.rerun()


def _render_device_summary(requested: str | None, resolved: str | None) -> None:
    req_label = helpers.device_label_from_value(requested) if requested else None
    resolved_label = helpers.device_label_from_value(resolved or requested)
    if not (req_label or resolved_label):
        return
    if req_label and resolved_label and req_label != resolved_label:
        caption = f"Device: requested {req_label} → resolved {resolved_label}"
        if req_label in {"CUDA", "CoreML", "MPS"} and resolved_label == "CPU":
            st.caption(f"⚠️ {caption}")
        else:
            st.caption(caption)
    else:
        st.caption(f"Device: {resolved_label or req_label}")


def _estimate_runtime_seconds(frames: int, device_value: str) -> float:
    per_device = {
        "cpu": 45.0,
        "cuda": 110.0,
        "coreml": 90.0,
        "mps": 70.0,
    }
    rate = per_device.get((device_value or "cpu").lower(), 40.0)
    if frames <= 0 or rate <= 0:
        return 0.0
    return frames / rate



@st.cache_data(ttl=30, show_spinner=False)
def _cached_count_manifest_rows(path_str: str, mtime: float) -> int | None:
    """Cache manifest row counts using path+mtime as cache key."""
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def _cached_manifest_has_rows(path_str: str, mtime: float) -> bool:
    """Cache manifest existence check using path+mtime as cache key."""
    path = Path(path_str)
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


def _count_manifest_rows(path: Path) -> int | None:
    """Count rows in manifest (cached by path+mtime)."""
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
        return _cached_count_manifest_rows(str(path), mtime)
    except OSError:
        return None


def _manifest_has_rows(path: Path) -> bool:
    """Check if manifest has rows (cached by path+mtime)."""
    if not path.exists() or not path.is_file():
        return False
    try:
        mtime = path.stat().st_mtime
        return _cached_manifest_has_rows(str(path), mtime)
    except OSError:
        return False


@st.cache_data(ttl=10, show_spinner=False)
def _cached_episode_details(ep_id: str, cache_key: float) -> Dict[str, Any]:
    """Cache episode details API response with 10s TTL."""
    return helpers.api_get(f"/episodes/{ep_id}")


@st.cache_data(ttl=10, show_spinner=False)
def _cached_episode_status(ep_id: str, cache_key: float) -> Dict[str, Any] | None:
    """Cache episode status API response with 10s TTL."""
    return helpers.get_episode_status(ep_id)


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
    if tracks_ready_flag:
        return "success", True, False
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
    filtered_episodes = [ep for ep in episodes if show_orphans or ep.get("exists_in_store")]
    if not filtered_episodes:
        st.warning("No tracked episodes available. Upload a video or enable orphan view above.")
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
        st.warning("⚠ Not tracked in episode store yet. Click 'Load Episode' to create it.")

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


def _ensure_local_artifacts(ep_id: str, details: Dict[str, Any]) -> bool:
    local_block = details.setdefault("local", {})
    video_path = get_path(ep_id, "video")
    if video_path.exists():
        local_block["path"] = str(video_path)
        local_block["exists"] = True
        return True
    s3_meta = details.get("s3") or {}
    if not (s3_meta.get("v2_exists") or s3_meta.get("v1_exists")):
        st.error("Episode is not mirrored in S3; mirror/upload the video before running this job.")
        return False
    mirror_path = f"/episodes/{ep_id}/mirror"
    with st.spinner("Mirroring artifacts from S3…"):
        try:
            resp = helpers.api_post(mirror_path)
        except requests.RequestException as exc:
            st.error(helpers.describe_error(f"{cfg['api_base']}{mirror_path}", exc))
            return False
        st.success(
            f"Mirrored to {helpers.link_local(resp['local_video_path'])} " f"({helpers.human_size(resp.get('bytes'))})"
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
    running_state_key: str | None = None,
    *,
    active_job_key: str | None = None,
    detect_flag_key: str | None = None,
):
    current_local = local_exists
    if not current_local:
        if not _ensure_local_artifacts(ep_id, details):
            return current_local, None, "mirror_failed"
        current_local = True
    if running_state_key:
        st.session_state[running_state_key] = True
    if active_job_key:
        st.session_state[active_job_key] = True
    if detect_flag_key:
        st.session_state[detect_flag_key] = True
    try:
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
    finally:
        if running_state_key:
            st.session_state[running_state_key] = False
        if active_job_key:
            st.session_state[active_job_key] = False
        if detect_flag_key:
            st.session_state[detect_flag_key] = False
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
running_job_key = f"{ep_id}::pipeline_job_running"
if running_job_key not in st.session_state:
    st.session_state[running_job_key] = False
detect_running_key = _detect_job_state_key(ep_id)
if detect_running_key not in st.session_state:
    st.session_state[detect_running_key] = False
if _job_activity_key(ep_id) not in st.session_state:
    st.session_state[_job_activity_key(ep_id)] = False
job_running = bool(st.session_state.get(running_job_key))

# Cache API responses with 10s TTL to reduce repeated requests
cache_key = time.time() // 10

try:
    details = _cached_episode_details(ep_id, cache_key)
except requests.HTTPError as exc:
    if exc.response is not None and exc.response.status_code == 404:
        _handle_missing_episode(ep_id)
    st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/{ep_id}", exc))
    st.stop()
except requests.RequestException as exc:
    st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/{ep_id}", exc))
    st.stop()

status_cache_key = _status_cache_key(ep_id)
status_ts_key = _status_timestamp_key(ep_id)
fetch_token_key = _status_fetch_token_key(ep_id)
force_refresh_key = _status_force_refresh_key(ep_id)
force_refresh = bool(st.session_state.pop(force_refresh_key, False))
fetch_token = st.session_state.get(fetch_token_key, 0)
status_payload = st.session_state.get(status_cache_key)
should_refresh_status = force_refresh or _job_active(ep_id) or status_payload is None
if should_refresh_status:
    fetch_token += 1
    st.session_state[fetch_token_key] = fetch_token
    status_payload = _cached_episode_status(ep_id, fetch_token)
    st.session_state[status_cache_key] = status_payload
    st.session_state[status_ts_key] = time.time()
status_refreshed_at = st.session_state.get(status_ts_key)

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
st.write(f"Show `{details['show_slug']}` · Season {details['season_number']} Episode {details['episode_number']}")
st.write(f"S3 v2 → `{details['s3']['v2_key']}` (exists={details['s3']['v2_exists']})")
st.write(f"S3 v1 → `{details['s3']['v1_key']}` (exists={details['s3']['v1_exists']})")
if not details["s3"]["v2_exists"] and details["s3"]["v1_exists"]:
    st.warning("Legacy v1 object detected; mirroring will use it until the v2 path is populated.")
st.write(f"Local → {helpers.link_local(details['local']['path'])} (exists={details['local']['exists']})")
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
detect_status_value, tracks_ready, using_manifest_fallback = _compute_detect_track_effective_status(
    detect_phase_status,
    manifest_ready=manifest_state["manifest_ready"],
    tracks_ready_flag=tracks_ready_flag,
    job_state=detect_job_state,
)
status_running = (
    detect_status_value == "running"
    or faces_status_value == "running"
    or cluster_status_value == "running"
    or str(detect_job_state or "").lower() == "running"
)
if status_running:
    _set_job_active(ep_id, True)
elif not job_running:
    _set_job_active(ep_id, False)

# Other status values
faces_count_value = helpers.coerce_int(faces_phase_status.get("faces"))
identities_count_value = helpers.coerce_int(cluster_phase_status.get("identities"))
faces_manifest_count = _count_manifest_rows(faces_path)
faces_ready_state = False
if faces_status_value == "success":
    if faces_count_value is None and faces_manifest_count is not None:
        faces_count_value = faces_manifest_count
    # Accept success status even with zero faces (successful run with no detections)
    faces_ready_state = True
elif faces_status_value in {"missing", "unknown"}:
    if faces_manifest_count and faces_manifest_count > 0:
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
        if st.button("Refresh status", key="episode_status_refresh", use_container_width=True):
            now = time.time()
            last_click = float(st.session_state.get(_refresh_click_key(ep_id), 0.0))
            if now - last_click < 1.0:
                st.caption("Please wait ≥1s between refreshes.")
            else:
                st.session_state[_refresh_click_key(ep_id)] = now
                st.session_state[_status_force_refresh_key(ep_id)] = True
                st.rerun()
    if status_refreshed_at:
        refreshed_label = datetime.utcfromtimestamp(status_refreshed_at).strftime("%Y-%m-%d %H:%M:%S UTC")
        st.caption(f"Status refreshed at {refreshed_label}")
    else:
        st.caption("Status will refresh when a job starts or you press refresh.")
    coreml_available = status_payload.get("coreml_available")
    if coreml_available is False and helpers.is_apple_silicon():
        st.warning(
            "⚠️ CoreML acceleration isn't available on this host. Install `onnxruntime-coreml` to avoid CPU-only runs."
        )
    col1, col2, col3 = st.columns(3)

    with col1:
        detect_params: list[str] = []
        stride_state = helpers.coerce_int(detect_phase_status.get("stride"))
        if stride_state:
            detect_params.append(f"stride={stride_state}")
        det_thresh_state = helpers.coerce_float(detect_phase_status.get("det_thresh"))
        if det_thresh_state is not None:
            detect_params.append(f"det_thresh={det_thresh_state:.2f}")
        max_gap_state = helpers.coerce_int(detect_phase_status.get("max_gap"))
        if max_gap_state is not None:
            detect_params.append(f"max_gap={max_gap_state}")
        scene_thresh_state = helpers.coerce_float(detect_phase_status.get("scene_threshold"))
        if scene_thresh_state is not None:
            detect_params.append(f"scene={scene_thresh_state:.2f}")
        track_high_state = helpers.coerce_float(detect_phase_status.get("track_high_thresh"))
        if track_high_state is not None:
            detect_params.append(f"track_high={track_high_state:.2f}")
        new_track_state = helpers.coerce_float(detect_phase_status.get("new_track_thresh"))
        if new_track_state is not None:
            detect_params.append(f"new_track={new_track_state:.2f}")
        save_frames_state = detect_phase_status.get("save_frames")
        if save_frames_state is not None:
            detect_params.append(f"save_frames={'on' if save_frames_state else 'off'}")
        save_crops_state = detect_phase_status.get("save_crops")
        if save_crops_state is not None:
            detect_params.append(f"save_crops={'on' if save_crops_state else 'off'}")
        jpeg_state = helpers.coerce_int(detect_phase_status.get("jpeg_quality"))
        if jpeg_state:
            detect_params.append(f"jpeg={jpeg_state}")
        device_state = detect_phase_status.get("device")
        requested_device_state = detect_phase_status.get("requested_device")
        resolved_device_state = detect_phase_status.get("resolved_device")
        detect_runtime = _format_runtime(detect_phase_status.get("runtime_sec"))
        if requested_device_state and requested_device_state != device_state:
            detect_params.append(f"requested={helpers.device_label_from_value(requested_device_state)}")
        if device_state:
            detect_params.append(f"device={helpers.device_label_from_value(device_state)}")
        if detect_status_value == "success":
            runtime_label = detect_runtime or "n/a"
            st.success(f"✅ **Detect/Track**: Complete (Runtime: {runtime_label})")
            det = detect_phase_status.get("detector") or "--"
            trk = detect_phase_status.get("tracker") or "--"
            st.caption(f"{det} + {trk}")
            detections = detect_phase_status.get("detections")
            tracks = detect_phase_status.get("tracks")
            st.caption(f"{(detections or 0):,} detections, {(tracks or 0):,} tracks")
            ratio_value = helpers.coerce_float(
                detect_phase_status.get("track_to_detection_ratio") or detect_phase_status.get("track_ratio")
            )
            if ratio_value is not None:
                st.caption(f"Tracks / detections: {ratio_value:.2f}")
                if ratio_value < 0.1:
                    st.caption(
                        "⚠️ Track-to-detection ratio < 0.10. Consider lowering ByteTrack thresholds or rerunning detect/track."
                    )
            # Show manifest-fallback caption when status was inferred from manifests
            if using_manifest_fallback:
                st.caption("ℹ️ _Detect/Track completion inferred from manifests (status API missing/stale)._")
        elif detect_status_value == "running":
            st.info("⏳ **Detect/Track**: Running")
            if detect_job_record and detect_job_record.get("started_at"):
                st.caption(f"Started at {detect_job_record['started_at']}")
            st.caption("Live progress appears in the log panel below.")
        elif detect_status_value == "stale":
            st.warning("⚠️ **Detect/Track**: Status stale (manifests missing)")
            st.caption("Rerun Detect/Track Faces to rebuild detections/tracks for this episode.")
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
        if detect_params:
            st.caption("Params: " + ", ".join(detect_params))
        if jpeg_state:
            st.caption(f"JPEG quality: {jpeg_state}")
        _render_device_summary(requested_device_state, resolved_device_state or device_state)
        finished = _format_timestamp(detect_phase_status.get("finished_at"))
        if finished:
            st.caption(f"Last run: {finished}")
        if detect_status_value != "success" and detect_runtime:
            st.caption(f"Runtime: {detect_runtime}")
        elif detect_status_value == "success" and detect_runtime is None:
            st.caption("Runtime: n/a")

    with col2:
        faces_params: list[str] = []
        faces_device_state = faces_phase_status.get("device")
        faces_device_request = faces_phase_status.get("requested_device")
        faces_resolved_state = faces_phase_status.get("resolved_device")
        faces_runtime = _format_runtime(faces_phase_status.get("runtime_sec"))
        if faces_device_request and faces_device_request != faces_device_state:
            faces_params.append(f"requested={helpers.device_label_from_value(faces_device_request)}")
        if faces_device_state:
            faces_params.append(f"device={helpers.device_label_from_value(faces_device_state)}")
        save_frames_state = faces_phase_status.get("save_frames")
        if save_frames_state is not None:
            faces_params.append(f"save_frames={'on' if save_frames_state else 'off'}")
        save_crops_state = faces_phase_status.get("save_crops")
        if save_crops_state is not None:
            faces_params.append(f"save_crops={'on' if save_crops_state else 'off'}")
        thumb_size_state = helpers.coerce_int(faces_phase_status.get("thumb_size"))
        if thumb_size_state:
            faces_params.append(f"thumb={thumb_size_state}px")
        jpeg_state = helpers.coerce_int(faces_phase_status.get("jpeg_quality"))
        if jpeg_state:
            faces_params.append(f"jpeg={jpeg_state}")
        if faces_ready_state:
            runtime_label = faces_runtime or "n/a"
            st.success(f"✅ **Faces Harvest**: Complete (Runtime: {runtime_label})")
            face_count_label = helpers.format_count(faces_count_value) or "0"
            st.caption(f"Faces: {face_count_label} (harvest completed)")
        elif faces_status_value == "success":
            st.warning("⚠️ **Faces Harvest**: Manifest unavailable locally")
            st.caption("Faces completed on the backend, but faces.jsonl has not been mirrored locally yet.")
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
        if faces_params:
            st.caption("Params: " + ", ".join(faces_params))
        _render_device_summary(faces_device_request, faces_resolved_state or faces_device_state)
        finished = _format_timestamp(faces_phase_status.get("finished_at"))
        if finished:
            st.caption(f"Last run: {finished}")
        if not faces_ready_state:
            if faces_runtime:
                st.caption(f"Runtime: {faces_runtime}")
            elif faces_status_value == "success":
                st.caption("Runtime: n/a")

    with col3:
        cluster_params: list[str] = []
        cluster_device_state = cluster_phase_status.get("device")
        cluster_device_request = cluster_phase_status.get("requested_device")
        cluster_resolved_state = cluster_phase_status.get("resolved_device")
        cluster_runtime = _format_runtime(cluster_phase_status.get("runtime_sec"))
        if cluster_device_request and cluster_device_request != cluster_device_state:
            cluster_params.append(f"requested={helpers.device_label_from_value(cluster_device_request)}")
        if cluster_device_state:
            cluster_params.append(f"device={helpers.device_label_from_value(cluster_device_state)}")
        cluster_thresh_state = helpers.coerce_float(cluster_phase_status.get("cluster_thresh"))
        if cluster_thresh_state is not None:
            cluster_params.append(f"thresh={cluster_thresh_state:.2f}")
        min_cluster_state = helpers.coerce_int(cluster_phase_status.get("min_cluster_size"))
        if min_cluster_state is not None:
            cluster_params.append(f"min_cluster={min_cluster_state}")
        identities_label = helpers.format_count(identities_count_value) or "0"
        if cluster_status_value == "success":
            runtime_label = cluster_runtime or "n/a"
            st.success(f"✅ **Cluster**: Complete (Runtime: {runtime_label})")
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
        if cluster_params:
            st.caption("Params: " + ", ".join(cluster_params))
        _render_device_summary(cluster_device_request, cluster_resolved_state or cluster_device_state)
        finished = _format_timestamp(cluster_phase_status.get("finished_at"))
        if finished:
            st.caption(f"Last run: {finished}")
        if cluster_status_value != "success" and cluster_runtime:
            st.caption(f"Runtime: {cluster_runtime}")
        elif cluster_status_value == "success" and cluster_runtime is None:
            st.caption("Runtime: n/a")

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
    detect_job_defaults.get("requested_device"),
    detect_phase_status.get("requested_device"),
    detect_job_defaults.get("device"),
    detect_phase_status.get("device"),
    fallback=helpers.DEFAULT_DEVICE,
)
detect_device_label_default = helpers.device_label_from_value(detect_device_default_value)
detect_device_label_default = _resolved_device_label(detect_device_label_default)
detect_detector_label = helpers.detector_label_from_value(detect_detector_value)
detect_tracker_label = helpers.tracker_label_from_value(detect_tracker_value)

faces_device_default_value = _choose_value(
    faces_job_defaults.get("requested_device"),
    faces_job_defaults.get("device"),
    faces_job_defaults.get("embed_device"),
    faces_phase_status.get("requested_device"),
    faces_phase_status.get("device"),
    faces_phase_status.get("embed_device"),
    fallback=detect_device_default_value,
)
faces_device_label_default = helpers.device_label_from_value(faces_device_default_value)
faces_device_label_default = _resolved_device_label(faces_device_label_default)
faces_save_frames_default = faces_job_defaults.get("save_frames")
if faces_save_frames_default is None:
    faces_save_frames_default = True
faces_save_crops_default = faces_job_defaults.get("save_crops")
if faces_save_crops_default is None:
    faces_save_crops_default = True
faces_jpeg_quality_default = helpers.coerce_int(faces_job_defaults.get("jpeg_quality")) or 85

cluster_device_default_value = _choose_value(
    cluster_phase_status.get("requested_device"),
    cluster_job_defaults.get("device"),
    cluster_phase_status.get("device"),
    fallback=faces_device_default_value,
)
cluster_device_label_default = helpers.device_label_from_value(cluster_device_default_value)
cluster_device_label_default = _resolved_device_label(cluster_device_label_default)
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
min_cluster_size_default = helpers.coerce_int(cluster_job_defaults.get("min_cluster_size"))
if min_cluster_size_default is None:
    min_cluster_size_default = helpers.coerce_int(cluster_phase_status.get("min_cluster_size"))
if min_cluster_size_default is None:
    min_cluster_size_default = 2

col_hydrate, col_detect = st.columns(2)
with col_hydrate:
    detect_inflight = bool(st.session_state.get(detect_running_key))
    mirror_disabled = detect_inflight or job_running
    mirror_help = "Detect/Track automatically mirrors before running." if detect_inflight else None
    if st.button("Mirror from S3", use_container_width=True, disabled=mirror_disabled, help=mirror_help):
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
    if detect_inflight:
        st.caption("Detect/Track is mirroring required artifacts automatically…")
with col_detect:
    st.markdown("### Detect/Track Faces")
    session_prefix = f"episode_detail_detect::{ep_id}"

    stride_default = int(detect_job_defaults.get("stride") or helpers.DEFAULT_STRIDE)
    # Prefill FPS from video metadata if available
    fps_default = float(detect_job_defaults.get("fps") or 0.0)
    if fps_default == 0.0 and video_meta and video_meta.get("fps_detected"):
        fps_default = float(video_meta["fps_detected"])
    det_thresh_default = float(detect_job_defaults.get("det_thresh") or helpers.DEFAULT_DET_THRESH)
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

    stride_field = _detect_setting_key(ep_id, "stride")
    if stride_field not in st.session_state:
        st.session_state[stride_field] = int(stride_default)
    stride_value = st.number_input(
        "Stride",
        min_value=1,
        max_value=50,
        step=1,
        key=stride_field,
    )
    st.caption(
        "Stride 4 (sampling every fourth frame) is the standard baseline for 42-minute episodes; "
        "lower values tighten QA, higher values accelerate longer cuts."
    )
    fps_field = _detect_setting_key(ep_id, "fps")
    if fps_field not in st.session_state:
        st.session_state[fps_field] = float(fps_default)
    fps_value = st.number_input(
        "FPS",
        min_value=0.0,
        max_value=120.0,
        step=1.0,
        key=fps_field,
    )
    st.caption("Frames per second extracted from source video. Lower FPS reduces processing time and storage.")
    # Automatically save to S3
    save_frames_key = _detect_setting_key(ep_id, "save_frames")
    if save_frames_key not in st.session_state:
        st.session_state[save_frames_key] = bool(save_frames_default)
    save_frames = st.checkbox(
        "Save sampled frames",
        value=bool(st.session_state[save_frames_key]),
        help="Stores sampled RGB frames alongside detections for QA and future crops.",
        key=save_frames_key,
    )
    save_crops_key = _detect_setting_key(ep_id, "save_crops")
    if save_crops_key not in st.session_state:
        st.session_state[save_crops_key] = bool(save_crops_default)
    save_crops = st.checkbox(
        "Save crops",
        value=bool(st.session_state[save_crops_key]),
        help="Exports aligned face crops during detect/track. Disable when reusing previous crops.",
        key=save_crops_key,
    )
    jpeg_key = _detect_setting_key(ep_id, "jpeg_quality")
    if jpeg_key not in st.session_state:
        st.session_state[jpeg_key] = int(jpeg_quality_default)
    jpeg_quality = st.number_input(
        "JPEG quality",
        min_value=50,
        max_value=100,
        value=int(st.session_state[jpeg_key]),
        step=5,
        key=jpeg_key,
    )
    st.caption("Compression quality for saved face thumbnails and frame images. Higher = better quality, larger files.")

    max_gap_key = f"{session_prefix}::max_gap"
    max_gap_seed = int(st.session_state.get(max_gap_key, max_gap_default))
    max_gap_value = st.number_input("Max gap (frames)", min_value=1, max_value=240, value=max_gap_seed, step=1)
    st.caption("Maximum frames a face can be missing before track terminates. Higher values connect tracks across longer occlusions.")
    st.session_state[max_gap_key] = int(max_gap_value)

    det_thresh_key = f"{session_prefix}::det_thresh"
    det_thresh_seed = float(st.session_state.get(det_thresh_key, det_thresh_default))
    det_thresh_value = st.slider(
        "Detection threshold",
        min_value=0.1,
        max_value=0.9,
        value=float(det_thresh_seed),
        step=0.01,
    )
    st.caption("Confidence range for valid face detections. Lower values increase recall but may add false positives.")
    st.session_state[det_thresh_key] = float(det_thresh_value)

    track_high_default = helpers.coerce_float(detect_job_defaults.get("track_high_thresh"))
    if track_high_default is None:
        track_high_default = helpers.coerce_float(detect_phase_status.get("track_high_thresh"))
    if track_high_default is None:
        track_high_default = helpers.TRACK_HIGH_THRESH_DEFAULT
    track_new_default = helpers.coerce_float(detect_job_defaults.get("new_track_thresh"))
    if track_new_default is None:
        track_new_default = helpers.coerce_float(detect_phase_status.get("new_track_thresh"))
    if track_new_default is None:
        track_new_default = helpers.TRACK_NEW_THRESH_DEFAULT
    track_high_value: float | None = track_high_default
    track_new_value: float | None = track_new_default

    with st.expander("Advanced detect/track", expanded=False):
        scene_detector_session_key = f"{session_prefix}::scene_detector"
        scene_detector_seed = st.session_state.get(
            scene_detector_session_key, detect_job_defaults.get("scene_detector")
        )
        scene_detector_label = st.selectbox(
            "Scene detector",
            helpers.SCENE_DETECTOR_LABELS,
            index=helpers.scene_detector_label_index(scene_detector_seed),
            key=f"{scene_detector_session_key}::select",
        )
        st.caption("Automatically detects scene changes/cuts. PySceneDetect uses content detection; HSV is a fallback.")
        scene_detector_value = helpers.scene_detector_value_from_label(scene_detector_label)
        st.session_state[scene_detector_session_key] = scene_detector_value

        scene_thresh_key = f"{session_prefix}::scene_threshold"
        scene_thresh_seed = float(st.session_state.get(scene_thresh_key, scene_threshold_default))
        scene_threshold_value = st.number_input(
            "Scene cut threshold",
            min_value=0.0,
            value=scene_thresh_seed,
            step=0.05,
        )
        st.caption("Sensitivity for detecting scene changes. Lower = more sensitive (detects subtle changes), higher = only hard cuts.")
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
        st.caption("Prevents rapid consecutive scene cut detections. Enforces minimum gap between detected cuts.")
        st.session_state[scene_min_key] = int(scene_min_len_value)

        scene_warmup_key = f"{session_prefix}::scene_warmup"
        scene_warmup_seed = int(st.session_state.get(scene_warmup_key, scene_warmup_default))
        scene_warmup_value = st.number_input(
            "Warmup detections after cut",
            min_value=0,
            max_value=25,
            value=scene_warmup_seed,
            step=1,
        )
        st.caption("Number of 'fresh' detection passes after scene cut. Helps re-establish tracking after scene changes.")
        st.session_state[scene_warmup_key] = int(scene_warmup_value)

        if detect_tracker_value == "bytetrack":
            st.markdown("#### Advanced tracking")
            track_high_session_key = f"{session_prefix}::track_high_thresh"
            track_high_seed = float(st.session_state.get(track_high_session_key, track_high_default))
            track_high_value = st.slider(
                "track_high_thresh",
                min_value=0.30,
                max_value=0.95,
                value=float(track_high_seed),
                step=0.01,
            )
            st.caption("Confidence threshold for continuing existing tracks. Match must score within this range to extend a track.")
            st.session_state[track_high_session_key] = float(track_high_value)
            track_new_session_key = f"{session_prefix}::new_track_thresh"
            track_new_seed = float(st.session_state.get(track_new_session_key, track_new_default))
            track_new_value = st.slider(
                "new_track_thresh",
                min_value=0.30,
                max_value=0.95,
                value=float(track_new_seed),
                step=0.01,
            )
            st.caption("Confidence threshold for creating new tracks. Detection must score within this range to start a new track.")
            st.session_state[track_new_session_key] = float(track_new_value)
            st.caption(
                "Lower thresholds increase recall but may introduce more false tracks; higher thresholds are stricter."
            )

    detect_device_choice = st.selectbox(
        "Device (for face detection/tracking)",
        helpers.DEVICE_LABELS,
        index=helpers.device_label_index(detect_device_label_default),
        key="detect_device_choice",
    )
    st.caption("CPU recommended for detection; GPU/CoreML may bottleneck on M-series chips for YOLOv8.")
    detect_device_value = helpers.DEVICE_VALUE_MAP[detect_device_choice]
    detect_device_label = helpers.device_label_from_value(detect_device_value)

    if detect_tracker_value != "bytetrack":
        track_high_value = None
        track_new_value = None

    sampled_frames_est = _estimated_sampled_frames(video_meta, int(stride_value))
    if sampled_frames_est:
        est_seconds = _estimate_runtime_seconds(sampled_frames_est, detect_device_value)
        if est_seconds > 0:
            runtime_minutes = est_seconds / 60.0
            st.caption(
                f"≈{sampled_frames_est:,} frames scheduled; rough runtime ~{runtime_minutes:.1f} min on {detect_device_label}."
            )
            if sampled_frames_est > 200_000 and detect_device_value == "cpu":
                st.warning(
                    "High load: consider increasing stride or lowering FPS when running on CPU to avoid stalls."
                )
    if save_frames and sampled_frames_est:
        quality_factor = max(min(jpeg_quality / 85.0, 2.0), 0.5)
        est_frame_bytes = int(sampled_frames_est * FRAME_JPEG_SIZE_EST_BYTES * quality_factor)
        st.caption(
            f"Frames: ≈{helpers.human_size(est_frame_bytes)} for {sampled_frames_est:,} sampled frames (estimate)."
        )
    if save_crops:
        # Derive face count from detections manifest for real-time feedback
        estimated_faces = _count_manifest_rows(detections_path)
        if estimated_faces is None:
            estimated_faces = helpers.coerce_int(detect_phase_status.get("detections"))
        if estimated_faces is None and sampled_frames_est:
            estimated_faces = int(sampled_frames_est * AVG_FACES_PER_FRAME)
        if estimated_faces:
            est_crop_bytes = int(estimated_faces * CROP_JPEG_SIZE_EST_BYTES)
            st.caption(f"Crops: ≈{helpers.human_size(est_crop_bytes)} for approximately {estimated_faces:,} faces.")
    stride_hint = "every frame" if stride_value == 1 else f"every {stride_value}th frame"
    export_bits: list[str] = []
    if save_frames:
        export_bits.append("frames")
    if save_crops:
        export_bits.append("crops")
    export_text = "saving " + " & ".join(export_bits) if export_bits else "no frame/crop exports"
    st.info(
        f"**Detect/Track plan** → {detect_detector_label} + {detect_tracker_label} on {detect_device_choice} "
        f"· stride {int(stride_value)} ({stride_hint}), {export_text}."
    )
    if job_running:
        st.caption("Another job is running; Detect/Track controls will re-enable once it completes.")

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
            "cpu_threads": 2,  # Limit CPU threads to reduce contention
        }
    )
    job_payload["detector"] = detect_detector_value
    job_payload["tracker"] = detect_tracker_value
    if detect_tracker_value == "bytetrack" and track_high_value is not None and track_new_value is not None:
        job_payload["track_high_thresh"] = float(track_high_value)
        job_payload["new_track_thresh"] = float(track_new_value)
    if fps_value > 0:
        job_payload["fps"] = fps_value
    mode_label = f"{detect_detector_label} + {detect_tracker_label}"

    def _process_detect_result(summary: Dict[str, Any] | None, error_message: str | None) -> None:
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
        detector_is_scene = isinstance(detector_summary, str) and detector_summary in helpers.SCENE_DETECTOR_LABEL_MAP
        has_detections = detections is not None and detections > 0
        has_tracks = tracks is not None and tracks > 0
        issue_messages: list[str] = []
        if detector_is_scene:
            detector_label = helpers.SCENE_DETECTOR_LABEL_MAP.get(detector_summary, detector_summary)
            issue_messages.append(f"Pipeline stopped after scene detection ({detector_label}); detect/track never ran.")
        if not has_detections or not has_tracks:
            det_label = helpers.format_count(detections) or "0"
            track_label = helpers.format_count(tracks) or "0"
            issue_messages.append(f"No detections/tracks were created (detections={det_label}, tracks={track_label}).")
        if issue_messages:
            st.error(" ".join(issue_messages) + " Please rerun **Detect/Track Faces** to generate the manifests.")
            return
        if track_ratio_value is not None and track_ratio_value < 0.1:
            st.warning(
                "⚠️ Track-to-detection ratio is below 0.10. Consider lowering ByteTrack thresholds or inspecting the episode."
            )
        details_line = [
            (f"detections: {helpers.format_count(detections)}" if detections is not None else "detections: ?"),
            (f"tracks: {helpers.format_count(tracks)}" if tracks is not None else "tracks: ?"),
        ]
        if track_ratio_value is not None:
            details_line.append(f"tracks/detections: {track_ratio_value:.2f}")
        if frames_exported:
            details_line.append(f"frames exported: {helpers.format_count(frames_exported)}")
        if crops_exported:
            details_line.append(f"crops exported: {helpers.format_count(crops_exported)}")
        if detector_summary:
            details_line.append(f"detector: {helpers.detector_label_from_value(detector_summary)}")
        if tracker_summary:
            details_line.append(f"tracker: {helpers.tracker_label_from_value(tracker_summary)}")
        st.session_state["episode_detail_flash"] = "Detect/track complete · " + " · ".join(details_line)
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
            running_state_key=running_job_key,
            active_job_key=_job_activity_key(ep_id),
            detect_flag_key=detect_running_key,
        )
        _process_detect_result(summary, error_message)

    if not local_video_exists:
        st.info("Local mirror missing; Detect/Track will mirror automatically before starting.")

    # Display total frames from video metadata
    total_frames = None
    if video_meta:
        frames_val = video_meta.get("frames")
        fps_detected = video_meta.get("fps_detected")
        duration_sec = video_meta.get("duration_sec")
        try:
            if frames_val is not None:
                total_frames = int(frames_val)
            elif duration_sec and fps_detected:
                total_frames = int(float(duration_sec) * float(fps_detected))
        except (TypeError, ValueError):
            pass

    if total_frames:
        st.markdown(f"**Total Frames:** {total_frames:,}")

    run_label = "Detect/Track (auto-mirrors from S3)"
    detect_button_disabled = job_running or detect_status_value == "running"
    if st.button(run_label, use_container_width=True, disabled=detect_button_disabled):
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
            running_state_key=running_job_key,
            active_job_key=_job_activity_key(ep_id),
            detect_flag_key=detect_running_key,
        )
        _process_detect_result(summary, error_message)
    st.caption("Mirrors required video artifacts automatically before detect/track starts.")

faces_ready = faces_ready_state
detector_manifest_value = helpers.tracks_detector_value(ep_id)
tracker_manifest_value = helpers.tracks_tracker_value(ep_id)
detector_face_only = helpers.detector_is_face_only(ep_id, detect_phase_status)
combo_detector, combo_tracker = helpers.detect_tracker_combo(ep_id, detect_phase_status)
combo_supported_harvest = helpers.pipeline_combo_supported("harvest", combo_detector, combo_tracker)
combo_supported_cluster = helpers.pipeline_combo_supported("cluster", combo_detector, combo_tracker)
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
        "Device (for face embeddings)",
        helpers.DEVICE_LABELS,
        index=helpers.device_label_index(faces_device_label_default),
        key="faces_device_choice",
    )
    st.caption("CoreML/GPU strongly recommended for ArcFace embeddings; significantly faster than CPU.")
    faces_device_value = helpers.DEVICE_VALUE_MAP[faces_device_choice]
    # Automatically save frames and crops to S3 (no local storage)
    faces_save_frames = True
    faces_save_crops = True
    st.caption("Frames and crops are automatically saved to S3 during harvest.")
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
    if faces_device_value in {"mps", "coreml"}:
        st.caption(
            "ArcFace embeddings use Apple's CoreML backend on supported hardware and only fall back to CPU "
            "if the CoreML provider is unavailable."
        )
    resolved_embed_device = faces_phase_status.get("resolved_device")
    if isinstance(resolved_embed_device, str) and resolved_embed_device.strip():
        resolved_embed_device = resolved_embed_device.strip().lower()
        resolved_label = helpers.device_label_from_value(resolved_embed_device)
        if resolved_label == helpers.device_default_label() and resolved_embed_device:
            resolved_label = resolved_embed_device.upper()
        st.caption(f"Last harvest resolved to **{resolved_label}**.")
    harvest_frame_est = (
        helpers.coerce_int(detect_phase_status.get("frames_exported"))
        or helpers.coerce_int(detect_phase_status.get("sampled_frames"))
        or sampled_frames_est
    )
    # Derive face count from detections manifest for real-time feedback
    harvest_faces_est = _count_manifest_rows(detections_path)
    if harvest_faces_est is None:
        harvest_faces_est = helpers.coerce_int(detect_phase_status.get("detections"))
    if harvest_faces_est is None and harvest_frame_est:
        harvest_faces_est = int(harvest_frame_est * AVG_FACES_PER_FRAME)
    harvest_estimates: list[str] = []
    if faces_save_frames and harvest_frame_est:
        frame_bytes = int(harvest_frame_est * FRAME_JPEG_SIZE_EST_BYTES)
        harvest_estimates.append(f"frames ≈{helpers.human_size(frame_bytes)}")
    if faces_save_crops and harvest_faces_est:
        crop_bytes = int(harvest_faces_est * CROP_JPEG_SIZE_EST_BYTES)
        harvest_estimates.append(f"crops ≈{helpers.human_size(crop_bytes)}")
    if harvest_estimates:
        st.caption("Estimated output: " + " + ".join(harvest_estimates))

    # Improved messaging for when Harvest Faces is disabled
    if not local_video_exists:
        st.info("Local mirror missing; video will be mirrored from S3 automatically when Faces Harvest starts.")
    elif faces_status_value == "stale":
        st.warning(
            "**Harvest Faces is outdated**: Detect/Track was rerun after the last faces harvest.\n\n"
            "Track IDs have changed. Rerun **Faces Harvest** to rebuild embeddings for the new tracks."
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
        if st.button(
            "Rerun Detect/Track",
            key="faces_inline_detect",
            use_container_width=True,
            disabled=job_running,
        ):
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
                running_state_key=running_job_key,
                active_job_key=_job_activity_key(ep_id),
                detect_flag_key=detect_running_key,
            )
            _process_detect_result(summary, error_message)
    elif faces_status_value == "running":
        st.info("Faces harvest is running. Progress will update automatically; clustering remains disabled until completion.")
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
            disabled=job_running,
        ):
            _trigger_safe_detect_rerun(ep_id, "Starting Detect/Track with RetinaFace + ByteTrack…")
    elif not combo_supported_harvest:
        current_combo = f"{helpers.detector_label_from_value(combo_detector)} + {helpers.tracker_label_from_value(combo_tracker)}"
        st.error(
            f"Harvest requires RetinaFace + ByteTrack tracks. Last detect run used **{current_combo}**. "
            "Fix the detector/tracker combo and rerun detect/track."
        )
        if st.button(
            "Fix + rerun detect",
            key="faces_fix_combo",
            use_container_width=True,
            disabled=job_running,
        ):
            _trigger_safe_detect_rerun(ep_id, "Starting Detect/Track with RetinaFace + ByteTrack…")

    faces_disabled = (
        (not tracks_ready)
        or (not detector_face_only)
        or job_running
        or faces_status_value == "running"
        or (not combo_supported_harvest)
    )
    if st.button("Run Faces Harvest", use_container_width=True, disabled=faces_disabled):
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
            st.session_state[running_job_key] = True
            _set_job_active(ep_id, True)
            try:
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
            finally:
                st.session_state[running_job_key] = False
                _set_job_active(ep_id, False)
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
                flash_msg = "Faces harvest complete" + (" · " + ", ".join(details) if details else "")
                st.session_state["episode_detail_flash"] = flash_msg
                st.rerun()
with col_cluster:
    st.markdown("### Cluster Identities")
    st.caption(_format_phase_status("Cluster Identities", cluster_phase_status, "identities"))
    cluster_device_choice = st.selectbox(
        "Device (for clustering)",
        helpers.DEVICE_LABELS,
        index=helpers.device_label_index(cluster_device_label_default),
        key="cluster_device_choice",
    )
    st.caption("Device for similarity comparisons during clustering; GPU/CoreML provides faster batch processing.")
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
        st.info("Local mirror missing; artifacts will be mirrored automatically when clustering starts.")
    elif not tracks_ready:
        st.caption("Run detect/track first; clustering requires fresh tracks and faces.")
        if st.button(
            "Rerun Detect/Track",
            key="cluster_inline_detect",
            use_container_width=True,
            disabled=job_running,
        ):
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
                running_state_key=running_job_key,
                active_job_key=_job_activity_key(ep_id),
                detect_flag_key=detect_running_key,
            )
            _process_detect_result(summary, error_message)

    # Check if faces harvest succeeded with zero faces
    zero_faces_success = faces_status_value == "success" and (
        (faces_count_value is not None and faces_count_value == 0)
        or (faces_manifest_count == 0)
    )

    if zero_faces_success:
        st.info("Faces harvest completed with 0 faces. Clustering is disabled until faces are available.")
    elif cluster_status_value == "stale":
        st.warning(
            "**Cluster is outdated**: Detect/Track was rerun after the last clustering.\n\n"
            "Track IDs have changed. Rerun **Faces Harvest** and then **Cluster** to rebuild identities."
        )
    elif faces_status_value == "stale":
        st.warning(
            "**Faces are outdated**: Detect/Track was rerun after the last faces harvest.\n\n"
            "Rerun **Faces Harvest** first, then cluster."
        )
    elif not faces_ready:
        if faces_status_value == "running":
            st.caption("Faces harvest is running — wait for it to finish before clustering.")
        elif faces_status_value == "error":
            st.error("Faces harvest failed. Rerun harvest to generate embeddings before clustering.")
        elif faces_status_value == "success":
            st.warning("Faces manifest not mirrored locally. Mirror artifacts before clustering.")
        else:
            st.caption("Run faces harvest first.")
    elif not detector_face_only:
        st.warning("Current tracks were generated with a legacy detector. Rerun detect/track first.")
    elif not combo_supported_cluster:
        combo_label = f"{helpers.detector_label_from_value(combo_detector)} + {helpers.tracker_label_from_value(combo_tracker)}"
        st.error(
            f"Cluster requires RetinaFace + ByteTrack tracks. Last detect run used **{combo_label}**. "
            "Rerun detect/track with the supported combo before clustering."
        )
        if st.button(
            "Fix + rerun detect",
            key="cluster_fix_combo",
            use_container_width=True,
            disabled=job_running,
        ):
            _trigger_safe_detect_rerun(ep_id, "Starting Detect/Track with RetinaFace + ByteTrack…")

    cluster_disabled = (
        (not faces_ready)
        or (not detector_face_only)
        or (not tracks_ready)
        or job_running
        or zero_faces_success
        or (not combo_supported_cluster)
        or faces_status_value == "stale"
    )
    if st.button("Run Cluster", use_container_width=True, disabled=cluster_disabled):
        can_run_cluster = True
        if not local_video_exists:
            can_run_cluster = _ensure_local_artifacts(ep_id, details)
            if can_run_cluster:
                local_video_exists = True
        # Ensure faces manifest is mirrored locally before clustering
        if can_run_cluster and not faces_path.exists():
            with st.spinner("Mirroring faces manifest from S3…"):
                try:
                    helpers.api_post(f"/episodes/{ep_id}/mirror")
                    st.success("Faces manifest mirrored successfully.")
                except requests.RequestException as exc:
                    st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/{ep_id}/mirror", exc))
                    can_run_cluster = False
        if can_run_cluster:
            payload = {
                "ep_id": ep_id,
                "device": cluster_device_value,
                "cluster_thresh": float(cluster_thresh_value),
                "min_cluster_size": int(min_cluster_size_value),
            }
            st.session_state[running_job_key] = True
            _set_job_active(ep_id, True)
            try:
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
            finally:
                st.session_state[running_job_key] = False
                _set_job_active(ep_id, False)
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
                flash_msg = f"Clustered (thresh {cluster_thresh_value:.2f}, min {int(min_cluster_size_value)})" + (
                    " · " + ", ".join(details) if details else ""
                )
                st.session_state["episode_detail_flash"] = flash_msg
                st.rerun()
with col_screen:
    st.markdown("### Screentime")
    screentime_disabled = False
    if not local_video_exists:
        st.info("Local mirror missing; video will be mirrored automatically when screentime starts.")
    if st.button("Compute screentime", use_container_width=True, disabled=screentime_disabled):
        can_run_screen = True
        if not local_video_exists:
            can_run_screen = _ensure_local_artifacts(ep_id, details)
            if can_run_screen:
                local_video_exists = True
        if can_run_screen:
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
                    st.rerun()

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
                if st.button("Refresh progress", key="refresh_screentime_progress", use_container_width=True):
                    st.rerun()
            elif job_state == "succeeded":
                st.success("Screentime analysis complete.")
                st.caption(f"JSON → {helpers.link_local(helpers.DATA_ROOT / 'analytics' / ep_id / 'screentime.json')}")
                st.caption(f"CSV → {helpers.link_local(helpers.DATA_ROOT / 'analytics' / ep_id / 'screentime.csv')}")
                if st.button("Dismiss screentime status", key="dismiss_screentime_job_success"):
                    st.session_state.pop(SCREENTIME_JOB_KEY, None)
                    st.rerun()
            elif job_state == "failed":
                st.error(f"Screentime job failed: {job_progress_resp.get('error') or 'unknown error'}")
                if st.button("Dismiss screentime status", key="dismiss_screentime_job_failed"):
                    st.session_state.pop(SCREENTIME_JOB_KEY, None)
                    st.rerun()
            else:
                st.info(f"Screentime job status: {job_state or 'unknown'}")
                if st.button("Dismiss screentime status", key="dismiss_screentime_job_other"):
                    st.session_state.pop(SCREENTIME_JOB_KEY, None)
                    st.rerun()

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

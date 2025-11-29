from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple
from zoneinfo import ZoneInfo

import requests
import streamlit as st

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402

from py_screenalytics.artifacts import get_path  # noqa: E402

FRAME_JPEG_SIZE_EST_BYTES = 220_000
CROP_JPEG_SIZE_EST_BYTES = 40_000
AVG_FACES_PER_FRAME = 1.5
JPEG_DEFAULT = int(os.environ.get("SCREENALYTICS_JPEG_QUALITY", "72"))
MIN_FRAMES_BETWEEN_CROPS_DEFAULT = int(os.environ.get("SCREENALYTICS_MIN_FRAMES_BETWEEN_CROPS", "32"))
EST_TZ = ZoneInfo("America/New_York")


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
    try:
        est = dt.astimezone(EST_TZ)
    except Exception:
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return est.strftime("%Y-%m-%d %H:%M:%S ET")


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


def _format_video_duration(duration_sec: Any) -> str | None:
    """Format video duration in human-readable form (e.g., '45m 30s' or '1h 23m')."""
    try:
        total = float(duration_sec)
    except (TypeError, ValueError):
        return None
    if total <= 0:
        return None
    seconds = int(round(total))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        minutes, rem = divmod(seconds, 60)
        return f"{minutes}m {rem:02d}s"
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    return f"{hours}h {minutes:02d}m"


def _runtime_from_iso(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    try:
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return None
    delta = (end_dt - start_dt).total_seconds()
    return delta if delta >= 0 else None


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


def _status_mtimes_key(ep_id: str) -> str:
    return f"{ep_id}::status_mtimes"


def _navigate_to_upload(ep_id: str) -> None:
    helpers.set_ep_id(ep_id, rerun=False, origin="replace")
    params = st.query_params
    params["ep_id"] = ep_id
    st.query_params = params
    helpers.try_switch_page("pages/0_Upload_Video.py")


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
def _cached_episode_status(ep_id: str, cache_key: float, marker_mtimes: tuple) -> Dict[str, Any] | None:
    """Cache episode status API response with 10s TTL.

    Args:
        ep_id: Episode ID
        cache_key: Fetch token for manual cache busting
        marker_mtimes: Tuple of artifact mtimes to auto-invalidate cache (runs + manifests)
    """
    return helpers.get_episode_status(ep_id)


def _detect_track_manifests_ready(detections_path: Path, tracks_path: Path) -> dict:
    detections_ready = _manifest_has_rows(detections_path)
    tracks_ready = _manifest_has_rows(tracks_path)
    tracks_only = bool(tracks_ready and not detections_ready)
    manifest_ready = bool(detections_ready and tracks_ready)
    return {
        "detections_ready": detections_ready,
        "tracks_ready": tracks_ready,
        "manifest_ready": manifest_ready,
        "tracks_only_fallback": tracks_only,
    }


def _compute_detect_track_effective_status(
    detect_status: Dict[str, Any],
    *,
    manifest_ready: bool,
    tracks_only_fallback: bool,
    tracks_ready_flag: bool,
    job_state: str | None = None,
) -> tuple[str, bool, bool, bool]:
    normalized_job_state = str(job_state or "").strip().lower()
    if normalized_job_state == "running":
        return "running", False, False, tracks_only_fallback
    if tracks_ready_flag:
        return "success", True, False, tracks_only_fallback
    normalized_status = str(detect_status.get("status") or "missing").strip().lower()
    if not normalized_status:
        normalized_status = "missing"
    manifest_tracks_ready = bool(manifest_ready)
    if normalized_status == "success":
        if manifest_tracks_ready:
            return "success", True, False, tracks_only_fallback
        return "stale", False, False, tracks_only_fallback
    if manifest_tracks_ready:
        return "success", True, True, tracks_only_fallback
    return normalized_status, False, False, tracks_only_fallback


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
helpers.inject_log_container_css()  # Limit log container height with scrolling
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
    with st.spinner("Mirroring video from S3 (this may take several minutes for large files)…"):
        try:
            # Use longer timeout for S3 downloads (10 minutes)
            resp = helpers.api_post(mirror_path, timeout=600)
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
        # Use execution mode from UI settings (respects local/redis toggle)
        execution_mode = helpers.get_execution_mode(ep_id)
        mode_desc = "local" if execution_mode == "local" else "Celery"
        runner = helpers.run_pipeline_job_with_mode
        if execution_mode == "local":
            summary, error_message = runner(
                ep_id,
                "detect_track",
                job_payload,
                requested_device=device_value,
                requested_detector=detector_value,
                requested_tracker=tracker_value,
            )
        else:
            with st.spinner(f"Running detect/track via {mode_desc} ({mode_label} on {device_label})…"):
                summary, error_message = runner(
                    ep_id,
                    "detect_track",
                    job_payload,
                    requested_device=device_value,
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

# Hydrate logs for this episode on page load (local mode log persistence)
# This fetches any previously saved logs so they can be displayed without re-running jobs
helpers.hydrate_logs_for_episode(ep_id)

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
mtimes_key = _status_mtimes_key(ep_id)
force_refresh_key = _status_force_refresh_key(ep_id)
force_refresh = bool(st.session_state.pop(force_refresh_key, False))
fetch_token = st.session_state.get(fetch_token_key, 0)
status_payload = st.session_state.get(status_cache_key)
_manifests_dir = get_path(ep_id, "detections").parent
_runs_dir = _manifests_dir / "runs"
_track_metrics_path = _manifests_dir / "track_metrics.json"
current_mtimes = (
    (_runs_dir / "detect_track.json").stat().st_mtime if (_runs_dir / "detect_track.json").exists() else 0,
    (_runs_dir / "faces_embed.json").stat().st_mtime if (_runs_dir / "faces_embed.json").exists() else 0,
    (_runs_dir / "cluster.json").stat().st_mtime if (_runs_dir / "cluster.json").exists() else 0,
    (_manifests_dir / "detections.jsonl").stat().st_mtime if (_manifests_dir / "detections.jsonl").exists() else 0,
    (_manifests_dir / "tracks.jsonl").stat().st_mtime if (_manifests_dir / "tracks.jsonl").exists() else 0,
    (_manifests_dir / "faces.jsonl").stat().st_mtime if (_manifests_dir / "faces.jsonl").exists() else 0,
    # Track cluster metrics so status cache updates when clustering writes only metrics.
    _track_metrics_path.stat().st_mtime if _track_metrics_path.exists() else 0,
    (_manifests_dir / "identities.json").stat().st_mtime if (_manifests_dir / "identities.json").exists() else 0,
)
cached_mtimes = st.session_state.get(mtimes_key)
should_refresh_status = force_refresh or _job_active(ep_id) or status_payload is None or cached_mtimes != current_mtimes
if should_refresh_status:
    fetch_token += 1
    st.session_state[fetch_token_key] = fetch_token
    # Include manifests in the cache key so status refreshes when identities/faces/tracks change.
    status_payload = _cached_episode_status(ep_id, fetch_token, current_mtimes)
    st.session_state[status_cache_key] = status_payload
    st.session_state[status_ts_key] = time.time()
    st.session_state[mtimes_key] = current_mtimes
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
screentime_json_path = helpers.DATA_ROOT / "analytics" / ep_id / "screentime.json"
detect_job_defaults, detect_job_record = _load_job_defaults(ep_id, "detect_track")
faces_job_defaults, faces_job_record = _load_job_defaults(ep_id, "faces_embed")
cluster_job_defaults, cluster_job_record = _load_job_defaults(ep_id, "cluster")
_, screentime_job_record = _load_job_defaults(ep_id, "screen_time_analyze")
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


# =============================================================================
# Execution Mode Selector
# =============================================================================
# Store execution mode globally for this episode so all actions respect it
with st.expander("🔧 Execution Settings", expanded=False):
    exec_mode_col1, exec_mode_col2 = st.columns([2, 3])
    with exec_mode_col1:
        execution_mode = helpers.render_execution_mode_selector(ep_id, key_suffix="episode_detail")
    with exec_mode_col2:
        if execution_mode == "local":
            st.info("**Local Mode**: Jobs run synchronously in-process. No Redis/Celery needed.")
        else:
            st.info("**Redis Mode**: Jobs are queued via Celery for background processing.")


# =============================================================================
# Current Jobs Panel (Celery + subprocess background jobs)
# =============================================================================
with st.expander("⚙️ Current Jobs", expanded=False):
    try:
        all_jobs: list[dict] = []

        # Fetch Celery jobs
        celery_response = helpers.api_get("/celery_jobs")
        celery_jobs = celery_response.get("jobs", []) if celery_response else []
        for job in celery_jobs:
            all_jobs.append({
                "job_id": job.get("job_id", "unknown"),
                "name": job.get("name", "Celery Task"),
                "state": job.get("state", "unknown"),
                "worker": job.get("worker", ""),
                "ep_id": job.get("ep_id"),
                "source": "celery",
            })

        # Fetch subprocess-based jobs (filtered to current episode if set)
        jobs_response = helpers.api_get(f"/jobs?ep_id={ep_id}&limit=20")
        subprocess_jobs = jobs_response.get("jobs", []) if jobs_response else []
        for job in subprocess_jobs:
            # Only show running/queued jobs, not completed ones
            state = job.get("state", "unknown")
            if state in ("running", "queued", "in_progress"):
                all_jobs.append({
                    "job_id": job.get("job_id", "unknown"),
                    "name": job.get("job_type", "Pipeline Job"),
                    "state": state,
                    "worker": "",
                    "ep_id": job.get("ep_id"),
                    "source": "subprocess",
                })

        if not all_jobs:
            st.info("No background jobs currently running.")
        else:
            st.caption(f"Found {len(all_jobs)} active job(s) for this episode")
            for job in all_jobs:
                job_id = job.get("job_id", "unknown")
                job_name = job.get("name", "unknown")
                job_state = job.get("state", "unknown")
                worker = job.get("worker", "")
                source = job.get("source", "unknown")

                # State badge
                if job_state in ("in_progress", "running"):
                    badge = "🔄"
                elif job_state == "queued":
                    badge = "⏳"
                elif job_state == "scheduled":
                    badge = "📅"
                else:
                    badge = "❓"

                # Display job card with cancel button
                col1, col2, col3 = st.columns([2.5, 1, 0.5])
                with col1:
                    st.markdown(f"**{badge} {job_name}**")
                    short_id = f"{job_id[:12]}..." if len(job_id) > 12 else job_id
                    st.caption(f"ID: `{short_id}` ({source})")
                with col2:
                    st.caption(f"State: {job_state}")
                    if worker:
                        st.caption(f"Worker: {worker.split('@')[-1]}")
                with col3:
                    # Cancel button
                    cancel_key = f"cancel_{job_id}"
                    if st.button("❌", key=cancel_key, help="Cancel this job"):
                        try:
                            if source == "celery":
                                cancel_resp = helpers.api_post(f"/celery_jobs/{job_id}/cancel")
                            else:
                                cancel_resp = helpers.api_post(f"/jobs/{job_id}/cancel")
                            if cancel_resp:
                                st.success(f"Cancelled job {job_id[:8]}...")
                                st.rerun()
                            else:
                                st.error("Failed to cancel job")
                        except Exception as cancel_err:
                            st.error(f"Cancel failed: {cancel_err}")
                st.divider()
    except Exception as e:
        st.warning(f"Could not fetch job status: {e}")


with st.expander(f"Episode {ep_id}", expanded=False):
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
detect_status_value, tracks_ready, using_manifest_fallback, tracks_only_fallback = _compute_detect_track_effective_status(
    detect_phase_status,
    manifest_ready=manifest_state["manifest_ready"],
    tracks_only_fallback=manifest_state["tracks_only_fallback"],
    tracks_ready_flag=tracks_ready_flag,
    job_state=detect_job_state,
)
if cluster_status_value in {"missing", "unknown"}:
    identities_count_manifest = None
    cluster_metrics_block: dict[str, Any] | None = None
    artifact_mtime = 0.0
    if identities_path.exists():
        try:
            payload = json.loads(identities_path.read_text(encoding="utf-8"))
            identities_list = payload.get("identities") if isinstance(payload, dict) else None
            if isinstance(identities_list, list):
                identities_count_manifest = len(identities_list)
        except (json.JSONDecodeError, OSError, KeyError):
            # File may be corrupted or in unexpected format - silently skip
            pass
        try:
            artifact_mtime = identities_path.stat().st_mtime
        except OSError:
            artifact_mtime = 0.0
    if _track_metrics_path.exists():
        try:
            metrics_data = json.loads(_track_metrics_path.read_text(encoding="utf-8"))
            if isinstance(metrics_data, dict):
                block = metrics_data.get("cluster_metrics")
                cluster_metrics_block = block if isinstance(block, dict) else None
        except (json.JSONDecodeError, OSError, KeyError):
            # File may be corrupted or in unexpected format - silently skip
            cluster_metrics_block = None
        try:
            artifact_mtime = max(artifact_mtime, _track_metrics_path.stat().st_mtime)
        except OSError:
            pass
    if identities_count_manifest is None and isinstance(cluster_metrics_block, dict):
        identities_count_manifest = helpers.coerce_int(
            cluster_metrics_block.get("total_clusters_after") or cluster_metrics_block.get("total_clusters")
        )
    if identities_count_manifest is not None or cluster_metrics_block:
        cluster_phase_status = dict(cluster_phase_status)
        if isinstance(cluster_metrics_block, dict):
            cluster_phase_status.setdefault("singleton_stats", cluster_metrics_block.get("singleton_stats"))
            cluster_phase_status.setdefault("singleton_merge", cluster_metrics_block.get("singleton_merge"))
            cluster_phase_status.setdefault("singleton_fraction_before", cluster_metrics_block.get("singleton_fraction_before"))
            cluster_phase_status.setdefault("singleton_fraction_after", cluster_metrics_block.get("singleton_fraction_after"))
            cluster_phase_status.setdefault("total_clusters_before", cluster_metrics_block.get("total_clusters_before"))
            cluster_phase_status.setdefault("total_clusters_after", cluster_metrics_block.get("total_clusters_after"))
        cluster_phase_status["status"] = "success"
        cluster_phase_status["identities"] = identities_count_manifest
        cluster_phase_status["source"] = cluster_phase_status.get("source") or "manifest_fallback"
        # Read marker file for timestamps and device info FIRST (most authoritative source)
        _cluster_marker_path = _runs_dir / "cluster.json"
        if _cluster_marker_path.exists():
            try:
                _marker_data = json.loads(_cluster_marker_path.read_text(encoding="utf-8"))
                if isinstance(_marker_data, dict):
                    if not cluster_phase_status.get("started_at"):
                        cluster_phase_status["started_at"] = _marker_data.get("started_at")
                    if not cluster_phase_status.get("finished_at"):
                        cluster_phase_status["finished_at"] = _marker_data.get("finished_at")
                    if not cluster_phase_status.get("device"):
                        cluster_phase_status["device"] = _marker_data.get("device")
            except (json.JSONDecodeError, OSError):
                pass
        # Fallback to artifact mtime for finished_at if marker didn't have it
        if not cluster_phase_status.get("finished_at") and artifact_mtime:
            cluster_phase_status["finished_at"] = (
                datetime.fromtimestamp(artifact_mtime, tz=timezone.utc).replace(microsecond=0).isoformat() + "Z"
            )
        # Compute runtime_sec if timestamps are available
        if not cluster_phase_status.get("runtime_sec"):
            _runtime = _runtime_from_iso(
                cluster_phase_status.get("started_at"),
                cluster_phase_status.get("finished_at"),
            )
            if _runtime is not None:
                cluster_phase_status["runtime_sec"] = _runtime
        cluster_status_value = "success"
jpeg_state = helpers.coerce_int(detect_phase_status.get("jpeg_quality"))
device_state = detect_phase_status.get("device")
requested_device_state = detect_phase_status.get("requested_device")
resolved_device_state = detect_phase_status.get("resolved_device")
screentime_status_value = "missing"
screentime_error = None
screentime_started_at = None
screentime_finished_at = None
if screentime_job_record:
    job_state = str(screentime_job_record.get("state") or "").lower()
    screentime_error = screentime_job_record.get("error")
    screentime_started_at = screentime_job_record.get("started_at")
    screentime_finished_at = screentime_job_record.get("ended_at")
    if job_state == "running":
        screentime_status_value = "running"
    elif job_state in {"failed", "error"}:
        screentime_status_value = "error"
    elif job_state == "succeeded":
        screentime_status_value = "success"
if screentime_status_value == "missing" and screentime_json_path.exists():
    screentime_status_value = "success"
    if screentime_finished_at is None:
        screentime_finished_at = (
            datetime.fromtimestamp(screentime_json_path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat() + "Z"
        )
screentime_runtime = _format_runtime(_runtime_from_iso(screentime_started_at, screentime_finished_at))
status_running = (
    detect_status_value == "running"
    or faces_status_value == "running"
    or cluster_status_value == "running"
    or str(detect_job_state or "").lower() == "running"
    or screentime_status_value == "running"
)
if status_running:
    _set_job_active(ep_id, True)
elif not job_running:
    _set_job_active(ep_id, False)

# Other status values
faces_count_value = helpers.coerce_int(faces_phase_status.get("faces"))
identities_count_value = helpers.coerce_int(cluster_phase_status.get("identities"))
faces_manifest_count = None
faces_ready_state = False
faces_manifest_fallback = bool(faces_phase_status.get("faces_manifest_fallback"))
faces_manifest_exists = faces_path.exists()
if faces_manifest_exists:
    faces_manifest_count = _count_manifest_rows(faces_path) or 0
if faces_status_value == "success":
    faces_ready_state = True
elif faces_status_value in {"missing", "unknown", "stale"} and faces_manifest_exists:
    faces_ready_state = True
    faces_manifest_fallback = True
if faces_count_value is None and faces_manifest_count is not None:
    faces_count_value = faces_manifest_count

# If detect status is missing but manifests are present, synthesize a summary so the UI still shows completion.
if not detect_phase_status and manifest_state["manifest_ready"]:
    detect_phase_status = {
        "status": "success",
        "detections": _count_manifest_rows(detections_path) or 0,
        "tracks": _count_manifest_rows(tracks_path) or 0,
        "finished_at": None,
    }
    detect_status_value = "success"
    using_manifest_fallback = True

# Add pipeline state indicators (even if status API is temporarily unavailable)
with st.expander("Pipeline Status", expanded=False):
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
        refreshed_dt = datetime.fromtimestamp(status_refreshed_at, tz=timezone.utc).astimezone(EST_TZ)
        refreshed_label = refreshed_dt.strftime("%Y-%m-%d %H:%M:%S ET")
        st.caption(f"Status refreshed at {refreshed_label}")
    else:
        st.caption("Status will refresh when a job starts or you press refresh.")
    coreml_available = status_payload.get("coreml_available") if status_payload else None
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
        if jpeg_state:
            detect_params.append(f"jpeg={jpeg_state}")
        detect_runtime = _format_runtime(detect_phase_status.get("runtime_sec"))
        if requested_device_state and requested_device_state != device_state:
            detect_params.append(f"requested={helpers.device_label_from_value(requested_device_state)}")
        device_label = helpers.device_label_from_value(
            resolved_device_state or device_state or requested_device_state or helpers.DEFAULT_DEVICE
        )
        if device_label:
            detect_params.append(f"device={device_label}")
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
            if using_manifest_fallback or detect_phase_status.get("metadata_missing"):
                st.warning(
                    "⚠️ Detect/Track details inferred from manifests (metadata missing). "
                    "Detector/tracker and runtime may be inaccurate; rerun detect/track for fresh metadata."
                )
            if tracks_only_fallback:
                st.warning("⚠️ Tracks exist but detections are missing. Rerun detect/track to regenerate detections.")
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
        if tracks_only_fallback:
            st.warning(
                "⚠️ Tracks manifest is present but detections are missing. Rerun Detect/Track to regenerate detections "
                "before continuing."
            )
        if jpeg_state:
            st.caption(f"JPEG quality: {jpeg_state}")
        _render_device_summary(requested_device_state, resolved_device_state or device_state)
        finished = _format_timestamp(detect_phase_status.get("finished_at"))
        if finished:
            st.caption(f"Last run: {finished}")
        # Show video duration and run duration on separate lines
        video_duration = _format_video_duration(detect_phase_status.get("video_duration_sec"))
        if video_duration:
            st.caption(f"Video Duration: {video_duration}")
        if detect_runtime:
            st.caption(f"Run Duration: {detect_runtime}")
        elif detect_status_value == "success":
            st.caption("Run Duration: n/a")

    with col2:
        faces_params: list[str] = []
        faces_device_state = faces_phase_status.get("device")
        faces_device_request = faces_phase_status.get("requested_device")
        faces_resolved_state = faces_phase_status.get("resolved_device")
        faces_runtime = _format_runtime(faces_phase_status.get("runtime_sec"))
        faces_job_state = str((faces_job_record or {}).get("state") or "").lower()
        faces_error_msg = faces_phase_status.get("error") or (faces_job_record or {}).get("error")
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
        spacing_state = helpers.coerce_int(faces_phase_status.get("min_frames_between_crops"))
        if spacing_state:
            faces_params.append(f"spacing={spacing_state}")
        thumb_size_state = helpers.coerce_int(faces_phase_status.get("thumb_size"))
        if thumb_size_state:
            faces_params.append(f"thumb={thumb_size_state}px")
        faces_jpeg_state = helpers.coerce_int(faces_phase_status.get("jpeg_quality"))
        if faces_jpeg_state:
            faces_params.append(f"jpeg={faces_jpeg_state}")
        if faces_ready_state:
            runtime_label = faces_runtime or "n/a"
            st.success(f"✅ **Faces Harvest**: Complete (Runtime: {runtime_label})")
            face_count_label = helpers.format_count(faces_count_value) or "0"
            st.caption(f"Faces: {face_count_label} (harvest completed)")
            if faces_manifest_fallback:
                st.caption("ℹ️ Using manifest fallback; status may be stale.")
        elif faces_status_value == "success":
            st.warning("⚠️ **Faces Harvest**: Manifest unavailable locally")
            st.caption("Faces completed on the backend, but faces.jsonl has not been mirrored locally yet.")
        elif faces_job_state == "failed":
            st.error("⚠️ **Faces Harvest**: Failed")
            if faces_error_msg:
                st.caption(faces_error_msg)
        elif faces_status_value not in {"missing", "unknown"}:
            st.warning(f"⚠️ **Faces Harvest**: {faces_status_value.title()}")
            if faces_error_msg:
                st.caption(faces_error_msg)
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
        if faces_runtime:
            st.caption(f"Run Duration: {faces_runtime}")
        elif faces_status_value == "success":
            st.caption("Run Duration: n/a")

    with col3:
        cluster_params: list[str] = []
        cluster_device_state = cluster_phase_status.get("device")
        cluster_device_request = cluster_phase_status.get("requested_device")
        cluster_resolved_state = cluster_phase_status.get("resolved_device")
        cluster_runtime = _format_runtime(cluster_phase_status.get("runtime_sec"))
        cluster_job_state = str((cluster_job_record or {}).get("state") or "").lower()
        cluster_error_msg = cluster_phase_status.get("error") or (cluster_job_record or {}).get("error")
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
            if identities_count_value == 0:
                st.warning("Cluster finished but found 0 identities. Rerun after checking detect/track and faces outputs.")
        elif cluster_status_value == "running":
            st.info("⏳ **Cluster**: Running")
            started = _format_timestamp(cluster_phase_status.get("started_at"))
            if started:
                st.caption(f"Started at {started}")
            st.caption("Live progress appears in the log panel below.")
        elif cluster_job_state == "failed":
            st.error("⚠️ **Cluster**: Failed")
            if cluster_error_msg:
                st.caption(cluster_error_msg)
        elif cluster_status_value not in {"missing", "unknown"}:
            st.warning(f"⚠️ **Cluster**: {cluster_status_value.title()}")
            if cluster_error_msg:
                st.caption(cluster_error_msg)
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
        merge_block = cluster_phase_status.get("singleton_merge") or {}
        singleton_stats = cluster_phase_status.get("singleton_stats") or merge_block.get("singleton_stats") or {}
        if not isinstance(singleton_stats, dict):
            singleton_stats = {}
        before_block = singleton_stats.get("before") or {}
        after_block = singleton_stats.get("after") or {}
        if not isinstance(before_block, dict):
            before_block = {}
        if not isinstance(after_block, dict):
            after_block = {}
        before_frac = helpers.coerce_float(before_block.get("singleton_fraction"))
        if before_frac is None:
            before_frac = helpers.coerce_float(cluster_phase_status.get("singleton_fraction_before"))
        after_frac = helpers.coerce_float(after_block.get("singleton_fraction"))
        if after_frac is None:
            after_frac = helpers.coerce_float(cluster_phase_status.get("singleton_fraction_after"))
        threshold = helpers.coerce_float(singleton_stats.get("threshold"))
        if threshold is None:
            threshold = helpers.coerce_float(cluster_phase_status.get("singleton_merge_threshold"))
        clusters_before = helpers.coerce_int(before_block.get("cluster_count"))
        if clusters_before is None:
            clusters_before = helpers.coerce_int(cluster_phase_status.get("total_clusters_before"))
        clusters_after = helpers.coerce_int(after_block.get("cluster_count"))
        if clusters_after is None:
            clusters_after = helpers.coerce_int(cluster_phase_status.get("total_clusters_after"))
        merge_count = helpers.coerce_int(after_block.get("merge_count"))
        if merge_count is None:
            merge_count = helpers.coerce_int(merge_block.get("num_singleton_merges"))
        sim_thresh = merge_block.get("similarity_thresh") or merge_block.get("secondary_cluster_thresh")
        neighbor_top_k = merge_block.get("neighbor_top_k") or merge_block.get("max_pairs_per_track")
        merge_enabled = merge_block.get("enabled") if merge_block else False
        primary_frac = after_frac if after_frac is not None else before_frac
        has_metrics = any(val is not None for val in [before_frac, after_frac, clusters_before, clusters_after, merge_count])
        use_after_merge_label = bool(merge_enabled and after_frac is not None)
        if has_metrics:
            lines: list[str] = []
            if before_frac is not None and after_frac is not None:
                thresh_label = f"{threshold:.2f}" if threshold is not None else "?"
                lines.append(f"Singletons: {before_frac:.2f} → {after_frac:.2f} (threshold {thresh_label})")
            elif before_frac is not None:
                thresh_label = f" (threshold {threshold:.2f})" if threshold is not None else ""
                lines.append(f"Singletons: {before_frac:.2f}{thresh_label}")
            elif after_frac is not None:
                thresh_label = f" (threshold {threshold:.2f})" if threshold is not None else ""
                lines.append(f"Singletons (post-merge): {after_frac:.2f}{thresh_label}")
            if clusters_before is not None and clusters_after is not None:
                before_clusters = helpers.format_count(clusters_before) or str(clusters_before)
                after_clusters = helpers.format_count(clusters_after) or str(clusters_after)
                lines.append(f"Clusters: {before_clusters} → {after_clusters}")
            elif clusters_after is not None:
                after_clusters = helpers.format_count(clusters_after) or str(clusters_after)
                lines.append(f"Clusters: {after_clusters}")
            elif clusters_before is not None:
                before_clusters = helpers.format_count(clusters_before) or str(clusters_before)
                lines.append(f"Clusters: {before_clusters}")
            if merge_enabled and merge_count is not None:
                sim_label = f"{sim_thresh:.2f}" if isinstance(sim_thresh, (int, float)) else sim_thresh or "?"
                neighbor_label = neighbor_top_k if neighbor_top_k is not None else "?"
                lines.append(f"Merge: {merge_count} pairs (sim ≥ {sim_label}, top_k={neighbor_label})")
            for entry in lines:
                st.caption(entry)
            if threshold is not None and primary_frac is not None:
                high_label = "🚧 High singleton fraction after merge" if use_after_merge_label else "🚧 High singleton fraction"
                ok_label = (
                    "✅ Singletons reduced below threshold after merge"
                    if use_after_merge_label
                    else "✅ Singleton fraction below threshold"
                )
                if primary_frac > threshold:
                    st.warning(high_label)
                elif merge_enabled:
                    st.success(ok_label)
        else:
            if cluster_status_value == "success":
                st.caption("Singleton metrics unavailable for this run.")
            else:
                st.caption("Singleton metrics not available until clustering completes.")
        _render_device_summary(cluster_device_request, cluster_resolved_state or cluster_device_state)
        finished = _format_timestamp(cluster_phase_status.get("finished_at"))
        if finished:
            st.caption(f"Last run: {finished}")
        if cluster_runtime:
            st.caption(f"Run Duration: {cluster_runtime}")
        elif cluster_status_value == "success":
            st.caption("Run Duration: n/a")


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
    faces_save_frames_default = False
faces_save_crops_default = faces_job_defaults.get("save_crops")
if faces_save_crops_default is None:
    faces_save_crops_default = True
faces_jpeg_quality_default = helpers.coerce_int(faces_job_defaults.get("jpeg_quality")) or JPEG_DEFAULT
faces_min_frames_between_crops_default = helpers.coerce_int(
    faces_job_defaults.get("min_frames_between_crops")
)
if faces_min_frames_between_crops_default is None:
    faces_min_frames_between_crops_default = helpers.coerce_int(
        faces_phase_status.get("min_frames_between_crops")
    )
if faces_min_frames_between_crops_default is None:
    faces_min_frames_between_crops_default = MIN_FRAMES_BETWEEN_CROPS_DEFAULT

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

detect_inflight = bool(st.session_state.get(detect_running_key))
faces_ready = faces_ready_state
detector_manifest_value = helpers.tracks_detector_value(ep_id)
tracker_manifest_value = helpers.tracks_tracker_value(ep_id)
detector_face_only = helpers.detector_is_face_only(ep_id, detect_phase_status)
combo_detector, combo_tracker = helpers.detect_tracker_combo(ep_id, detect_phase_status)
combo_supported_harvest = helpers.pipeline_combo_supported("harvest", combo_detector, combo_tracker)
combo_supported_cluster = helpers.pipeline_combo_supported("cluster", combo_detector, combo_tracker)
col_detect, col_faces, col_cluster = st.columns(3)

# Check for running jobs for each phase
running_detect_job = helpers.get_running_job_for_episode(ep_id, "detect_track")
running_faces_job = helpers.get_running_job_for_episode(ep_id, "faces_embed")
running_cluster_job = helpers.get_running_job_for_episode(ep_id, "cluster")

# Session state keys for cancel confirmation dialogs
confirm_cancel_detect_key = f"{ep_id}::confirm_cancel_detect"
confirm_cancel_faces_key = f"{ep_id}::confirm_cancel_faces"
confirm_cancel_cluster_key = f"{ep_id}::confirm_cancel_cluster"

with col_detect:
    st.markdown("### Detect/Track Faces")
    session_prefix = f"episode_detail_detect::{ep_id}"

    # Show running job progress if a job is active
    if running_detect_job:
        job_id = running_detect_job.get("job_id", "unknown")
        progress_pct = running_detect_job.get("progress_pct", 0)
        frames_done = running_detect_job.get("frames_done", 0)
        frames_total = running_detect_job.get("frames_total", 0)
        state = running_detect_job.get("state", "running")

        st.info(f"🔄 **Detect/Track job running** ({state})")
        if frames_total > 0:
            st.progress(min(progress_pct / 100, 1.0))
            st.caption(f"Progress: {frames_done:,} / {frames_total:,} frames ({progress_pct:.1f}%)")
        else:
            st.caption(f"Progress: {progress_pct:.1f}%")

        # Refresh and Cancel buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🔄 Refresh", key=f"refresh_detect_{job_id}", use_container_width=True):
                st.rerun()
        with btn_col2:
            if st.button("❌ Cancel", key=f"cancel_detect_{job_id}", use_container_width=True):
                success, msg = helpers.cancel_running_job(job_id)
                if success:
                    st.success(msg)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(msg)

        st.divider()

    profile_default_value = detect_job_defaults.get("profile") or detect_phase_status.get("profile")
    if not profile_default_value:
        profile_default_value = helpers.default_profile_for_device(detect_device_default_value)
    profile_key = _detect_setting_key(ep_id, "profile")
    profile_seed_value = helpers.profile_value_from_state(st.session_state.get(profile_key, profile_default_value))
    # Sanitize the selectbox session state to use label format (Streamlit stores widget value directly)
    profile_widget_key = f"{session_prefix}::profile"
    if profile_widget_key in st.session_state:
        stored_value = st.session_state[profile_widget_key]
        if stored_value not in helpers.PROFILE_LABELS:
            # Convert internal value to label format
            sanitized = helpers.PROFILE_LABEL_MAP.get(str(stored_value).lower())
            if sanitized:
                st.session_state[profile_widget_key] = sanitized
            else:
                del st.session_state[profile_widget_key]  # Remove invalid value
    profile_label = st.selectbox(
        "Performance profile",
        helpers.PROFILE_LABELS,
        index=helpers.profile_label_index(profile_seed_value),
        key=profile_widget_key,
        help="Low Power is recommended on Apple laptops (higher stride, capped FPS, fewer exports).",
    )
    profile_value = helpers.PROFILE_VALUE_MAP.get(profile_label, profile_seed_value)
    profile_changed = profile_value != profile_seed_value
    profile_defaults = helpers.profile_defaults(profile_value)

    stride_default = helpers.coerce_int(detect_job_defaults.get("stride"))
    if stride_default is None:
        stride_default = helpers.coerce_int(profile_defaults.get("stride")) or helpers.DEFAULT_STRIDE
    # Prefill FPS from video metadata if available
    fps_default = helpers.coerce_float(detect_job_defaults.get("fps")) or 0.0
    if fps_default == 0.0 and video_meta and video_meta.get("fps_detected"):
        fps_default = float(video_meta["fps_detected"])
    if fps_default == 0.0 and profile_defaults.get("fps") is not None:
        try:
            fps_default = float(profile_defaults["fps"])
        except (TypeError, ValueError):
            fps_default = 0.0
    det_thresh_default = float(detect_job_defaults.get("det_thresh") or helpers.DEFAULT_DET_THRESH)
    save_frames_default = detect_job_defaults.get("save_frames")
    if save_frames_default is None:
        save_frames_default = profile_defaults.get("save_frames")
    if save_frames_default is None:
        save_frames_default = True
    save_crops_default = detect_job_defaults.get("save_crops")
    if save_crops_default is None:
        save_crops_default = profile_defaults.get("save_crops")
    if save_crops_default is None:
        save_crops_default = True
    cpu_threads_default = helpers.coerce_int(detect_job_defaults.get("cpu_threads"))
    if cpu_threads_default is None:
        cpu_threads_default = helpers.coerce_int(profile_defaults.get("cpu_threads"))
    if cpu_threads_default is None and profile_value == "low_power":
        cpu_threads_default = 2
    jpeg_quality_default = int(detect_job_defaults.get("jpeg_quality") or JPEG_DEFAULT)
    max_gap_default = int(detect_job_defaults.get("max_gap") or helpers.DEFAULT_MAX_GAP)
    scene_threshold_default = float(detect_job_defaults.get("scene_threshold") or helpers.SCENE_THRESHOLD_DEFAULT)
    scene_min_len_default = int(detect_job_defaults.get("scene_min_len") or helpers.SCENE_MIN_LEN_DEFAULT)
    scene_warmup_default = int(detect_job_defaults.get("scene_warmup_dets") or helpers.SCENE_WARMUP_DETS_DEFAULT)

    stride_field = _detect_setting_key(ep_id, "stride")
    fps_field = _detect_setting_key(ep_id, "fps")
    save_frames_key = _detect_setting_key(ep_id, "save_frames")
    save_crops_key = _detect_setting_key(ep_id, "save_crops")
    cpu_threads_key = _detect_setting_key(ep_id, "cpu_threads")

    if profile_changed:
        if profile_defaults.get("stride") is not None:
            st.session_state[stride_field] = int(profile_defaults["stride"])
        if profile_defaults.get("fps") is not None:
            try:
                st.session_state[fps_field] = float(profile_defaults["fps"])
            except (TypeError, ValueError):
                pass
        if profile_defaults.get("save_frames") is not None:
            st.session_state[save_frames_key] = bool(profile_defaults["save_frames"])
        if profile_defaults.get("save_crops") is not None:
            st.session_state[save_crops_key] = bool(profile_defaults["save_crops"])
        if profile_defaults.get("cpu_threads") is not None:
            try:
                st.session_state[cpu_threads_key] = int(profile_defaults["cpu_threads"])
            except (TypeError, ValueError):
                pass

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
    if save_frames_key not in st.session_state:
        st.session_state[save_frames_key] = bool(save_frames_default)
    save_frames = st.checkbox(
        "Save sampled frames",
        value=bool(st.session_state[save_frames_key]),
        help="Stores sampled RGB frames alongside detections for QA and future crops.",
        key=save_frames_key,
    )
    if save_crops_key not in st.session_state:
        st.session_state[save_crops_key] = bool(save_crops_default)
    save_crops = st.checkbox(
        "Save crops",
        value=bool(st.session_state[save_crops_key]),
        help="Exports aligned face crops during detect/track. Disable when reusing previous crops.",
        key=save_crops_key,
    )
    cpu_options = [1, 2, 4]
    cpu_seed = int(st.session_state.get(cpu_threads_key, cpu_threads_default or 2))
    if cpu_seed not in cpu_options:
        cpu_seed = 2
    if cpu_threads_key not in st.session_state:
        st.session_state[cpu_threads_key] = cpu_seed
    cpu_default_index = cpu_options.index(cpu_seed) if cpu_seed in cpu_options else 1
    cpu_threads_value = st.selectbox(
        "CPU threads (cap)",
        options=cpu_options,
        index=cpu_default_index,
        key=cpu_threads_key,
        help="Caps BLAS/ONNX threads. Use 2 for laptop-friendly Low Power runs; increase if you need throughput.",
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
        key=f"{ep_id}::detect_device_choice",
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
        f"· stride {int(stride_value)} ({stride_hint}), {export_text}, profile {helpers.profile_label_from_value(profile_value)}."
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
            "profile": profile_value,
            "save_frames": bool(save_frames),
            "save_crops": bool(save_crops),
            "jpeg_quality": int(jpeg_quality),
            "max_gap": int(max_gap_value),
            "scene_detector": scene_detector_value,
            "scene_threshold": float(scene_threshold_value),
            "scene_min_len": int(scene_min_len_value),
            "scene_warmup_dets": int(scene_warmup_value),
            "cpu_threads": int(cpu_threads_value),
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
                st.error("Failed to mirror video from S3. Check that the video exists in S3 and you have network connectivity.")
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
        # Force status refresh after job completion to pick up new detect/track status
        st.session_state[_status_force_refresh_key(ep_id)] = True
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
        s3_meta = details.get("s3") or {}
        s3_exists = s3_meta.get("v2_exists") or s3_meta.get("v1_exists")
        if s3_exists:
            st.info("Local mirror missing; Detect/Track will mirror automatically from S3 before starting.")
        else:
            st.warning("Video not found locally or in S3. Upload the video first via the Upload page.")

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
    # Disable button if any job is running (ours or detected from API)
    detect_button_disabled = job_running or detect_status_value == "running" or running_detect_job is not None

    if running_detect_job:
        # Show warning that a job is already running
        st.warning(f"⚠️ A detect/track job is already running ({running_detect_job.get('progress_pct', 0):.1f}% complete). Cancel it above to start a new one.")

    if st.button(run_label, use_container_width=True, disabled=detect_button_disabled):
        # Keep runtime logs anchored just below the button for local mode runs.
        detect_log_container = st.container()
        with detect_log_container:
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

    # Show previous run logs (only in local mode, collapsed by default)
    if helpers.get_execution_mode(ep_id) == "local":
        helpers.render_previous_logs(ep_id, "detect_track", expanded=False)

with col_faces:
    st.markdown("### Faces Harvest")
    st.caption(_format_phase_status("Faces Harvest", faces_phase_status, "faces"))

    # Show running job progress if a job is active
    if running_faces_job:
        job_id = running_faces_job.get("job_id", "unknown")
        progress_pct = running_faces_job.get("progress_pct", 0)
        state = running_faces_job.get("state", "running")

        st.info(f"🔄 **Faces Harvest job running** ({state})")
        st.progress(min(progress_pct / 100, 1.0))
        st.caption(f"Progress: {progress_pct:.1f}%")

        # Refresh and Cancel buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🔄 Refresh", key=f"refresh_faces_{job_id}", use_container_width=True):
                st.rerun()
        with btn_col2:
            if st.button("❌ Cancel", key=f"cancel_faces_{job_id}", use_container_width=True):
                success, msg = helpers.cancel_running_job(job_id)
                if success:
                    st.success(msg)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(msg)

        st.divider()

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
        key=f"{ep_id}::faces_device_choice",
    )
    st.caption("CoreML/GPU strongly recommended for ArcFace embeddings; significantly faster than CPU.")
    faces_device_value = helpers.DEVICE_VALUE_MAP[faces_device_choice]
    faces_save_frames = st.toggle(
        "Save full frames",
        value=bool(faces_save_frames_default),
        help="When off, only face crops are stored; reduces storage and CPU during harvest.",
        key=f"{ep_id}::faces_save_frames_toggle",
    )
    faces_save_crops = st.toggle(
        "Save face crops",
        value=bool(faces_save_crops_default),
        help="Exports aligned face crops for review and embeddings.",
        key=f"{ep_id}::faces_save_crops_toggle",
    )
    faces_min_frames_between_crops = st.number_input(
        "Minimum frames between crops",
        min_value=1,
        max_value=600,
        value=int(faces_min_frames_between_crops_default),
        step=1,
        help="Number of frames between successive crops on the same track; higher values reduce near-duplicate faces.",
        key=f"{ep_id}::faces_min_frames_between_crops",
    )
    st.caption("Default spacing ~1–2s keeps crops lean for laptop runs.")

    faces_thumb_size_default = int(faces_job_defaults.get("thumb_size") or 256)
    quality_options = sorted({60, 70, 80, int(faces_jpeg_quality_default)})
    with st.expander("Advanced face exports", expanded=False):
        faces_jpeg_quality = st.select_slider(
            "Image quality (frames/crops)",
            options=quality_options,
            value=int(faces_jpeg_quality_default),
            help="Lower JPEG quality reduces S3 size; 60–80 is usually sufficient for UI flows.",
            key=f"{ep_id}::faces_jpeg_quality_detail",
        )
        faces_thumb_size = st.number_input(
            "Thumbnail size",
            min_value=64,
            max_value=512,
            value=faces_thumb_size_default,
            step=32,
            key=f"{ep_id}::faces_thumb_size_detail",
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
        s3_meta = details.get("s3") or {}
        s3_exists = s3_meta.get("v2_exists") or s3_meta.get("v1_exists")
        if s3_exists:
            st.info("Local mirror missing; video will be mirrored from S3 automatically when Faces Harvest starts.")
        else:
            st.warning("Video not found locally or in S3. Upload the video first.")
    elif faces_status_value == "stale":
        st.warning(
            "**Harvest Faces is outdated**: Detect/Track was rerun after the last faces harvest.\n\n"
            "Track IDs have changed. Rerun **Faces Harvest** to rebuild embeddings for the new tracks."
        )
    elif not tracks_ready:
        message = (
            "**Harvest Faces is unavailable**: Face detection/tracking has not run yet.\n\n"
            "Run **Detect/Track Faces** first to generate `detections.jsonl` and `tracks.jsonl` for this episode."
        )
        if tracks_only_fallback:
            message = (
                "**Harvest Faces is unavailable**: Tracks exist but detections are missing.\n\n"
                "Run **Detect/Track Faces** again to regenerate detections before harvesting faces."
            )
        st.warning(message)
        if detect_track_info and detect_track_info.get("detector") == "pyscenedetect":
            st.error(
                "⚠️ **Scene detection only**: Your last run only executed scene detection (PySceneDetect), "
                "not full face detection + tracking. Please run **Detect/Track Faces** again to generate tracks."
            )
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
                "with a supported detector/tracker before harvesting."
            )
    elif not combo_supported_harvest:
        current_combo = f"{helpers.detector_label_from_value(combo_detector)} + {helpers.tracker_label_from_value(combo_tracker)}"
        st.error(
            f"Harvest requires a supported detector/tracker combo. Last detect run used **{current_combo}**. "
            "Select a supported combo (e.g., RetinaFace + ByteTrack/StrongSORT) and rerun detect/track."
        )

    faces_disabled = (
        (not tracks_ready)
        or (not detector_face_only)
        or job_running
        or faces_status_value == "running"
        or (not combo_supported_harvest)
        or tracks_only_fallback
        or running_faces_job is not None
    )

    if running_faces_job:
        st.warning(f"⚠️ A faces harvest job is already running ({running_faces_job.get('progress_pct', 0):.1f}% complete). Cancel it above to start a new one.")

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
                "min_frames_between_crops": int(faces_min_frames_between_crops),
                "jpeg_quality": int(faces_jpeg_quality),
                "thumb_size": int(faces_thumb_size),
            }
            st.session_state[running_job_key] = True
            _set_job_active(ep_id, True)
            try:
                # Use execution mode from UI settings (respects local/redis toggle)
                execution_mode = helpers.get_execution_mode(ep_id)
                mode_desc = "local" if execution_mode == "local" else "Celery"
                if execution_mode == "local":
                    # Local mode handles its own UI - no spinner needed
                    summary, error_message = helpers.run_pipeline_job_with_mode(
                        ep_id,
                        "faces_embed",
                        payload,
                        requested_device=faces_device_value,
                        requested_detector=helpers.tracks_detector_value(ep_id),
                        requested_tracker=helpers.tracks_tracker_value(ep_id),
                    )
                else:
                    with st.spinner(f"Running faces harvest via {mode_desc}…"):
                        summary, error_message = helpers.run_pipeline_job_with_mode(
                            ep_id,
                            "faces_embed",
                            payload,
                            requested_device=faces_device_value,
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
                flash_parts = []
                if isinstance(faces_count, int):
                    flash_parts.append(f"faces: {faces_count:,}")
                if crops_exported:
                    flash_parts.append(f"crops exported: {crops_exported:,}")
                flash_parts.append(f"thumb size: {int(faces_thumb_size)}px")
                flash_msg = "Faces harvest complete" + (" · " + ", ".join(flash_parts) if flash_parts else "")
                st.session_state["episode_detail_flash"] = flash_msg
                # Force status refresh after job completion to pick up new faces status
                st.session_state[_status_force_refresh_key(ep_id)] = True
                st.rerun()

    # Show previous run logs (only in local mode, collapsed by default)
    if helpers.get_execution_mode(ep_id) == "local":
        helpers.render_previous_logs(ep_id, "faces_embed", expanded=False)

with col_cluster:
    st.markdown("### Cluster Identities")
    st.caption(_format_phase_status("Cluster Identities", cluster_phase_status, "identities"))

    # Show running job progress if a job is active
    if running_cluster_job:
        job_id = running_cluster_job.get("job_id", "unknown")
        progress_pct = running_cluster_job.get("progress_pct", 0)
        state = running_cluster_job.get("state", "running")

        st.info(f"🔄 **Cluster job running** ({state})")
        st.progress(min(progress_pct / 100, 1.0))
        st.caption(f"Progress: {progress_pct:.1f}%")

        # Refresh and Cancel buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🔄 Refresh", key=f"refresh_cluster_{job_id}", use_container_width=True):
                st.rerun()
        with btn_col2:
            if st.button("❌ Cancel", key=f"cancel_cluster_{job_id}", use_container_width=True):
                success, msg = helpers.cancel_running_job(job_id)
                if success:
                    st.success(msg)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(msg)

        st.divider()

    cluster_device_choice = st.selectbox(
        "Device (for clustering)",
        helpers.DEVICE_LABELS,
        index=helpers.device_label_index(cluster_device_label_default),
        key=f"{ep_id}::cluster_device_choice",
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
        key=f"{ep_id}::cluster_similarity_threshold",
    )
    # Provide threshold guidance based on selected value
    if cluster_thresh_value >= 0.80:
        st.caption("🔴 **Very strict**: May over-split same person into multiple clusters.")
    elif cluster_thresh_value >= 0.70:
        st.caption("🟡 **Strict**: Good for distinguishing similar-looking people.")
    elif cluster_thresh_value >= 0.55:
        st.caption("🟢 **Balanced**: Recommended for most content.")
    else:
        st.caption("🟠 **Lenient**: May merge different people into same cluster.")

    min_cluster_size_value = st.number_input(
        "Minimum tracks per identity",
        min_value=1,
        max_value=50,
        value=int(min_cluster_size_default),
        step=1,
        help="Clusters smaller than this are discarded as noise. Recommended: 2+ for cleaner results.",
        key=f"{ep_id}::cluster_min_tracks_per_identity",
    )
    if min_cluster_size_value == 1:
        st.caption("⚠️ Single-track clusters may contain noise/false detections.")
    if not local_video_exists:
        s3_meta = details.get("s3") or {}
        s3_exists = s3_meta.get("v2_exists") or s3_meta.get("v1_exists")
        if s3_exists:
            st.info("Local mirror missing; artifacts will be mirrored automatically when clustering starts.")
        else:
            st.warning("Video not found locally or in S3. Upload the video first.")
    elif not tracks_ready:
        st.caption("Run detect/track first; clustering requires fresh tracks and faces.")

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
    elif cluster_status_value == "running":
        st.info("Clustering is currently running. Wait for it to complete before starting another run.")

    cluster_disabled = (
        (not faces_ready)
        or (not detector_face_only)
        or (not tracks_ready)
        or job_running
        or zero_faces_success
        or (not combo_supported_cluster)
        or faces_status_value == "stale"
        or cluster_status_value == "running"
        or running_cluster_job is not None
    )

    if running_cluster_job:
        st.warning(f"⚠️ A cluster job is already running ({running_cluster_job.get('progress_pct', 0):.1f}% complete). Cancel it above to start a new one.")

    if st.button("Run Cluster", use_container_width=True, disabled=cluster_disabled):
        can_run_cluster = True
        if not local_video_exists:
            can_run_cluster = _ensure_local_artifacts(ep_id, details)
            if can_run_cluster:
                local_video_exists = True
        # Ensure faces manifest is mirrored locally before clustering
        if can_run_cluster and not faces_path.exists():
            with st.spinner("Mirroring faces artifacts from S3…"):
                try:
                    # Use the new mirror_artifacts endpoint that actually mirrors faces/identities
                    mirror_resp = helpers.api_post(
                        f"/episodes/{ep_id}/mirror_artifacts",
                        json={"artifacts": ["faces", "identities"]},
                    )
                    if mirror_resp.get("faces_manifest_exists"):
                        st.success("Faces manifest mirrored successfully.")
                    else:
                        errors = mirror_resp.get("errors", {})
                        error_msg = errors.get("faces", "Faces manifest not found in S3")
                        st.error(f"Failed to mirror faces: {error_msg}")
                        can_run_cluster = False
                except requests.RequestException as exc:
                    st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/{ep_id}/mirror_artifacts", exc))
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
                # Use execution mode from UI settings (respects local/redis toggle)
                execution_mode = helpers.get_execution_mode(ep_id)
                mode_desc = "local" if execution_mode == "local" else "Celery"
                if execution_mode == "local":
                    # Local mode handles its own UI - no spinner needed
                    summary, error_message = helpers.run_pipeline_job_with_mode(
                        ep_id,
                        "cluster",
                        payload,
                        requested_device=cluster_device_value,
                        requested_detector=helpers.tracks_detector_value(ep_id),
                        requested_tracker=helpers.tracks_tracker_value(ep_id),
                    )
                else:
                    with st.spinner(f"Clustering faces via {mode_desc}…"):
                        summary, error_message = helpers.run_pipeline_job_with_mode(
                            ep_id,
                            "cluster",
                            payload,
                            requested_device=cluster_device_value,
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
                cluster_flash_parts = []
                if isinstance(identities_count, int):
                    cluster_flash_parts.append(f"identities: {identities_count:,}")
                if isinstance(faces_count, int):
                    cluster_flash_parts.append(f"faces: {faces_count:,}")
                flash_msg = f"Clustered (thresh {cluster_thresh_value:.2f}, min {int(min_cluster_size_value)})" + (
                    " · " + ", ".join(cluster_flash_parts) if cluster_flash_parts else ""
                )
                st.session_state["episode_detail_flash"] = flash_msg
                # Force status refresh after job completion to pick up new cluster status
                st.session_state[_status_force_refresh_key(ep_id)] = True
                st.rerun()

    # Show previous run logs (only in local mode, collapsed by default)
    if helpers.get_execution_mode(ep_id) == "local":
        helpers.render_previous_logs(ep_id, "cluster", expanded=False)

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


def _read_json_artifact(path: Path, max_lines: int = 2000) -> tuple[str | None, str | None]:
    """Return (content, error) for a JSON/JSONL artifact with defensive limits."""
    if not path.exists():
        return None, f"{path.name} does not exist."
    try:
        if path.suffix.lower() == ".jsonl":
            lines = []
            with path.open("r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle, start=1):
                    if idx > max_lines:
                        lines.append(f"... truncated after {max_lines} lines ...")
                        break
                    lines.append(line.rstrip("\n"))
            return "\n".join(lines), None
        if path.suffix.lower() == ".json":
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            return json.dumps(payload, indent=2, ensure_ascii=False), None
    except Exception as exc:
        return None, f"Failed to load {path.name}: {exc}"
    return None, f"Unsupported file type for {path.name}"


st.subheader("Debug: Raw JSON artifacts")
artifact_groups = {
    "Detect / Faces / Tracks": [
        get_path(ep_id, "detections"),
        get_path(ep_id, "tracks"),
        faces_path,
    ],
    "Cluster": [
        identities_path,
        manifests_dir / "track_metrics.json",
    ],
    "Screentime": [
        analytics_dir / "screentime.json",
    ],
}
for group, paths in artifact_groups.items():
    existing = [p for p in paths if p.exists()]
    with st.expander(group, expanded=False):
        if not existing:
            st.caption("No artifacts found for this stage.")
            continue
        labels = [p.name for p in existing]
        selected = st.selectbox(
            "Choose artifact",
            labels,
            key=f"{group}::artifact_selector",
        )
        chosen_path = next((p for p in existing if p.name == selected), None)
        if not chosen_path:
            st.caption("Select a file to view its contents.")
            continue
        st.caption(f"Path: {helpers.link_local(chosen_path)}")
        content, err = _read_json_artifact(chosen_path)
        if err:
            st.error(err)
            continue
        st.code(content or "", language="json")


# =============================================================================
# Auto-refresh when jobs are running
# =============================================================================
# If ANY job is running for this episode, auto-refresh every 3 seconds
# to poll for updates. This MUST be at the end of the page so content
# fully renders before the refresh.

_any_job_running = running_detect_job or running_faces_job or running_cluster_job
if _any_job_running:
    import time as _time
    _running_ops = []
    if running_detect_job:
        _running_ops.append("Detect/Track")
    if running_faces_job:
        _running_ops.append("Faces Harvest")
    if running_cluster_job:
        _running_ops.append("Cluster")
    st.caption(f"⏳ Auto-refreshing for running job(s): {', '.join(_running_ops)}...")
    _time.sleep(3)
    st.rerun()

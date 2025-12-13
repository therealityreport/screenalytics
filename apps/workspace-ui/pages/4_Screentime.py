from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, NoReturn

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yaml

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
PROJECT_ROOT = WORKSPACE_DIR.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402


def load_pipeline_configs() -> Dict[str, Any]:
    """Load pipeline config values from YAML files."""
    config_dir = PROJECT_ROOT / "config" / "pipeline"
    configs = {
        "detection": {},
        "tracking": {},
        "embedding": {},
        "clustering": {},
    }

    # Detection config
    det_path = config_dir / "detection.yaml"
    if det_path.exists():
        with det_path.open() as f:
            det = yaml.safe_load(f) or {}
            configs["detection"] = {
                "confidence_th": det.get("confidence_th", 0.50),
                "min_size": det.get("min_size", 16),
            }

    # Tracking config
    trk_path = config_dir / "tracking.yaml"
    if trk_path.exists():
        with trk_path.open() as f:
            trk = yaml.safe_load(f) or {}
            configs["tracking"] = {
                "track_thresh": trk.get("track_thresh", 0.55),
                "match_thresh": trk.get("match_thresh", 0.65),
                "track_buffer": trk.get("track_buffer", 90),
                "new_track_thresh": trk.get("new_track_thresh", 0.60),
            }

    # Embedding/sampling config
    emb_path = config_dir / "faces_embed_sampling.yaml"
    if emb_path.exists():
        with emb_path.open() as f:
            emb = yaml.safe_load(f) or {}
            qg = emb.get("quality_gating", {})
            configs["embedding"] = {
                "min_quality_score": qg.get("min_quality_score", 1.5),
                "min_confidence": qg.get("min_confidence", 0.45),
                "min_blur_score": qg.get("min_blur_score", 18.0),
                "max_yaw_angle": qg.get("max_yaw_angle", 60.0),
                "max_pitch_angle": qg.get("max_pitch_angle", 45.0),
            }

    # Clustering config
    cls_path = config_dir / "clustering.yaml"
    if cls_path.exists():
        with cls_path.open() as f:
            cls = yaml.safe_load(f) or {}
            configs["clustering"] = {
                "cluster_thresh": cls.get("cluster_thresh", 0.52),
                "min_cluster_size": cls.get("min_cluster_size", 1),
                "min_identity_sim": cls.get("min_identity_sim", 0.45),
            }

    return configs


def _load_scene_cuts(ep_id: str) -> dict | None:
    """Load scene cuts from track_metrics.json."""
    metrics_path = helpers.DATA_ROOT / "manifests" / ep_id / "track_metrics.json"
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            return data.get("scene_cuts")
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _load_video_fps(ep_id: str) -> float | None:
    """Load video FPS from video_meta API."""
    try:
        resp = helpers.api_get(f"/episodes/{ep_id}/video_meta")
        return resp.get("fps_detected")
    except Exception:
        return None


def _confidence_to_color(conf: float) -> str:
    """Map confidence score to color (red/yellow/green)."""
    if conf >= 0.9:
        return "#22c55e"  # green
    elif conf >= 0.75:
        return "#eab308"  # yellow
    else:
        return "#ef4444"  # red


def _compute_overlap_regions(timeline_data: list) -> list[tuple[float, float, int]]:
    """Compute time regions where multiple people appear simultaneously.

    Returns list of (start_s, end_s, person_count) for overlapping regions.
    """
    # Collect all interval boundaries as events
    events: list[tuple[float, int]] = []  # (time, +1 for start, -1 for end)
    for person in timeline_data:
        for start_s, end_s in person.get("intervals", []):
            events.append((start_s, 1))
            events.append((end_s, -1))

    if not events:
        return []

    # Sort by time, starts before ends at same time
    events.sort(key=lambda x: (x[0], -x[1]))

    regions: list[tuple[float, float, int]] = []
    count = 0
    prev_time: float | None = None
    for time_val, delta in events:
        if prev_time is not None and count >= 2 and time_val > prev_time:
            regions.append((prev_time, time_val, count))
        count += delta
        prev_time = time_val

    return regions


def _load_tracks(ep_id: str) -> list[dict]:
    """Load track data from tracks.jsonl."""
    tracks_path = helpers.DATA_ROOT / "manifests" / ep_id / "tracks.jsonl"
    if not tracks_path.exists():
        return []
    tracks = []
    try:
        for line in tracks_path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                tracks.append(json.loads(line))
    except (json.JSONDecodeError, OSError):
        return []
    return tracks


def _load_identities(ep_id: str) -> dict:
    """Load cluster/identity data from identities.json."""
    identities_path = helpers.DATA_ROOT / "manifests" / ep_id / "identities.json"
    if not identities_path.exists():
        return {"identities": []}
    try:
        return json.loads(identities_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"identities": []}


def _load_skipped_faces(ep_id: str) -> list[dict]:
    """Load faces that were skipped due to quality issues."""
    faces_path = helpers.DATA_ROOT / "manifests" / ep_id / "faces.jsonl"
    if not faces_path.exists():
        return []
    skipped = []
    try:
        for line in faces_path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                face = json.loads(line)
                if face.get("skip"):
                    skipped.append(face)
    except (json.JSONDecodeError, OSError):
        return []
    return skipped


def _compute_no_face_regions(
    tracks: list[dict], video_duration: float, min_gap: float = 2.0
) -> list[tuple[float, float]]:
    """Compute time regions where no faces are detected (B-roll).

    Args:
        tracks: List of track dicts with first_ts and last_ts
        video_duration: Total video duration in seconds
        min_gap: Minimum gap duration to consider as B-roll

    Returns:
        List of (start_s, end_s) tuples for gaps without faces
    """
    if not tracks or video_duration <= 0:
        return []

    # Collect all track intervals
    intervals = [(t.get("first_ts", 0), t.get("last_ts", 0)) for t in tracks]
    intervals = [(s, e) for s, e in intervals if s is not None and e is not None]
    if not intervals:
        return []

    # Sort by start time
    intervals.sort()

    # Merge overlapping intervals
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Find gaps
    gaps = []
    # Gap at start
    if merged[0][0] > min_gap:
        gaps.append((0, merged[0][0]))
    # Gaps between intervals
    for i in range(len(merged) - 1):
        gap_start = merged[i][1]
        gap_end = merged[i + 1][0]
        if gap_end - gap_start >= min_gap:
            gaps.append((gap_start, gap_end))
    # Gap at end
    if video_duration - merged[-1][1] >= min_gap:
        gaps.append((merged[-1][1], video_duration))

    return gaps


def _get_export_job_info(ep_id: str, job_id_hint: str | None = None) -> dict | None:
    """Return the most relevant overlay export job (succeeded > running > hinted)."""
    try:
        jobs_resp = helpers.api_get(f"/jobs?ep_id={ep_id}&job_type=video_export")
        jobs = jobs_resp.get("jobs", [])
    except requests.RequestException:
        jobs = []

    job_lookup = {j.get("job_id"): j for j in jobs}
    selected = job_lookup.get(job_id_hint) if job_id_hint else None
    if not selected:
        selected = next((j for j in jobs if j.get("state") == "succeeded"), None)
    if not selected:
        selected = next((j for j in jobs if j.get("state") == "running"), None)
    if not selected and job_id_hint:
        selected = job_lookup.get(job_id_hint)
    if not selected:
        return None

    try:
        progress_resp = helpers.api_get(f"/jobs/{selected['job_id']}/progress")
        state = progress_resp.get("state", selected.get("state"))
        progress_data = progress_resp.get("progress", {})
    except requests.RequestException:
        state = selected.get("state")
        progress_data = {}

    url = None
    if isinstance(progress_data, dict):
        url = progress_data.get("url") or progress_data.get("output_path")

    return {
        "job_id": selected.get("job_id"),
        "state": state,
        "progress": progress_data,
        "url": url,
    }


def _ensure_export_job(ep_id: str, include_unidentified: bool = True) -> dict | None:
    """Fetch existing overlay export job or start a new one after screentime completes."""
    export_job_key = f"{ep_id}::overlay_export_job"
    job_hint = st.session_state.get(export_job_key)

    export_info = _get_export_job_info(ep_id, job_hint)
    if export_info:
        st.session_state[export_job_key] = export_info["job_id"]
        return export_info

    try:
        resp = helpers.api_post(
            "/jobs/video_export",
            {
                "ep_id": ep_id,
                "include_unidentified": include_unidentified,
            },
        )
        job_id = resp.get("job_id")
        if job_id:
            st.session_state[export_job_key] = job_id
            return {
                "job_id": job_id,
                "state": "running",
                "progress": {},
                "url": None,
            }
    except requests.RequestException as exc:
        st.warning(helpers.describe_error("video export", exc))

    return None


cfg = helpers.init_page("Screentime")
st.title("Screen Time Analysis")


def _stop_forever() -> NoReturn:
    """Wrapper so type-checkers understand we don't return after st.stop()."""
    st.stop()
    raise RuntimeError("Streamlit stop returned unexpectedly")


def _require_episode() -> str:
    ep_id = helpers.get_ep_id()
    if ep_id:
        return ep_id
    try:
        payload = helpers.api_get("/episodes")
    except requests.RequestException as exc:
        st.error(helpers.describe_error(f"{cfg['api_base']}/episodes", exc))
        _stop_forever()
    episodes = payload.get("episodes", [])
    if not episodes:
        st.info("No episodes yet.")
        _stop_forever()
    option_lookup = {ep["ep_id"]: ep for ep in episodes}
    selection = st.selectbox(
        "Episode",
        list(option_lookup.keys()),
        format_func=lambda eid: f"{eid} ({option_lookup[eid]['show_slug']})",
    )
    if st.button("Load episode", use_container_width=True):
        helpers.set_ep_id(selection)
        return selection
    _stop_forever()


ep_id = _require_episode()
helpers.set_ep_id(ep_id)

analytics_dir = helpers.DATA_ROOT / "analytics" / ep_id
json_path = analytics_dir / "screentime.json"
csv_path = analytics_dir / "screentime.csv"

# Job history section
st.subheader("Recent Jobs")
try:
    jobs_resp = helpers.api_get(f"/jobs?ep_id={ep_id}&job_type=screen_time_analyze")
    jobs_list = jobs_resp.get("jobs", [])
    if jobs_list:
        job_table = []
        for job in jobs_list[:5]:  # Show last 5 jobs
            started_at = job.get("started_at") or "N/A"
            ended_at = job.get("ended_at")
            job_table.append(
                {
                    "Job ID": (job.get("job_id") or "unknown")[:8] + "...",
                    "Status": job.get("state", "unknown"),
                    "Started": started_at[:19].replace("T", " ") if len(started_at) >= 19 else started_at,
                    "Ended": ended_at[:19].replace("T", " ") if ended_at and len(ended_at) >= 19 else ("Running" if not ended_at else ended_at),
                }
            )
        st.dataframe(job_table, use_container_width=True, hide_index=True)
    else:
        st.info("No screen time jobs have been run yet for this episode.")
except requests.RequestException as e:
    st.warning(f"Could not fetch job history: {e}")

st.divider()

# Configuration options - preset defaults from config/pipeline/screen_time_v2.yaml (bravo_default)
PRESET_DEFAULTS = {
    "bravo_default": {
        "quality_min": 0.48,
        "gap_tolerance_s": 1.2,
        "use_video_decode": True,
        "screen_time_mode": "tracks",
        "edge_padding_s": 0.2,
        "track_coverage_min": 0.35,
    },
    "stricter": {
        "quality_min": 0.65,
        "gap_tolerance_s": 0.6,
        "use_video_decode": True,
        "screen_time_mode": "faces",
        "edge_padding_s": 0.05,
        "track_coverage_min": 0.5,
    },
    "strict": {
        "quality_min": 0.48,
        "gap_tolerance_s": 0.1,
        "use_video_decode": True,
        "screen_time_mode": "tracks",
        "edge_padding_s": 0.0,
        "track_coverage_min": 0.6,
    },
}
DEFAULT_PRESET = "bravo_default"

with st.expander("Analysis Configuration", expanded=False):
    # Preset selector
    preset = st.selectbox(
        "Preset",
        options=list(PRESET_DEFAULTS.keys()),
        index=0,
        help="Select a configuration preset. 'bravo_default' is optimized for reality TV with quick cuts.",
    )
    preset_values = PRESET_DEFAULTS.get(preset, PRESET_DEFAULTS[DEFAULT_PRESET])

    st.caption(f"Using preset: **{preset}**")

    col1, col2 = st.columns(2)
    with col1:
        quality_min = st.slider(
            "Minimum face quality threshold",
            min_value=0.0,
            max_value=1.0,
            value=preset_values["quality_min"],
            step=0.05,
            help="Only face samples with quality >= this value will be counted",
        )
        gap_tolerance_s = st.number_input(
            "Gap tolerance (seconds)",
            min_value=0.0,
            max_value=5.0,
            value=preset_values["gap_tolerance_s"],
            step=0.1,
            help="Maximum gap between face samples to be considered continuous",
        )
        use_video_decode = st.checkbox(
            "Use video decode for timestamps",
            value=preset_values["use_video_decode"],
            help="Use video decoding for accurate timestamps (slower but more accurate)",
        )

    with col2:
        screen_time_mode = st.selectbox(
            "Screen time mode",
            options=["tracks", "faces"],
            index=0 if preset_values["screen_time_mode"] == "tracks" else 1,
            help="'tracks' uses full track spans (recommended), 'faces' uses per-face intervals",
        )
        edge_padding_s = st.number_input(
            "Edge padding (seconds)",
            min_value=0.0,
            max_value=1.0,
            value=preset_values["edge_padding_s"],
            step=0.05,
            help="Pad each interval to better match human 'in/out' perception",
        )
        track_coverage_min = st.slider(
            "Track coverage minimum",
            min_value=0.0,
            max_value=1.0,
            value=preset_values["track_coverage_min"],
            step=0.05,
            help="Require at least this detection coverage for tracks (only in 'tracks' mode)",
        )

# Job launch section
col1, col2 = st.columns([3, 1])
with col1:
    if st.button("Analyze Screen Time", use_container_width=True, type="primary"):
        try:
            payload = {
                "ep_id": ep_id,
                "quality_min": quality_min,
                "gap_tolerance_s": gap_tolerance_s,
                "use_video_decode": use_video_decode,
                "screen_time_mode": screen_time_mode,
                "edge_padding_s": edge_padding_s,
                "track_coverage_min": track_coverage_min,
                "preset": preset,
            }
            resp = helpers.api_post("/jobs/screen_time/analyze", payload)
            if not resp or not resp.get("job_id"):
                st.error("Failed to start job: No job ID returned from API")
            else:
                job_id = resp.get("job_id")
                st.session_state[f"{ep_id}::current_screentime_job"] = job_id
                # Kick off overlay export in parallel so Timestamp Search has a fresh video
                _ensure_export_job(ep_id)
                st.success(f"Screen time analysis job started: {job_id[:12]}...")
                st.rerun()
        except requests.RequestException as exc:
            st.error(helpers.describe_error(f"{cfg['api_base']}/jobs/screen_time/analyze", exc))
with col2:
    if st.button("Refresh", use_container_width=True):
        st.rerun()

# Progress monitoring section
# Screen time job phases: init -> loading -> analyzing -> writing -> done
SCREEN_TIME_PHASES = ["init", "loading", "analyzing", "writing", "done"]
PHASE_LABELS = {
    "init": "Initializing...",
    "loading": "Loading episode data...",
    "analyzing": "Analyzing screen time...",
    "writing": "Writing outputs...",
    "done": "Complete",
    "error": "Error",
}

current_job_id = st.session_state.get(f"{ep_id}::current_screentime_job")
if current_job_id:
    try:
        job_progress_resp = helpers.api_get(f"/jobs/{current_job_id}/progress")
        job_state = job_progress_resp.get("state")
        progress_data = job_progress_resp.get("progress", {})

        if job_state == "running":
            st.info(f"Job {current_job_id[:12]}... is currently running")

            # Show progress based on phase (screen time jobs use phase-based progress)
            if progress_data:
                phase = progress_data.get("phase", "init")
                message = progress_data.get("message", "")
                cast_count = progress_data.get("cast_count")

                # Calculate progress based on phase
                if phase in SCREEN_TIME_PHASES:
                    phase_idx = SCREEN_TIME_PHASES.index(phase)
                    progress_pct = (phase_idx + 1) / len(SCREEN_TIME_PHASES)
                else:
                    progress_pct = 0.1

                st.progress(progress_pct)

                col1, col2 = st.columns(2)
                col1.metric("Phase", PHASE_LABELS.get(phase, phase.title()))
                if cast_count is not None:
                    col2.metric("Cast Members Found", cast_count)
                else:
                    col2.metric("Status", message[:50] if message else "Processing...")

            # Auto-refresh every 2 seconds
            time.sleep(2)
            st.rerun()

        elif job_state in ("succeeded", "failed"):
            if job_state == "succeeded":
                st.success(f"Job {current_job_id[:12]}... completed successfully!")
                export_info = _ensure_export_job(ep_id)
                if export_info:
                    export_state = export_info.get("state")
                    export_progress = export_info.get("progress", {}) or {}
                    export_url = export_info.get("url")
                    export_job_id = export_info.get("job_id", "")
                    if export_state == "running":
                        st.info(f"Overlay export {export_job_id[:12]}... is running")
                        percent = export_progress.get("percent")
                        if isinstance(percent, (int, float)):
                            st.progress(max(0.0, min(1.0, percent / 100)))
                        phase = export_progress.get("phase")
                        message = export_progress.get("message")
                        if message or phase:
                            st.caption(f"{phase or 'encoding'}: {message or 'processing...'}")
                    elif export_state == "succeeded":
                        st.success("Overlay video ready.")
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            if st.button("View Overlays in Interactive Viewer", key=f"{ep_id}::open_overlay_iv", use_container_width=True):
                                if export_url:
                                    st.session_state[f"{ep_id}::interactive_video_url"] = export_url
                                st.switch_page("pages/9_Interactive_Viewer.py")
                        with c2:
                            if export_url:
                                st.markdown(
                                    f'<a href="{export_url}" target="_blank" style="'
                                    f'display:inline-block;padding:10px 16px;width:100%;text-align:center;'
                                    f'background:#1f6feb;color:white;text-decoration:none;border-radius:5px;">'
                                    f'Open Overlay Video</a>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.caption("Overlay URL not available yet.")
                    elif export_state == "failed":
                        err_msg = export_progress.get("message", "Overlay export failed")
                        st.error(err_msg)
                    else:
                        st.caption(f"Overlay export status: {export_state}")
            else:
                error_msg = progress_data.get("message", "Unknown error") if progress_data else "Unknown error"
                st.error(f"Job {current_job_id[:12]}... failed: {error_msg}")

            # Clear the current job from session state
            if st.button("Clear Job Status", key=f"{ep_id}::clear_job_1"):
                st.session_state.pop(f"{ep_id}::current_screentime_job", None)
                st.rerun()

    except requests.RequestException:
        # Job not found or API error - clear from session
        if st.button("Clear Job Status", key=f"{ep_id}::clear_job_2"):
            st.session_state.pop(f"{ep_id}::current_screentime_job", None)
            st.rerun()

st.divider()

if json_path.exists():
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    st.subheader("Screen Time Results")

    # Display metadata
    generated_at = data.get("generated_at", "unknown")
    st.caption(f"Generated: {generated_at[:19].replace('T', ' ')}")
    metadata = data.get("metadata") or {}
    diagnostics = data.get("diagnostics") or {}
    body_tracking_enabled = bool(metadata.get("body_tracking_enabled", False))
    body_metrics_available = bool(metadata.get("body_metrics_available", False))
    speaking_time_computed = bool(diagnostics.get("speaking_time_computed", False))
    st.caption(
        "Metadata: "
        f"body_tracking_enabled={body_tracking_enabled} · "
        f"body_metrics_available={body_metrics_available} · "
        f"speaking_time_computed={speaking_time_computed}"
    )

    # Overlay export quick actions tied to last generated results
    export_info = _ensure_export_job(ep_id)
    export_url = export_info.get("url") if export_info else None
    export_state = export_info.get("state") if export_info else None
    export_job_id = export_info.get("job_id") if export_info else None
    with st.expander("Overlay Video", expanded=True):
        if export_state == "running":
            st.info(f"Overlay export {export_job_id[:12]}... running")
            percent = (export_info.get("progress") or {}).get("percent")
            if isinstance(percent, (int, float)):
                st.progress(max(0.0, min(1.0, percent / 100)))
        elif export_state == "succeeded":
            st.success("Overlay video ready for Interactive Viewer.")
        elif export_state == "failed":
            st.error("Overlay export failed; click Analyze again to retry.")
        if st.button("View Interactive Viewer", key=f"{ep_id}::results_overlay_iv", use_container_width=True):
            if export_url:
                st.session_state[f"{ep_id}::interactive_video_url"] = export_url
            st.switch_page("pages/9_Interactive_Viewer.py")
        if export_url:
            st.caption(f"Overlay: {export_url}")
        elif export_job_id:
            st.caption(f"Overlay job: {export_job_id[:12]}...")

    metrics = data.get("metrics", [])
    if metrics:
        # Summary stats at the top
        def _face_visible_seconds(metric: dict) -> float:
            value = metric.get("face_visible_seconds")
            if value is None:
                value = metric.get("visual_s", 0.0)
            return float(value or 0.0)

        def _safe_seconds(metric: dict, key: str) -> float:
            value = metric.get(key)
            return float(value or 0.0) if isinstance(value, (int, float)) else 0.0

        total_time = sum(_face_visible_seconds(m) for m in metrics)
        total_speaking = sum(_safe_seconds(m, "speaking_s") for m in metrics)
        total_body_visible = (
            sum(_safe_seconds(m, "body_visible_seconds") for m in metrics) if body_metrics_available else 0.0
        )
        total_body_only = (
            sum(_safe_seconds(m, "body_only_seconds") for m in metrics) if body_metrics_available else 0.0
        )
        total_tracks = sum(m.get("tracks_count", 0) for m in metrics)
        total_faces = sum(m.get("faces_count", 0) for m in metrics)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Face Visible", f"{total_time:.1f}s ({total_time / 60:.1f} min)")
        col2.metric("Cast Members", len(metrics))
        col3.metric("Total Speaking", f"{total_speaking:.1f}s")
        col4.metric("Total Faces", total_faces)
        if body_metrics_available:
            st.caption(
                f"Body metrics: total_body_visible={total_body_visible:.1f}s · total_body_only={total_body_only:.1f}s"
            )

        st.divider()

        # Helper function to format seconds as SS.MS (e.g., 19.12 = 19 seconds, 12 hundredths)
        def format_time(seconds):
            return f"{seconds:.2f}"

        def parse_time_to_seconds(time_str: str) -> float:
            """Parse time string to seconds.

            Supported formats:
            - SS.MS: 19.12 = 19.12 seconds (decimal seconds - primary format)
            - HH:MM:SS: 01:30:45 = 5445 seconds
            - MM:SS: 01:30 = 90 seconds
            """
            time_str = time_str.strip()
            if not time_str or time_str == "-":
                return 0.0

            # Parse time formats with colons (HH:MM:SS or MM:SS)
            if ":" in time_str:
                parts = time_str.split(":")
                if len(parts) == 3:
                    # HH:MM:SS format
                    try:
                        hours = int(parts[0])
                        minutes = int(parts[1])
                        seconds = float(parts[2])
                        return hours * 3600 + minutes * 60 + seconds
                    except ValueError:
                        pass
                elif len(parts) == 2:
                    # MM:SS or MM:SS.ms format
                    try:
                        minutes = int(parts[0])
                        seconds = float(parts[1])
                        return minutes * 60 + seconds
                    except ValueError:
                        pass

            # Try as decimal seconds (e.g., "19.12" = 19.12 seconds)
            try:
                return float(time_str)
            except ValueError:
                pass

            return 0.0

        # File-based persistence for manual screentime values
        manual_json_path = analytics_dir / "manual_screentime.json"

        def save_manual_values(values: dict) -> None:
            """Save manual screentime values to file."""
            analytics_dir.mkdir(parents=True, exist_ok=True)
            with manual_json_path.open("w", encoding="utf-8") as f:
                json.dump(values, f, indent=2)

        def load_manual_values() -> dict:
            """Load manual screentime values from file."""
            if manual_json_path.exists():
                try:
                    with manual_json_path.open("r", encoding="utf-8") as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError):
                    return {}
            return {}

        def clear_manual_values() -> None:
            """Delete manual screentime values file."""
            if manual_json_path.exists():
                manual_json_path.unlink()

        # Load manual values from file into session state if not already loaded
        if f"{ep_id}::manual_screentime" not in st.session_state:
            persisted = load_manual_values()
            if persisted:
                st.session_state[f"{ep_id}::manual_screentime"] = persisted

        # Manual screentime comparison section
        st.subheader("Manual Comparison")

        # Show status if manual values are loaded
        if manual_json_path.exists():
            st.caption(f"✅ Manual values loaded from `{manual_json_path.name}`")

        with st.expander("Upload Manual Screentime Values", expanded=False):
            st.caption("Upload a CSV with 'Name' and 'Time' columns, or paste values below.")
            st.caption("Time format: SS.MS (e.g., 19.12 = 19.12 seconds) or MM:SS")

            uploaded_file = st.file_uploader(
                "Upload CSV",
                type=["csv"],
                key=f"{ep_id}::manual_csv_upload",
                help="CSV should have 'Name' and 'Time' columns",
            )

            if uploaded_file is not None:
                try:
                    manual_df = pd.read_csv(uploaded_file)
                    if "Name" in manual_df.columns and "Time" in manual_df.columns:
                        manual_values = {}
                        for _, row in manual_df.iterrows():
                            name = str(row["Name"]).strip()
                            time_val = row["Time"]
                            if isinstance(time_val, str):
                                manual_values[name] = parse_time_to_seconds(time_val)
                            else:
                                manual_values[name] = float(time_val)
                        st.session_state[f"{ep_id}::manual_screentime"] = manual_values
                        save_manual_values(manual_values)  # Persist to file
                        st.success(f"Loaded and saved {len(manual_values)} manual values")
                    else:
                        st.error("CSV must have 'Name' and 'Time' columns")
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

            st.divider()

            # Manual text input for quick entry
            st.caption("Or paste values (one per line: Name, Time)")
            manual_text = st.text_area(
                "Manual values",
                value="",
                height=150,
                placeholder="BRONWYN 19.12\nWHITNEY 21.48\nANGIE 16.00\n...",
                key=f"{ep_id}::manual_text_input",
            )

            if st.button("Apply Manual Values", key=f"{ep_id}::apply_manual"):
                manual_values = {}
                for line in manual_text.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    # Parse "Name, Time" or "Name\tTime" or "Name  Time" format
                    parts = None
                    if "," in line:
                        parts = line.split(",", 1)
                    elif "\t" in line:
                        parts = line.split("\t", 1)
                    else:
                        # Try splitting by 2+ spaces or last space before time pattern
                        import re
                        # Match: NAME followed by spaces then MM:SS, M:SS, or MM.SS
                        match = re.match(r'^(.+?)\s+(\d{1,2}[:.]\d{2}(?:\.\d+)?)$', line)
                        if match:
                            parts = [match.group(1), match.group(2)]
                    if parts and len(parts) == 2:
                        name = parts[0].strip()
                        time_str = parts[1].strip()
                        manual_values[name] = parse_time_to_seconds(time_str)
                if manual_values:
                    st.session_state[f"{ep_id}::manual_screentime"] = manual_values
                    save_manual_values(manual_values)  # Persist to file
                    # Show what was parsed for verification
                    parsed_summary = ", ".join([f"{k}: {v:.0f}s" for k, v in list(manual_values.items())[:5]])
                    if len(manual_values) > 5:
                        parsed_summary += f"... (+{len(manual_values) - 5} more)"
                    st.success(f"Saved {len(manual_values)} values: {parsed_summary}")
                    st.rerun()
                else:
                    st.warning("No valid values found. Use format: Name, Time (e.g., ANGIE 16.00 for 16 seconds)")

            if st.button("Clear Manual Values", key=f"{ep_id}::clear_manual"):
                st.session_state.pop(f"{ep_id}::manual_screentime", None)
                clear_manual_values()  # Delete file
                st.success("Manual values cleared")
                st.rerun()

        # Get manual values from session state
        manual_screentime = st.session_state.get(f"{ep_id}::manual_screentime", {})

        def find_manual_value(cast_name: str, manual_dict: dict) -> float:
            """Find manual value with flexible matching (exact, case-insensitive, first name)."""
            if not manual_dict or not cast_name:
                return 0.0
            # Exact match
            if cast_name in manual_dict:
                return manual_dict[cast_name]
            # Case-insensitive exact match
            cast_lower = cast_name.lower()
            for key, val in manual_dict.items():
                if key.lower() == cast_lower:
                    return val
            # First name match (case-insensitive)
            cast_first = cast_name.split()[0].lower() if cast_name else ""
            for key, val in manual_dict.items():
                key_lower = key.lower()
                # Manual key matches cast first name
                if key_lower == cast_first:
                    return val
                # Cast first name matches manual key's first word
                key_first = key.split()[0].lower() if key else ""
                if key_first == cast_first:
                    return val
            return 0.0

        # Prepare data for display
        display_rows = []
        has_speaking = any(_safe_seconds(m, "speaking_s") > 0 for m in metrics)
        for m in metrics:
            face_s = _face_visible_seconds(m)
            speaking_s = _safe_seconds(m, "speaking_s")
            body_visible_s = m.get("body_visible_seconds") if body_metrics_available else None
            body_only_s = m.get("body_only_seconds") if body_metrics_available else None
            gap_bridged_s = m.get("gap_bridged_seconds") if body_metrics_available else None
            name = m.get("name", "Unknown")

            row = {
                "Name": name,
                "Face Visible": format_time(face_s),
                "Percentage": f"{(face_s / max(total_time, 1)) * 100:.1f}%",
                "Speaking": format_time(speaking_s) if has_speaking else None,
                "Tracks": m.get("tracks_count", 0),
                "Faces": m.get("faces_count", 0),
                "Confidence": f"{m.get('confidence', 0.0):.3f}",
                "_face_s": face_s,  # Hidden column for sorting
            }
            if body_metrics_available:
                row["Body Visible"] = format_time(body_visible_s) if isinstance(body_visible_s, (int, float)) else "-"
                row["Body Only"] = format_time(body_only_s) if isinstance(body_only_s, (int, float)) else "-"
                row["Gap Bridged"] = format_time(gap_bridged_s) if isinstance(gap_bridged_s, (int, float)) else "-"

            # Add manual comparison columns if manual values exist
            if manual_screentime:
                manual_s = find_manual_value(name, manual_screentime)
                if manual_s > 0:
                    difference = face_s - manual_s
                    error_pct = abs(difference) / manual_s * 100
                    row["Manual"] = format_time(manual_s)
                    row["Diff"] = f"{difference:+.1f}s"
                    row["Error %"] = f"{error_pct:.1f}%"
                    row["_manual_s"] = manual_s
                    row["_error_pct"] = error_pct
                else:
                    row["Manual"] = "-"
                    row["Diff"] = "-"
                    row["Error %"] = "-"
                    row["_manual_s"] = 0.0
                    row["_error_pct"] = 0.0

            display_rows.append(row)

        # Convert to DataFrame for better display
        df = pd.DataFrame(display_rows)

        # Show summary stats if manual values present
        if manual_screentime:
            matched_count = sum(1 for r in display_rows if r.get("_manual_s", 0) > 0)
            avg_error = sum(r.get("_error_pct", 0) for r in display_rows if r.get("_manual_s", 0) > 0)
            if matched_count > 0:
                avg_error /= matched_count
            st.info(f"Comparing {matched_count}/{len(display_rows)} cast members | Average Error: {avg_error:.1f}%")

        # Interactive table with sorting
        st.subheader("Per-Cast Breakdown")
        sort_options = ["Face Visible", "Percentage", "Tracks", "Faces", "Confidence", "Name"]
        if has_speaking:
            sort_options.insert(2, "Speaking")
        if body_metrics_available:
            sort_options.extend(["Body Visible", "Body Only", "Gap Bridged"])
        if manual_screentime:
            sort_options.extend(["Manual", "Error %"])

        sort_by = st.selectbox(
            "Sort by",
            sort_options,
            index=0,
        )
        ascending = st.checkbox("Ascending order", value=False)

        # Sort the dataframe
        if sort_by == "Face Visible":
            df_sorted = df.sort_values("_face_s", ascending=ascending)
        elif sort_by == "Manual" and "_manual_s" in df.columns:
            df_sorted = df.sort_values("_manual_s", ascending=ascending)
        elif sort_by == "Error %" and "_error_pct" in df.columns:
            df_sorted = df.sort_values("_error_pct", ascending=ascending)
        else:
            df_sorted = df.sort_values(sort_by, ascending=ascending)

        # Display without hidden columns, in specific order
        hidden_cols = ["_face_s", "_manual_s", "_error_pct"]
        if manual_screentime:
            desired_order = ["Name", "Face Visible", "Manual", "Diff", "Error %", "Percentage"]
        else:
            desired_order = ["Name", "Face Visible", "Percentage"]
        if has_speaking:
            desired_order.append("Speaking")
        if body_metrics_available:
            desired_order.extend(["Body Visible", "Body Only", "Gap Bridged"])
        desired_order.extend(["Tracks", "Faces", "Confidence"])
        display_cols = [c for c in desired_order if c in df_sorted.columns and c not in hidden_cols]
        display_df = df_sorted[display_cols]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.divider()

        # Charts section
        st.subheader("Visualizations")

        # Tab layout for different chart types
        tab1, tab2, tab3 = st.tabs(["Bar Chart", "Pie Chart", "Time Distribution"])

        with tab1:
            # Bar chart of screen time - use name for display
            chart_data = pd.DataFrame(
                {
                    "Cast Member": [m.get("name", m.get("person_id", "unknown")) for m in metrics],
                    "Face Visible (s)": [_face_visible_seconds(m) for m in metrics],
                }
            ).sort_values("Face Visible (s)", ascending=False)

            st.bar_chart(chart_data.set_index("Cast Member"), height=400)

        with tab2:
            # Pie chart showing percentage distribution
            pie_data = pd.DataFrame(
                {
                    "Cast Member": [m.get("name", m.get("person_id", "unknown")) for m in metrics],
                    "Face Visible": [_face_visible_seconds(m) for m in metrics],
                }
            )

            fig = px.pie(
                pie_data,
                values="Face Visible",
                names="Cast Member",
                title="Face Visible Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # Scatter plot: Tracks vs Faces
            scatter_data = pd.DataFrame(
                {
                    "Cast Member": [m.get("name", m.get("person_id", "unknown")) for m in metrics],
                    "Tracks": [m.get("tracks_count", 0) for m in metrics],
                    "Faces": [m.get("faces_count", 0) for m in metrics],
                    "Face Visible (s)": [_face_visible_seconds(m) for m in metrics],
                }
            )

            fig2 = px.scatter(
                scatter_data,
                x="Tracks",
                y="Faces",
                size="Face Visible (s)",
                hover_data=["Cast Member"],
                title="Tracks vs Faces (bubble size = face visible)",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # Timeline Visualization section
        st.subheader("Timeline Visualization")

        # Load scene cuts and FPS data
        scene_cuts_data = _load_scene_cuts(ep_id)
        video_fps = _load_video_fps(ep_id) or 24.0  # fallback to 24fps
        scene_cuts_available = (
            scene_cuts_data is not None
            and scene_cuts_data.get("detector") != "off"
            and scene_cuts_data.get("count", 0) > 0
        )

        # Load additional data for advanced options
        tracks_data = _load_tracks(ep_id)
        identities_data = _load_identities(ep_id)
        skipped_faces_data = _load_skipped_faces(ep_id)
        video_duration = 0.0
        try:
            video_meta = helpers.api_get(f"/episodes/{ep_id}/video_meta")
            video_duration = video_meta.get("duration_sec", 0) or 0
        except Exception:
            pass

        # Display option checkboxes (2x2 grid)
        col1, col2 = st.columns(2)
        with col1:
            show_cast_intervals = st.checkbox(
                "Show Cast Intervals",
                value=True,
                key=f"{ep_id}::timeline_show_cast",
            )
            show_confidence_heatmap = st.checkbox(
                "Color by Confidence",
                value=False,
                key=f"{ep_id}::timeline_confidence_heatmap",
                help="Color bars by detection confidence (red=low, yellow=medium, green=high)",
            )
        with col2:
            show_scene_cuts = st.checkbox(
                "Show Scene Cuts",
                value=scene_cuts_available,
                disabled=not scene_cuts_available,
                key=f"{ep_id}::timeline_show_cuts",
                help="Scene cuts from PySceneDetect" if scene_cuts_available else "No scene cuts detected for this episode",
            )
            show_multi_person = st.checkbox(
                "Show Multi-Person Regions",
                value=False,
                key=f"{ep_id}::timeline_multi_person",
                help="Highlight time regions where multiple people appear together",
            )

        # Advanced timeline options
        with st.expander("Advanced Timeline Options", expanded=False):
            adv_col1, adv_col2 = st.columns(2)
            with adv_col1:
                show_track_intervals = st.checkbox(
                    "Show Track Intervals",
                    value=False,
                    key=f"{ep_id}::timeline_show_tracks",
                    help="Show individual face tracks (different color per track)",
                    disabled=not tracks_data,
                )
                show_cluster_intervals = st.checkbox(
                    "Show Cluster Intervals",
                    value=False,
                    key=f"{ep_id}::timeline_show_clusters",
                    help="Show clusters/identities (tracks grouped by person)",
                    disabled=not identities_data.get("identities"),
                )
            with adv_col2:
                show_broll_regions = st.checkbox(
                    "Show B-Roll / No-Face Regions",
                    value=False,
                    key=f"{ep_id}::timeline_show_broll",
                    help="Highlight time regions with no detected faces",
                    disabled=not tracks_data or not video_duration,
                )
                show_skipped_frames = st.checkbox(
                    "Show Skipped Frames",
                    value=False,
                    key=f"{ep_id}::timeline_show_skipped",
                    help="Show frames skipped due to quality issues (blur, pose, etc.)",
                    disabled=not skipped_faces_data,
                )

            # Info about available data
            st.caption(
                f"Tracks: {len(tracks_data)} | "
                f"Clusters: {len(identities_data.get('identities', []))} | "
                f"Skipped: {len(skipped_faces_data)} | "
                f"Duration: {video_duration:.1f}s"
            )

        # Build confidence lookup from metrics
        confidence_by_name: dict[str, float] = {}
        for m in metrics:
            name = m.get("name", m.get("person_id", "unknown"))
            confidence_by_name[name] = m.get("confidence", 0.5)

        # Timeline / Gantt chart showing when each person appears
        timeline_data = data.get("timeline", [])
        if timeline_data or (show_scene_cuts and scene_cuts_available):
            # Build Gantt-style chart data
            fig_timeline = go.Figure()

            # Color palette for cast members
            colors = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel1
            unassigned_color = "#888888"  # Gray for unassigned

            # Add each person's intervals as horizontal bars
            if show_cast_intervals and timeline_data:
                for idx, person_data in enumerate(timeline_data):
                    name = person_data.get("name", "Unknown")
                    intervals = person_data.get("intervals", [])
                    cast_id = person_data.get("cast_id")

                    # Determine bar color
                    if show_confidence_heatmap:
                        # Color by confidence score
                        conf = confidence_by_name.get(name, 0.5)
                        color = _confidence_to_color(conf)
                    elif cast_id is None:
                        # Gray for unassigned
                        color = unassigned_color
                    else:
                        # Cycle through color palette
                        color = colors[idx % len(colors)]

                    for interval in intervals:
                        start_s, end_s = interval
                        # Ensure minimum visible width
                        duration = max(end_s - start_s, 0.5)

                        fig_timeline.add_trace(go.Bar(
                            x=[duration],
                            y=[name],
                            base=[start_s],
                            orientation="h",
                            marker=dict(color=color),
                            name=name,
                            showlegend=False,
                            hovertemplate=(
                                f"<b>{name}</b><br>"
                                f"Start: %{{base:.1f}}s<br>"
                                f"End: %{{customdata[0]:.1f}}s<br>"
                                f"Duration: %{{x:.1f}}s"
                                "<extra></extra>"
                            ),
                            customdata=[[end_s]],
                        ))

            # Add scene cuts as vertical lines
            if show_scene_cuts and scene_cuts_available and scene_cuts_data:
                scene_indices = scene_cuts_data.get("indices", [])
                for frame_idx in scene_indices:
                    time_s = frame_idx / video_fps
                    fig_timeline.add_vline(
                        x=time_s,
                        line_dash="dash",
                        line_color="red",
                        line_width=1,
                        opacity=0.7,
                    )

            # Add multi-person regions as shaded areas
            if show_multi_person and timeline_data:
                overlap_regions = _compute_overlap_regions(timeline_data)
                for start_s, end_s, person_count in overlap_regions:
                    # Color intensity by count: 2=light, 3+=darker
                    opacity = min(0.4, 0.15 + (person_count - 2) * 0.1)
                    fig_timeline.add_vrect(
                        x0=start_s,
                        x1=end_s,
                        fillcolor="purple",
                        opacity=opacity,
                        line_width=0,
                        layer="below",
                    )

            # Add B-Roll / No-Face regions as gray shaded areas
            if show_broll_regions and tracks_data and video_duration > 0:
                no_face_regions = _compute_no_face_regions(tracks_data, video_duration)
                for start_s, end_s in no_face_regions:
                    fig_timeline.add_vrect(
                        x0=start_s,
                        x1=end_s,
                        fillcolor="gray",
                        opacity=0.25,
                        line_width=0,
                        layer="below",
                    )

            # Track color palette for individual tracks
            track_colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24

            # Build cluster-to-track mapping for cluster intervals
            cluster_track_map: dict[str, list[dict]] = {}
            if identities_data.get("identities"):
                for identity in identities_data["identities"]:
                    cluster_id = identity.get("cluster_id", "unknown")
                    track_ids = identity.get("track_ids", [])
                    cluster_track_map[cluster_id] = [
                        t for t in tracks_data if t.get("track_id") in track_ids
                    ]

            # Add individual track intervals (grouped by cast member, one row per cast)
            cast_tracks: dict[str, list[dict]] = {}  # Initialize for row count calculation
            if show_track_intervals and tracks_data:
                # Build track -> cast member mapping
                # First: track_id -> identity_id
                track_to_identity: dict[int, str] = {}
                identity_to_name: dict[str, str] = {}
                if identities_data.get("identities"):
                    for identity in identities_data["identities"]:
                        identity_id = identity.get("identity_id") or identity.get("cluster_id")
                        person_id = identity.get("person_id")
                        for tid in identity.get("track_ids", []):
                            track_to_identity[int(tid)] = identity_id

                # Then: identity_id -> cast name (from timeline_data)
                if timeline_data:
                    for person_data in timeline_data:
                        cast_id = person_data.get("cast_id")
                        name = person_data.get("name", "Unknown")
                        # Match by cast_id prefix in identity
                        if cast_id:
                            for identity in identities_data.get("identities", []):
                                identity_id = identity.get("identity_id") or identity.get("cluster_id")
                                person_id = identity.get("person_id")
                                if person_id and cast_id in str(person_id):
                                    identity_to_name[identity_id] = name

                # Group tracks by cast member name
                cast_tracks: dict[str, list[dict]] = {}
                unassigned_tracks: list[dict] = []
                for track in tracks_data:
                    track_id = track.get("track_id")
                    if track_id is None:
                        continue
                    identity_id = track_to_identity.get(int(track_id))
                    cast_name = identity_to_name.get(identity_id, None) if identity_id else None

                    if cast_name:
                        if cast_name not in cast_tracks:
                            cast_tracks[cast_name] = []
                        cast_tracks[cast_name].append(track)
                    else:
                        unassigned_tracks.append(track)

                # Add "Unassigned" group if there are unassigned tracks
                if unassigned_tracks:
                    cast_tracks["Unassigned (Tracks)"] = unassigned_tracks

                # Render one row per cast member with different colors per track
                track_color_idx = 0
                for cast_name, cast_track_list in cast_tracks.items():
                    y_label = f"{cast_name} (Tracks)"

                    for track in cast_track_list:
                        track_id = track.get("track_id", "?")
                        first_ts = track.get("first_ts", 0) or 0
                        last_ts = track.get("last_ts", 0) or 0
                        duration = max(last_ts - first_ts, 0.5)

                        fig_timeline.add_trace(go.Bar(
                            x=[duration],
                            y=[y_label],
                            base=[first_ts],
                            orientation="h",
                            marker=dict(color=track_colors[track_color_idx % len(track_colors)]),
                            name=y_label,
                            showlegend=False,
                            hovertemplate=(
                                f"<b>{cast_name}</b><br>"
                                f"Track: {track_id}<br>"
                                f"Start: %{{base:.1f}}s<br>"
                                f"End: %{{customdata[0]:.1f}}s<br>"
                                f"Duration: %{{x:.1f}}s"
                                "<extra></extra>"
                            ),
                            customdata=[[last_ts]],
                        ))
                        track_color_idx += 1

            # Add cluster intervals (grouped tracks by identity)
            if show_cluster_intervals and cluster_track_map:
                cluster_colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set3
                for c_idx, (cluster_id, cluster_tracks) in enumerate(cluster_track_map.items()):
                    if not cluster_tracks:
                        continue

                    y_label = f"Cluster {cluster_id}"
                    color = cluster_colors[c_idx % len(cluster_colors)]

                    for track in cluster_tracks:
                        first_ts = track.get("first_ts", 0) or 0
                        last_ts = track.get("last_ts", 0) or 0
                        duration = max(last_ts - first_ts, 0.5)
                        track_id = track.get("track_id", "?")

                        fig_timeline.add_trace(go.Bar(
                            x=[duration],
                            y=[y_label],
                            base=[first_ts],
                            orientation="h",
                            marker=dict(color=color),
                            name=y_label,
                            showlegend=False,
                            hovertemplate=(
                                f"<b>{y_label}</b><br>"
                                f"Track: {track_id}<br>"
                                f"Start: %{{base:.1f}}s<br>"
                                f"End: %{{customdata[0]:.1f}}s<br>"
                                f"Duration: %{{x:.1f}}s"
                                "<extra></extra>"
                            ),
                            customdata=[[last_ts]],
                        ))

            # Add skipped frames as vertical dashed lines
            if show_skipped_frames and skipped_faces_data:
                # Group skipped faces by timestamp to avoid too many lines
                skipped_times_set: set[float] = set()
                for face in skipped_faces_data:
                    ts = face.get("timestamp") or face.get("ts")
                    if ts is not None:
                        # Round to 0.1s to reduce visual clutter
                        skipped_times_set.add(round(ts, 1))

                # Limit to prevent chart overload (max 100 markers)
                skipped_times = sorted(skipped_times_set)[:100]

                for ts in skipped_times:
                    fig_timeline.add_vline(
                        x=ts,
                        line_dash="dot",
                        line_color="orange",
                        line_width=1,
                        opacity=0.5,
                    )

            # Configure layout - calculate height based on all visible rows
            row_count = 0
            if show_cast_intervals and timeline_data:
                row_count += len(timeline_data)
            if show_track_intervals and cast_tracks:
                # Count cast members with tracks (grouped), not individual tracks
                row_count += len(cast_tracks)
            if show_cluster_intervals and cluster_track_map:
                row_count += len(cluster_track_map)
            chart_height = max(400, row_count * 35 + 100)

            fig_timeline.update_layout(
                title="Screen Time Timeline",
                xaxis_title="Time (seconds)",
                yaxis_title="",
                barmode="overlay",
                height=chart_height,
                xaxis=dict(
                    tickformat=".0f",
                    ticksuffix="s",
                ),
                yaxis=dict(
                    categoryorder="trace",  # preserve order as added
                ),
                showlegend=False,
            )

            st.plotly_chart(fig_timeline, use_container_width=True)

            # Add legend/explanation
            captions = []
            if show_cast_intervals:
                if show_confidence_heatmap:
                    captions.append("Bars colored by confidence: green (≥0.9), yellow (0.75-0.9), red (<0.75).")
                else:
                    captions.append("Colored bars = time intervals where each person was detected. Gray = unassigned.")
            if show_scene_cuts and scene_cuts_available and scene_cuts_data:
                detector = scene_cuts_data.get("detector", "unknown")
                count = scene_cuts_data.get("count", 0)
                captions.append(f"Red dashed lines = scene cuts ({count} via {detector}).")
            if show_multi_person and timeline_data:
                captions.append("Purple shading = multiple people on screen (darker = more people).")
            if show_broll_regions:
                no_face_count = len(_compute_no_face_regions(tracks_data, video_duration)) if tracks_data else 0
                captions.append(f"Gray shading = B-roll / no-face regions ({no_face_count} gaps).")
            if show_track_intervals and cast_tracks:
                total_tracks = sum(len(tl) for tl in cast_tracks.values())
                captions.append(f"Track rows = face tracks grouped by cast ({total_tracks} tracks across {len(cast_tracks)} rows, different colors per track).")
            if show_cluster_intervals:
                captions.append(f"Cluster rows = tracks grouped by identity ({len(cluster_track_map)} clusters).")
            if show_skipped_frames:
                skip_count = len(skipped_faces_data)
                captions.append(f"Orange dotted lines = skipped frames ({skip_count} faces skipped).")
            if captions:
                st.caption(" ".join(captions))

            # Show interval counts
            if timeline_data:
                interval_counts = []
                for person_data in timeline_data:
                    name = person_data.get("name", "Unknown")
                    intervals = person_data.get("intervals", [])
                    total_duration = sum(end - start for start, end in intervals)
                    interval_counts.append({
                        "Name": name,
                        "Intervals": len(intervals),
                        "Total Duration": format_time(total_duration),
                    })
                st.dataframe(interval_counts, use_container_width=True, hide_index=True)
        else:
            st.info(
                "Timeline data not available. Re-run the screen time analysis to generate timeline data."
            )

        st.divider()

        # PDF Export Section
        st.subheader("Export Screen Time PDF")
        st.caption("Export current view as PDF with table and timeline visualization")

        pdf_col1, pdf_col2 = st.columns([3, 1])
        with pdf_col1:
            # Show which options will be included
            pdf_options_desc = []
            if st.session_state.get(f"{ep_id}::timeline_confidence_heatmap", False):
                pdf_options_desc.append("Color by confidence")
            if st.session_state.get(f"{ep_id}::timeline_show_cuts", False) and scene_cuts_available:
                pdf_options_desc.append("Scene cuts")
            if st.session_state.get(f"{ep_id}::timeline_multi_person", False):
                pdf_options_desc.append("Multi-person regions")
            if pdf_options_desc:
                st.caption(f"Options: {', '.join(pdf_options_desc)}")
            else:
                st.caption("Options: Color by cast (default)")

        with pdf_col2:
            if st.button("Generate PDF", key=f"{ep_id}::export_screentime_pdf", type="primary", use_container_width=True):
                with st.spinner("Generating PDF..."):
                    try:
                        from py_screenalytics.reports.screentime_pdf import ScreenTimePDF

                        options = {
                            "color_by_confidence": st.session_state.get(f"{ep_id}::timeline_confidence_heatmap", False),
                            "show_scene_cuts": st.session_state.get(f"{ep_id}::timeline_show_cuts", False),
                            "show_multi_person": st.session_state.get(f"{ep_id}::timeline_multi_person", False),
                        }

                        pdf_gen = ScreenTimePDF(
                            metrics=metrics,
                            timeline=timeline_data,
                            ep_id=ep_id,
                            options=options,
                            scene_cuts_data=scene_cuts_data if scene_cuts_available else None,
                            video_fps=video_fps,
                            generated_at=data.get("generated_at"),
                        )
                        pdf_bytes = pdf_gen.generate()
                        st.session_state[f"{ep_id}::screentime_pdf_bytes"] = pdf_bytes
                        st.success("PDF generated successfully!")
                    except ImportError as e:
                        st.error(f"Missing dependency: {e}")
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")

        # Download button for generated PDF
        if st.session_state.get(f"{ep_id}::screentime_pdf_bytes"):
            st.download_button(
                "Download PDF",
                data=st.session_state[f"{ep_id}::screentime_pdf_bytes"],
                file_name=f"{ep_id}_screentime.pdf",
                mime="application/pdf",
                key=f"{ep_id}::download_screentime_pdf",
                use_container_width=True,
            )
            if st.button("Clear PDF", key=f"{ep_id}::clear_screentime_pdf"):
                st.session_state.pop(f"{ep_id}::screentime_pdf_bytes", None)
                st.rerun()

        st.divider()

        # Detailed metrics expansion
        with st.expander("View Raw Metrics JSON"):
            st.json(metrics)

    else:
        # Show diagnostic-based empty state messages
        diagnostics = data.get("diagnostics", {})

        st.warning("No cast members found with screen time.")

        # Provide specific guidance based on diagnostics
        if diagnostics:
            with st.expander("Diagnostic Information", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Faces Loaded", diagnostics.get("faces_loaded", "N/A"))
                    st.metric("Tracks Loaded", diagnostics.get("tracks_loaded", "N/A"))
                    st.metric("Identities Loaded", diagnostics.get("identities_loaded", "N/A"))
                with col2:
                    st.metric(
                        "People with Cast ID",
                        diagnostics.get("people_with_cast_id", "N/A"),
                    )
                    st.metric(
                        "Tracks Mapped to Identity",
                        diagnostics.get("tracks_mapped_to_identity", "N/A"),
                    )
                    st.metric(
                        "Tracks with Cast ID",
                        diagnostics.get("tracks_with_cast_id", "N/A"),
                    )

                # Show specific issues
                if diagnostics.get("tracks_missing_identity", 0) > 0:
                    st.info(f"⚠️ {diagnostics['tracks_missing_identity']} tracks have no identity assignment")
                if diagnostics.get("tracks_missing_person", 0) > 0:
                    st.info(f"⚠️ {diagnostics['tracks_missing_person']} identities have no person mapping")
                if diagnostics.get("tracks_missing_cast", 0) > 0:
                    st.info(f"⚠️ {diagnostics['tracks_missing_cast']} people have no cast_id assignment")

        # Specific guidance based on what's missing
        people_with_cast = diagnostics.get("people_with_cast_id", 0)
        tracks_with_cast = diagnostics.get("tracks_with_cast_id", 0)

        if people_with_cast == 0:
            st.error("**No cast members are linked to this show.**")
            st.info("➡️ Please assign cast members in the **Cast page** before running screen time analysis.")
        elif tracks_with_cast == 0:
            st.error("**No clusters in this episode are assigned to cast members.**")
            identities_count = diagnostics.get("identities_loaded", 0)
            st.info(
                f"This episode has {identities_count} cluster(s) but none are linked to cast members. "
                "Use **Faces Review** to assign clusters to cast members."
            )
            # Add a direct link to Faces Review with auto-assign action
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 Go to Faces Review", key="goto_faces_review_from_screentime", use_container_width=True):
                    st.switch_page("pages/3_Faces_Review.py")
            with col2:
                st.caption("Use the **Auto-Assign All** button in Faces Review to quickly assign clusters to cast.")
        else:
            st.info("Make sure cast members are assigned in the Cast page before running screen time analysis.")
else:
    st.info("No screentime analytics yet. Click 'Analyze Screen Time' to generate results.")

st.divider()

# Timeline Export Section
st.subheader("Timeline Data Export")
st.caption("Export second-by-second detection data showing who appears at each timestamp.")

tl_col1, tl_col2, tl_col3 = st.columns([1, 1, 1])
with tl_col1:
    tl_interval = st.selectbox(
        "Time interval",
        options=[0.5, 1.0, 2.0, 5.0],
        index=1,
        format_func=lambda x: f"{x}s intervals",
        key=f"{ep_id}::tl_interval",
    )
with tl_col2:
    tl_include_unassigned = st.checkbox(
        "Include unassigned tracks",
        value=True,
        key=f"{ep_id}::tl_include_unassigned",
    )
with tl_col3:
    tl_format = st.selectbox(
        "Format",
        options=["csv", "json"],
        index=0,
        key=f"{ep_id}::tl_format",
    )

if st.button("Export Timeline Data", key=f"{ep_id}::export_timeline"):
    with st.spinner("Generating timeline export..."):
        try:
            resp = helpers.api_get(
                f"/episodes/{ep_id}/timeline_export",
                params={
                    "interval_s": tl_interval,
                    "include_unassigned": str(tl_include_unassigned).lower(),
                    "format": tl_format,
                },
            )
            st.session_state[f"{ep_id}::timeline_export_result"] = resp
        except requests.RequestException as exc:
            st.error(helpers.describe_error("Timeline export", exc))

# Display timeline export result
tl_result = st.session_state.get(f"{ep_id}::timeline_export_result")
if tl_result:
    if tl_result.get("format") == "csv":
        csv_url = tl_result.get("url")
        if csv_url:
            st.success("Timeline CSV exported!")
            st.markdown(f"**Download**: [Timeline CSV]({csv_url})")
        elif tl_result.get("content"):
            # Local mode - provide download button
            st.success("Timeline CSV generated!")
            st.download_button(
                "Download Timeline CSV",
                data=tl_result["content"],
                file_name=f"{ep_id}_timeline.csv",
                mime="text/csv",
                key=f"{ep_id}::download_timeline_csv",
            )
    else:
        # JSON format - show summary
        summary = tl_result.get("summary", {})
        st.success(f"Timeline data: {summary.get('total_intervals', 0)} intervals")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Intervals", summary.get("total_intervals", 0))
            st.metric("Unassigned Intervals", summary.get("unassigned_intervals", 0))
        with col2:
            cast_appearances = summary.get("cast_appearances", {})
            if cast_appearances:
                top_3 = sorted(cast_appearances.items(), key=lambda x: x[1], reverse=True)[:3]
                st.write("**Top appearances:**")
                for name, count in top_3:
                    st.write(f"• {name}: {count} intervals")

        with st.expander("View Timeline JSON"):
            st.json(tl_result)

    # Show pipeline configs that produced these results
    with st.expander("Pipeline Settings (used for this run)", expanded=False):
        pipeline_cfg = load_pipeline_configs()
        st.markdown("**Detection** (`config/pipeline/detection.yaml`)")
        det_cfg = pipeline_cfg.get("detection", {})
        st.code(f"confidence_th: {det_cfg.get('confidence_th', 'N/A')}\nmin_size: {det_cfg.get('min_size', 'N/A')}")

        st.markdown("**Tracking** (`config/pipeline/tracking.yaml`)")
        trk_cfg = pipeline_cfg.get("tracking", {})
        st.code(
            f"track_thresh: {trk_cfg.get('track_thresh', 'N/A')}\n"
            f"new_track_thresh: {trk_cfg.get('new_track_thresh', 'N/A')}\n"
            f"match_thresh: {trk_cfg.get('match_thresh', 'N/A')}\n"
            f"track_buffer: {trk_cfg.get('track_buffer', 'N/A')}"
        )

        st.markdown("**Embedding/Quality** (`config/pipeline/faces_embed_sampling.yaml`)")
        emb_cfg = pipeline_cfg.get("embedding", {})
        st.code(
            f"min_quality_score: {emb_cfg.get('min_quality_score', 'N/A')}\n"
            f"min_confidence: {emb_cfg.get('min_confidence', 'N/A')}\n"
            f"min_blur_score: {emb_cfg.get('min_blur_score', 'N/A')}\n"
            f"max_yaw_angle: {emb_cfg.get('max_yaw_angle', 'N/A')}\n"
            f"max_pitch_angle: {emb_cfg.get('max_pitch_angle', 'N/A')}"
        )

        st.markdown("**Clustering** (`config/pipeline/clustering.yaml`)")
        cls_cfg = pipeline_cfg.get("clustering", {})
        st.code(
            f"cluster_thresh: {cls_cfg.get('cluster_thresh', 'N/A')}\n"
            f"min_cluster_size: {cls_cfg.get('min_cluster_size', 'N/A')}\n"
            f"min_identity_sim: {cls_cfg.get('min_identity_sim', 'N/A')}"
        )

        # Provide copyable summary
        config_summary = {
            "detection": det_cfg,
            "tracking": trk_cfg,
            "embedding": emb_cfg,
            "clustering": cls_cfg,
        }
        st.markdown("**Copy as JSON:**")
        st.code(json.dumps(config_summary, indent=2), language="json")

    if st.button("Clear Export", key=f"{ep_id}::clear_timeline"):
        st.session_state.pop(f"{ep_id}::timeline_export_result", None)
        st.rerun()

# ── Episode Pipeline Report Export ────────────────────────────────────────────
st.divider()
st.subheader("Episode Pipeline Report")
st.caption(
    "Generate a comprehensive PDF report of this episode's full pipeline run, "
    "including metadata, configs, face/audio analysis, and screen time metrics."
)

report_col1, report_col2 = st.columns([2, 1])
with report_col1:
    report_include_appendix = st.checkbox(
        "Include raw data appendix",
        value=False,
        key=f"{ep_id}::report_include_appendix",
        help="Include truncated JSON excerpts of raw manifest data in the appendix section",
    )
with report_col2:
    report_format = st.selectbox(
        "Format",
        options=["pdf", "snapshot"],
        index=0,
        key=f"{ep_id}::report_format",
        format_func=lambda x: "PDF Report" if x == "pdf" else "JSON Snapshot",
    )

if st.button(
    "Generate Report",
    key=f"{ep_id}::generate_report",
    type="primary",
    use_container_width=True,
):
    with st.spinner("Generating report..."):
        try:
            params = {
                "include_appendix": str(report_include_appendix).lower(),
                "format": report_format,
            }
            response = requests.get(
                f"{cfg['api_base']}/episodes/{ep_id}/report.pdf",
                params=params,
                timeout=120,
            )
            response.raise_for_status()
            st.session_state[f"{ep_id}::report_bytes"] = response.content
            st.session_state[f"{ep_id}::report_format_result"] = report_format
            st.success("Report generated successfully!")
        except requests.RequestException as exc:
            st.error(helpers.describe_error("Report generation", exc))

# Display download button if report is ready
if st.session_state.get(f"{ep_id}::report_bytes"):
    fmt = st.session_state.get(f"{ep_id}::report_format_result", "pdf")
    ext = "pdf" if fmt == "pdf" else "json"
    mime = "application/pdf" if fmt == "pdf" else "application/json"

    st.download_button(
        f"Download {ext.upper()} Report",
        data=st.session_state[f"{ep_id}::report_bytes"],
        file_name=f"{ep_id}_report.{ext}",
        mime=mime,
        key=f"{ep_id}::download_report",
        use_container_width=True,
    )

    if st.button("Clear Report", key=f"{ep_id}::clear_report"):
        st.session_state.pop(f"{ep_id}::report_bytes", None)
        st.session_state.pop(f"{ep_id}::report_format_result", None)
        st.rerun()

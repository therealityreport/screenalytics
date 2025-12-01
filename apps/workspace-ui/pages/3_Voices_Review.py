"""Voices Review page for labeling voice clusters.

This page allows reviewing and labeling unique voice clusters identified
by the audio pipeline for an episode.
"""

from __future__ import annotations

import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402

PAGE_TITLE = "Voices Review - Screenalytics"

DATA_ROOT = helpers.DATA_ROOT


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins:02d}m"


def _get_audio_paths(ep_id: str) -> Dict[str, Path]:
    """Get paths for audio artifacts."""
    audio_dir = DATA_ROOT / "audio" / ep_id
    manifests_dir = DATA_ROOT / "manifests" / ep_id
    artifacts_dir = DATA_ROOT / "artifacts" / ep_id

    return {
        "audio_final": audio_dir / "episode_final_voice_only.wav",
        "audio_vocals_enhanced": audio_dir / "episode_vocals_enhanced.wav",
        "audio_vocals": audio_dir / "episode_vocals.wav",
        "audio_original": audio_dir / "episode_original.wav",
        "voice_clusters": manifests_dir / "audio_voice_clusters.json",
        "voice_mapping": manifests_dir / "audio_voice_mapping.json",
        "transcript_jsonl": manifests_dir / "episode_transcript.jsonl",
        "asr_raw": manifests_dir / "audio_asr_raw.jsonl",
        "audio_qc": manifests_dir / "audio_qc.json",
        "voice_segments": artifacts_dir / "voices",
        "archived_segments": manifests_dir / "audio_archived_segments.json",
    }


def _get_segment_audio_path(ep_id: str, cluster_id: str, start: float, end: float) -> Path:
    """Get path for a segment audio file.

    Uses timestamps in filename to ensure cache validity when segments change.
    """
    paths = _get_audio_paths(ep_id)
    segments_dir = paths["voice_segments"] / cluster_id
    # Use timestamps to create unique filename (avoids stale cache after re-clustering)
    start_ms = int(start * 1000)
    end_ms = int(end * 1000)
    return segments_dir / f"seg_{start_ms}_{end_ms}.wav"


def _extract_segment_audio(ep_id: str, cluster_id: str, segment_idx: int, start: float, end: float) -> Optional[Path]:
    """Extract a segment from the source audio and save it as a clip.

    Returns the path to the segment file, or None if extraction fails.
    """
    segment_path = _get_segment_audio_path(ep_id, cluster_id, start, end)

    # Return cached file if it exists
    if segment_path.exists():
        return segment_path

    # Get source audio
    audio_path = _get_audio_file_for_playback(ep_id)
    if not audio_path or not audio_path.exists():
        return None

    try:
        import subprocess

        # Create output directory
        segment_path.parent.mkdir(parents=True, exist_ok=True)

        # Use ffmpeg to extract segment
        duration = end - start
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-threads", "1",  # avoid runaway CPU on laptops
            "-ss", str(start),
            "-t", str(duration),
            "-i", str(audio_path),
            "-c:a", "pcm_s16le",
            "-ar", "16000",
            str(segment_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=max(10, duration + 5))

        return segment_path if segment_path.exists() else None
    except Exception as e:
        # Log error but don't crash
        import logging
        logging.warning(f"Failed to extract segment audio: {e}")
        return None


@lru_cache(maxsize=8)
def _load_audio_bytes(audio_path: Path) -> Optional[bytes]:
    """Load audio bytes with caching to avoid repeated disk reads."""
    if not audio_path.exists():
        return None
    try:
        return audio_path.read_bytes()
    except Exception:
        return None


def _load_voice_clusters(ep_id: str) -> List[Dict]:
    """Load voice clusters from file."""
    paths = _get_audio_paths(ep_id)
    clusters_path = paths["voice_clusters"]

    if not clusters_path.exists():
        return []

    try:
        with clusters_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _load_voice_mapping(ep_id: str) -> List[Dict]:
    """Load voice mapping from file."""
    paths = _get_audio_paths(ep_id)
    mapping_path = paths["voice_mapping"]

    if not mapping_path.exists():
        return []

    try:
        with mapping_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _load_transcript(ep_id: str) -> List[Dict]:
    """Load transcript rows from JSONL file."""
    paths = _get_audio_paths(ep_id)
    jsonl_path = paths["transcript_jsonl"]

    if not jsonl_path.exists():
        return []

    try:
        rows = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    except Exception:
        return []


def _load_qc_status(ep_id: str) -> Dict:
    """Load QC status from file."""
    paths = _get_audio_paths(ep_id)
    qc_path = paths["audio_qc"]

    if not qc_path.exists():
        return {}

    try:
        with qc_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_asr_raw(ep_id: str) -> List[Dict]:
    """Load raw ASR segments from JSONL file."""
    paths = _get_audio_paths(ep_id)
    asr_path = paths["asr_raw"]

    if not asr_path.exists():
        return []

    try:
        rows = []
        with asr_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    except Exception:
        return []


def _load_archived_segments(ep_id: str) -> List[Dict]:
    """Load archived (excluded) transcript segments."""
    paths = _get_audio_paths(ep_id)
    archive_path = paths.get("archived_segments")

    if not archive_path or not archive_path.exists():
        return []

    try:
        with archive_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _is_segment_archived(
    archived_segments: List[Dict],
    start: float,
    end: float,
    tolerance: float = 0.1,
) -> bool:
    """Check if a segment is archived (by time range)."""
    for seg in archived_segments:
        if abs(seg.get("start", 0) - start) < tolerance and abs(seg.get("end", 0) - end) < tolerance:
            return True
    return False


def _get_transcript_for_segment(
    asr_rows: List[Dict],
    seg_start: float,
    seg_end: float,
    tolerance: float = 0.5,
) -> List[Dict]:
    """Get ASR segments that overlap with a voice cluster segment.

    Returns list of ASR segments with their text, sorted by start time.
    Each ASR segment represents a distinct utterance (potentially different speaker).
    Includes confidence scores for highlighting.
    """
    matching = []
    for row in asr_rows:
        asr_start = row.get("start", 0)
        asr_end = row.get("end", 0)

        # Check for overlap (with tolerance)
        if asr_start < (seg_end + tolerance) and asr_end > (seg_start - tolerance):
            matching.append({
                "start": asr_start,
                "end": asr_end,
                "text": row.get("text", ""),
                "speaker": row.get("speaker"),
                "confidence": row.get("confidence"),
            })

    # Sort by start time
    matching.sort(key=lambda x: x["start"])
    return matching


def _get_confidence_color(confidence: Optional[float]) -> str:
    """Get color code for confidence level.

    Returns CSS color based on confidence:
    - Green (>= 0.8): High confidence
    - Yellow (0.6-0.8): Medium confidence
    - Red (< 0.6): Low confidence
    - Gray: Unknown
    """
    if confidence is None:
        return "#888"  # Gray for unknown
    if confidence >= 0.8:
        return "#4CAF50"  # Green
    if confidence >= 0.6:
        return "#FFC107"  # Yellow/amber
    return "#F44336"  # Red


def _get_confidence_badge(confidence: Optional[float]) -> str:
    """Get emoji badge for confidence level."""
    if confidence is None:
        return "❓"
    if confidence >= 0.8:
        return "✅"
    if confidence >= 0.6:
        return "⚠️"
    return "🔴"


def _get_show_id(ep_id: str) -> str:
    """Extract show_id from ep_id.

    ep_id format: show-sXXeYY (e.g., rhoslc-s06e02)
    Returns the show slug in uppercase (e.g., 'RHOSLC').
    """
    parts = ep_id.split("-")
    if parts:
        return parts[0].upper()
    return ep_id.upper()


def _load_cast_members(show_id: str) -> List[Dict]:
    """Load cast members for a show."""
    try:
        resp = helpers.api_get(f"/shows/{show_id}/cast")
        return resp.get("cast", [])
    except requests.RequestException:
        return []


def _assign_voice(ep_id: str, voice_cluster_id: str, cast_id: Optional[str], custom_label: Optional[str]) -> Dict:
    """Assign a voice cluster to a cast member or custom label."""
    payload = {
        "voice_cluster_id": voice_cluster_id,
        "cast_id": cast_id,
        "custom_label": custom_label,
    }

    try:
        resp = helpers.api_post(f"/jobs/episodes/{ep_id}/audio/voices/assign", json=payload)
        return resp
    except requests.RequestException as e:
        return {"error": str(e)}


def _get_audio_file_for_playback(ep_id: str) -> Optional[Path]:
    """Get the best available audio file for playback."""
    paths = _get_audio_paths(ep_id)

    # Prefer final voice-only, then enhanced vocals, then original vocals
    for key in ["audio_final", "audio_vocals_enhanced", "audio_vocals", "audio_original"]:
        if paths[key].exists():
            return paths[key]

    return None


def _render_audio_snippet(audio_path: Path, start_s: float, end_s: float, key: str) -> None:
    """Render an audio player for a specific time range.

    Note: Streamlit's audio player doesn't support time ranges natively,
    so we show the full audio with a note about the time range.
    """
    try:
        audio_bytes = audio_path.read_bytes()
        st.audio(audio_bytes, format="audio/wav")
        st.caption(f"Segment: {_format_duration(start_s)} - {_format_duration(end_s)}")
    except Exception as e:
        st.warning(f"Could not load audio: {e}")


# =============================================================================
# Page Logic
# =============================================================================

# Initialize page (sets page config first)
helpers.init_page(PAGE_TITLE)

# Get ep_id from query params or session state
ep_id = helpers.get_ep_id_from_query_params()

if not ep_id:
    st.title("Voices Review")
    st.warning("No episode selected. Please select an episode first.")
    helpers.render_sidebar_episode_selector()
    st.stop()

# Parse episode info
parsed = helpers.parse_ep_id(ep_id)
if parsed:
    show_name = parsed["show"].upper()
    season = parsed["season"]
    episode = parsed["episode"]
    st.title(f"Voices Review - {show_name} S{season:02d}E{episode:02d}")
else:
    st.title(f"Voices Review - {ep_id}")
    show_name = ep_id

# Render sidebar
helpers.render_sidebar_episode_selector()

# Check for voice artifacts and any running jobs
paths = _get_audio_paths(ep_id)
has_voice_clusters = paths["voice_clusters"].exists()
has_voice_mapping = paths["voice_mapping"].exists()
running_audio_job = helpers.get_running_job_for_episode(ep_id, "audio_pipeline")

# =============================================================================
# Audio Pipeline Section (allow running directly from Voices Review)
# =============================================================================

if not has_voice_clusters and not has_voice_mapping:
    st.warning("Audio pipeline has not been run for this episode yet.")

# Show running job progress
if running_audio_job:
    st.markdown("### 🔄 Audio Pipeline Running")

    # Handle both audio pipeline progress format (step, progress, message)
    # and generic job format (progress_pct, step_name, etc.)
    raw_progress = running_audio_job.get("progress", 0)
    progress_pct = running_audio_job.get("progress_pct", raw_progress * 100 if raw_progress <= 1 else raw_progress)
    state = running_audio_job.get("state", "running")
    message = running_audio_job.get("message", "")
    # Audio pipeline uses "step" field, map to step_name
    step_name = running_audio_job.get("step_name", "") or running_audio_job.get("step", "")
    step_order = running_audio_job.get("step_order", 0)
    total_steps = running_audio_job.get("total_steps", 9)
    completed_steps = running_audio_job.get("completed_steps", [])
    secs_done = running_audio_job.get("secs_done", 0)

    st.progress(min(progress_pct / 100, 1.0))

    if secs_done > 0:
        mins = int(secs_done // 60)
        secs = int(secs_done % 60)
        elapsed_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
    else:
        elapsed_str = ""

    # Display step info
    step_names_map = {
        "extract": "Extract", "separate": "Separate", "enhance": "Enhance",
        "diarize": "Diarize", "voices": "Voices", "transcribe": "Transcribe",
        "fuse": "Fuse", "export": "Export", "qc": "QC", "s3_sync": "S3 Sync"
    }

    if step_name:
        step_display = step_names_map.get(step_name, step_name.replace("_", " ").title())
        if step_order > 0:
            step_info = f"**Step {step_order}/{total_steps}: {step_display}**"
        else:
            step_info = f"**{step_display}**"
        if elapsed_str:
            step_info += f" ({elapsed_str})"
        st.markdown(step_info)

    if message:
        st.caption(f"📍 {message} ({progress_pct:.1f}%)")
    else:
        st.caption(f"Progress: {progress_pct:.1f}%")

    if completed_steps:
        completed_display = " → ".join(
            f"✅ {step_names_map.get(s, s)}" for s in completed_steps[:5]
        )
        if len(completed_steps) > 5:
            completed_display += f" + {len(completed_steps) - 5} more"
        st.caption(completed_display)

    job_id = running_audio_job.get("job_id", "unknown")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🔄 Refresh", key=f"refresh_audio_voices_{job_id}", use_container_width=True):
            st.rerun()
    with btn_col2:
        if st.button("❌ Cancel", key=f"cancel_audio_voices_{job_id}", use_container_width=True):
            success, msg = helpers.cancel_running_job(job_id)
            if success:
                st.success(msg)
                time.sleep(1)
                st.rerun()
            else:
                st.error(msg)

audio_job_running = running_audio_job is not None

with st.expander("🎙️ Audio Pipeline", expanded=not has_voice_clusters):
    if has_voice_mapping:
        st.success("✅ Audio pipeline complete - transcript available")
    elif has_voice_clusters:
        st.info("⏳ Audio pipeline in progress - voice clusters generated")
    else:
        st.info("Run the audio pipeline to generate transcript and voice clusters")

    # Execution mode selector (Local vs Redis/Celery)
    exec_col1, exec_col2 = st.columns([1, 2])
    with exec_col1:
        execution_mode = helpers.render_execution_mode_selector(ep_id, key_suffix="voices_review")
    with exec_col2:
        if execution_mode == "local":
            st.info("**Local Mode**: Jobs run synchronously in-process. Progress shown in real-time.")
        else:
            st.info("**Redis Mode**: Jobs are queued via Celery for background processing.")

    st.markdown("---")

    audio_overwrite = st.checkbox(
        "Overwrite existing audio artifacts",
        value=False,
        key=f"audio_overwrite_voices_{ep_id}",
        disabled=audio_job_running,
    )
    asr_provider = st.selectbox(
        "ASR Provider",
        options=["openai_whisper", "gemini"],
        index=0,
        key=f"audio_asr_provider_voices_{ep_id}",
        disabled=audio_job_running,
    )

    if st.button(
        "🎙️ Generate Audio + Transcript",
        key=f"run_audio_pipeline_voices_{ep_id}",
        disabled=audio_job_running,
        use_container_width=True,
    ):
        # Use the execution_mode from the selector above
        run_mode = "local" if execution_mode == "local" else "queue"

        # If a job is already running, cancel it before starting a new one
        existing = helpers.get_running_job_for_episode(ep_id, "audio_pipeline")
        if existing and existing.get("job_id"):
            cancel_ok, cancel_msg = helpers.cancel_running_job(existing["job_id"])
            if cancel_ok:
                st.info(f"Cancelled existing audio job {existing['job_id']}")
                time.sleep(1)
            else:
                st.warning(f"Could not cancel existing job: {cancel_msg}")

        if run_mode == "local":
            # Use streaming function for real-time logs and progress
            result, error = helpers.run_audio_pipeline_with_streaming(
                ep_id=ep_id,
                overwrite=audio_overwrite,
                asr_provider=asr_provider,
            )

            if result and result.get("status") == "succeeded":
                time.sleep(1)
                st.rerun()
            elif error:
                # Error already displayed by streaming function
                pass
        else:
            # Redis mode - queue via Celery with streaming progress
            result, error = helpers.run_audio_pipeline_with_celery_streaming(
                ep_id=ep_id,
                overwrite=audio_overwrite,
                asr_provider=asr_provider,
            )
            if result and result.get("status") == "succeeded":
                time.sleep(1)
                st.rerun()

# Show previous logs for audio pipeline OUTSIDE the expander to avoid nesting
helpers.render_previous_logs(ep_id, "audio_pipeline", expanded=False, show_if_none=True)

# =============================================================================
# Incremental Pipeline Reruns (Feature #9)
# =============================================================================

with st.expander("⚡ Incremental Reruns", expanded=False):
    st.markdown(
        "Re-run specific pipeline stages without redoing earlier stages. "
        "Useful for trying different settings or fixing issues."
    )

    # Get execution mode for streaming support
    _exec_mode = helpers.get_execution_mode(ep_id)
    run_mode = "local" if _exec_mode == "local" else "queue"

    rerun_col1, rerun_col2 = st.columns(2)

    with rerun_col1:
        st.write("**Re-run Transcription Only**")
        st.caption("Keep existing diarization, just re-transcribe")
        rerun_asr = st.selectbox(
            "ASR Provider",
            options=["openai_whisper", "gemini_3"],
            key=f"rerun_asr_{ep_id}",
        )
        if st.button("🎙️ Re-transcribe", key=f"rerun_transcribe_{ep_id}", use_container_width=True):
            payload = {"asr_provider": rerun_asr, "run_mode": run_mode}

            if run_mode == "local":
                # Use streaming for real-time progress
                result, error = helpers.run_incremental_with_streaming(
                    ep_id=ep_id,
                    operation="transcribe_only",
                    endpoint=f"/jobs/episodes/{ep_id}/audio/transcribe_only",
                    payload=payload,
                    operation_display_name="Re-transcription",
                )
                if result and result.get("status") == "succeeded":
                    st.success(f"Re-transcribed! {result.get('segment_count', '?')} segments")
                    time.sleep(1)
                    st.rerun()
                elif error:
                    st.error(f"Failed: {error}")
            else:
                # Queue mode - async request
                with st.spinner("Re-running transcription..."):
                    try:
                        resp = helpers.api_post(
                            f"/jobs/episodes/{ep_id}/audio/transcribe_only",
                            json=payload,
                        )
                        if resp.get("success"):
                            st.success(f"Re-transcribed! {resp.get('segment_count', 0)} segments")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Failed: {resp}")
                    except requests.RequestException as exc:
                        st.error(f"API error: {exc}")

    with rerun_col2:
        st.write("**Re-run Diarization Only**")
        st.caption("Keep audio files, re-do speaker segmentation")

        # Calculate suggested speaker count from cast
        _rerun_show_id = _get_show_id(ep_id)
        _rerun_cast = _load_cast_members(_rerun_show_id)
        # Parse season from ep_id (e.g., "rhoslc-s06e02" -> "S06")
        _season_match = None
        for part in ep_id.split("-"):
            if part.startswith("s") and len(part) >= 3:
                _season_match = part[:3].upper()  # "s06" -> "S06"
                break
        # Filter cast by season if available
        _season_cast = []
        if _season_match and _rerun_cast:
            _season_cast = [c for c in _rerun_cast if _season_match in (c.get("seasons") or [])]
            _cast_count = len(_season_cast) if _season_cast else len(_rerun_cast)
        else:
            _cast_count = len(_rerun_cast)
        _suggested_speakers = _cast_count + 2 if _cast_count > 0 else 0

        # Show suggested value and input
        _num_input_col, _use_suggested_col = st.columns([2, 1])
        with _num_input_col:
            rerun_num_speakers = st.number_input(
                "Force speaker count",
                min_value=0,
                max_value=30,
                value=st.session_state.get(f"suggested_speakers_{ep_id}", 0),
                key=f"rerun_num_speakers_{ep_id}",
                help=f"Set to 0 for auto-detect (often under-counts). Suggested: {_suggested_speakers} based on cast count + buffer for guests.",
            )
        with _use_suggested_col:
            st.write("")  # spacing
            if _suggested_speakers > 0:
                if st.button(f"Use {_suggested_speakers}", key=f"use_suggested_{ep_id}", help=f"{_cast_count} cast + 2 guests"):
                    st.session_state[f"suggested_speakers_{ep_id}"] = _suggested_speakers
                    st.rerun()

        if _suggested_speakers > 0:
            st.caption(f"💡 {_cast_count} cast members for this season + 2 buffer for guests")

        if st.button("👥 Re-diarize", key=f"rerun_diarize_{ep_id}", use_container_width=True):
            payload = {"run_mode": run_mode}
            if rerun_num_speakers > 0:
                payload["num_speakers"] = rerun_num_speakers

            if run_mode == "local":
                # Use streaming for real-time progress
                result, error = helpers.run_incremental_with_streaming(
                    ep_id=ep_id,
                    operation="diarize_only",
                    endpoint=f"/jobs/episodes/{ep_id}/audio/diarize_only",
                    payload=payload,
                    operation_display_name="Re-diarization",
                )
                if result and result.get("status") == "succeeded":
                    speaker_count = result.get("speaker_count", "?")
                    segment_count = result.get("segment_count", "?")
                    st.success(f"Re-diarized! {segment_count} segments, {speaker_count} speakers")
                    time.sleep(1)
                    st.rerun()
                elif error:
                    st.error(f"Failed: {error}")
            else:
                # Queue mode - async request
                with st.spinner("Re-running diarization..."):
                    try:
                        resp = helpers.api_post(
                            f"/jobs/episodes/{ep_id}/audio/diarize_only",
                            json=payload,
                            timeout=600,
                        )
                        if resp.get("success"):
                            st.success(f"Re-diarized! {resp.get('segment_count', 0)} segments, {resp.get('speaker_count', 0)} speakers")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Failed: {resp}")
                    except requests.RequestException as exc:
                        st.error(f"API error: {exc}")

    st.markdown("---")
    st.caption("💡 Use these options to iterate on specific stages without losing prior work.")

# =============================================================================
# Clustering Controls (moved here to be adjacent to Incremental Reruns)
# =============================================================================

with st.expander("🔧 Clustering Settings", expanded=False):
    st.markdown(
        "Adjust clustering parameters to group similar voices together. "
        "**Lower threshold = fewer clusters** (more aggressive grouping), "
        "**Higher threshold = more clusters** (more separation)."
    )

    cluster_col1, cluster_col2 = st.columns(2)

    with cluster_col1:
        new_similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.15,
            max_value=0.70,
            value=0.30,
            step=0.05,
            key=f"cluster_threshold_{ep_id}",
            help="Cosine similarity threshold for grouping segments. Lower = more aggressive grouping.",
        )

    with cluster_col2:
        new_min_segments = st.selectbox(
            "Min Segments per Cluster",
            options=[1, 2, 3],
            index=0,
            key=f"cluster_min_seg_{ep_id}",
            help="Minimum segments required to form a cluster. Set to 1 for trailers/short content.",
        )

    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if st.button(
            "👁️ Preview Changes",
            key=f"preview_cluster_{ep_id}",
            use_container_width=True,
        ):
            with st.spinner("Previewing..."):
                try:
                    payload = {
                        "similarity_threshold": new_similarity_threshold,
                        "min_segments_per_cluster": new_min_segments,
                    }
                    preview_resp = helpers.api_post(f"/jobs/episodes/{ep_id}/audio/clusters/preview", json=payload)

                    if "error" in preview_resp:
                        st.error(preview_resp["error"])
                    else:
                        proposed = preview_resp.get("proposed_cluster_count", 0)
                        current = preview_resp.get("current_cluster_count", 0)
                        change = preview_resp.get("change", 0)

                        if change > 0:
                            st.info(f"📈 Preview: {current} → {proposed} clusters (+{change})")
                        elif change < 0:
                            st.info(f"📉 Preview: {current} → {proposed} clusters ({change})")
                        else:
                            st.info(f"→ Preview: No change ({proposed} clusters)")

                        # Show preview clusters
                        preview_clusters = preview_resp.get("clusters", [])[:5]
                        if preview_clusters:
                            st.markdown("**Proposed clusters (top 5):**")
                            for pc in preview_clusters:
                                st.caption(
                                    f"• {pc['cluster_id']}: {pc['segment_count']} segments, "
                                    f"{pc['total_duration']:.1f}s total"
                                )
                except requests.RequestException as exc:
                    st.error(f"Preview error: {exc}")

    with btn_col2:
        if st.button(
            "🔄 Re-cluster Voices",
            key=f"recluster_{ep_id}",
            use_container_width=True,
            type="primary",
        ):
            # Get execution mode
            _cluster_exec = helpers.get_execution_mode(ep_id)
            cluster_run_mode = "local" if _cluster_exec == "local" else "queue"
            payload = {
                "similarity_threshold": new_similarity_threshold,
                "min_segments_per_cluster": new_min_segments,
                "run_mode": cluster_run_mode,
            }

            if cluster_run_mode == "local":
                # Use streaming for real-time progress
                result, error = helpers.run_incremental_with_streaming(
                    ep_id=ep_id,
                    operation="voices_only",
                    endpoint=f"/jobs/episodes/{ep_id}/audio/clusters/recluster",
                    payload=payload,
                    operation_display_name="Re-clustering",
                )
                if result and result.get("status") == "succeeded":
                    cluster_count = result.get("cluster_count", "?")
                    st.success(f"Re-clustered! Now have {cluster_count} voice clusters.")
                    time.sleep(1)
                    st.rerun()
                elif error:
                    st.error(f"Failed to recluster: {error}")
            else:
                # Queue mode - async request
                with st.spinner("Re-clustering voices..."):
                    try:
                        resp = helpers.api_post(f"/jobs/episodes/{ep_id}/audio/clusters/recluster", json=payload)

                        if resp.get("success"):
                            new_count = resp.get("cluster_count", 0)
                            st.success(f"Re-clustered! Now have {new_count} voice clusters.")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Failed to recluster: {resp}")
                    except requests.RequestException as exc:
                        st.error(f"API error: {exc}")

if not has_voice_clusters and not has_voice_mapping:
    st.info("Run the audio pipeline above to generate voice clusters for review.")
    st.stop()

# Load data
voice_clusters = _load_voice_clusters(ep_id)
voice_mapping = _load_voice_mapping(ep_id)
transcript = _load_transcript(ep_id)
asr_raw = _load_asr_raw(ep_id)  # Raw ASR for transcript text per segment
qc_data = _load_qc_status(ep_id)
archived_segments = _load_archived_segments(ep_id)  # Segments excluded from voicebank
show_id = _get_show_id(ep_id)
cast_members = _load_cast_members(show_id)

# Build mapping lookup - safely handle missing voice_cluster_id key
mapping_lookup = {m.get("voice_cluster_id", ""): m for m in voice_mapping if m.get("voice_cluster_id")}

# Calculate stats - use mapping count for accurate labeled/unlabeled
voice_cluster_count = len(voice_clusters)
labeled_count = sum(1 for m in voice_mapping if m.get("similarity") is not None)
# Unlabeled count should fall back to clusters minus labeled when mapping is incomplete
unlabeled_from_mapping = sum(1 for m in voice_mapping if m.get("similarity") is None)
unlabeled_count = max(voice_cluster_count - labeled_count, unlabeled_from_mapping)

# =============================================================================
# Page Header and Status
# =============================================================================

st.markdown("---")

# Status card
col1, col2, col3 = st.columns(3)

with col1:
    pipeline_status = "Succeeded" if has_voice_mapping else "Partial"
    qc_status = qc_data.get("status", "unknown")
    # Handle all QC status values properly
    qc_badge_map = {
        "ok": "✅",
        "warn": "⚠️",
        "needs_review": "🔍",
        "unknown": "❓",
        "failed": "🔴",
    }
    qc_badge = qc_badge_map.get(qc_status, "❓")
    st.metric("Pipeline Status", f"{pipeline_status} {qc_badge}")

with col2:
    st.metric("Total Voices", voice_cluster_count)

with col3:
    st.metric("Labeled / Unlabeled", f"{labeled_count} / {unlabeled_count}")

st.markdown(
    "> **Voice segments** generated via MDX-Extra stem separation, "
    "Resemble-enhanced vocals, and pyannote.audio diarization + speaker embeddings."
)

st.markdown("---")

# =============================================================================
# A/B Diarization Comparison (Feature #3)
# =============================================================================

with st.expander("🔄 Diarization Comparison (Pyannote vs GPT-4o)", expanded=False):
    st.markdown(
        "Compare speaker segmentation from different diarization providers. "
        "Useful for understanding which provider works better for your content."
    )

    try:
        comparison_resp = helpers.api_get(f"/jobs/episodes/{ep_id}/audio/diarization/comparison")

        has_pyannote = comparison_resp.get("has_pyannote", False)
        has_gpt4o = comparison_resp.get("has_gpt4o", False)
        summary = comparison_resp.get("summary", {})

        if has_pyannote or has_gpt4o:
            # Show summary comparison
            cmp_col1, cmp_col2 = st.columns(2)

            with cmp_col1:
                st.markdown("**📊 Pyannote**")
                pyannote_info = summary.get("pyannote", {})
                if pyannote_info:
                    st.write(f"- Speakers: {pyannote_info.get('speaker_count', '?')}")
                    st.write(f"- Segments: {pyannote_info.get('segment_count', '?')}")
                    st.write(f"- Speech duration: {pyannote_info.get('total_speech_duration_s', 0):.1f}s")
                    st.write(f"- Avg segment: {pyannote_info.get('avg_segment_duration_s', 0):.1f}s")
                elif has_pyannote:
                    pyannote_segs = comparison_resp.get("pyannote_segments", [])
                    speakers = set(s.get("speaker") for s in pyannote_segs)
                    st.write(f"- Speakers: {len(speakers)}")
                    st.write(f"- Segments: {len(pyannote_segs)}")
                else:
                    st.caption("Pyannote data not available")

            with cmp_col2:
                st.markdown("**🤖 GPT-4o**")
                gpt4o_info = summary.get("gpt4o", {})
                if gpt4o_info:
                    st.write(f"- Speakers: {gpt4o_info.get('speaker_count', '?')}")
                    st.write(f"- Segments: {gpt4o_info.get('segment_count', '?')}")
                    st.write(f"- Speech duration: {gpt4o_info.get('total_speech_duration_s', 0):.1f}s")
                    st.write(f"- Avg segment: {gpt4o_info.get('avg_segment_duration_s', 0):.1f}s")
                elif has_gpt4o:
                    gpt4o_segs = comparison_resp.get("gpt4o_segments", [])
                    speakers = set(s.get("speaker") for s in gpt4o_segs)
                    st.write(f"- Speakers: {len(speakers)}")
                    st.write(f"- Segments: {len(gpt4o_segs)}")
                else:
                    st.caption("GPT-4o data not available")

            # Show comparison metrics if available
            cmp_metrics = summary.get("comparison", {})
            if cmp_metrics:
                st.markdown("---")
                st.markdown("**Comparison:**")
                speaker_diff = cmp_metrics.get("speaker_count_diff", 0)
                if speaker_diff > 0:
                    st.write(f"- GPT-4o found **{speaker_diff} more** speakers")
                elif speaker_diff < 0:
                    st.write(f"- Pyannote found **{abs(speaker_diff)} more** speakers")
                else:
                    st.write("- Both found the same number of speakers")

                duration_diff = cmp_metrics.get("duration_diff_s", 0)
                if abs(duration_diff) > 1:
                    if duration_diff > 0:
                        st.write(f"- GPT-4o detected **{duration_diff:.1f}s more** speech")
                    else:
                        st.write(f"- Pyannote detected **{abs(duration_diff):.1f}s more** speech")

            # Show timeline-aligned segments comparison
            if has_pyannote and has_gpt4o:
                st.markdown("---")
                st.markdown("**📊 Speaker Segmentation Comparison**")
                st.caption("Compare speaker detection between Pyannote (SPEAKER_XX) vs GPT-4o (A, B, C). Disagreements show where providers differ on who is speaking.")

                pyannote_segs = comparison_resp.get("pyannote_segments", [])
                gpt4o_segs = comparison_resp.get("gpt4o_segments", [])

                # Build unified timeline events
                events = []
                for s in pyannote_segs:
                    events.append({"time": s["start"], "type": "start", "provider": "pyannote", "seg": s})
                    events.append({"time": s["end"], "type": "end", "provider": "pyannote", "seg": s})
                for s in gpt4o_segs:
                    events.append({"time": s["start"], "type": "start", "provider": "gpt4o", "seg": s})
                    events.append({"time": s["end"], "type": "end", "provider": "gpt4o", "seg": s})

                # Sort by time
                events.sort(key=lambda x: (x["time"], 0 if x["type"] == "start" else 1))

                # Find time windows where segments overlap/differ
                time_windows = []
                active_pyannote = []
                active_gpt4o = []
                last_time = 0.0

                for evt in events:
                    if evt["time"] > last_time and (active_pyannote or active_gpt4o):
                        time_windows.append({
                            "start": last_time,
                            "end": evt["time"],
                            "pyannote": list(active_pyannote),
                            "gpt4o": list(active_gpt4o),
                        })

                    if evt["provider"] == "pyannote":
                        if evt["type"] == "start":
                            active_pyannote.append(evt["seg"])
                        else:
                            active_pyannote = [s for s in active_pyannote if s != evt["seg"]]
                    else:
                        if evt["type"] == "start":
                            active_gpt4o.append(evt["seg"])
                        else:
                            active_gpt4o = [s for s in active_gpt4o if s != evt["seg"]]

                    last_time = evt["time"]

                # Helper to determine if a window has a disagreement
                def is_disagreement(window):
                    has_pyannote = bool(window["pyannote"])
                    has_gpt4o = bool(window["gpt4o"])
                    # Disagreement if one has speech and other doesn't
                    if has_pyannote != has_gpt4o:
                        return True
                    # Both have speech - could compare speaker counts as proxy for disagreement
                    if has_pyannote and has_gpt4o:
                        pyannote_speakers = set(s.get("speaker", "?") for s in window["pyannote"])
                        gpt4o_speakers = set(s.get("speaker", "?") for s in window["gpt4o"])
                        # Different number of speakers in same window = disagreement
                        if len(pyannote_speakers) != len(gpt4o_speakers):
                            return True
                    return False

                # Filter toggle
                show_only_disagreements = st.checkbox(
                    "🔍 Show only disagreements",
                    value=False,
                    help="Filter to show only time windows where Pyannote and GPT-4o detected different results"
                )

                # Filter windows
                filtered_windows = []
                for window in time_windows:
                    if window["end"] - window["start"] < 0.3:
                        continue  # Skip very short windows
                    if show_only_disagreements and not is_disagreement(window):
                        continue
                    filtered_windows.append(window)

                # Count stats
                total_disagreements = sum(1 for w in time_windows if w["end"] - w["start"] >= 0.3 and is_disagreement(w))
                total_valid_windows = sum(1 for w in time_windows if w["end"] - w["start"] >= 0.3)

                if show_only_disagreements:
                    st.info(f"Showing {len(filtered_windows)} disagreements out of {total_valid_windows} total time windows")
                else:
                    st.caption(f"{total_disagreements} disagreements out of {total_valid_windows} time windows")

                # Build table data
                table_rows = []
                max_rows = 25
                for window in filtered_windows[:max_rows]:
                    w_start = window["start"]
                    w_end = window["end"]
                    w_duration = w_end - w_start
                    disagreement = is_disagreement(window)

                    # Format Pyannote column
                    if window["pyannote"]:
                        pyannote_speakers = [s.get("speaker", "?") for s in window["pyannote"]]
                        pyannote_text = ", ".join(pyannote_speakers)
                    else:
                        pyannote_text = "—"

                    # Format GPT-4o column (speaker labels only for fair diarization comparison)
                    if window["gpt4o"]:
                        gpt4o_speakers = [s.get("speaker", "?") for s in window["gpt4o"]]
                        gpt4o_text = ", ".join(gpt4o_speakers)
                    else:
                        gpt4o_text = "—"

                    table_rows.append({
                        "Time": f"{w_start:.1f}–{w_end:.1f}s",
                        "Duration": f"{w_duration:.1f}s",
                        "🔵 Pyannote": pyannote_text,
                        "🟢 GPT-4o": gpt4o_text,
                        "⚠️": "⚠️" if disagreement else "✓",
                    })

                if table_rows:
                    import pandas as pd
                    df = pd.DataFrame(table_rows)
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Time": st.column_config.TextColumn("Time", width="small"),
                            "Duration": st.column_config.TextColumn("Dur", width="small"),
                            "🔵 Pyannote": st.column_config.TextColumn("🔵 Pyannote", width="medium"),
                            "🟢 GPT-4o": st.column_config.TextColumn("🟢 GPT-4o", width="medium"),
                            "⚠️": st.column_config.TextColumn("", width="small"),
                        }
                    )

                    if len(filtered_windows) > max_rows:
                        st.caption(f"... and {len(filtered_windows) - max_rows} more time windows")
                else:
                    if show_only_disagreements:
                        st.success("✅ No disagreements found! Both providers agree on all time windows.")
                    else:
                        st.info("No time windows to display.")

                # Detailed segment lists for each provider
                st.markdown("---")
                st.markdown("**📋 Detailed Segment Lists**")
                st.caption("View all segments detected by each provider")

                seg_tab1, seg_tab2 = st.tabs(["🔵 Pyannote Segments", "🟢 GPT-4o Segments"])

                with seg_tab1:
                    if pyannote_segs:
                        # Group by speaker
                        pyannote_by_speaker = {}
                        for seg in pyannote_segs:
                            spk = seg.get("speaker", "?")
                            if spk not in pyannote_by_speaker:
                                pyannote_by_speaker[spk] = []
                            pyannote_by_speaker[spk].append(seg)

                        st.write(f"**{len(pyannote_segs)} segments** from **{len(pyannote_by_speaker)} speakers**")

                        for spk, segs in sorted(pyannote_by_speaker.items()):
                            total_dur = sum(s["end"] - s["start"] for s in segs)
                            st.markdown(f"**{spk}** ({len(segs)} segments, {total_dur:.1f}s total)")
                            seg_rows = []
                            for s in segs[:20]:
                                seg_rows.append({
                                    "Start": f"{s['start']:.1f}s",
                                    "End": f"{s['end']:.1f}s",
                                    "Duration": f"{s['end'] - s['start']:.1f}s",
                                })
                            st.dataframe(seg_rows, use_container_width=True, hide_index=True)
                            if len(segs) > 20:
                                st.caption(f"... and {len(segs) - 20} more segments")
                    else:
                        st.info("No Pyannote segments available")

                with seg_tab2:
                    if gpt4o_segs:
                        # Group by speaker
                        gpt4o_by_speaker = {}
                        for seg in gpt4o_segs:
                            spk = seg.get("speaker", "?")
                            if spk not in gpt4o_by_speaker:
                                gpt4o_by_speaker[spk] = []
                            gpt4o_by_speaker[spk].append(seg)

                        st.write(f"**{len(gpt4o_segs)} segments** from **{len(gpt4o_by_speaker)} speakers**")

                        for spk, segs in sorted(gpt4o_by_speaker.items()):
                            total_dur = sum(s["end"] - s["start"] for s in segs)
                            st.markdown(f"**Speaker {spk}** ({len(segs)} segments, {total_dur:.1f}s total)")
                            seg_rows = []
                            for s in segs[:20]:
                                text_preview = s.get("text", "")[:50] + "..." if len(s.get("text", "")) > 50 else s.get("text", "")
                                seg_rows.append({
                                    "Start": f"{s['start']:.1f}s",
                                    "End": f"{s['end']:.1f}s",
                                    "Duration": f"{s['end'] - s['start']:.1f}s",
                                    "Text": text_preview,
                                })
                            st.dataframe(seg_rows, use_container_width=True, hide_index=True)
                            if len(segs) > 20:
                                st.caption(f"... and {len(segs) - 20} more segments")
                    else:
                        st.info("No GPT-4o segments available")

        else:
            st.info("No diarization comparison data available. Run the full pipeline to generate comparison.")

    except requests.RequestException as e:
        st.warning(f"Could not load diarization comparison: {e}")

st.markdown("---")

# =============================================================================
# Voice References Section (Feature #1)
# =============================================================================

with st.expander("🎤 Voice References", expanded=False):
    st.markdown(
        "Upload voice reference clips for cast members to improve speaker identification accuracy. "
        "References should be 5-30 seconds of clear speech from a single speaker."
    )

    # List existing voice references
    try:
        refs_resp = helpers.api_get(f"/jobs/shows/{show_id}/voice_references")
        existing_refs = refs_resp.get("references", [])

        if existing_refs:
            st.info(f"**{len(existing_refs)}** cast member(s) have voice references")
            for ref in existing_refs:
                ref_col1, ref_col2, ref_col3 = st.columns([3, 2, 1])
                with ref_col1:
                    # Find cast member name
                    ref_cast_id = ref.get("cast_id", "")
                    cast_name = ref_cast_id
                    for cm in cast_members:
                        if cm.get("cast_id", "").lower() == ref_cast_id.lower():
                            cast_name = cm.get("name", ref_cast_id)
                            break
                    st.write(f"✅ **{cast_name}**")
                with ref_col2:
                    file_size_kb = ref.get("file_size_bytes", 0) / 1024
                    st.caption(f"{file_size_kb:.1f} KB")
                with ref_col3:
                    if st.button("🗑️", key=f"del_ref_{ref_cast_id}", help=f"Delete reference for {cast_name}"):
                        del_resp = helpers.api_delete(f"/jobs/shows/{show_id}/cast/{ref_cast_id}/voice_reference")
                        if del_resp.get("success"):
                            st.success(f"Deleted reference for {cast_name}")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"Failed to delete: {del_resp}")
        else:
            st.caption("No voice references uploaded yet")

        st.markdown("---")

        # Upload new reference
        st.write("**Upload Voice Reference:**")
        upload_col1, upload_col2 = st.columns([2, 3])

        with upload_col1:
            # Cast member selector
            upload_cast_options = ["(Select cast member)"] + [
                f"{m.get('name', m.get('cast_id', 'Unknown'))}" for m in cast_members
            ]
            upload_cast_values = [None] + [m.get("cast_id") for m in cast_members]
            upload_cast_idx = st.selectbox(
                "Cast Member",
                options=range(len(upload_cast_options)),
                format_func=lambda i: upload_cast_options[i],
                key=f"upload_ref_cast_{ep_id}",
            )
            selected_upload_cast = upload_cast_values[upload_cast_idx] if upload_cast_idx > 0 else None

        with upload_col2:
            uploaded_file = st.file_uploader(
                "Audio File",
                type=["wav", "mp3", "m4a"],
                key=f"voice_ref_upload_{ep_id}",
                help="Upload 5-30 seconds of clear speech",
            )

        if selected_upload_cast and uploaded_file:
            if st.button("📤 Upload Voice Reference", key="upload_ref_btn", use_container_width=True):
                with st.spinner("Uploading and processing..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "audio/wav")}
                        upload_resp = helpers.api_post_files(
                            f"/jobs/shows/{show_id}/cast/{selected_upload_cast}/voice_reference",
                            files=files,
                        )
                        if upload_resp.get("success"):
                            st.success(f"Uploaded voice reference for {upload_cast_options[upload_cast_idx]}")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"Upload failed: {upload_resp}")
                    except Exception as e:
                        st.error(f"Upload error: {e}")

    except requests.RequestException as e:
        st.warning(f"Could not load voice references: {e}")

st.markdown("---")

# =============================================================================
# Suggested Merges Section (Feature #6)
# =============================================================================

with st.expander("🔗 Suggested Merges", expanded=False):
    st.markdown(
        "Voice clusters with similar voice embeddings that may belong to the same speaker. "
        "Review and merge to consolidate."
    )

    merge_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.70,
        max_value=0.95,
        value=0.85,
        step=0.05,
        key=f"merge_threshold_{ep_id}",
        help="Minimum cosine similarity to suggest a merge",
    )

    try:
        suggestions_resp = helpers.api_get(
            f"/jobs/episodes/{ep_id}/audio/clusters/suggest_merges",
            params={"min_similarity": merge_threshold},
        )
        suggestions = suggestions_resp.get("suggestions", [])

        if suggestions:
            st.info(f"Found **{len(suggestions)}** potential merge(s) above {merge_threshold:.0%} similarity")

            # Accept All button
            if st.button("✅ Accept All Suggested Merges", key="accept_all_merges", use_container_width=True):
                # Build merge list - merge smaller into larger (by duration)
                merges = []
                for s in suggestions:
                    ca = s["cluster_a"]
                    cb = s["cluster_b"]
                    # Keep the larger cluster as target
                    if ca["total_duration"] >= cb["total_duration"]:
                        merges.append({
                            "source_cluster_id": cb["voice_cluster_id"],
                            "target_cluster_id": ca["voice_cluster_id"],
                        })
                    else:
                        merges.append({
                            "source_cluster_id": ca["voice_cluster_id"],
                            "target_cluster_id": cb["voice_cluster_id"],
                        })

                with st.spinner("Merging clusters..."):
                    merge_resp = helpers.api_post(
                        f"/jobs/episodes/{ep_id}/audio/clusters/bulk_merge",
                        json=merges,
                    )
                if merge_resp.get("success"):
                    st.success(f"Merged {merge_resp.get('merged_count', 0)} cluster pairs!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"Merge failed: {merge_resp}")

            st.markdown("---")

            # Individual suggestions
            for idx, s in enumerate(suggestions):
                ca = s["cluster_a"]
                cb = s["cluster_b"]
                sim = s["similarity"]

                col_a, col_sim, col_b, col_action = st.columns([3, 1, 3, 2])

                with col_a:
                    label_a = "✅" if ca["is_labeled"] else "❓"
                    st.write(f"{label_a} **{ca['display_name']}**")
                    st.caption(f"{ca['voice_cluster_id']} | {ca['segment_count']} segs | {ca['total_duration']:.1f}s")

                with col_sim:
                    st.metric("Similarity", f"{sim:.0%}")

                with col_b:
                    label_b = "✅" if cb["is_labeled"] else "❓"
                    st.write(f"{label_b} **{cb['display_name']}**")
                    st.caption(f"{cb['voice_cluster_id']} | {cb['segment_count']} segs | {cb['total_duration']:.1f}s")

                with col_action:
                    # Merge smaller into larger
                    if ca["total_duration"] >= cb["total_duration"]:
                        source_id, target_id = cb["voice_cluster_id"], ca["voice_cluster_id"]
                        target_name = ca["display_name"]
                    else:
                        source_id, target_id = ca["voice_cluster_id"], cb["voice_cluster_id"]
                        target_name = cb["display_name"]

                    if st.button(f"Merge → {target_name}", key=f"merge_{idx}", use_container_width=True):
                        with st.spinner("Merging..."):
                            merge_resp = helpers.api_post(
                                f"/jobs/episodes/{ep_id}/audio/clusters/merge",
                                json={
                                    "source_cluster_id": source_id,
                                    "target_cluster_id": target_id,
                                },
                            )
                        if merge_resp.get("success"):
                            st.success(f"Merged into {target_name}")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"Failed: {merge_resp}")

                if idx < len(suggestions) - 1:
                    st.markdown("---")
        else:
            st.success("No merge suggestions - all clusters are sufficiently distinct!")

    except requests.RequestException as e:
        st.warning(f"Could not load merge suggestions: {e}")

st.markdown("---")

# =============================================================================
# Archived Segments Section
# =============================================================================

if archived_segments:
    with st.expander(f"🗄️ Archived Segments ({len(archived_segments)})", expanded=False):
        st.markdown(
            "Segments excluded from voicebank. These won't be used for cast voice profiles. "
            "You can restore them if archived by mistake."
        )

        for arc_idx, arc_seg in enumerate(archived_segments):
            arc_col1, arc_col2 = st.columns([5, 1])

            with arc_col1:
                arc_start = arc_seg.get("start", 0)
                arc_end = arc_seg.get("end", 0)
                arc_text = arc_seg.get("text", "")
                arc_cluster = arc_seg.get("voice_cluster_id", "")
                arc_reason = arc_seg.get("reason", "")
                arc_time = arc_seg.get("archived_at", "")[:10] if arc_seg.get("archived_at") else ""

                st.markdown(
                    f'<span style="color:#888;font-size:0.85em">'
                    f'[{_format_duration(arc_start)}-{_format_duration(arc_end)}]</span> '
                    f'<span style="color:#666">"{arc_text}"</span>',
                    unsafe_allow_html=True,
                )
                cluster_info = f"Cluster: {arc_cluster}" if arc_cluster else ""
                reason_info = f" | Reason: {arc_reason}" if arc_reason else ""
                time_info = f" | {arc_time}" if arc_time else ""
                st.caption(f"{cluster_info}{reason_info}{time_info}")

            with arc_col2:
                if st.button("↩️ Restore", key=f"restore_arc_{arc_idx}", help="Restore to voicebank"):
                    try:
                        resp = helpers.api_post(
                            f"/jobs/episodes/{ep_id}/audio/segments/restore",
                            json={
                                "start": arc_start,
                                "end": arc_end,
                                "text": arc_text,
                                "voice_cluster_id": arc_cluster,
                            },
                        )
                        if resp.get("success"):
                            st.success("Restored!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"Failed: {resp}")
                    except requests.RequestException as exc:
                        st.error(f"Error: {exc}")

            if arc_idx < len(archived_segments) - 1:
                st.markdown("---")

st.markdown("---")

# =============================================================================
# Audio Waveform Visualization (Feature #4) - Above Voice Clusters
# =============================================================================

with st.expander("🎵 Audio Waveform", expanded=False):
    st.markdown(
        "Interactive audio waveform with speaker-colored regions. "
        "Navigate through the audio timeline to see voice segments."
    )

    # Word search for finding timestamps
    word_search_col, word_result_col = st.columns([2, 3])
    with word_search_col:
        search_word = st.text_input(
            "Search transcript",
            placeholder="Enter word(s) to find...",
            key=f"wave_search_{ep_id}",
            help="Search for words in the transcript to jump to their location",
        )
    with word_result_col:
        if search_word and search_word.strip():
            # Search ASR data for matching words
            search_matches = []
            search_lower = search_word.lower().strip()
            for row in asr_raw:
                text = row.get("text", "").lower()
                if search_lower in text:
                    search_matches.append({
                        "start": row.get("start", 0),
                        "end": row.get("end", 0),
                        "text": row.get("text", ""),
                    })

            if search_matches:
                st.caption(f"Found {len(search_matches)} match(es)")
                # Show first 5 matches as quick-jump buttons
                for match in search_matches[:5]:
                    match_start = match["start"]
                    match_text = match["text"][:40] + "..." if len(match["text"]) > 40 else match["text"]
                    if st.button(
                        f"⏱️ {_format_duration(match_start)}: \"{match_text}\"",
                        key=f"jump_{ep_id}_{int(match_start*1000)}",
                        use_container_width=True,
                    ):
                        # Set the waveform start time to this position
                        st.session_state[f"wave_start_{ep_id}"] = max(0, match_start - 5)
                        st.rerun()
            else:
                st.caption("No matches found")

    st.markdown("---")

    # Time range selection with audio playback
    wave_col1, wave_col2, wave_col3, wave_col4 = st.columns([2, 2, 1, 2])

    with wave_col1:
        wave_start = st.number_input(
            "Start (seconds)",
            min_value=0.0,
            max_value=3600.0,
            value=st.session_state.get(f"wave_start_{ep_id}", 0.0),
            step=10.0,
            key=f"wave_start_{ep_id}",
        )
    with wave_col2:
        wave_duration = st.selectbox(
            "Window",
            options=[30, 60, 90, 120],
            index=1,
            format_func=lambda x: f"{x}s",
            key=f"wave_duration_{ep_id}",
        )
    with wave_col3:
        st.write("")
        st.write("")
        load_waveform = st.button("📈 Load", key=f"load_wave_{ep_id}")

    with wave_col4:
        # Timestamp audio playback
        st.write("Play from timestamp:")
        play_ts = st.number_input(
            "Seconds",
            min_value=0.0,
            max_value=3600.0,
            value=wave_start,
            step=1.0,
            key=f"play_ts_{ep_id}",
            label_visibility="collapsed",
        )

    # Audio player for selected timestamp
    audio_path = _get_audio_file_for_playback(ep_id)
    if audio_path and audio_path.exists():
        audio_bytes = _load_audio_bytes(audio_path)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav", start_time=int(play_ts))
            st.caption(f"Playing from {_format_duration(play_ts)}")

    if load_waveform or st.session_state.get(f"wave_data_{ep_id}"):
        try:
            wave_resp = helpers.api_get(
                f"/jobs/episodes/{ep_id}/audio/waveform",
                params={
                    "start": wave_start,
                    "end": wave_start + wave_duration,
                    "resolution": 500,
                },
            )

            waveform = wave_resp.get("waveform", [])
            speaker_regions = wave_resp.get("speaker_regions", [])
            total_duration = wave_resp.get("total_duration", 0)

            if waveform:
                import plotly.graph_objects as go

                # Create waveform figure
                times = [p["time"] for p in waveform]
                mins = [p["min"] for p in waveform]
                maxs = [p["max"] for p in waveform]

                fig = go.Figure()

                # Add waveform envelope (filled area between min and max)
                fig.add_trace(go.Scatter(
                    x=times + times[::-1],
                    y=maxs + mins[::-1],
                    fill="toself",
                    fillcolor="rgba(100, 149, 237, 0.3)",
                    line=dict(color="cornflowerblue", width=1),
                    name="Waveform",
                    hoverinfo="skip",
                ))

                # Add speaker region annotations
                colors = [
                    "rgba(76, 175, 80, 0.3)",   # Green
                    "rgba(255, 152, 0, 0.3)",   # Orange
                    "rgba(233, 30, 99, 0.3)",   # Pink
                    "rgba(156, 39, 176, 0.3)",  # Purple
                    "rgba(33, 150, 243, 0.3)",  # Blue
                    "rgba(255, 193, 7, 0.3)",   # Yellow
                    "rgba(0, 188, 212, 0.3)",   # Cyan
                    "rgba(121, 85, 72, 0.3)",   # Brown
                ]

                cluster_color_map = {}
                for region in speaker_regions:
                    cluster_id = region.get("cluster_id", "")
                    if cluster_id not in cluster_color_map:
                        cluster_color_map[cluster_id] = colors[len(cluster_color_map) % len(colors)]

                    fig.add_vrect(
                        x0=region["start"],
                        x1=region["end"],
                        fillcolor=cluster_color_map[cluster_id],
                        layer="below",
                        line_width=0,
                        annotation_text=region.get("speaker", cluster_id)[:10],
                        annotation_position="top left",
                        annotation=dict(font_size=8),
                    )

                fig.update_layout(
                    title=f"Audio Timeline ({wave_start:.0f}s - {wave_start + wave_duration:.0f}s)",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Amplitude",
                    height=300,
                    showlegend=False,
                    xaxis=dict(range=[wave_start, wave_start + wave_duration]),
                    yaxis=dict(range=[-1, 1]),
                    margin=dict(l=50, r=20, t=40, b=40),
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show legend
                if speaker_regions:
                    st.caption(f"Total duration: {total_duration:.1f}s | Showing {len(speaker_regions)} speaker regions")
            else:
                st.info("No waveform data available")

        except requests.RequestException as e:
            st.warning(f"Could not load waveform: {e}")
        except ImportError:
            st.warning("Plotly not installed. Run: pip install plotly")

st.markdown("---")

# =============================================================================
# Voice Table
# =============================================================================

st.subheader("Voice Clusters")

# Show which audio file is being used for all audio operations
_audio_source_path = _get_audio_file_for_playback(ep_id)
if _audio_source_path and _audio_source_path.exists():
    _audio_source_name = _audio_source_path.name
    _audio_source_type = {
        "episode_final_voice_only.wav": "🎧 Final Voice-Only (enhanced, separated)",
        "episode_vocals_enhanced.wav": "🎤 Enhanced Vocals (resemble-enhance)",
        "episode_vocals.wav": "🔊 Raw Vocals (MDX separation)",
    }.get(_audio_source_name, f"📁 {_audio_source_name}")

    st.info(
        f"**Audio Source:** {_audio_source_type}\n\n"
        f"`{_audio_source_path}`\n\n"
        "ℹ️ *This same file is used for diarization, transcription, and segment extraction.*"
    )
else:
    st.warning(
        "⚠️ **No audio file found.** Expected one of:\n"
        "- `episode_final_voice_only.wav`\n"
        "- `episode_vocals_enhanced.wav`\n"
        "- `episode_vocals.wav`"
    )

if not voice_clusters:
    st.info("No voice clusters found.")
    st.stop()

# Filters (Feature #7 + #5)
filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 2])
with filter_col1:
    show_low_confidence = st.checkbox(
        "🔴 Low Confidence",
        key=f"filter_low_conf_{ep_id}",
        help="Filter to show only segments with ASR confidence < 60%",
    )
with filter_col2:
    show_poor_quality = st.checkbox(
        "⚠️ Poor Quality",
        key=f"filter_poor_quality_{ep_id}",
        help="Filter to show only segments with poor audio quality (noisy/clipping)",
    )
with filter_col3:
    # Calculate stats
    low_conf_count = sum(
        1 for row in asr_raw
        if row.get("confidence") is not None and row.get("confidence") < 0.6
    )
    total_asr = len([r for r in asr_raw if r.get("confidence") is not None])
    if total_asr > 0:
        st.caption(f"📊 {low_conf_count}/{total_asr} low confidence")
    else:
        st.caption("📊 No confidence data")

# Load segment quality data (Feature #5)
segment_quality_lookup = {}
try:
    quality_resp = helpers.api_get(f"/jobs/episodes/{ep_id}/audio/segments/quality")
    for q in quality_resp.get("segments", []):
        key = (round(q["start"], 1), round(q["end"], 1))
        segment_quality_lookup[key] = q
    quality_summary = quality_resp.get("summary", {})
    if quality_summary.get("poor", 0) > 0:
        st.caption(
            f"🔊 Quality: {quality_summary.get('good', 0)} good, "
            f"{quality_summary.get('fair', 0)} fair, "
            f"{quality_summary.get('poor', 0)} poor"
        )
except requests.RequestException:
    pass  # Quality data optional

# Sort clusters by total duration (descending)
sorted_clusters = sorted(voice_clusters, key=lambda c: c.get("total_duration", 0), reverse=True)

# Session state for expanded cluster
if "expanded_cluster" not in st.session_state:
    st.session_state.expanded_cluster = None

# Render voice table
for cluster in sorted_clusters:
    cluster_id = cluster.get("voice_cluster_id", "")
    segments = cluster.get("segments", [])
    total_duration = cluster.get("total_duration", 0.0)
    segment_count = cluster.get("segment_count", len(segments))

    # Get mapping info
    map_entry = mapping_lookup.get(cluster_id, {})
    speaker_id = map_entry.get("speaker_id", "SPK_UNKNOWN")
    speaker_display_name = map_entry.get("speaker_display_name", "Unknown")
    voice_bank_id = map_entry.get("voice_bank_id", "")
    similarity = map_entry.get("similarity")
    is_labeled = similarity is not None

    # Determine status badge
    if is_labeled:
        status_badge = "✅"
        status_text = "Labeled"
    else:
        status_badge = "❓"
        status_text = "Unlabeled"

    # Create expander for this cluster
    with st.expander(
        f"{status_badge} **{cluster_id}** - {speaker_display_name} "
        f"({_format_duration(total_duration)}, {segment_count} segments)",
        expanded=st.session_state.expanded_cluster == cluster_id,
    ):
        # Cluster metadata
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.write(f"**Cluster ID:** {cluster_id}")
            st.write(f"**Speaker ID:** {speaker_id}")
            st.write(f"**Display Name:** {speaker_display_name}")
        with meta_col2:
            st.write(f"**Voice Bank ID:** {voice_bank_id}")
            st.write(f"**Similarity:** {similarity:.2f}" if similarity is not None else "**Similarity:** N/A")
            st.write(f"**Status:** {status_text}")

        st.markdown("---")

        # Audio snippets section - show ALL segments with extracted clips
        st.write(f"**All Audio Segments ({len(segments)}):**")

        # Guardrails: default to first 5 clips to avoid spiking CPU with dozens of ffmpeg extractions
        default_clip_count = 5
        clip_limit_key = f"voice_clip_limit::{cluster_id}"
        if clip_limit_key not in st.session_state:
            st.session_state[clip_limit_key] = default_clip_count

        max_clips = st.session_state[clip_limit_key]
        more_available = len(segments) > max_clips
        visible_segments = segments[:max_clips]

        # Build dropdown options for moving segments - show cast members instead of cluster IDs
        # Build a mapping from cast_id to cluster that's assigned to them
        cast_to_cluster = {}
        for m in voice_mapping:
            m_cluster_id = m.get("voice_cluster_id", "")
            m_speaker_id = m.get("speaker_id", "")
            # Extract cast_id from speaker_id (format: SPK_CASTID)
            if m_speaker_id.startswith("SPK_") and m.get("similarity") is not None:
                cast_id_from_speaker = m_speaker_id[4:].lower()  # Remove SPK_ prefix
                cast_to_cluster[cast_id_from_speaker] = m_cluster_id

        # Build move options from cast members
        move_options = ["(keep here)"]
        move_values = [None]
        for cm in cast_members:
            cm_id = cm.get("cast_id", "")
            cm_name = cm.get("name", cm_id)
            # Check if this cast member has a cluster assigned
            existing_cluster = cast_to_cluster.get(cm_id.lower())
            if existing_cluster and existing_cluster != cluster_id:
                move_options.append(f"→ {cm_name}")
                move_values.append({"type": "cast", "cast_id": cm_id, "target_cluster": existing_cluster})
            elif not existing_cluster:
                move_options.append(f"+ {cm_name} (new)")
                move_values.append({"type": "cast_new", "cast_id": cm_id, "name": cm_name})

        # Also add other clusters that aren't assigned to anyone (for unlabeled voices)
        for c in sorted_clusters:
            c_id = c.get("voice_cluster_id", "")
            if c_id == cluster_id:
                continue
            # Check if this cluster is assigned to a cast member
            c_map = mapping_lookup.get(c_id, {})
            if c_map.get("similarity") is None:
                # Unlabeled cluster - show as option
                c_name = c_map.get("speaker_display_name", f"Unlabeled {c_id}")
                move_options.append(f"→ {c_name} ({c_id})")
                move_values.append({"type": "cluster", "target_cluster": c_id})

        if segments:
            # Apply filters if enabled
            if show_low_confidence or show_poor_quality:
                filtered_visible = []
                for seg in visible_segments:
                    seg_start = seg.get("start", 0)
                    seg_end = seg.get("end", 0)

                    passes_filter = False

                    # Check low confidence filter
                    if show_low_confidence:
                        seg_transcripts = _get_transcript_for_segment(asr_raw, seg_start, seg_end)
                        has_low_conf = any(
                            t.get("confidence") is not None and t.get("confidence") < 0.6
                            for t in seg_transcripts
                        )
                        if has_low_conf:
                            passes_filter = True

                    # Check poor quality filter
                    if show_poor_quality:
                        quality_key = (round(seg_start, 1), round(seg_end, 1))
                        quality = segment_quality_lookup.get(quality_key, {})
                        if quality.get("status") in ("poor", "fair"):
                            passes_filter = True

                    if passes_filter:
                        filtered_visible.append(seg)

                visible_segments = filtered_visible
                if not visible_segments:
                    st.info("No segments matching filter in this cluster")

            for i, seg in enumerate(visible_segments):
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)
                seg_duration = seg_end - seg_start

                # Get quality info for this segment (Feature #5)
                quality_key = (round(seg_start, 1), round(seg_end, 1))
                seg_quality = segment_quality_lookup.get(quality_key, {})
                quality_badge = seg_quality.get("badge", "")
                quality_snr = seg_quality.get("snr_db")
                has_overlap = seg_quality.get("has_overlap", False)

                col_time, col_audio, col_move = st.columns([1, 2, 1])
                with col_time:
                    # Show quality badge next to segment number
                    quality_tip = ""
                    if quality_snr is not None:
                        quality_tip = f"SNR: {quality_snr:.1f}dB"
                        if has_overlap:
                            quality_tip += " | Overlap detected"
                    st.write(f"Seg {i+1}: {quality_badge}")
                    st.caption(f"{_format_duration(seg_start)} - {_format_duration(seg_end)} ({seg_duration:.1f}s)")
                    if quality_tip:
                        st.caption(quality_tip)
                with col_audio:
                    # Skip extremely long segments to avoid runaway ffmpeg CPU usage
                    if seg_duration > 120:
                        st.caption("Segment >120s, skip clip (too heavy)")
                    else:
                        # Extract and play individual segment clip
                        segment_file = _extract_segment_audio(ep_id, cluster_id, i, seg_start, seg_end)
                        if segment_file and segment_file.exists():
                            st.audio(str(segment_file), format="audio/wav")
                        else:
                            # Fallback to cached full audio bytes (single read per file)
                            audio_path = _get_audio_file_for_playback(ep_id)
                            audio_bytes = _load_audio_bytes(audio_path) if audio_path else None
                            if audio_bytes:
                                st.audio(audio_bytes, format="audio/wav", start_time=int(seg_start))
                            else:
                                st.caption("Audio unavailable")
                with col_move:
                    # Move segment dropdown - shows cast members and unlabeled clusters
                    move_idx = st.selectbox(
                        "Assign to:",
                        options=range(len(move_options)),
                        format_func=lambda idx: move_options[idx],
                        key=f"move_{cluster_id}_{i}",
                        label_visibility="collapsed",
                    )
                    move_val = move_values[move_idx]
                    if move_val:
                        if st.button("Assign", key=f"move_btn_{cluster_id}_{i}", use_container_width=True):
                            try:
                                if move_val.get("type") == "cast_new":
                                    # Create new cluster for this cast member and move segment there
                                    payload = {
                                        "segment_start": seg_start,
                                        "segment_end": seg_end,
                                        "from_cluster_id": cluster_id,
                                        "cast_id": move_val["cast_id"],
                                    }
                                    with st.spinner(f"Creating voice cluster for {move_val['name']}..."):
                                        resp = helpers.api_post(f"/jobs/episodes/{ep_id}/audio/segments/assign_cast", json=payload)
                                    if resp.get("success"):
                                        new_cid = resp.get("new_cluster_id", "new cluster")
                                        st.success(f"✅ Created {new_cid} for {move_val['name']}")
                                        time.sleep(0.5)
                                        st.rerun()
                                    else:
                                        st.error(f"Failed: {resp}")
                                elif move_val.get("type") == "cast":
                                    # Move segment to existing cluster assigned to cast member
                                    target_cluster = move_val["target_cluster"]
                                    payload = {
                                        "segment_start": seg_start,
                                        "segment_end": seg_end,
                                        "from_cluster_id": cluster_id,
                                        "to_cluster_id": target_cluster,
                                    }
                                    resp = helpers.api_post(f"/jobs/episodes/{ep_id}/audio/segments/move", json=payload)
                                    if resp.get("success"):
                                        st.success(f"Moved to {move_val['cast_id']}'s cluster")
                                        time.sleep(0.5)
                                        st.rerun()
                                    else:
                                        st.error(f"Failed: {resp}")
                                else:
                                    # Move to another unlabeled cluster
                                    target_cluster = move_val["target_cluster"]
                                    payload = {
                                        "segment_start": seg_start,
                                        "segment_end": seg_end,
                                        "from_cluster_id": cluster_id,
                                        "to_cluster_id": target_cluster,
                                    }
                                    resp = helpers.api_post(f"/jobs/episodes/{ep_id}/audio/segments/move", json=payload)
                                    if resp.get("success"):
                                        st.success(f"Moved to {target_cluster}")
                                        time.sleep(0.5)
                                        st.rerun()
                                    else:
                                        st.error(f"Failed: {resp}")
                            except requests.RequestException as exc:
                                st.error(f"API error: {exc}")

                # Show transcript text below audio player
                seg_transcripts = _get_transcript_for_segment(asr_raw, seg_start, seg_end)
                if seg_transcripts:
                    # Check if multiple utterances (potential multiple speakers)
                    if len(seg_transcripts) > 1:
                        warn_col, split_col = st.columns([3, 1])
                        with warn_col:
                            st.warning(f"⚠️ Multiple utterances ({len(seg_transcripts)}) - may contain different speakers")
                        with split_col:
                            if st.button("🔀 Smart Split", key=f"split_{cluster_id}_{i}", help="Split and auto-assign by voice similarity"):
                                try:
                                    # Build split points from ASR boundaries
                                    split_points = [{"start": t["start"], "end": t["end"]} for t in seg_transcripts]
                                    payload = {
                                        "cluster_id": cluster_id,
                                        "segment_start": seg_start,
                                        "segment_end": seg_end,
                                        "split_points": split_points,
                                        "auto_assign": True,
                                        "min_similarity": 0.65,
                                    }
                                    with st.spinner("Analyzing voice embeddings..."):
                                        resp = helpers.api_post(f"/jobs/episodes/{ep_id}/audio/segments/smart_split", json=payload)
                                    if resp.get("success"):
                                        moved = resp.get("moved_to_other_clusters", 0)
                                        stayed = resp.get("stayed_in_original", 0)
                                        suggestions = resp.get("voice_bank_suggestions", [])

                                        if moved > 0:
                                            st.success(f"✅ Split: {moved} moved to matching clusters, {stayed} stayed")
                                        else:
                                            st.info(f"Split into {len(split_points)} segments (all kept in this cluster)")

                                        # Show cast suggestions from voice bank
                                        if suggestions:
                                            st.markdown("**🎭 Suggested Cast Assignments:**")
                                            for sug in suggestions:
                                                st.write(
                                                    f"• {sug['cluster_id']} → **{sug['suggested_name']}** "
                                                    f"({sug['similarity']:.0%} match)"
                                                )
                                        time.sleep(0.8)
                                        st.rerun()
                                    else:
                                        st.error(f"Failed: {resp}")
                                except requests.RequestException as exc:
                                    st.error(f"API error: {exc}")
                    for t_idx, t in enumerate(seg_transcripts):
                        t_start = t["start"]
                        t_end = t["end"]
                        t_text = t["text"]
                        t_conf = t.get("confidence")

                        # Check if this segment is archived
                        is_archived = _is_segment_archived(archived_segments, t_start, t_end)

                        # Format: [time] [confidence badge] "text" with color based on confidence
                        time_badge = f"[{_format_duration(t_start)}-{_format_duration(t_end)}]"
                        conf_color = _get_confidence_color(t_conf)
                        conf_badge = _get_confidence_badge(t_conf)
                        conf_pct = f"{t_conf:.0%}" if t_conf is not None else "?"

                        # Use columns for text + archive button
                        t_col_text, t_col_btn = st.columns([6, 1])

                        with t_col_text:
                            if is_archived:
                                # Show as archived (strikethrough, muted)
                                st.markdown(
                                    f'<span style="color:#888;font-size:0.8em">{time_badge}</span> '
                                    f'<span style="color:#666;font-size:0.75em">🗄️</span> '
                                    f'<span style="color:#666;text-decoration:line-through">"{t_text}"</span>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f'<span style="color:#888;font-size:0.8em">{time_badge}</span> '
                                    f'<span style="color:{conf_color};font-size:0.75em" title="Confidence: {conf_pct}">{conf_badge}</span> '
                                    f'<span style="color:{conf_color}">"{t_text}"</span>',
                                    unsafe_allow_html=True,
                                )

                        with t_col_btn:
                            btn_key = f"archive_{cluster_id}_{i}_{t_idx}_{int(t_start*1000)}"
                            if is_archived:
                                # Show restore button
                                if st.button("↩️", key=btn_key, help="Restore to voicebank"):
                                    try:
                                        resp = helpers.api_post(
                                            f"/jobs/episodes/{ep_id}/audio/segments/restore",
                                            json={
                                                "start": t_start,
                                                "end": t_end,
                                                "text": t_text,
                                                "voice_cluster_id": cluster_id,
                                            },
                                        )
                                        if resp.get("success"):
                                            st.toast(f"Restored segment")
                                            time.sleep(0.3)
                                            st.rerun()
                                    except requests.RequestException as exc:
                                        st.error(f"Error: {exc}")
                            else:
                                # Show archive button
                                if st.button("🗑️", key=btn_key, help="Archive (exclude from voicebank)"):
                                    try:
                                        resp = helpers.api_post(
                                            f"/jobs/episodes/{ep_id}/audio/segments/archive",
                                            json={
                                                "start": t_start,
                                                "end": t_end,
                                                "text": t_text,
                                                "voice_cluster_id": cluster_id,
                                            },
                                        )
                                        if resp.get("success"):
                                            st.toast(f"Archived segment")
                                            time.sleep(0.3)
                                            st.rerun()
                                    except requests.RequestException as exc:
                                        st.error(f"Error: {exc}")
                else:
                    st.caption("(no transcript)")
        else:
            st.info("No segments for this cluster.")

        if more_available:
            if st.button(
                f"Show next {min(10, len(segments) - max_clips)} segments",
                key=f"more_segments_{cluster_id}",
                use_container_width=True,
            ):
                st.session_state[clip_limit_key] = min(len(segments), max_clips + 10)
                st.rerun()

        st.markdown("---")

        # Transcript excerpts
        st.write("**Transcript Excerpts:**")
        cluster_transcript = [t for t in transcript if t.get("voice_cluster_id") == cluster_id]

        if cluster_transcript:
            # Show up to 5 transcript excerpts
            for row in cluster_transcript[:5]:
                start = row.get("start", 0)
                text = row.get("text", "")
                truncated = text[:200] + "..." if len(text) > 200 else text
                st.write(f"[{_format_duration(start)}] {truncated}")
        else:
            st.info("No transcript excerpts for this cluster.")

        st.markdown("---")

        # Assignment controls
        st.write("**Assign Voice:**")

        assign_col1, assign_col2 = st.columns(2)

        with assign_col1:
            # Cast member dropdown
            cast_options = ["(None)"] + [f"{m.get('name', m.get('cast_id', 'Unknown'))} ({m.get('cast_id', '')})" for m in cast_members]
            cast_values = [None] + [m.get("cast_id") for m in cast_members]

            selected_cast_idx = st.selectbox(
                "Assign to cast member:",
                options=range(len(cast_options)),
                format_func=lambda i: cast_options[i],
                key=f"cast_select_{cluster_id}",
            )
            selected_cast_id = cast_values[selected_cast_idx] if selected_cast_idx > 0 else None

        with assign_col2:
            # Custom label input
            custom_label = st.text_input(
                "Or enter custom label:",
                placeholder="e.g., Narrator, Producer VO",
                key=f"custom_label_{cluster_id}",
            )

        # Save button
        if st.button(f"Save Assignment", key=f"save_{cluster_id}", type="primary"):
            if not selected_cast_id and not custom_label:
                st.warning("Please select a cast member or enter a custom label.")
            else:
                with st.spinner("Saving assignment..."):
                    result = _assign_voice(ep_id, cluster_id, selected_cast_id, custom_label if not selected_cast_id else None)

                    if "error" in result:
                        st.error(f"Failed to save: {result['error']}")
                    else:
                        st.success(
                            f"Saved: {cluster_id} → {result.get('speaker_display_name', 'Unknown')}"
                        )
                        # Note: The API persists the mapping to file, so we just need to rerun
                        # to reload fresh data. No need to manually update local mapping.
                        # Rerun to refresh
                        st.rerun()

st.markdown("---")

# =============================================================================
# Speaker Similarity Heatmap (Feature #2) - At bottom
# =============================================================================

with st.expander("📊 Speaker Similarity Heatmap", expanded=False):
    st.markdown(
        "Visual matrix showing cosine similarity between voice cluster centroids. "
        "Higher values (warmer colors) indicate voices that may be confused."
    )

    try:
        matrix_resp = helpers.api_get(f"/jobs/episodes/{ep_id}/audio/clusters/similarity_matrix")
        labels = matrix_resp.get("labels", [])
        matrix = matrix_resp.get("matrix", [])
        confusable = matrix_resp.get("confusable_pairs", [])

        if matrix and len(matrix) >= 2:
            import plotly.graph_objects as go
            import plotly.express as px

            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=matrix,
                x=labels,
                y=labels,
                colorscale=[
                    [0, '#2E7D32'],      # Green for low similarity
                    [0.5, '#FFC107'],    # Yellow for medium
                    [0.7, '#FF9800'],    # Orange for high
                    [1, '#F44336'],      # Red for very high
                ],
                zmin=0,
                zmax=1,
                hoverongaps=False,
                hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Similarity: %{z:.1%}<extra></extra>",
            ))

            fig.update_layout(
                title="Voice Cluster Similarity Matrix",
                xaxis_title="",
                yaxis_title="",
                width=600,
                height=500,
                xaxis=dict(side="bottom"),
                yaxis=dict(autorange="reversed"),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show confusable pairs
            if confusable:
                st.markdown("**⚠️ Confusable Voice Pairs (>70% similarity):**")
                for pair in confusable[:5]:
                    st.write(
                        f"- **{pair['cluster_a']}** ↔ **{pair['cluster_b']}**: "
                        f"{pair['similarity']:.0%} similarity"
                    )
        else:
            st.info(matrix_resp.get("message", "Not enough clusters for similarity matrix"))

    except requests.RequestException as e:
        st.warning(f"Could not load similarity matrix: {e}")
    except ImportError:
        st.warning("Plotly not installed. Run: pip install plotly")

st.markdown("---")

# =============================================================================
# Voice Analytics (Feature #8) - Moved to bottom
# =============================================================================

with st.expander("📈 Voice Analytics (Cross-Episode)", expanded=False):
    st.markdown(
        "Track speaking patterns across all episodes in this show. "
        "See which cast members appear most and their total speaking time."
    )

    try:
        analytics_resp = helpers.api_get(f"/jobs/shows/{show_id}/voice_analytics")

        episode_count = analytics_resp.get("episode_count", 0)
        cast_count = analytics_resp.get("cast_count", 0)
        cast_stats = analytics_resp.get("cast_stats", [])
        chart_data = analytics_resp.get("chart_data", [])

        if episode_count > 0:
            st.info(f"**{cast_count}** cast members across **{episode_count}** episodes with voice data")

            # Top speakers table
            if cast_stats:
                st.markdown("**Top Speakers (by total speaking time):**")

                for i, cast in enumerate(cast_stats[:10]):
                    name = cast.get("name", "Unknown")
                    total_time = cast.get("total_speaking_time", 0)
                    ep_count = cast.get("episode_count", 0)
                    first_ep = cast.get("first_appearance", "")

                    # Format time as mm:ss
                    mins = int(total_time // 60)
                    secs = int(total_time % 60)
                    time_str = f"{mins}:{secs:02d}"

                    st.write(
                        f"**{i+1}. {name}**: {time_str} total | "
                        f"{ep_count} episodes | "
                        f"First: {first_ep.split('-')[-1] if first_ep else '?'}"
                    )

            # Show chart if plotly is available
            if chart_data:
                st.markdown("---")
                st.markdown("**Speaking Time by Episode:**")

                try:
                    import plotly.express as px
                    import pandas as pd

                    df = pd.DataFrame(chart_data)

                    fig = px.bar(
                        df,
                        x="episode",
                        y="speaking_time",
                        color="cast_member",
                        barmode="stack",
                        title="Speaking Time per Cast Member",
                        labels={
                            "episode": "Episode",
                            "speaking_time": "Speaking Time (s)",
                            "cast_member": "Cast Member",
                        },
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                except ImportError:
                    st.caption("Install plotly and pandas for visualization: pip install plotly pandas")
        else:
            st.info(analytics_resp.get("message", "No voice data found for this show"))

    except requests.RequestException as e:
        st.warning(f"Could not load voice analytics: {e}")

st.markdown("---")

# Navigation
st.write("**Navigation:**")
nav_col1, nav_col2 = st.columns(2)

with nav_col1:
    if st.button("← Back to Episode Detail", use_container_width=True):
        helpers.set_ep_id(ep_id, rerun=False)
        helpers.try_switch_page("pages/2_Episode_Detail.py")

with nav_col2:
    if st.button("Refresh Data", use_container_width=True):
        st.rerun()

# No page-level auto-refresh; streaming handlers update logs/progress live

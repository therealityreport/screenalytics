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
        "audio_vocals": audio_dir / "episode_vocals_enhanced.wav",
        "audio_original": audio_dir / "episode_vocals.wav",
        "voice_clusters": manifests_dir / "audio_voice_clusters.json",
        "voice_mapping": manifests_dir / "audio_voice_mapping.json",
        "transcript_jsonl": manifests_dir / "episode_transcript.jsonl",
        "audio_qc": manifests_dir / "audio_qc.json",
        "voice_segments": artifacts_dir / "voices",
    }


def _get_segment_audio_path(ep_id: str, cluster_id: str, segment_idx: int) -> Path:
    """Get path for a segment audio file."""
    paths = _get_audio_paths(ep_id)
    segments_dir = paths["voice_segments"] / cluster_id
    return segments_dir / f"segment_{segment_idx:03d}.wav"


def _extract_segment_audio(ep_id: str, cluster_id: str, segment_idx: int, start: float, end: float) -> Optional[Path]:
    """Extract a segment from the source audio and save it as a clip.

    Returns the path to the segment file, or None if extraction fails.
    """
    segment_path = _get_segment_audio_path(ep_id, cluster_id, segment_idx)

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
    for key in ["audio_final", "audio_vocals", "audio_original"]:
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
            # Redis mode - queue via Celery
            try:
                payload = {
                    "ep_id": ep_id,
                    "overwrite": audio_overwrite,
                    "asr_provider": asr_provider,
                    "run_mode": "queue",
                }
                resp = helpers.api_post("/jobs/episode_audio_pipeline", json=payload, timeout=30)

                job_id = resp.get("job_id")
                if job_id:
                    helpers.store_celery_job_id(ep_id, "audio_pipeline", job_id)
                    st.success(f"Audio pipeline queued via Celery: {job_id}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Failed to start audio pipeline: {resp}")
            except requests.RequestException as exc:
                st.error(helpers.describe_error("/jobs/episode_audio_pipeline", exc))

# Show previous logs for audio pipeline OUTSIDE the expander to avoid nesting
if execution_mode == "local":
    helpers.render_previous_logs(ep_id, "audio_pipeline")

if not has_voice_clusters and not has_voice_mapping:
    st.info("Run the audio pipeline above to generate voice clusters for review.")
    st.stop()

# Load data
voice_clusters = _load_voice_clusters(ep_id)
voice_mapping = _load_voice_mapping(ep_id)
transcript = _load_transcript(ep_id)
qc_data = _load_qc_status(ep_id)
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
# Voice Table
# =============================================================================

st.subheader("Voice Clusters")

if not voice_clusters:
    st.info("No voice clusters found.")
    st.stop()

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

        if segments:
            for i, seg in enumerate(visible_segments):
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)
                seg_duration = seg_end - seg_start

                col_time, col_audio = st.columns([1, 3])
                with col_time:
                    st.write(f"Seg {i+1}:")
                    st.caption(f"{_format_duration(seg_start)} - {_format_duration(seg_end)} ({seg_duration:.1f}s)")
                with col_audio:
                    # Skip extremely long segments to avoid runaway ffmpeg CPU usage
                    if seg_duration > 120:
                        st.caption("Segment >120s, skip clip (too heavy)")
                        continue

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

# Auto-refresh when audio pipeline is running
# Use st.rerun with a session state counter to avoid blocking the UI
if running_audio_job:
    # Store refresh timestamp in session state
    auto_refresh_key = f"audio_autorefresh_{ep_id}"
    last_refresh = st.session_state.get(auto_refresh_key, 0)
    current_time = time.time()

    # Only auto-refresh every 5 seconds (non-blocking check)
    if current_time - last_refresh >= 5:
        st.session_state[auto_refresh_key] = current_time
        st.rerun()

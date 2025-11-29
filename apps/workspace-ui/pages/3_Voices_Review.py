"""Voices Review page for labeling voice clusters.

This page allows reviewing and labeling unique voice clusters identified
by the audio pipeline for an episode.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402

# Page config
st.set_page_config(
    page_title="Voices Review - Screenalytics",
    page_icon="🎙️",
    layout="wide",
)

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

    return {
        "audio_final": audio_dir / "episode_final_voice_only.wav",
        "audio_vocals": audio_dir / "episode_vocals_enhanced.wav",
        "audio_original": audio_dir / "episode_vocals.wav",
        "voice_clusters": manifests_dir / "audio_voice_clusters.json",
        "voice_mapping": manifests_dir / "audio_voice_mapping.json",
        "transcript_jsonl": manifests_dir / "episode_transcript.jsonl",
        "audio_qc": manifests_dir / "audio_qc.json",
    }


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
    """Extract show_id from ep_id."""
    if "-" in ep_id:
        return ep_id.rsplit("-", 2)[0].lower()
    return ep_id.lower()


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

# Initialize page
helpers.init_page()

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

# Check for voice artifacts
paths = _get_audio_paths(ep_id)
has_voice_clusters = paths["voice_clusters"].exists()
has_voice_mapping = paths["voice_mapping"].exists()

if not has_voice_clusters and not has_voice_mapping:
    st.error(
        "Audio pipeline has not produced voice clusters yet.\n\n"
        "Go to **Episode Detail → Audio & Transcript** and run the pipeline first."
    )
    if st.button("Go to Episode Detail"):
        helpers.set_ep_id(ep_id, rerun=False)
        helpers.try_switch_page("pages/2_Episode_Detail.py")
    st.stop()

# Load data
voice_clusters = _load_voice_clusters(ep_id)
voice_mapping = _load_voice_mapping(ep_id)
transcript = _load_transcript(ep_id)
qc_data = _load_qc_status(ep_id)
show_id = _get_show_id(ep_id)
cast_members = _load_cast_members(show_id)

# Build mapping lookup
mapping_lookup = {m["voice_cluster_id"]: m for m in voice_mapping}

# Calculate stats
voice_cluster_count = len(voice_clusters)
labeled_count = sum(1 for m in voice_mapping if m.get("similarity") is not None)
unlabeled_count = voice_cluster_count - labeled_count

# =============================================================================
# Page Header and Status
# =============================================================================

st.markdown("---")

# Status card
col1, col2, col3 = st.columns(3)

with col1:
    pipeline_status = "Succeeded" if has_voice_mapping else "Partial"
    qc_status = qc_data.get("status", "unknown")
    qc_badge = "✅" if qc_status == "ok" else ("⚠️" if qc_status == "warn" else "🔴")
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
            st.write(f"**Similarity:** {similarity:.2f}" if similarity else "**Similarity:** N/A")
            st.write(f"**Status:** {status_text}")

        st.markdown("---")

        # Audio snippets section
        st.write("**Sample Audio Snippets:**")

        audio_path = _get_audio_file_for_playback(ep_id)
        if audio_path and segments:
            # Show up to 3 sample segments
            sample_segments = segments[:3]
            for i, seg in enumerate(sample_segments):
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)
                col_time, col_audio = st.columns([1, 3])
                with col_time:
                    st.write(f"Segment {i+1}:")
                    st.caption(f"{_format_duration(seg_start)} - {_format_duration(seg_end)}")
                with col_audio:
                    # Note: Full audio file - user can seek manually
                    st.audio(str(audio_path), format="audio/wav", start_time=int(seg_start))
        else:
            st.info("Audio file not available for playback.")

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
                        # Update local mapping
                        if cluster_id in mapping_lookup:
                            mapping_lookup[cluster_id]["speaker_id"] = result.get("speaker_id", "")
                            mapping_lookup[cluster_id]["speaker_display_name"] = result.get("speaker_display_name", "")
                            mapping_lookup[cluster_id]["voice_bank_id"] = result.get("voice_bank_id", "")
                            mapping_lookup[cluster_id]["similarity"] = 1.0
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

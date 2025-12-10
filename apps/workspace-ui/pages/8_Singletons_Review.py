"""Singletons Review Page - Review and assign singleton clusters.

This page shows all singleton clusters (clusters with only 1 track) for review
AFTER the main Faces Review workflow is complete. Singletons often include:
- Brief appearances of cast members
- Background extras
- Low-quality/skipped faces that couldn't be clustered
- Noise detections

Use this page to:
1. Assign singletons to known cast members
2. Mark singletons as "noise" to exclude from screentime
3. Review and clean up after main cast assignment
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import os

import ui_helpers as helpers  # noqa: E402
from similarity_badges import (  # noqa: E402
    SimilarityType,
    render_similarity_badge,
    render_singleton_risk_badge,
)

LOGGER = logging.getLogger(__name__)

cfg = helpers.init_page("Singletons Review")

st.title("Singletons Review")

# ============================================================================
# API Client Setup
# ============================================================================

API_BASE = st.session_state.get("api_base") or os.environ.get("API_BASE", "http://127.0.0.1:8000")


def _api_session() -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _api_get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Make a GET request to the API."""
    try:
        session = _api_session()
        url = f"{API_BASE}{endpoint}"
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        LOGGER.error("API GET %s failed: %s", endpoint, e)
        return {"error": str(e)}


def _api_post(endpoint: str, json_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Make a POST request to the API."""
    try:
        session = _api_session()
        url = f"{API_BASE}{endpoint}"
        resp = session.post(url, json=json_data or {}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        LOGGER.error("API POST %s failed: %s", endpoint, e)
        return {"error": str(e)}


# ============================================================================
# Data Loading
# ============================================================================


@st.cache_data(ttl=30)
def load_singletons(ep_id: str) -> Dict[str, Any]:
    """Load singleton clusters AND unclustered tracks for an episode.

    Returns both:
    1. Existing singleton clusters from identities.json (identity_id set)
    2. Unclustered tracks from faces.jsonl (identity_id=None, has track_id)
    """
    # Load existing identities
    result = _api_get(f"/episodes/{ep_id}/identities_with_metrics")
    if "error" in result:
        return {"clusters": [], "singleton_health": {}, "error": result["error"]}

    # API returns "identities" array with track_ids list
    identities = result.get("identities", [])

    # Enrich with counts structure for compatibility
    for identity in identities:
        track_ids = identity.get("track_ids", [])
        faces = identity.get("faces", 0)
        identity["counts"] = {
            "tracks": len(track_ids),
            "faces": faces,
        }
        # Map metrics fields to expected names
        metrics = identity.get("metrics", {})
        identity["identity_similarity"] = metrics.get("cohesion")
        identity["cohesion"] = metrics.get("cohesion")
        # Determine singleton risk based on track and face counts
        if len(track_ids) == 1 and faces == 1:
            identity["singleton_risk"] = "HIGH"
        elif len(track_ids) == 1 and faces <= 3:
            identity["singleton_risk"] = "MEDIUM"
        elif len(track_ids) == 1:
            identity["singleton_risk"] = "LOW"
        else:
            identity["singleton_risk"] = None

    # Filter to only singleton clusters (1 track)
    singletons = [c for c in identities if c.get("counts", {}).get("tracks", 0) == 1]

    # ALSO load unclustered tracks (tracks in faces.jsonl but not in any identity)
    unclustered_result = _api_get(f"/episodes/{ep_id}/unclustered_tracks")
    unclustered_tracks = []
    if "error" not in unclustered_result:
        raw_unclustered = unclustered_result.get("unclustered_tracks", [])
        for track in raw_unclustered:
            # Convert to same format as singletons for unified display
            unclustered_tracks.append({
                "identity_id": None,  # Not yet clustered
                "track_id": track.get("track_id"),  # Store track_id for assignment
                "track_ids": [track.get("track_id")],
                "faces": track.get("faces", 0),
                "counts": {
                    "tracks": 1,
                    "faces": track.get("faces", 0),
                },
                "rep_thumbnail_url": track.get("rep_thumbnail_url"),
                "singleton_risk": track.get("singleton_risk", "HIGH"),
                "person_id": None,
                "name": None,
                "is_unclustered": True,  # Flag to identify unclustered tracks
                "faces_skipped": track.get("faces_skipped", 0),
                "faces_with_embeddings": track.get("faces_with_embeddings", 0),
            })

    # Combine existing singletons and unclustered tracks
    all_items = singletons + unclustered_tracks

    # Calculate singleton health metrics (including unclustered)
    total_identities = len(identities)
    total_items = len(all_items)
    singleton_count = len(singletons)
    unclustered_count = len(unclustered_tracks)
    high_risk_count = sum(1 for s in all_items if s.get("singleton_risk") == "HIGH")
    single_frame_count = sum(1 for s in all_items if s.get("counts", {}).get("faces", 0) == 1)

    singleton_health = {
        "singleton_fraction": singleton_count / total_identities if total_identities > 0 else 0,
        "high_risk_count": high_risk_count,
        "single_frame_tracks": single_frame_count,
        "unclustered_count": unclustered_count,
        "health_status": "critical" if unclustered_count > 50 else (
            "warning" if unclustered_count > 20 else "healthy"
        ),
    }

    return {
        "clusters": all_items,
        "singleton_health": singleton_health,
        "total_clusters": total_identities,
        "unclustered_count": unclustered_count,
    }


@st.cache_data(ttl=60)
def load_cast_members(ep_id: str) -> List[Dict[str, Any]]:
    """Load cast members for assignment dropdown."""
    # Extract show from ep_id (e.g., rhoslc-s06e99 -> RHOSLC)
    import re
    match = re.match(r"^(?P<show>.+)-s\d{2}e\d{2}$", ep_id, re.IGNORECASE)
    if not match:
        return []

    show_id = match.group("show").upper()
    result = _api_get(f"/shows/{show_id}/cast")
    if "error" in result:
        return []

    return result.get("cast", [])


@st.cache_data(ttl=30)
def load_cast_suggestions(ep_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load cast suggestions for all clusters, keyed by cluster_id."""
    result = _api_get(f"/episodes/{ep_id}/cast_suggestions")
    if "error" in result:
        return {}

    suggestions = result.get("suggestions", [])
    # Build map: cluster_id -> list of cast suggestions
    suggestions_map = {}
    for s in suggestions:
        cluster_id = s.get("cluster_id")
        cast_suggestions = s.get("cast_suggestions", [])
        if cluster_id and cast_suggestions:
            suggestions_map[cluster_id] = cast_suggestions

    return suggestions_map


def get_thumbnail_url(ep_id: str, cluster: Dict[str, Any]) -> Optional[str]:
    """Get the best thumbnail URL for a cluster."""
    # Try rep_thumbnail_url first (API provides signed S3 URL)
    rep_url = cluster.get("rep_thumbnail_url")
    if rep_url:
        return rep_url

    # Try rep_media_url as fallback
    rep_media = cluster.get("rep_media_url")
    if rep_media:
        return rep_media

    # Try rep_crop for local path resolution
    rep_crop = cluster.get("rep_crop")
    if rep_crop:
        return helpers.resolve_image_url(ep_id, rep_crop)

    # Try tracks[0].thumb_rel_path
    tracks = cluster.get("tracks", [])
    if tracks:
        thumb_path = tracks[0].get("thumb_rel_path")
        if thumb_path:
            return helpers.resolve_image_url(ep_id, thumb_path)

    return None


# ============================================================================
# Actions
# ============================================================================


def assign_to_cast(ep_id: str, cluster_id: str, cast_id: str) -> Dict[str, Any]:
    """Assign a singleton cluster to a cast member."""
    # Use the manual assign endpoint with cluster_ids list
    return _api_post(
        f"/episodes/{ep_id}/clusters/manual",
        {"cluster_ids": [cluster_id], "cast_id": cast_id},
    )


def assign_track_to_cast(ep_id: str, track_id: int, cast_name: str, show: str) -> Dict[str, Any]:
    """Assign an unclustered track to a cast member by name.

    This creates a singleton identity for the track and assigns it in one step.
    """
    return _api_post(
        f"/episodes/{ep_id}/tracks/{track_id}/name",
        {"name": cast_name, "show": show},
    )


def mark_as_noise(ep_id: str, cluster_id: str) -> Dict[str, Any]:
    """Mark a singleton as noise WITHOUT deleting it from identities.json.

    IMPORTANT: This function does NOT delete the cluster. It only marks it as
    noise/dismissed in the tracking system. The cluster remains in identities.json
    so it can be restored later if needed.

    Previously this was calling DELETE which caused data loss by permanently
    removing clusters from identities.json.
    """
    try:
        session = _api_session()
        # Use the dismissed_suggestions endpoint to mark as skipped/noise
        # This preserves the cluster in identities.json
        url = f"{API_BASE}/episodes/{ep_id}/dismissed_suggestions"
        resp = session.post(url, json={"suggestion_ids": [cluster_id]}, timeout=30)
        resp.raise_for_status()
        LOGGER.info("Marked cluster %s as noise (dismissed) in %s", cluster_id, ep_id)
        return {"status": "dismissed", "cluster_id": cluster_id}
    except requests.RequestException as e:
        LOGGER.error("API POST dismissed_suggestions %s failed: %s", cluster_id, e)
        return {"error": str(e)}


def mark_track_as_noise(ep_id: str, track_id: int) -> Dict[str, Any]:
    """Mark an unclustered track as noise by creating and archiving a singleton."""
    # First, we need to create a singleton for this track, then archive it
    # For now, we'll just skip it - unclustered tracks don't have identities to delete
    # TODO: Implement proper noise marking for unclustered tracks
    return {"status": "skipped", "message": "Track not yet clustered - no identity to archive"}


def merge_with_cluster(ep_id: str, source_id: str, target_id: str) -> Dict[str, Any]:
    """Merge a singleton into another cluster."""
    return _api_post(
        f"/episodes/{ep_id}/identities/merge",
        {"source_id": source_id, "target_id": target_id},
    )


# ============================================================================
# UI Components
# ============================================================================


def render_singleton_card(
    ep_id: str,
    cluster: Dict[str, Any],
    cast_members: List[Dict[str, Any]],
    suggestions: List[Dict[str, Any]],
    idx: int,
) -> Optional[str]:
    """Render a singleton cluster card. Returns action taken if any."""
    import re

    cluster_id = cluster.get("identity_id", "")
    track_id = cluster.get("track_id")  # For unclustered tracks
    is_unclustered = cluster.get("is_unclustered", False)
    person_id = cluster.get("person_id")
    # API uses "name" field (can be None)
    person_name = cluster.get("name") or cluster.get("person_name")
    counts = cluster.get("counts", {})
    track_count = counts.get("tracks", 0)
    face_count = counts.get("faces", 0)
    risk_level = cluster.get("singleton_risk", "LOW")

    # Get metrics
    identity_sim = cluster.get("identity_similarity")

    # Get top suggestion if available
    top_suggestion = suggestions[0] if suggestions else None

    # Check if already assigned
    is_assigned = bool(person_id)

    # Display name: for unclustered tracks, show track_id
    if is_unclustered:
        display_name = f"T{track_id}"
        unique_key = f"track_{track_id}"
    else:
        display_name = person_name if person_name else cluster_id
        unique_key = cluster_id or f"unknown_{idx}"

    # Extract show from ep_id for assignment
    match = re.match(r"^(?P<show>.+)-s\d{2}e\d{2}$", ep_id, re.IGNORECASE)
    show_slug = match.group("show").upper() if match else None

    # Card container
    with st.container():
        cols = st.columns([1, 3, 2])

        # Column 1: Thumbnail
        with cols[0]:
            thumb_url = get_thumbnail_url(ep_id, cluster)
            if thumb_url:
                st.image(thumb_url, width=100)
            else:
                st.markdown("🖼️ *No image*")

        # Column 2: Info
        with cols[1]:
            # Title with status badges
            title_html = f"**{display_name}**"
            if is_assigned:
                title_html += " ✅"
            if is_unclustered:
                title_html += " 🆕"  # New/unclustered indicator
            st.markdown(title_html)

            # Risk badge
            risk_badge = render_singleton_risk_badge(track_count, face_count)
            st.markdown(risk_badge, unsafe_allow_html=True)

            # Stats - show extra info for unclustered tracks
            if is_unclustered:
                faces_skipped = cluster.get("faces_skipped", 0)
                faces_with_emb = cluster.get("faces_with_embeddings", 0)
                st.caption(f"Faces: {face_count} (skipped: {faces_skipped}, emb: {faces_with_emb})")
            else:
                st.caption(f"Faces: {face_count} | Tracks: {track_count}")

            # Show top cast suggestion if available and not assigned
            if top_suggestion and not is_assigned:
                sug_name = top_suggestion.get("name", "Unknown")
                sug_sim = top_suggestion.get("similarity", 0)
                sug_conf = top_suggestion.get("confidence", "low")
                # Color code by confidence
                conf_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(sug_conf, "⚪")
                st.markdown(
                    f"**Suggested:** {sug_name} ({sug_sim:.0%}) {conf_color}",
                    help=f"Confidence: {sug_conf}, Similarity: {sug_sim:.2f}",
                )

            # Metrics
            if identity_sim is not None:
                sim_badge = render_similarity_badge(
                    identity_sim,
                    SimilarityType.IDENTITY,
                    show_label=True,
                )
                st.markdown(sim_badge, unsafe_allow_html=True)

        # Column 3: Actions
        with cols[2]:
            action_taken = None

            if not is_assigned:
                # Quick assign button if suggestion available (only for clustered items)
                if top_suggestion and not is_unclustered:
                    sug_cast_id = top_suggestion.get("cast_id", "")
                    sug_name = top_suggestion.get("name", "Unknown")
                    sug_conf = top_suggestion.get("confidence", "low")
                    btn_label = f"⚡ {sug_name[:12]}"
                    btn_disabled = sug_conf == "low"  # Disable for low confidence
                    if st.button(
                        btn_label,
                        key=f"btn_quick_{unique_key}_{idx}",
                        use_container_width=True,
                        disabled=btn_disabled,
                        help=f"Quick assign to {sug_name}" if not btn_disabled else "Low confidence - use manual assign",
                    ):
                        result = assign_to_cast(ep_id, cluster_id, sug_cast_id)
                        if "error" not in result:
                            st.success(f"Assigned to {sug_name}")
                            action_taken = "assigned"
                        else:
                            st.error(f"Failed: {result.get('error')}")

                # Manual assignment dropdown
                cast_options = ["-- Select Cast --"] + [
                    f"{c.get('name', 'Unknown')} ({c.get('cast_id', '')})"
                    for c in cast_members
                ]
                selected = st.selectbox(
                    "Assign to:",
                    cast_options,
                    key=f"assign_{unique_key}_{idx}",
                    label_visibility="collapsed",
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("✅ Assign", key=f"btn_assign_{unique_key}_{idx}", use_container_width=True):
                        if selected != "-- Select Cast --":
                            # Extract cast name from selection for track-based assignment
                            cast_name = selected.split(" (")[0]
                            cast_id = selected.split("(")[-1].rstrip(")")
                            if is_unclustered and show_slug:
                                # For unclustered tracks, use track-based assignment
                                result = assign_track_to_cast(ep_id, track_id, cast_name, show_slug)
                            else:
                                # For clustered singletons, use identity-based assignment
                                result = assign_to_cast(ep_id, cluster_id, cast_id)
                            if "error" not in result:
                                st.success(f"Assigned to {cast_name}")
                                action_taken = "assigned"
                            else:
                                st.error(f"Failed: {result.get('error')}")

                with col_b:
                    if st.button("🗑️ Noise", key=f"btn_noise_{unique_key}_{idx}", use_container_width=True):
                        if is_unclustered:
                            # For unclustered tracks, use track-based noise marking
                            result = mark_track_as_noise(ep_id, track_id)
                        else:
                            result = mark_as_noise(ep_id, cluster_id)
                        if "error" not in result:
                            st.success("Marked as noise")
                            action_taken = "deleted"
                        else:
                            st.error(f"Failed: {result.get('error')}")
            else:
                st.success(f"Assigned: {person_name}")

                # View in Faces Review button (only for clustered items)
                if not is_unclustered:
                    if st.button(
                        "👁️ View",
                        key=f"btn_view_{unique_key}_{idx}",
                        use_container_width=True,
                    ):
                        st.query_params["ep_id"] = ep_id
                        st.query_params["view"] = "cluster"
                        st.query_params["cluster"] = cluster_id
                        st.switch_page("pages/3_Faces_Review.py")

        st.divider()
        return action_taken


# ============================================================================
# Main Page
# ============================================================================

# Episode selector
ep_id = st.query_params.get("ep_id", "")
if not ep_id:
    # Show episode picker
    episodes_resp = _api_get("/episodes")
    episodes = episodes_resp.get("episodes", []) if "error" not in episodes_resp else []

    if not episodes:
        st.warning("No episodes found. Process an episode first.")
        st.stop()

    ep_options = [e.get("ep_id", "") for e in episodes if e.get("ep_id")]
    ep_id = st.selectbox("Select Episode:", ep_options)
    if ep_id:
        st.query_params["ep_id"] = ep_id
        st.rerun()
    st.stop()

st.markdown(f"**Episode:** `{ep_id}`")

# Load data
with st.spinner("Loading singletons and suggestions..."):
    data = load_singletons(ep_id)
    cast_members = load_cast_members(ep_id)
    suggestions_map = load_cast_suggestions(ep_id)

if "error" in data:
    st.error(f"Failed to load data: {data['error']}")
    st.stop()

singletons = data["clusters"]
singleton_health = data["singleton_health"]
total_clusters = data.get("total_clusters", 0)

# Count how many singletons have suggestions
singletons_with_suggestions = sum(
    1 for s in singletons
    if s.get("identity_id") in suggestions_map and not s.get("person_id")
)

# ============================================================================
# Summary Header
# ============================================================================

st.markdown("---")

# Count unclustered tracks
unclustered_count = sum(1 for s in singletons if s.get("is_unclustered", False))
clustered_count = len(singletons) - unclustered_count

# Health metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric(
        "Total Items",
        len(singletons),
        help="Singleton clusters + unclustered tracks",
    )
with col2:
    st.metric(
        "Unclustered",
        unclustered_count,
        help="Tracks not yet assigned to any identity cluster",
    )
with col3:
    singleton_frac = singleton_health.get("singleton_fraction", 0)
    st.metric(
        "Singleton %",
        f"{singleton_frac * 100:.1f}%",
        help="Percentage of all clusters that are singletons",
    )
with col4:
    high_risk = singleton_health.get("high_risk_count", 0)
    st.metric(
        "High Risk",
        high_risk,
        help="Single-track, single-frame clusters (unreliable)",
    )
with col5:
    single_frame = singleton_health.get("single_frame_tracks", 0)
    st.metric(
        "Single-Frame",
        single_frame,
        help="Tracks with only 1 frame",
    )
with col6:
    st.metric(
        "With Suggestions",
        singletons_with_suggestions,
        help="Unassigned singletons with cast suggestions",
    )

# Health status
health_status = singleton_health.get("health_status", "unknown")
if health_status == "critical":
    st.error(
        f"⚠️ High singleton fraction ({singleton_frac * 100:.0f}%) - "
        "Consider re-running clustering with looser thresholds or reviewing detections."
    )
elif health_status == "warning":
    st.warning(
        f"⚠️ Moderate singleton fraction ({singleton_frac * 100:.0f}%) - "
        "Some singletons may be valid brief appearances."
    )

st.markdown("---")

# ============================================================================
# Filters
# ============================================================================

st.subheader("Filters")

filter_cols = st.columns([1, 1, 1, 1, 1, 1])

with filter_cols[0]:
    filter_type = st.selectbox(
        "Type:",
        ["All", "Clustered Only", "Unclustered Only"],
        key="filter_type",
        help="Clustered = existing singleton identities, Unclustered = tracks with no identity",
    )

with filter_cols[1]:
    filter_assigned = st.selectbox(
        "Assignment Status:",
        ["All", "Unassigned Only", "Assigned Only"],
        key="filter_assigned",
    )

with filter_cols[2]:
    filter_risk = st.selectbox(
        "Risk Level:",
        ["All", "HIGH", "MEDIUM", "LOW"],
        key="filter_risk",
    )

with filter_cols[3]:
    filter_suggestions = st.selectbox(
        "Suggestions:",
        ["All", "Has Suggestion", "No Suggestion"],
        key="filter_suggestions",
    )

with filter_cols[4]:
    sort_by = st.selectbox(
        "Sort By:",
        ["Suggestion Confidence", "Risk (High First)", "Faces (Most First)", "Track ID"],
        key="sort_by",
    )

with filter_cols[5]:
    page_size = st.selectbox(
        "Per Page:",
        [10, 25, 50, 100],
        index=1,
        key="page_size",
    )

# Apply filters
filtered = singletons

# Type filter (clustered vs unclustered)
if filter_type == "Clustered Only":
    filtered = [s for s in filtered if not s.get("is_unclustered", False)]
elif filter_type == "Unclustered Only":
    filtered = [s for s in filtered if s.get("is_unclustered", False)]

if filter_assigned == "Unassigned Only":
    filtered = [s for s in filtered if not s.get("person_id")]
elif filter_assigned == "Assigned Only":
    filtered = [s for s in filtered if s.get("person_id")]

if filter_risk != "All":
    filtered = [s for s in filtered if s.get("singleton_risk") == filter_risk]

if filter_suggestions == "Has Suggestion":
    filtered = [s for s in filtered if s.get("identity_id") in suggestions_map]
elif filter_suggestions == "No Suggestion":
    filtered = [s for s in filtered if s.get("identity_id") not in suggestions_map]

# Helper to get suggestion confidence for sorting
def get_suggestion_confidence(s):
    cluster_id = s.get("identity_id", "")
    suggestions = suggestions_map.get(cluster_id, [])
    if not suggestions:
        return (3, 0)  # No suggestion = lowest priority
    top = suggestions[0]
    conf = top.get("confidence", "low")
    sim = top.get("similarity", 0)
    conf_order = {"high": 0, "medium": 1, "low": 2}
    return (conf_order.get(conf, 3), -sim)  # Sort by confidence, then by similarity descending

# Helper to get sort key by track ID
def get_track_id_key(s):
    # For unclustered tracks, use track_id; for clustered, get first track from track_ids
    if s.get("is_unclustered"):
        return s.get("track_id", 0)
    track_ids = s.get("track_ids", [])
    return track_ids[0] if track_ids else 0

# Apply sorting
if sort_by == "Suggestion Confidence":
    filtered = sorted(filtered, key=get_suggestion_confidence)
elif sort_by == "Risk (High First)":
    risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    filtered = sorted(filtered, key=lambda s: risk_order.get(s.get("singleton_risk", "LOW"), 3))
elif sort_by == "Faces (Most First)":
    filtered = sorted(filtered, key=lambda s: s.get("counts", {}).get("faces", 0), reverse=True)
else:
    filtered = sorted(filtered, key=get_track_id_key)

st.markdown(f"**Showing {len(filtered)} of {len(singletons)} items**")

# ============================================================================
# Pagination
# ============================================================================

total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
current_page = st.session_state.get("singletons_page", 1)
current_page = min(current_page, total_pages)

if total_pages > 1:
    page_cols = st.columns([1, 3, 1])
    with page_cols[0]:
        if st.button("⬅️ Prev", disabled=current_page <= 1):
            st.session_state["singletons_page"] = current_page - 1
            st.rerun()
    with page_cols[1]:
        st.markdown(f"<center>Page {current_page} of {total_pages}</center>", unsafe_allow_html=True)
    with page_cols[2]:
        if st.button("Next ➡️", disabled=current_page >= total_pages):
            st.session_state["singletons_page"] = current_page + 1
            st.rerun()

# ============================================================================
# Singleton Cards
# ============================================================================

start_idx = (current_page - 1) * page_size
end_idx = min(start_idx + page_size, len(filtered))
page_singletons = filtered[start_idx:end_idx]

if not page_singletons:
    if filtered:
        st.info("No singletons on this page.")
    else:
        st.success("🎉 No singletons matching the filters! All clusters have been processed.")
else:
    actions_taken = []
    for idx, singleton in enumerate(page_singletons):
        cluster_id = singleton.get("identity_id", "")
        # Get suggestions for this singleton
        cluster_suggestions = suggestions_map.get(cluster_id, [])
        action = render_singleton_card(ep_id, singleton, cast_members, cluster_suggestions, idx)
        if action:
            actions_taken.append(action)

    # If actions were taken, clear cache and rerun
    if actions_taken:
        load_singletons.clear()
        load_cast_suggestions.clear()
        time.sleep(0.5)  # Brief delay for API to process
        st.rerun()

# ============================================================================
# Bulk Actions
# ============================================================================

st.markdown("---")
st.subheader("Bulk Actions")

bulk_cols = st.columns([2, 2, 2])

with bulk_cols[0]:
    high_risk_unassigned = [
        s for s in singletons
        if not s.get("person_id") and s.get("singleton_risk") == "HIGH" and not s.get("is_unclustered")
    ]
    high_risk_count = len(high_risk_unassigned)

    # Two-step confirmation for bulk dismissal
    confirm_key = f"{ep_id}::bulk_dismiss_confirm"
    if st.session_state.get(confirm_key, False):
        st.warning(
            f"⚠️ **Confirm:** This will mark {high_risk_count} HIGH-risk singletons as noise. "
            "They will be hidden from suggestions but NOT deleted from identities.json."
        )
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("✓ Yes, Mark as Noise", key="confirm_bulk_yes", type="primary"):
                dismissed = 0
                for s in high_risk_unassigned:
                    identity_id = s.get("identity_id", "")
                    if identity_id:
                        result = mark_as_noise(ep_id, identity_id)
                        if "error" not in result:
                            dismissed += 1
                st.success(f"Marked {dismissed} singletons as noise (hidden from view)")
                st.session_state[confirm_key] = False
                load_singletons.clear()
                st.rerun()
        with col_no:
            if st.button("✗ Cancel", key="confirm_bulk_no"):
                st.session_state[confirm_key] = False
                st.rerun()
    else:
        if st.button(
            f"🚫 Mark HIGH Risk as Noise ({high_risk_count})",
            disabled=high_risk_count == 0,
            use_container_width=True,
            help="Mark unassigned HIGH-risk singletons as noise (hidden but not deleted)"
        ):
            st.session_state[confirm_key] = True
            st.rerun()

with bulk_cols[1]:
    # Restore hidden singletons
    try:
        dismissed_resp = api_get(f"/episodes/{ep_id}/dismissed_suggestions")
        dismissed_ids = set(dismissed_resp.get("dismissed_ids", [])) if dismissed_resp else set()
    except Exception:
        dismissed_ids = set()

    hidden_count = len(dismissed_ids)
    if hidden_count > 0:
        if st.button(f"👁️ Restore Hidden ({hidden_count})", use_container_width=True, help="Restore hidden clusters"):
            try:
                session = _api_session()
                resp = session.delete(f"{API_BASE}/episodes/{ep_id}/dismissed_suggestions", timeout=30)
                if resp.status_code == 200:
                    st.success(f"Restored {hidden_count} hidden cluster(s)")
                    load_singletons.clear()
                    st.rerun()
                else:
                    st.error("Failed to restore hidden clusters")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        if st.button("🔄 Refresh Data", use_container_width=True):
            load_singletons.clear()
            load_cast_members.clear()
            st.rerun()

with bulk_cols[2]:
    if st.button("👈 Back to Faces Review", use_container_width=True):
        st.query_params["ep_id"] = ep_id
        st.switch_page("pages/3_Faces_Review.py")

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.caption(
    "**Tip:** Review singletons AFTER completing the main Faces Review workflow. "
    "HIGH-risk singletons (single-frame, single-track) are often noise, while "
    "MEDIUM/LOW-risk singletons may be valid brief appearances of cast members."
)

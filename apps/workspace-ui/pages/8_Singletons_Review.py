"""Singletons Review Page - Batch review and assign singleton clusters.

This page performs batch analysis of ALL singleton clusters (clusters with 1 track)
against ALL assigned track embeddings from the episode. Results are grouped by
suggested cast member for efficient review.

Features:
- Batch comparison (single API call instead of N+1)
- Person-grouped display (all Lisa suggestions together, etc.)
- Bulk accept/reject per person group
- Archive similarity matching
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

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

API_BASE = st.session_state.get("api_base") or os.environ.get("API_BASE", "http://localhost:8000")

_retry_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Get or create retry session."""
    global _retry_session
    if _retry_session is None:
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _retry_session = session
    return _retry_session


@dataclass
class ApiResult:
    """Structured API result with explicit error handling."""

    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.data is not None

    @property
    def error_message(self) -> str:
        if not self.error:
            return ""
        if self.status_code:
            return f"API error {self.status_code}: {self.error}"
        return f"API error: {self.error}"


def _api_get(path: str, params: Optional[Dict] = None, timeout: int = 30) -> ApiResult:
    """GET request with structured error handling."""
    url = f"{API_BASE}{path}"
    session = _get_session()
    try:
        resp = session.get(url, params=params or {}, timeout=timeout)
        if resp.status_code == 200:
            return ApiResult(data=resp.json())
        try:
            error_detail = resp.json().get("detail", resp.text or resp.reason)
        except Exception:
            error_detail = resp.text or resp.reason or "Unknown error"
        return ApiResult(error=error_detail, status_code=resp.status_code)
    except requests.Timeout:
        return ApiResult(error=f"Request timed out ({timeout}s)")
    except requests.ConnectionError:
        return ApiResult(error="Connection failed - API may be unavailable")
    except Exception as e:
        return ApiResult(error=f"Unexpected error: {e}")


def _api_post(path: str, payload: Optional[Dict] = None, timeout: int = 60) -> ApiResult:
    """POST request with structured error handling."""
    url = f"{API_BASE}{path}"
    session = _get_session()
    try:
        resp = session.post(url, json=payload or {}, timeout=timeout)
        if resp.status_code in (200, 201):
            return ApiResult(data=resp.json())
        try:
            error_detail = resp.json().get("detail", resp.text or resp.reason)
        except Exception:
            error_detail = resp.text or resp.reason or "Unknown error"
        return ApiResult(error=error_detail, status_code=resp.status_code)
    except requests.Timeout:
        return ApiResult(error=f"Request timed out ({timeout}s)")
    except requests.ConnectionError:
        return ApiResult(error="Connection failed - API may be unavailable")
    except Exception as e:
        return ApiResult(error=f"Unexpected error: {e}")


# ============================================================================
# Session State Keys
# ============================================================================


def _analysis_key(ep_id: str) -> str:
    return f"{ep_id}::singleton_analysis"


def _analysis_timestamp_key(ep_id: str) -> str:
    return f"{ep_id}::singleton_analysis_ts"


def _selections_key(ep_id: str) -> str:
    return f"{ep_id}::singleton_selections"


def _expanded_key(ep_id: str) -> str:
    return f"{ep_id}::expanded_person_groups"


# ============================================================================
# Data Loading
# ============================================================================


def load_singleton_analysis(
    ep_id: str,
    include_archive: bool = False,
    min_similarity: float = 0.30,
    force_refresh: bool = False,
) -> Optional[Dict[str, Any]]:
    """Load singleton analysis from API (cached in session state)."""
    cache_key = _analysis_key(ep_id)
    ts_key = _analysis_timestamp_key(ep_id)

    if not force_refresh and cache_key in st.session_state:
        return st.session_state[cache_key]

    with st.spinner("Analyzing singletons... (batch comparison)"):
        result = _api_post(
            f"/episodes/{ep_id}/singleton_analysis",
            {
                "include_archive": include_archive,
                "min_similarity": min_similarity,
                "top_k": 3,
            },
            timeout=120,  # May take longer for large episodes
        )

    if result.ok:
        st.session_state[cache_key] = result.data
        st.session_state[ts_key] = time.strftime("%H:%M:%S")
        st.toast("Analysis generated successfully!", icon="✅")
        return result.data
    else:
        st.error(f"Failed to analyze singletons: {result.error_message}")
        return None


def load_cast_members(show_id: str) -> List[Dict[str, Any]]:
    """Load cast members for dropdown selection."""
    cache_key = f"cast_members:{show_id}"
    if cache_key not in st.session_state:
        result = _api_get(f"/shows/{show_id}/cast")
        if result.ok and result.data:
            # API returns {"cast": [...]} format
            cast_list = result.data.get("cast", []) if isinstance(result.data, dict) else result.data
            st.session_state[cache_key] = cast_list
        else:
            st.session_state[cache_key] = []
    return st.session_state.get(cache_key, [])


def get_show_id_from_ep(ep_id: str) -> Optional[str]:
    """Extract show ID from episode ID (e.g., 'rhobh-s05e02' -> 'rhobh')."""
    if "-" in ep_id:
        return ep_id.split("-")[0].lower()
    return None


# ============================================================================
# Actions
# ============================================================================


def assign_singleton_to_cast(
    ep_id: str,
    identity_id: str,
    track_id: int,
    cast_id: str,
    cast_name: str,
) -> bool:
    """Assign a singleton cluster to a cast member."""
    show_id = get_show_id_from_ep(ep_id)
    if not show_id:
        st.error("Could not determine show ID")
        return False

    # Use the cluster group endpoint for assignment
    result = _api_post(
        f"/episodes/{ep_id}/clusters/group",
        {
            "cluster_ids": [identity_id],
            "cast_id": cast_id,
            "name": cast_name,
            "strategy": "manual",
        },
    )

    if result.ok:
        return True
    else:
        st.error(f"Failed to assign: {result.error_message}")
        return False


def mark_singleton_as_noise(
    ep_id: str,
    identity_id: str,
    track_id: int,
    thumbnail_url: Optional[str] = None,
) -> bool:
    """Mark a singleton as noise (archive it)."""
    show_id = get_show_id_from_ep(ep_id)
    if not show_id:
        st.error("Could not determine show ID")
        return False

    result = _api_post(
        f"/archive/shows/{show_id}/clusters",
        {
            "episode_id": ep_id,
            "cluster_id": identity_id,
            "reason": "noise",
            "rep_crop_url": thumbnail_url,
        },
    )

    if result.ok:
        return True
    else:
        st.error(f"Failed to archive: {result.error_message}")
        return False


# ============================================================================
# Batch Operations
# ============================================================================


@dataclass
class BatchProgress:
    """Track batch operation progress."""

    total: int
    completed: int = 0
    succeeded: int = 0
    failed: int = 0
    errors: List[str] = field(default_factory=list)

    def increment(self, success: bool = True, error: str = "") -> None:
        self.completed += 1
        if success:
            self.succeeded += 1
        else:
            self.failed += 1
            if error:
                self.errors.append(error)


def batch_assign_person_group(
    ep_id: str,
    person_id: str,
    cast_id: str,
    cast_name: str,
    matches: List[Dict[str, Any]],
) -> BatchProgress:
    """Assign all matches in a person group to that cast member."""
    progress = BatchProgress(total=len(matches))
    progress_bar = st.progress(0, text=f"Assigning to {cast_name}: 0/{len(matches)}")

    for i, match in enumerate(matches):
        identity_id = match.get("identity_id")
        track_id = match.get("track_id")

        if not identity_id:
            progress.increment(success=False, error=f"Track {track_id}: missing identity_id")
            continue

        success = assign_singleton_to_cast(
            ep_id=ep_id,
            identity_id=identity_id,
            track_id=track_id,
            cast_id=cast_id,
            cast_name=cast_name,
        )
        progress.increment(success=success, error="" if success else f"Track {track_id} failed")

        progress_bar.progress(
            (i + 1) / len(matches),
            text=f"Assigning to {cast_name}: {i + 1}/{len(matches)}",
        )

    progress_bar.empty()
    return progress


def batch_archive_matches(
    ep_id: str,
    matches: List[Dict[str, Any]],
) -> BatchProgress:
    """Archive all matches as noise."""
    progress = BatchProgress(total=len(matches))
    progress_bar = st.progress(0, text=f"Archiving: 0/{len(matches)}")

    for i, match in enumerate(matches):
        identity_id = match.get("identity_id")
        track_id = match.get("track_id")
        thumbnail_url = match.get("thumbnail_url")

        if not identity_id:
            progress.increment(success=False, error=f"Track {track_id}: missing identity_id")
            continue

        success = mark_singleton_as_noise(
            ep_id=ep_id,
            identity_id=identity_id,
            track_id=track_id,
            thumbnail_url=thumbnail_url,
        )
        progress.increment(success=success, error="" if success else f"Track {track_id} failed")

        progress_bar.progress(
            (i + 1) / len(matches),
            text=f"Archiving: {i + 1}/{len(matches)}",
        )

    progress_bar.empty()
    return progress


# ============================================================================
# UI Components
# ============================================================================


def render_confidence_badge(confidence: str, similarity: float) -> str:
    """Render confidence badge with color."""
    if confidence == "high":
        color = "#28a745"  # Green
        icon = "H"
    elif confidence == "medium":
        color = "#ffc107"  # Yellow
        icon = "M"
    else:
        color = "#dc3545"  # Red
        icon = "L"

    return f'<span style="background-color:{color};color:white;padding:2px 6px;border-radius:4px;font-size:12px;font-weight:bold;">{icon} {similarity:.0%}</span>'


def render_singleton_card(
    ep_id: str,
    match: Dict[str, Any],
    cast_id: str,
    cast_name: str,
    card_key: str,
) -> Optional[str]:
    """Render a single singleton match card. Returns action taken or None."""
    track_id = match.get("track_id")
    identity_id = match.get("identity_id")
    similarity = match.get("similarity", 0)
    confidence = match.get("confidence", "low")
    face_count = match.get("face_count", 0)
    singleton_risk = match.get("singleton_risk", "MEDIUM")
    thumbnail_url = match.get("thumbnail_url")

    # Resolve S3 key to presigned URL
    resolved_thumb = helpers.resolve_thumb(thumbnail_url) if thumbnail_url else None

    cols = st.columns([0.5, 2, 1, 1, 1])

    with cols[0]:
        # Thumbnail
        if resolved_thumb:
            try:
                st.image(resolved_thumb, width=60)
            except Exception:
                st.text("No img")
        else:
            st.text("No img")

    with cols[1]:
        # Info
        st.markdown(
            f"**T{track_id}** ({face_count} faces) "
            f"{render_confidence_badge(confidence, similarity)}",
            unsafe_allow_html=True,
        )
        risk_badge = render_singleton_risk_badge(1, face_count)
        st.markdown(f"Risk: {risk_badge}", unsafe_allow_html=True)

    with cols[2]:
        # Accept button
        if st.button(f"Accept", key=f"accept_{card_key}", type="primary"):
            success = assign_singleton_to_cast(
                ep_id=ep_id,
                identity_id=identity_id,
                track_id=track_id,
                cast_id=cast_id,
                cast_name=cast_name,
            )
            if success:
                return "accepted"

    with cols[3]:
        # Skip/Noise button
        if st.button(f"Noise", key=f"noise_{card_key}"):
            success = mark_singleton_as_noise(
                ep_id=ep_id,
                identity_id=identity_id,
                track_id=track_id,
                thumbnail_url=thumbnail_url,
            )
            if success:
                return "archived"

    with cols[4]:
        # Alternative suggestions
        alternatives = match.get("alternative_suggestions", [])
        if alternatives:
            alt_text = ", ".join([f"{a.get('person_id', '?')[:8]}:{a.get('similarity', 0):.0%}" for a in alternatives[:2]])
            st.caption(f"Alt: {alt_text}")

    return None


def render_person_group(
    ep_id: str,
    group: Dict[str, Any],
    group_index: int,
    show_id: str,
) -> int:
    """Render a person group accordion. Returns count of actions taken."""
    person_name = group.get("person_name", "Unknown")
    person_id = group.get("person_id", "")
    cast_id = group.get("cast_id")
    matches = group.get("matches", [])
    avg_similarity = group.get("avg_similarity", 0)
    confidence_breakdown = group.get("confidence_breakdown", {})

    high_count = confidence_breakdown.get("high", 0)
    medium_count = confidence_breakdown.get("medium", 0)
    low_count = confidence_breakdown.get("low", 0)

    # Header with stats
    header = f"{person_name} ({len(matches)} singletons | {avg_similarity:.0%} avg"
    if high_count:
        header += f" | {high_count}H"
    if medium_count:
        header += f" | {medium_count}M"
    if low_count:
        header += f" | {low_count}L"
    header += ")"

    actions_taken = 0

    with st.expander(header, expanded=(group_index == 0)):
        # Bulk action buttons
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button(f"Accept All ({len(matches)})", key=f"accept_all_{person_id}", type="primary"):
                if cast_id:
                    progress = batch_assign_person_group(
                        ep_id=ep_id,
                        person_id=person_id,
                        cast_id=cast_id,
                        cast_name=person_name,
                        matches=matches,
                    )
                    st.success(f"Assigned {progress.succeeded}/{progress.total} to {person_name}")
                    if progress.failed:
                        st.warning(f"{progress.failed} failed: {', '.join(progress.errors[:3])}")
                    actions_taken = progress.succeeded
                else:
                    st.error("No cast_id for this person")

        with col2:
            if st.button(f"Archive All", key=f"archive_all_{person_id}"):
                progress = batch_archive_matches(ep_id=ep_id, matches=matches)
                st.info(f"Archived {progress.succeeded}/{progress.total}")
                actions_taken = progress.succeeded

        st.divider()

        # Individual cards
        for i, match in enumerate(matches):
            card_key = f"{person_id}_{match.get('track_id', i)}_{group_index}"
            action = render_singleton_card(
                ep_id=ep_id,
                match=match,
                cast_id=cast_id or "",
                cast_name=person_name,
                card_key=card_key,
            )
            if action:
                actions_taken += 1
            if i < len(matches) - 1:
                st.divider()

    return actions_taken


def render_unmatched_section(
    ep_id: str,
    unmatched: List[Dict[str, Any]],
    cast_members: List[Dict[str, Any]],
) -> int:
    """Render the unmatched singletons section."""
    if not unmatched:
        return 0

    actions_taken = 0

    with st.expander(f"No Suggestions ({len(unmatched)} singletons)", expanded=False):
        st.warning("These singletons had no good matches. You can manually assign or archive them.")

        # Cast member dropdown for manual assignment
        # Handle both dict format {"name": ..., "cast_id": ...} and string format (just cast_id)
        cast_options = {}
        for m in cast_members:
            if isinstance(m, dict):
                name = m.get("name", m.get("cast_id", "?"))
                cast_id = m.get("cast_id")
            else:
                # m is a string (cast_id)
                name = str(m)
                cast_id = str(m)
            cast_options[name] = cast_id

        for i, item in enumerate(unmatched):
            track_id = item.get("track_id")
            identity_id = item.get("identity_id")
            face_count = item.get("face_count", 0)
            singleton_risk = item.get("singleton_risk", "MEDIUM")
            thumbnail_url = item.get("thumbnail_url")
            # Resolve S3 key to presigned URL
            resolved_thumb = helpers.resolve_thumb(thumbnail_url) if thumbnail_url else None

            cols = st.columns([0.5, 2, 2, 1])

            with cols[0]:
                if resolved_thumb:
                    try:
                        st.image(resolved_thumb, width=60)
                    except Exception:
                        st.text("No img")
                else:
                    st.text("No img")

            with cols[1]:
                st.markdown(f"**T{track_id}** ({face_count} faces)")
                risk_badge = render_singleton_risk_badge(1, face_count)
                st.markdown(f"Risk: {risk_badge}", unsafe_allow_html=True)

            with cols[2]:
                selected_name = st.selectbox(
                    "Assign to:",
                    options=["-- Select --"] + list(cast_options.keys()),
                    key=f"manual_assign_{track_id}_{i}",
                    label_visibility="collapsed",
                )
                if selected_name and selected_name != "-- Select --":
                    if st.button("Assign", key=f"do_assign_{track_id}_{i}"):
                        cast_id = cast_options[selected_name]
                        success = assign_singleton_to_cast(
                            ep_id=ep_id,
                            identity_id=identity_id,
                            track_id=track_id,
                            cast_id=cast_id,
                            cast_name=selected_name,
                        )
                        if success:
                            st.success(f"Assigned T{track_id} to {selected_name}")
                            actions_taken += 1

            with cols[3]:
                if st.button("Noise", key=f"noise_unmatched_{track_id}_{i}"):
                    success = mark_singleton_as_noise(
                        ep_id=ep_id,
                        identity_id=identity_id,
                        track_id=track_id,
                        thumbnail_url=thumbnail_url,
                    )
                    if success:
                        st.info(f"Archived T{track_id}")
                        actions_taken += 1

            if i < len(unmatched) - 1:
                st.divider()

    return actions_taken


def render_archive_matches_section(
    ep_id: str,
    archive_matches: List[Dict[str, Any]],
) -> int:
    """Render singletons that match archived items."""
    if not archive_matches:
        return 0

    actions_taken = 0

    with st.expander(f"Similar to Archived ({len(archive_matches)} matches)", expanded=False):
        st.info("These singletons are similar to previously archived items. Consider archiving them too.")

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Archive All Similar", key=f"{ep_id}::archive_all_similar"):
                progress = batch_archive_matches(ep_id=ep_id, matches=archive_matches)
                st.success(f"Archived {progress.succeeded}/{progress.total}")
                actions_taken = progress.succeeded

        for i, item in enumerate(archive_matches):
            track_id = item.get("track_id")
            identity_id = item.get("identity_id")
            similarity = item.get("similarity", 0)
            archived_name = item.get("archived_name", "noise")
            thumbnail_url = item.get("thumbnail_url")
            # Resolve S3 key to presigned URL
            resolved_thumb = helpers.resolve_thumb(thumbnail_url) if thumbnail_url else None

            cols = st.columns([0.5, 2, 1, 1])

            with cols[0]:
                if resolved_thumb:
                    try:
                        st.image(resolved_thumb, width=60)
                    except Exception:
                        st.text("No img")
                else:
                    st.text("No img")

            with cols[1]:
                st.markdown(f"**T{track_id}** - {similarity:.0%} similar to '{archived_name}'")

            with cols[2]:
                if st.button("Archive", key=f"archive_similar_{track_id}_{i}"):
                    success = mark_singleton_as_noise(
                        ep_id=ep_id,
                        identity_id=identity_id,
                        track_id=track_id,
                        thumbnail_url=thumbnail_url,
                    )
                    if success:
                        st.info(f"Archived T{track_id}")
                        actions_taken += 1

            with cols[3]:
                if st.button("Keep", key=f"keep_similar_{track_id}_{i}"):
                    st.info(f"Keeping T{track_id}")

    return actions_taken


# ============================================================================
# Main Page
# ============================================================================

# Episode selector
sidebar_ep_id = None
try:
    sidebar_ep_id = helpers.render_sidebar_episode_selector()
except Exception:
    sidebar_ep_id = None

ep_id = helpers.get_ep_id()
if not ep_id and sidebar_ep_id:
    helpers.set_ep_id(sidebar_ep_id, rerun=False)
    ep_id = sidebar_ep_id
if not ep_id:
    ep_id_from_query = helpers.get_ep_id_from_query_params()
    if ep_id_from_query:
        helpers.set_ep_id(ep_id_from_query, rerun=False)
        ep_id = ep_id_from_query
if not ep_id:
    st.warning("Select an episode from the sidebar to continue.")
    st.stop()

show_id = get_show_id_from_ep(ep_id)

# Controls
st.markdown("---")
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    include_archive = st.checkbox("Include archive comparison", value=False, key=f"{ep_id}::include_archive")

with col2:
    min_sim = st.slider("Min similarity", 0.0, 1.0, 0.30, 0.05, key=f"{ep_id}::min_sim")

with col3:
    if st.button("Generate Analysis", type="primary"):
        # Force refresh - clear cache and rerun
        st.session_state.pop(_analysis_key(ep_id), None)
        st.session_state.pop(_analysis_timestamp_key(ep_id), None)
        st.toast("Regenerating analysis...", icon="🔄")
        st.rerun()

with col4:
    if st.button("Refresh"):
        # Force refresh - clear cache and rerun
        st.session_state.pop(_analysis_key(ep_id), None)
        st.session_state.pop(_analysis_timestamp_key(ep_id), None)
        st.toast("Refreshing...", icon="🔄")
        st.rerun()

# Load analysis
analysis = load_singleton_analysis(
    ep_id=ep_id,
    include_archive=include_archive,
    min_similarity=min_sim,
)

if not analysis:
    st.info("Click 'Generate Analysis' to analyze singletons.")
    st.stop()

# Stats summary
stats = analysis.get("stats", {})
person_groups = analysis.get("person_groups", [])
archive_matches = analysis.get("archive_matches", [])
unmatched = analysis.get("unmatched", [])

total_singletons = stats.get("total_singletons", 0)
with_suggestions = stats.get("with_suggestions", 0)
unmatched_count = stats.get("unmatched_count", 0)
persons_with_matches = stats.get("persons_with_matches", 0)

# Show last generated timestamp
ts_key = _analysis_timestamp_key(ep_id)
last_generated = st.session_state.get(ts_key, "")

st.markdown("---")
summary_text = f"**Summary**: {total_singletons} singletons | {with_suggestions} with suggestions | {unmatched_count} unmatched | {persons_with_matches} person groups"
if last_generated:
    summary_text += f" | *Generated at {last_generated}*"
st.markdown(summary_text)

if total_singletons == 0:
    st.success("No unassigned singletons found. Great job!")
    st.stop()

# Load cast members for manual assignment
cast_members = load_cast_members(show_id) if show_id else []

# Track total actions for refresh
total_actions = 0

# Render person groups
st.markdown("### Suggested Assignments")
if person_groups:
    for i, group in enumerate(person_groups):
        actions = render_person_group(
            ep_id=ep_id,
            group=group,
            group_index=i,
            show_id=show_id or "",
        )
        total_actions += actions
else:
    st.info("No person groups with suggestions.")

# Render archive matches
if archive_matches:
    st.markdown("### Archive Matches")
    actions = render_archive_matches_section(ep_id=ep_id, archive_matches=archive_matches)
    total_actions += actions

# Render unmatched
st.markdown("### Unmatched Singletons")
actions = render_unmatched_section(
    ep_id=ep_id,
    unmatched=unmatched,
    cast_members=cast_members,
)
total_actions += actions

# Auto-refresh if actions were taken
if total_actions > 0:
    st.info(f"Completed {total_actions} action(s). Click 'Refresh' to update the list.")

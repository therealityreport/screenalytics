"""Smart Suggestions Page - Sub-page of Faces Review.

This page shows cast assignment suggestions for unassigned clusters,
allowing batch review and one-click accept/dismiss actions.

Each suggestion is displayed as its own row with up to 6 frame thumbnails.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402
from similarity_badges import (  # noqa: E402
    SimilarityType,
    render_similarity_badge,
    render_singleton_risk_badge,
    render_singleton_fraction_badge,
)

LOGGER = logging.getLogger(__name__)

cfg = helpers.init_page("Smart Suggestions")
st.title("Smart Suggestions")


# --- Undo History ---
@dataclass
class UndoAction:
    """Record of an action that can be undone."""

    action_type: str  # "assign" | "dismiss" | "link"
    ep_id: str
    cluster_id: Optional[str] = None
    person_id: Optional[str] = None
    cast_id: Optional[str] = None
    cast_name: Optional[str] = None
    timestamp: float = 0.0

    def describe(self) -> str:
        """Human-readable description of the action."""
        if self.action_type == "assign":
            return f"Assigned cluster to {self.cast_name or self.cast_id}"
        elif self.action_type == "dismiss":
            return f"Dismissed {self.cluster_id or self.person_id}"
        elif self.action_type == "link":
            return f"Linked person to {self.cast_name or self.cast_id}"
        return "Unknown action"


def _get_undo_history_key(ep_id: str) -> str:
    return f"undo_history:{ep_id}"


def _push_undo_action(ep_id: str, action: UndoAction) -> None:
    """Add an action to the undo history (max 10 actions)."""
    key = _get_undo_history_key(ep_id)
    history = st.session_state.get(key, [])
    action.timestamp = time.time()
    history.append(action)
    # Keep only last 10 actions
    st.session_state[key] = history[-10:]


def _pop_undo_action(ep_id: str) -> Optional[UndoAction]:
    """Pop the last action from undo history."""
    key = _get_undo_history_key(ep_id)
    history = st.session_state.get(key, [])
    if not history:
        return None
    action = history.pop()
    st.session_state[key] = history
    return action


def _get_last_undo_action(ep_id: str) -> Optional[UndoAction]:
    """Peek at the last action without removing it."""
    key = _get_undo_history_key(ep_id)
    history = st.session_state.get(key, [])
    return history[-1] if history else None


# --- API Result Types ---
@dataclass
class ApiResult:
    """Structured result from API calls with explicit error handling."""

    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.data is not None

    @property
    def error_message(self) -> str:
        """Human-readable error message."""
        if not self.error:
            return ""
        if self.status_code:
            return f"API error {self.status_code}: {self.error}"
        return f"API error: {self.error}"

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

# API helpers
_api_base = st.session_state.get("api_base", "http://localhost:8000")

# Default timeout values (overridden by config if available)
_DEFAULT_TIMEOUT = 30
_HEAVY_TIMEOUT = 60

# Default thresholds (overridden by config if available)
_DEFAULT_THRESHOLDS = {
    "cast_auto_assign": 0.95,  # ≥95% - auto-accept
    "cast_high": 0.65,         # 65-94% - high confidence
    "cast_medium": 0.35,       # 35-64% - medium confidence
    "cast_low": 0.15,          # 15-34% - low confidence
    # <15% = "rest" - very weak matches
}


def _validate_thumbnail_url(url: str | None) -> str | None:
    """Validate and sanitize thumbnail URL to prevent XSS.

    Args:
        url: Raw URL string from API response

    Returns:
        Sanitized URL if valid, None otherwise
    """
    if not url or not isinstance(url, str):
        return None

    # Only allow http/https URLs
    url = url.strip()
    if not url.startswith(("http://", "https://", "/")):
        LOGGER.warning(f"[Smart Suggestions] Rejected invalid thumbnail URL scheme: {url[:50]}")
        return None

    # Reject URLs with suspicious content (basic XSS prevention)
    suspicious = ["javascript:", "data:", "vbscript:", "<script", "onerror=", "onload="]
    url_lower = url.lower()
    for pattern in suspicious:
        if pattern in url_lower:
            LOGGER.warning(f"[Smart Suggestions] Rejected suspicious thumbnail URL: {url[:50]}")
            return None

    return url


def _format_counts(faces: int | None, tracks: int | None) -> str:
    """Format face/track counts with graceful handling of missing data.

    Args:
        faces: Number of faces, or None if unknown
        tracks: Number of tracks, or None if unknown

    Returns:
        Human-readable string like "5 faces · 2 tracks" or "Unknown" if both missing
    """
    parts = []
    if faces is not None and faces > 0:
        parts.append(f"{faces} face{'s' if faces != 1 else ''}")
    if tracks is not None and tracks > 0:
        parts.append(f"{tracks} track{'s' if tracks != 1 else ''}")

    if not parts:
        return "No data available"
    return " · ".join(parts)


def _fetch_config_thresholds() -> Dict[str, Any]:
    """Fetch thresholds from config API (cached in session state)."""
    cache_key = "config_thresholds"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    # Try to fetch from API
    try:
        url = f"{_api_base}/config/thresholds"
        resp = requests.get(url, timeout=5)  # Short timeout for config
        if resp.status_code == 200:
            data = resp.json()
            thresholds = {
                "cast_auto_assign": data.get("suggestion", {}).get("cast_auto_assign", _DEFAULT_THRESHOLDS["cast_auto_assign"]),
                "cast_high": data.get("suggestion", {}).get("cast_high", _DEFAULT_THRESHOLDS["cast_high"]),
                "cast_medium": data.get("suggestion", {}).get("cast_medium", _DEFAULT_THRESHOLDS["cast_medium"]),
                "cast_low": data.get("suggestion", {}).get("cast_low", _DEFAULT_THRESHOLDS["cast_low"]),
                "cast_high_label": data.get("suggestion", {}).get("cast_high_label", "≥70%"),
                "cast_medium_label": data.get("suggestion", {}).get("cast_medium_label", "≥30%"),
                "has_overrides": data.get("has_overrides", False),
            }
            st.session_state[cache_key] = thresholds
            return thresholds
    except Exception as e:
        LOGGER.debug(f"[Smart Suggestions] Failed to fetch config thresholds: {e}")

    # Fall back to defaults
    return {
        **_DEFAULT_THRESHOLDS,
        "cast_high_label": f"≥{int(_DEFAULT_THRESHOLDS['cast_high'] * 100)}%",
        "cast_medium_label": f"≥{int(_DEFAULT_THRESHOLDS['cast_medium'] * 100)}%",
        "has_overrides": False,
    }


def _get_confidence_level(
    similarity: float,
    face_count: int | None = None,
    track_count: int | None = None,
) -> str:
    """Get confidence level based on config thresholds with singleton penalties.

    Args:
        similarity: Cast similarity score (0-1)
        face_count: Optional total face count for singleton penalty
        track_count: Optional track count for singleton penalty

    Returns:
        Confidence level: "auto", "high", "medium", "low", or "rest"
        - auto: ≥95% - can auto-assign
        - high: 65-94% - strong match
        - medium: 35-64% - reasonable match
        - low: 15-34% - weak match
        - rest: <15% - very weak
    """
    thresholds = _fetch_config_thresholds()

    # Apply singleton penalty for unreliable clusters
    # Single-track + single-frame = 15% penalty (per SINGLETONS_PLAN.md)
    # Single-track + multi-frame = 5% penalty
    effective_sim = similarity
    if track_count == 1 and face_count == 1:
        effective_sim -= 0.15  # HIGH risk singleton
    elif track_count == 1:
        effective_sim -= 0.05  # MEDIUM risk singleton

    if effective_sim >= thresholds.get("cast_auto_assign", 0.90):
        return "auto"
    elif effective_sim >= thresholds.get("cast_high", 0.70):
        return "high"
    elif effective_sim >= thresholds.get("cast_medium", 0.30):
        return "medium"
    elif effective_sim >= thresholds.get("cast_low", 0.10):
        return "low"
    return "rest"


def _safe_api_get(path: str, params: Dict[str, Any] | None = None, timeout: int | None = None) -> ApiResult:
    """Fetch from API with structured error handling.

    Returns ApiResult with data on success, or error details on failure.
    Never silently swallows errors.
    """
    if timeout is None:
        timeout = _DEFAULT_TIMEOUT
    url = f"{_api_base}{path}"
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 200:
            return ApiResult(data=resp.json())
        # Non-200 status - extract error detail
        try:
            error_detail = resp.json().get("detail", resp.text or resp.reason)
        except Exception:
            error_detail = resp.text or resp.reason or "Unknown error"
        LOGGER.warning(f"[Smart Suggestions] GET {path} returned {resp.status_code}: {error_detail}")
        return ApiResult(error=error_detail, status_code=resp.status_code)
    except requests.Timeout:
        LOGGER.error(f"[Smart Suggestions] GET {path} timed out after {timeout}s")
        return ApiResult(error=f"Request timed out ({timeout}s)")
    except requests.ConnectionError as e:
        LOGGER.error(f"[Smart Suggestions] GET {path} connection error: {e}")
        return ApiResult(error="Connection failed - API may be unavailable")
    except Exception as e:
        LOGGER.exception(f"[Smart Suggestions] GET {path} unexpected error: {e}")
        return ApiResult(error=f"Unexpected error: {type(e).__name__}: {e}")


def _api_post(path: str, payload: Dict[str, Any] | None = None, timeout: int | None = None) -> ApiResult:
    """POST to API with structured error handling.

    Returns ApiResult with data on success, or error details on failure.
    Never silently swallows errors.
    """
    if timeout is None:
        timeout = _DEFAULT_TIMEOUT
    url = f"{_api_base}{path}"
    try:
        resp = requests.post(url, json=payload or {}, timeout=timeout)
        if resp.status_code in (200, 201):
            return ApiResult(data=resp.json())
        # Non-success status - extract error detail
        try:
            error_detail = resp.json().get("detail", resp.text or resp.reason)
        except Exception:
            error_detail = resp.text or resp.reason or "Unknown error"
        LOGGER.warning(f"[Smart Suggestions] POST {path} returned {resp.status_code}: {error_detail}")
        return ApiResult(error=error_detail, status_code=resp.status_code)
    except requests.Timeout:
        LOGGER.error(f"[Smart Suggestions] POST {path} timed out after {timeout}s")
        return ApiResult(error=f"Request timed out ({timeout}s)")
    except requests.ConnectionError as e:
        LOGGER.error(f"[Smart Suggestions] POST {path} connection error: {e}")
        return ApiResult(error="Connection failed - API may be unavailable")
    except Exception as e:
        LOGGER.exception(f"[Smart Suggestions] POST {path} unexpected error: {e}")
        return ApiResult(error=f"Unexpected error: {type(e).__name__}: {e}")


def _api_patch(path: str, payload: Dict[str, Any] | None = None, timeout: int | None = None) -> ApiResult:
    """PATCH to API with structured error handling.

    Returns ApiResult with data on success, or error details on failure.
    Never silently swallows errors.
    """
    if timeout is None:
        timeout = _DEFAULT_TIMEOUT
    url = f"{_api_base}{path}"
    try:
        resp = requests.patch(url, json=payload or {}, timeout=timeout)
        if resp.status_code in (200, 201):
            return ApiResult(data=resp.json())
        # Non-success status - extract error detail
        try:
            error_detail = resp.json().get("detail", resp.text or resp.reason)
        except Exception:
            error_detail = resp.text or resp.reason or "Unknown error"
        LOGGER.warning(f"[Smart Suggestions] PATCH {path} returned {resp.status_code}: {error_detail}")
        return ApiResult(error=error_detail, status_code=resp.status_code)
    except requests.Timeout:
        LOGGER.error(f"[Smart Suggestions] PATCH {path} timed out after {timeout}s")
        return ApiResult(error=f"Request timed out ({timeout}s)")
    except requests.ConnectionError as e:
        LOGGER.error(f"[Smart Suggestions] PATCH {path} connection error: {e}")
        return ApiResult(error="Connection failed - API may be unavailable")
    except Exception as e:
        LOGGER.exception(f"[Smart Suggestions] PATCH {path} unexpected error: {e}")
        return ApiResult(error=f"Unexpected error: {type(e).__name__}: {e}")


@dataclass
class RecomputeResult:
    """Result of recomputing suggestions with explicit success/error status."""

    suggestions: Dict[str, List[Dict[str, Any]]]
    mismatched_embeddings: List[Dict[str, Any]]
    quality_only_clusters: List[Dict[str, Any]] = None  # Clusters with no embeddings (blurry faces)
    error: Optional[str] = None

    def __post_init__(self):
        if self.quality_only_clusters is None:
            self.quality_only_clusters = []

    @property
    def ok(self) -> bool:
        return self.error is None


def _recompute_cast_suggestions(ep_id: str, show_slug: str | None) -> RecomputeResult:
    """Persist assignments and recompute cast suggestions from latest embeddings.

    Atomic state update: only clears old state AFTER new data is successfully fetched.
    Returns structured result with error information and dimension mismatch warnings.
    """
    # First, persist current assignments (don't clear state yet)
    save_result = _api_post(f"/episodes/{ep_id}/save_assignments", {})
    if not save_result.ok:
        # Non-fatal: log warning but continue with suggestion fetch
        LOGGER.warning(f"[{ep_id}] Failed to save assignments before recompute: {save_result.error}")

    # Fetch new suggestions into local variables (NOT directly into session_state)
    # Pass min_similarity=0.01 to get ALL clusters including weak matches
    # The UI will group them by confidence tier (auto/high/medium/low/rest)
    cache_buster = int(time.time() * 1000)
    suggestions_result = _safe_api_get(
        f"/episodes/{ep_id}/cast_suggestions",
        params={"_t": cache_buster, "min_similarity": 0.01},
    )

    if not suggestions_result.ok:
        # API failed - return error, keep existing state intact
        return RecomputeResult(
            suggestions={},
            mismatched_embeddings=[],
            error=suggestions_result.error_message,
        )

    suggestions_data = suggestions_result.data or {}
    suggestions_map: Dict[str, List[Dict[str, Any]]] = {}
    rescued_clusters: set[str] = set()  # Track clusters with rescued/force-embedded faces
    for entry in suggestions_data.get("suggestions", []):
        cid = entry.get("cluster_id")
        if not cid:
            continue
        suggestions_map[cid] = entry.get("cast_suggestions", []) or []
        # Track if this cluster was rescued (force-embedded after quality gate)
        if entry.get("rescued"):
            rescued_clusters.add(cid)

    # Extract dimension mismatch warnings from API response
    mismatched_embeddings = suggestions_data.get("mismatched_embeddings", [])

    # Extract quality-only clusters (no embeddings, all faces skipped)
    quality_only_clusters = suggestions_data.get("quality_only_clusters", [])

    # SUCCESS: Now atomically update session state
    # Clear old state and replace with new data in one consistent operation
    st.session_state[f"cast_suggestions:{ep_id}"] = suggestions_map
    st.session_state[f"rescued_clusters:{ep_id}"] = rescued_clusters
    st.session_state[f"embedding_mismatches:{ep_id}"] = mismatched_embeddings
    st.session_state[f"quality_only_clusters:{ep_id}"] = quality_only_clusters
    st.session_state.pop(f"dismissed_suggestions:{ep_id}", None)
    if show_slug:
        st.session_state.pop(f"people_cache:{show_slug}", None)
    st.session_state.pop(f"cluster_tracks:{ep_id}", None)
    st.session_state.pop(f"identities:{ep_id}", None)

    return RecomputeResult(
        suggestions=suggestions_map,
        mismatched_embeddings=mismatched_embeddings,
        quality_only_clusters=quality_only_clusters,
    )


def _fetch_people_cached(show_id: str) -> Dict[str, Any] | None:
    """Fetch people for a show (cached)."""
    cache_key = f"people_cache:{show_id}"
    if cache_key not in st.session_state:
        result = _safe_api_get(f"/shows/{show_id}/people")
        st.session_state[cache_key] = result.data if result.ok else None
    return st.session_state.get(cache_key)


# CSS for fixed 147x184px thumbnails (4:5 aspect ratio) + carousel styling
THUMB_CSS = """
<style>
/* Fixed size thumbnail frames */
.thumb-row {
    display: flex;
    gap: 8px;
    align-items: flex-start;
}
.thumb-frame {
    width: 147px;
    height: 184px;
    border-radius: 6px;
    overflow: hidden;
    background: #2a2a2a;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}
.thumb-frame img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
/* Placeholder for failed images */
.thumb-frame .placeholder {
    color: #666;
    font-size: 48px;
    text-align: center;
}
/* Carousel container */
.carousel-container {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.carousel-strip {
    display: flex;
    gap: 6px;
    align-items: center;
}
.carousel-thumbs {
    display: flex;
    gap: 6px;
    overflow: hidden;
}
.carousel-nav {
    font-size: 0.75em;
    color: #888;
    text-align: center;
}
/* Outlier risk badges */
.risk-badge {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.75em;
    font-weight: bold;
    margin-left: 4px;
}
.risk-low { background-color: #4CAF50; color: white; }
.risk-medium { background-color: #FF9800; color: white; }
.risk-high { background-color: #F44336; color: white; }
</style>
"""
st.markdown(THUMB_CSS, unsafe_allow_html=True)


# Carousel constants
CAROUSEL_PAGE_SIZE = 5


def _get_carousel_page_key(entity_id: str) -> str:
    """Get session state key for carousel page index."""
    return f"carousel_page:{entity_id}"


def _render_carousel_thumbnails(
    thumb_urls: List[str],
    entity_id: str,
    page_size: int = CAROUSEL_PAGE_SIZE,
) -> None:
    """Render thumbnails with carousel navigation if needed.

    Args:
        thumb_urls: List of all thumbnail URLs
        entity_id: Unique ID for this entity (for session state keying)
        page_size: Number of thumbnails to show per page
    """
    total = len(thumb_urls)
    if total == 0:
        st.caption("No thumbnails available")
        return

    # Get current page index from session state
    page_key = _get_carousel_page_key(entity_id)
    page_index = st.session_state.get(page_key, 0)

    # Calculate page bounds
    page_count = (total + page_size - 1) // page_size  # ceil division
    page_index = max(0, min(page_index, page_count - 1))  # Clamp to valid range

    start_idx = page_index * page_size
    end_idx = min(start_idx + page_size, total)
    visible_urls = thumb_urls[start_idx:end_idx]

    # Render navigation if multiple pages
    if page_count > 1:
        nav_cols = st.columns([1, 6, 1])
        with nav_cols[0]:
            if st.button("◀", key=f"carousel_prev_{entity_id}", disabled=page_index == 0, use_container_width=True):
                st.session_state[page_key] = page_index - 1
                st.rerun()
        with nav_cols[1]:
            # Render visible thumbnails
            thumbs_html = '<div class="carousel-thumbs">'
            for url in visible_urls:
                thumbs_html += (
                    f'<div class="thumb-frame">'
                    f'<img src="{url}" onerror="this.style.display=\'none\';'
                    f'this.parentElement.innerHTML=\'<span class=placeholder>👤</span>\';"/>'
                    f'</div>'
                )
            thumbs_html += '</div>'
            st.markdown(thumbs_html, unsafe_allow_html=True)
            # Page indicator
            st.caption(f"{start_idx + 1}–{end_idx} of {total}")
        with nav_cols[2]:
            if st.button("▶", key=f"carousel_next_{entity_id}", disabled=page_index >= page_count - 1, use_container_width=True):
                st.session_state[page_key] = page_index + 1
                st.rerun()
    else:
        # No pagination needed - just render all thumbnails
        thumbs_html = '<div class="thumb-row">'
        for url in visible_urls:
            thumbs_html += (
                f'<div class="thumb-frame">'
                f'<img src="{url}" onerror="this.style.display=\'none\';'
                f'this.parentElement.innerHTML=\'<span class=placeholder>👤</span>\';"/>'
                f'</div>'
            )
        thumbs_html += '</div>'
        st.markdown(thumbs_html, unsafe_allow_html=True)


def _render_outlier_risk_badge(outlier_risk: Dict[str, Any]) -> str:
    """Render outlier risk badge as HTML.

    Args:
        outlier_risk: Dict with risk_level, cluster_outliers, cluster_total, etc.

    Returns:
        HTML string for the badge
    """
    risk_level = outlier_risk.get("risk_level", "LOW")
    cluster_outliers = outlier_risk.get("cluster_outliers", 0)
    cluster_total = outlier_risk.get("cluster_total", 0)
    low_cohesion = outlier_risk.get("low_cohesion_clusters", 0)

    risk_class = f"risk-{risk_level.lower()}"

    # Build tooltip text
    tooltip_parts = []
    if cluster_outliers > 0:
        tooltip_parts.append(f"{cluster_outliers}/{cluster_total} clusters flagged as outliers")
    if low_cohesion > 0:
        tooltip_parts.append(f"{low_cohesion} cluster(s) with low cohesion")
    if not tooltip_parts:
        tooltip_parts.append("No outliers detected")

    tooltip = " | ".join(tooltip_parts)

    return (
        f'<span class="risk-badge {risk_class}" title="{tooltip}">'
        f'OUTLIER RISK: {risk_level}</span>'
    )

# Get episode metadata
ep_meta = helpers.parse_ep_id(ep_id) or {}
show_slug = ep_meta.get("show_slug", ep_id.rsplit("-", 2)[0] if "-" in ep_id else ep_id)
season_label = None
season_value = ep_meta.get("season")
if isinstance(season_value, int):
    season_label = f"S{season_value:02d}"

# Fetch required data with proper error handling
cluster_result = _safe_api_get(f"/episodes/{ep_id}/cluster_tracks")
cast_result = _safe_api_get(f"/shows/{show_slug}/cast" + (f"?season={season_label}" if season_label else ""))
people_resp = _fetch_people_cached(show_slug)
unlinked_result = _safe_api_get(f"/episodes/{ep_id}/unlinked_entities")

# Track API errors for display
_api_errors: List[str] = []
if not cluster_result.ok:
    _api_errors.append(f"Failed to load clusters: {cluster_result.error}")
if not cast_result.ok:
    _api_errors.append(f"Failed to load cast: {cast_result.error}")
if not unlinked_result.ok:
    _api_errors.append(f"Failed to load unlinked entities: {unlinked_result.error}")

# Show critical API errors
if _api_errors:
    for err in _api_errors:
        st.error(f"⚠️ {err}")

cluster_payload = cluster_result.data if cluster_result.ok else {"clusters": []}
cast_api_resp = cast_result.data if cast_result.ok else None
unlinked_resp = unlinked_result.data if unlinked_result.ok else None

# Build cluster lookup
cluster_lookup: Dict[str, Dict[str, Any]] = {}
for cluster in cluster_payload.get("clusters", []):
    identity_id = cluster.get("identity_id")
    if identity_id:
        cluster_lookup[identity_id] = cluster

# Build cast options
cast_options: Dict[str, str] = {}
cast_members_list: List[Dict[str, Any]] = []
if cast_api_resp:
    cast_members_list = cast_api_resp.get("cast", [])
    cast_options = {
        cm.get("cast_id"): cm.get("name")
        for cm in cast_members_list
        if cm.get("cast_id") and cm.get("name")
    }

# Warn if no cast members defined (makes suggestions less useful)
if not cast_options and cast_result.ok:
    st.warning(
        "⚠️ **No cast members defined for this show.** "
        "Smart Suggestions work best when cast members are imported. "
        "[Add cast members in Show Management](/Show_Management) to enable cast matching."
    )

# Display singleton health warning if applicable
singleton_health = cluster_payload.get("singleton_health", {})
if singleton_health:
    singleton_frac = singleton_health.get("singleton_fraction", 0)
    health_status = singleton_health.get("health_status", "healthy")
    singleton_count = singleton_health.get("singleton_clusters", 0)
    total_clusters = singleton_health.get("total_clusters", 0)
    high_risk = singleton_health.get("high_risk_count", 0)
    single_frame = singleton_health.get("single_frame_tracks", 0)

    if health_status == "critical":
        # Gate: Show strong warning for critical singleton fraction
        st.error(
            f"🎯 **Singleton Fraction Critical: {int(singleton_frac * 100)}%** "
            f"({singleton_count}/{total_clusters} clusters)\n\n"
            f"High-risk singletons: {high_risk} · Single-frame tracks: {single_frame}\n\n"
            "Consider running **Singleton Merge** or adjusting clustering threshold before proceeding."
        )
    elif health_status == "warning":
        st.warning(
            f"🎯 **Singleton Fraction: {int(singleton_frac * 100)}%** "
            f"({singleton_count}/{total_clusters} clusters) - Above target (<25%)\n\n"
            f"High-risk: {high_risk} · Single-frame: {single_frame}"
        )
    # Show badge in metrics strip area
    if singleton_frac > 0:
        badge_html = render_singleton_fraction_badge(singleton_count, total_clusters, single_frame)
        st.markdown(
            f'<div style="margin: 8px 0;">Singleton Health: {badge_html}</div>',
            unsafe_allow_html=True,
        )

# Identify auto-clustered people (people without cast_id)
people = people_resp.get("people", []) if people_resp else []
people_by_id = {p.get("person_id"): p for p in people if p.get("person_id")}
# Build person_id -> cast_id lookup to know which persons are fully assigned
person_to_cast: Dict[str, str] = {
    p.get("person_id"): p.get("cast_id")
    for p in people
    if p.get("person_id") and p.get("cast_id")
}
unlinked_entities = unlinked_resp.get("entities", []) if unlinked_resp else []
auto_clustered_people: List[Dict[str, Any]] = []
unlinked_cluster_ids: set[str] = set()
for entity in unlinked_entities:
    cluster_ids = [cid for cid in entity.get("cluster_ids", []) if cid]
    if not cluster_ids:
        continue
    if entity.get("entity_type") == "person":
        person_id = entity.get("entity_id")
        person = people_by_id.get(person_id) or entity.get("person") or {"person_id": person_id}
        auto_clustered_people.append({
            "person": person,
            "person_id": person_id,
            "name": person.get("name") or f"Person {person_id}",
            "episode_clusters": cluster_ids,
        })
    else:
        unlinked_cluster_ids.update(cluster_ids)

# Navigation back to Faces Review
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("← Faces Review"):
        helpers.try_switch_page("pages/3_Faces_Review.py")

# Sync to S3 option for missing thumbnails
# Check storage backend to show appropriate UI
_storage_backend = os.getenv("STORAGE_BACKEND", "s3")
if _storage_backend == "s3":
    with st.expander("🔧 Thumbnails showing placeholders?", expanded=False):
        st.info(
            "👤 **Placeholder icons** indicate thumbnails haven't been synced to S3 yet. "
            "Click below to upload local thumbnails to S3 for proper display."
        )
        if st.button("📤 Sync Thumbnails to S3", key="sync_thumbs_s3_smart", use_container_width=True):
            with st.spinner("Syncing thumbnails and crops to S3..."):
                try:
                    resp = requests.post(f"{_api_base}/episodes/{ep_id}/sync_thumbnails_to_s3", timeout=120)
                    if resp.status_code == 200:
                        data = resp.json()
                        uploaded = data.get("uploaded_thumbs", 0) + data.get("uploaded_crops", 0)
                        if uploaded > 0:
                            st.success(f"✅ Uploaded {uploaded} file(s) to S3!")
                            st.rerun()
                        else:
                            st.info("All thumbnails already in S3 (or no local files found).")
                    else:
                        st.error(f"Sync failed: {resp.status_code}")
                except Exception as e:
                    st.error(f"Sync failed: {e}")

st.markdown("---")

# --- State Management Keys ---
suggestions_key = f"cast_suggestions:{ep_id}"
dismissed_key = f"dismissed_suggestions:{ep_id}"
mismatches_key = f"embedding_mismatches:{ep_id}"
status_key = f"smart_suggestions_status:{ep_id}"
error_key = f"smart_suggestions_error:{ep_id}"
auto_attempt_key = f"cast_suggestions_attempted:{ep_id}"
dismissed_loaded_key = f"dismissed_loaded:{ep_id}"
quality_only_key = f"quality_only_clusters:{ep_id}"
rescued_clusters_key = f"rescued_clusters:{ep_id}"

# Get or fetch cast suggestions with proper status tracking
cast_suggestions = st.session_state.get(suggestions_key, {})
embedding_mismatches = st.session_state.get(mismatches_key, [])
quality_only_clusters = st.session_state.get(quality_only_key, [])
rescued_clusters: set[str] = st.session_state.get(rescued_clusters_key, set())


def _load_dismissed_from_api() -> set:
    """Load dismissed suggestions from API (persisted to disk)."""
    try:
        result = _safe_api_get(f"/episodes/{ep_id}/dismissed_suggestions", timeout=5)
        if result.ok and result.data:
            return set(result.data.get("dismissed", []))
    except Exception as e:
        LOGGER.debug(f"[Smart Suggestions] Failed to load dismissed from API: {e}")
    return set()


def _save_dismissed_to_api(suggestion_ids: List[str]) -> bool:
    """Save dismissed suggestions to API."""
    try:
        result = _api_post(f"/episodes/{ep_id}/dismissed_suggestions", {"suggestion_ids": suggestion_ids}, timeout=5)
        return result.ok
    except Exception:
        return False


def _clear_dismissed_via_api() -> bool:
    """Clear all dismissed suggestions via API."""
    try:
        url = f"{_api_base}/episodes/{ep_id}/dismissed_suggestions"
        resp = requests.delete(url, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# Dismissed suggestions tracking - load from API once per session
if dismissed_loaded_key not in st.session_state:
    api_dismissed = _load_dismissed_from_api()
    st.session_state[dismissed_key] = api_dismissed
    st.session_state[dismissed_loaded_key] = True

dismissed = st.session_state.setdefault(dismissed_key, set())

# Status: "idle" | "loading" | "loaded" | "error"
suggestions_status = st.session_state.get(status_key, "idle")
suggestions_error = st.session_state.get(error_key)

# Auto-refresh suggestions once when none are cached
if not cast_suggestions and not st.session_state.get(auto_attempt_key, False):
    st.session_state[status_key] = "loading"
    with st.spinner("Analyzing cast vs unassigned clusters..."):
        result = _recompute_cast_suggestions(ep_id, show_slug)
        st.session_state[auto_attempt_key] = True
        if result.ok:
            cast_suggestions = result.suggestions
            embedding_mismatches = result.mismatched_embeddings
            st.session_state[status_key] = "loaded"
            st.session_state.pop(error_key, None)
        else:
            # Error state - keep existing suggestions (if any), show error banner
            st.session_state[status_key] = "error"
            st.session_state[error_key] = result.error
            suggestions_error = result.error
        dismissed = st.session_state.setdefault(dismissed_key, set())

# Show error banner if in error state
if suggestions_status == "error" and suggestions_error:
    st.error(f"⚠️ **Smart Suggestions API failed:** {suggestions_error}")
    col_retry, col_spacer = st.columns([1, 4])
    with col_retry:
        if st.button("🔄 Retry", key="retry_suggestions_top"):
            st.session_state[status_key] = "loading"
            with st.spinner("Retrying..."):
                result = _recompute_cast_suggestions(ep_id, show_slug)
                if result.ok:
                    cast_suggestions = result.suggestions
                    embedding_mismatches = result.mismatched_embeddings
                    st.session_state[status_key] = "loaded"
                    st.session_state.pop(error_key, None)
                    st.rerun()
                else:
                    st.session_state[error_key] = result.error

# Show embedding mismatch warnings if any clusters were skipped
if embedding_mismatches:
    mismatch_count = len(embedding_mismatches)
    with st.expander(f"⚠️ {mismatch_count} cluster(s) skipped due to embedding dimension mismatch", expanded=False):
        st.warning(
            "Some clusters couldn't be suggested because their embeddings are incompatible "
            "with the current facebank. This usually means different embedding models were used. "
            "Re-run clustering with a consistent model to resolve."
        )
        # Show diagnostics
        st.markdown("**Affected clusters:**")
        for mismatch in embedding_mismatches[:10]:  # Show first 10
            cluster_id = mismatch.get("cluster_id", "unknown")
            message = mismatch.get("message", "Dimension mismatch")
            st.text(f"  • {cluster_id}: {message}")
        if mismatch_count > 10:
            st.caption(f"... and {mismatch_count - 10} more")

# Collect clusters with suggestions, sorted by confidence
suggestion_entries = []
for cluster_id, cluster_data in cluster_lookup.items():
    if unlinked_cluster_ids and cluster_id not in unlinked_cluster_ids:
        continue
    # Skip if dismissed
    if cluster_id in dismissed:
        continue

    # Only skip if the cluster's person has a cast_id assigned
    # Clusters with person_id but NO cast_id still need suggestions
    person_id = cluster_data.get("person_id")
    if person_id and person_id in person_to_cast:
        continue  # This person is fully assigned to a cast member

    # Collect ALL track thumbnails with URL validation (carousel will limit display)
    tracks = cluster_data.get("tracks", [])
    thumb_urls = []
    for track in tracks:
        url = _validate_thumbnail_url(track.get("rep_thumb_url") or track.get("rep_media_url"))
        if url:
            thumb_urls.append(url)

    # Get cast suggestions for this cluster
    cluster_suggestions = cast_suggestions.get(cluster_id, [])
    if cluster_suggestions:
        best_suggestion = cluster_suggestions[0]

        suggestion_entries.append({
            "cluster_id": cluster_id,
            "cluster_data": cluster_data,
            "suggestion": best_suggestion,
            "all_suggestions": cluster_suggestions,
            "faces": cluster_data.get("counts", {}).get("faces", 0),
            "tracks": cluster_data.get("counts", {}).get("tracks", 0),
            "cohesion": cluster_data.get("cohesion"),
            "thumb_urls": thumb_urls,
            "rescued": cluster_id in rescued_clusters,  # Flag for force-embedded clusters
        })
    else:
        # Track clusters with NO suggestions (no centroids or no reference embeddings)
        # These still need assignment but have no comparison data
        suggestion_entries.append({
            "cluster_id": cluster_id,
            "cluster_data": cluster_data,
            "suggestion": {"similarity": 0.0, "name": "(No suggestions)", "cast_id": None},
            "all_suggestions": [],
            "faces": cluster_data.get("counts", {}).get("faces", 0),
            "tracks": cluster_data.get("counts", {}).get("tracks", 0),
            "cohesion": cluster_data.get("cohesion"),
            "thumb_urls": thumb_urls,
            "no_suggestions": True,  # Flag to identify these in UI
        })

# Collect person-level suggestions by aggregating cluster suggestions
person_suggestion_entries = []
for person_entry in auto_clustered_people:
    person_id = person_entry["person_id"]
    if f"person:{person_id}" in dismissed:
        continue
    episode_clusters = person_entry["episode_clusters"]
    suggestion_bucket: Dict[str, Dict[str, Any]] = {}
    total_faces = 0
    total_tracks = 0
    thumb_urls: List[str] = []

    # Collect quality metrics for outlier detection
    cluster_cohesions: List[float] = []
    cluster_similarities: List[float] = []  # Per-cluster best similarity to top cast
    track_internal_sims: List[float] = []  # Track-level internal consistency

    for cluster_id in episode_clusters:
        cluster_data = cluster_lookup.get(cluster_id, {})
        counts = cluster_data.get("counts", {}) if isinstance(cluster_data, dict) else {}
        total_faces += counts.get("faces", 0)
        total_tracks += counts.get("tracks", 0)

        # Collect cohesion for this cluster
        cohesion = cluster_data.get("cohesion")
        if cohesion is not None:
            cluster_cohesions.append(cohesion)

        # Collect track-level internal similarities
        tracks = cluster_data.get("tracks", [])
        for track in tracks:
            internal_sim = track.get("internal_similarity")
            if internal_sim is not None:
                track_internal_sims.append(internal_sim)

        # Aggregate suggestions by cast_id, keep the strongest similarity
        cluster_suggestions = cast_suggestions.get(cluster_id, [])
        best_cluster_sim = 0.0
        for sugg in cluster_suggestions:
            cast_id = sugg.get("cast_id")
            if not cast_id:
                continue
            sim = sugg.get("similarity", 0)
            if sim > best_cluster_sim:
                best_cluster_sim = sim
            current = suggestion_bucket.get(cast_id)
            if not current or sim > current.get("similarity", 0):
                suggestion_bucket[cast_id] = dict(sugg)
        if best_cluster_sim > 0:
            cluster_similarities.append(best_cluster_sim)

        # Collect ALL thumbnails with URL validation (carousel will limit display)
        for track in tracks:
            url = _validate_thumbnail_url(track.get("rep_thumb_url") or track.get("rep_media_url"))
            if url:
                thumb_urls.append(url)

    # Handle persons without suggestions - still include them for manual assignment
    if not suggestion_bucket:
        person_suggestion_entries.append({
            "person": person_entry["person"],
            "person_id": person_id,
            "name": person_entry["name"],
            "episode_clusters": episode_clusters,
            "best_suggestion": None,
            "all_suggestions": [],
            "faces": total_faces,
            "tracks": total_tracks,
            "thumb_urls": thumb_urls,
            "quality_metrics": {},
            "cluster_count": len(episode_clusters),
            "no_suggestions": True,  # Flag for UI grouping
        })
        continue

    sorted_suggs = sorted(suggestion_bucket.values(), key=lambda x: x.get("similarity", 0), reverse=True)

    # Compute quality metrics
    quality_metrics: Dict[str, Any] = {}
    if cluster_cohesions:
        quality_metrics["cohesion_min"] = min(cluster_cohesions)
        quality_metrics["cohesion_max"] = max(cluster_cohesions)
        quality_metrics["cohesion_avg"] = sum(cluster_cohesions) / len(cluster_cohesions)
    if cluster_similarities:
        quality_metrics["sim_min"] = min(cluster_similarities)
        quality_metrics["sim_max"] = max(cluster_similarities)
        quality_metrics["sim_avg"] = sum(cluster_similarities) / len(cluster_similarities)
        quality_metrics["sim_spread"] = max(cluster_similarities) - min(cluster_similarities)
    if track_internal_sims:
        quality_metrics["track_sim_min"] = min(track_internal_sims)
        quality_metrics["track_sim_avg"] = sum(track_internal_sims) / len(track_internal_sims)

    # Compute outlier risk metrics
    # Outliers are clusters/tracks that don't match the best cast suggestion well
    cluster_count = len(episode_clusters)
    outlier_risk: Dict[str, Any] = {"risk_level": "LOW", "cluster_outliers": 0, "cluster_total": cluster_count}

    if cluster_similarities and cluster_count > 1:
        # Threshold for identifying outlier clusters (similarity significantly below person average)
        # Use 70% of average or absolute threshold of 0.50, whichever is stricter
        avg_sim = sum(cluster_similarities) / len(cluster_similarities)
        outlier_thresh = max(avg_sim * 0.70, 0.50)

        # Count clusters with similarity below threshold
        cluster_outliers = sum(1 for sim in cluster_similarities if sim < outlier_thresh)
        outlier_fraction = cluster_outliers / cluster_count if cluster_count > 0 else 0

        outlier_risk["cluster_outliers"] = cluster_outliers
        outlier_risk["outlier_threshold"] = round(outlier_thresh, 3)

        # Determine risk level
        if outlier_fraction > 0.30:
            outlier_risk["risk_level"] = "HIGH"
        elif outlier_fraction > 0.10:
            outlier_risk["risk_level"] = "MEDIUM"
        else:
            outlier_risk["risk_level"] = "LOW"

    # Also flag if cohesion is very low for any cluster (inconsistent cluster)
    if cluster_cohesions:
        low_cohesion_count = sum(1 for c in cluster_cohesions if c < 0.60)
        outlier_risk["low_cohesion_clusters"] = low_cohesion_count
        if low_cohesion_count > 0 and outlier_risk["risk_level"] == "LOW":
            outlier_risk["risk_level"] = "MEDIUM"  # Upgrade to medium if cohesion issues

    quality_metrics["outlier_risk"] = outlier_risk

    person_suggestion_entries.append({
        "person": person_entry["person"],
        "person_id": person_id,
        "name": person_entry["name"],
        "episode_clusters": episode_clusters,
        "best_suggestion": sorted_suggs[0],
        "all_suggestions": sorted_suggs,
        "faces": total_faces,
        "tracks": total_tracks,
        "thumb_urls": thumb_urls,
        "quality_metrics": quality_metrics,
        "cluster_count": len(episode_clusters),
    })

# Sort options
SORT_OPTIONS = {
    "Similarity (High → Low)": ("similarity", True),
    "Similarity (Low → High)": ("similarity", False),
    "Frames (Most → Least)": ("faces", True),
    "Frames (Least → Most)": ("faces", False),
    "Tracks (Most → Least)": ("tracks", True),
    "Tracks (Least → Most)": ("tracks", False),
    "Suggested Name (A → Z)": ("name", False),
    "Suggested Name (Z → A)": ("name", True),
}


def _sort_entries(entries: List[Dict[str, Any]], sort_key: str, reverse: bool) -> List[Dict[str, Any]]:
    """Sort suggestion entries by the specified key."""
    if sort_key == "similarity":
        return sorted(entries, key=lambda x: x["suggestion"].get("similarity", 0), reverse=reverse)
    elif sort_key == "faces":
        return sorted(entries, key=lambda x: x.get("faces", 0), reverse=reverse)
    elif sort_key == "tracks":
        return sorted(entries, key=lambda x: x.get("tracks", 0), reverse=reverse)
    elif sort_key == "name":
        return sorted(
            entries,
            key=lambda x: (x["suggestion"].get("name") or "").lower(),
            reverse=reverse,
        )
    return entries


# Stats bar (fallback to legacy unassigned count if endpoint empty)
total_unassigned = len(unlinked_entities) or sum(1 for c in cluster_lookup.values() if not c.get("person_id"))
with_suggestions = len(suggestion_entries) + len(person_suggestion_entries)
dismissed_count = len(dismissed)
auto_clustered_count = len(auto_clustered_people)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Needs Cast Assignment", total_unassigned)
with col2:
    st.metric("Auto-Clustered People", auto_clustered_count)
with col3:
    st.metric("With Suggestions", with_suggestions)
with col4:
    st.metric("Dismissed", dismissed_count)
with col5:
    # Refresh button with proper error handling
    if st.button("Refresh Suggestions", key="refresh_all_suggestions", use_container_width=True):
        st.session_state[status_key] = "loading"
        with st.spinner("Recomputing suggestions from latest cast assignments..."):
            result = _recompute_cast_suggestions(ep_id, show_slug)
            st.session_state[auto_attempt_key] = True
            if result.ok:
                st.session_state[status_key] = "loaded"
                st.session_state.pop(error_key, None)
                if result.suggestions:
                    st.toast(f"Found {len(result.suggestions)} cluster(s) with suggestions")
                    if result.mismatched_embeddings:
                        st.toast(f"⚠️ {len(result.mismatched_embeddings)} cluster(s) skipped (embedding mismatch)", icon="⚠️")
                    st.rerun()
                else:
                    st.info("No suggestions available - all clusters may already be assigned")
            else:
                st.session_state[status_key] = "error"
                st.session_state[error_key] = result.error
                st.error(f"⚠️ Failed to refresh: {result.error}")

# Undo button (if there's history)
last_action = _get_last_undo_action(ep_id)
if last_action:
    undo_col1, undo_col2 = st.columns([1, 5])
    with undo_col1:
        if st.button(f"↩️ Undo: {last_action.describe()}", key="undo_last_action", use_container_width=True):
            action = _pop_undo_action(ep_id)
            if action:
                if action.action_type == "dismiss":
                    # Restore dismissed suggestion
                    if action.cluster_id:
                        dismissed.discard(action.cluster_id)
                        # Also restore via API
                        try:
                            url = f"{_api_base}/episodes/{ep_id}/dismissed_suggestions/{action.cluster_id}"
                            requests.delete(url, timeout=5)
                        except Exception:
                            pass
                    elif action.person_id:
                        dismissed.discard(f"person:{action.person_id}")
                        try:
                            url = f"{_api_base}/episodes/{ep_id}/dismissed_suggestions/person:{action.person_id}"
                            requests.delete(url, timeout=5)
                        except Exception:
                            pass
                    st.toast("Restored dismissed suggestion")
                    st.rerun()
                elif action.action_type == "assign":
                    # Note: Full undo of assignment requires unassigning the cluster
                    # This is more complex - for now just show a message
                    st.warning(
                        f"To undo assignment of cluster `{action.cluster_id}`, "
                        f"go to Faces Review and manually unassign it."
                    )
                elif action.action_type == "link":
                    st.warning(
                        f"To undo linking of person `{action.person_id}`, "
                        f"go to Faces Review and manually unlink it."
                    )

st.markdown("---")

# Sort control with episode-specific persistence
sort_pref_key = f"smart_suggestions_sort:{ep_id}"
if sort_pref_key not in st.session_state:
    st.session_state[sort_pref_key] = "Similarity (High → Low)"

sort_col1, sort_col2, bulk_col = st.columns([1, 1, 2])
with sort_col1:
    selected_sort = st.selectbox(
        "Sort by",
        options=list(SORT_OPTIONS.keys()),
        index=list(SORT_OPTIONS.keys()).index(st.session_state.get(sort_pref_key, "Similarity (High → Low)")),
        key=f"sort_select_{ep_id}",
    )
    # Persist selection
    if selected_sort != st.session_state.get(sort_pref_key):
        st.session_state[sort_pref_key] = selected_sort

# Get config thresholds for bulk action labels
config_thresholds = _fetch_config_thresholds()
high_label = config_thresholds.get("cast_high_label", "≥68%")

with bulk_col:
    # Bulk action buttons
    bulk_c1, bulk_c2, bulk_c3 = st.columns(3)
    with bulk_c1:
        # Count high-confidence suggestions (with singleton penalty applied)
        high_conf_entries = [
            e for e in suggestion_entries
            if _get_confidence_level(
                e["suggestion"].get("similarity", 0),
                face_count=e.get("faces"),
                track_count=e.get("tracks"),
            ) == "high"
        ]
        high_count = len(high_conf_entries)
        if st.button(f"✓ Accept All High ({high_count})", key="bulk_accept_high", disabled=high_count == 0, use_container_width=True):
            accepted = 0
            for entry in high_conf_entries:
                cluster_id = entry["cluster_id"]
                suggestion = entry["suggestion"]
                cast_id = suggestion.get("cast_id")
                if not cast_id:
                    continue
                # Find or create person
                people_resp = _fetch_people_cached(show_slug)
                people = people_resp.get("people", []) if people_resp else []
                target_person = next((p for p in people if p.get("cast_id") == cast_id), None)
                target_person_id = target_person.get("person_id") if target_person else None
                payload = {
                    "strategy": "manual",
                    "cluster_ids": [cluster_id],
                    "target_person_id": target_person_id,
                    "cast_id": cast_id,
                }
                result = _api_post(f"/episodes/{ep_id}/clusters/group", payload, timeout=10)
                if result.ok:
                    accepted += 1
                    if cluster_id in cast_suggestions:
                        del cast_suggestions[cluster_id]
            if accepted > 0:
                st.toast(f"Assigned {accepted} high-confidence clusters")
                st.rerun()

    with bulk_c2:
        # Dismiss all low confidence (with singleton penalty applied)
        low_conf_entries = [
            e for e in suggestion_entries
            if _get_confidence_level(
                e["suggestion"].get("similarity", 0),
                face_count=e.get("faces"),
                track_count=e.get("tracks"),
            ) == "low"
        ]
        low_count = len(low_conf_entries)
        if st.button(f"✗ Dismiss Low ({low_count})", key="bulk_dismiss_low", disabled=low_count == 0, use_container_width=True):
            to_dismiss = [e["cluster_id"] for e in low_conf_entries]
            for cid in to_dismiss:
                dismissed.add(cid)
            _save_dismissed_to_api(to_dismiss)
            st.toast(f"Dismissed {low_count} low-confidence suggestions")
            st.rerun()

    with bulk_c3:
        # Special action for HIGH risk singletons with MEDIUM+ similarity
        # These need extra review since they're unreliable
        high_risk_singletons = [
            e for e in suggestion_entries
            if e.get("cluster_data", {}).get("singleton_risk") == "HIGH"
            and e["suggestion"].get("similarity", 0) >= config_thresholds.get("cast_medium", 0.50)
        ]
        singleton_count = len(high_risk_singletons)
        if st.button(
            f"🎯 Triage Singletons ({singleton_count})",
            key="bulk_triage_singletons",
            disabled=singleton_count == 0,
            use_container_width=True,
            help="Review HIGH-risk singletons with decent matches"
        ):
            # Set filter to show only these
            st.session_state["singleton_triage_mode"] = True
            st.rerun()

# Apply sort
sort_key, sort_reverse = SORT_OPTIONS[selected_sort]
suggestion_entries = _sort_entries(suggestion_entries, sort_key, sort_reverse)

# Singleton triage mode - filter to only show high-risk singletons
singleton_triage_mode = st.session_state.get("singleton_triage_mode", False)
if singleton_triage_mode:
    original_count = len(suggestion_entries)
    suggestion_entries = [
        e for e in suggestion_entries
        if e.get("cluster_data", {}).get("singleton_risk") == "HIGH"
    ]
    triage_count = len(suggestion_entries)

    st.info(
        f"🎯 **Singleton Triage Mode**: Showing {triage_count} HIGH-risk singletons "
        f"(filtered from {original_count} total suggestions)"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Exit Triage", key="exit_singleton_triage"):
            st.session_state["singleton_triage_mode"] = False
            st.rerun()

    st.markdown("---")

if not suggestion_entries and not person_suggestion_entries:
    # Distinguish between "no suggestions available" (success) and "API failed" (error)
    if suggestions_status == "error":
        # Error state already shown above; don't show "no suggestions" message
        pass
    else:
        st.info("No pending suggestions. Click 'Refresh Suggestions' to find matches for unassigned clusters.")
    # Clear dismissed button if there are dismissed entries
    if dismissed:
        if st.button("Clear All Dismissed", key="clear_all_dismissed", use_container_width=True):
            _clear_dismissed_via_api()  # Persist to disk
            st.session_state[dismissed_key] = set()
            st.rerun()
    st.stop()


def render_suggestion_row(entry: Dict[str, Any], idx: int) -> None:
    """Render a single suggestion as a full-width row with thumbnails."""
    cluster_id = entry["cluster_id"]
    cluster_data = entry["cluster_data"]
    suggestion = entry["suggestion"]
    cast_id = suggestion.get("cast_id")
    cast_name = suggestion.get("name") or cast_options.get(cast_id, cast_id) if cast_id else "(No suggestions)"
    similarity = suggestion.get("similarity", 0)
    confidence = suggestion.get("confidence", "low")
    source = suggestion.get("source", "facebank")
    faces_used = suggestion.get("faces_used")
    faces = entry["faces"]
    tracks = entry["tracks"]
    cohesion = entry["cohesion"]
    thumb_urls = entry["thumb_urls"]
    track_list = cluster_data.get("tracks", [])
    has_suggestions = not entry.get("no_suggestions", False)
    is_rescued = entry.get("rescued", False)  # Cluster was force-embedded via quality rescue

    # Build source label with extra info for frame-based suggestions
    source_label = source
    if source == "frame" and faces_used is not None and faces_used > 0:
        source_label = f"frame ({faces_used} face{'s' if faces_used != 1 else ''})"

    # Determine which similarity to show: cluster cohesion for multi-track, internal similarity for single-track
    similarity_value = None
    similarity_label = None
    similarity_type = None
    if tracks > 1 and cohesion is not None:
        # Multi-track cluster: show cluster cohesion
        similarity_value = cohesion
        similarity_label = "Cluster Similarity"
        similarity_type = SimilarityType.CLUSTER
    elif track_list:
        # Single/few-track cluster: show track internal similarity (frame consistency within track)
        # Try internal_similarity first (how similar rep frame is to track centroid)
        track = track_list[0]
        internal_sim = track.get("internal_similarity")
        if internal_sim is not None:
            similarity_value = internal_sim
            similarity_label = "Track Consistency"
            similarity_type = SimilarityType.TRACK
        else:
            # Fallback to track-to-cluster similarity if available
            track_sim = track.get("similarity")
            if track_sim is not None:
                similarity_value = track_sim
                similarity_label = "Track Similarity"
                similarity_type = SimilarityType.TRACK

    with st.container(border=True):
        # Row layout: thumbnails on left (fixed 147x184px each, max 3), info on right
        thumb_col, info_col, action_col = st.columns([3, 3, 1])

        with thumb_col:
            # Use carousel for cluster thumbnails (shows 5 at a time with navigation)
            _render_carousel_thumbnails(thumb_urls, f"cluster_{cluster_id}")

        with info_col:
            # Cluster info with singleton risk badge
            singleton_risk = cluster_data.get("singleton_risk", "LOW")
            singleton_origin = cluster_data.get("singleton_origin")
            risk_badge = ""
            if singleton_risk in ("HIGH", "MEDIUM"):
                risk_badge = " " + render_singleton_risk_badge(tracks, faces, singleton_origin)
            st.markdown(f"**Cluster `{cluster_id}`**{risk_badge}", unsafe_allow_html=True)
            st.caption(_format_counts(faces, tracks))

            if similarity_value is not None and similarity_label and similarity_type:
                sim_badge = render_similarity_badge(similarity_value, similarity_type, show_label=True, custom_label=similarity_label)
                st.markdown(sim_badge, unsafe_allow_html=True)

            st.markdown("")  # Spacer

            if has_suggestions:
                # Suggestion with confidence aligned to cast thresholds (68/50)
                # Apply singleton penalty for unreliable clusters
                cast_badge = render_similarity_badge(similarity, SimilarityType.CAST, show_label=True)
                conf_level = _get_confidence_level(similarity, face_count=faces, track_count=tracks)
                conf_colors = {"high": "#4CAF50", "medium": "#FF9800", "low": "#F44336"}
                conf_color = conf_colors.get(conf_level, "#9E9E9E")

                # Ambiguity margin (top1 vs top2) - show "Tie" if margin is 0
                margin_html = ""
                all_suggestions = entry.get("all_suggestions") or []
                if len(all_suggestions) > 1:
                    runner_up = all_suggestions[1].get("similarity", 0) or 0
                    margin_pct = int(max(similarity - runner_up, 0) * 100)
                    if margin_pct == 0:
                        margin_html = " · ⚠️ Tie (ambiguous)"
                    else:
                        margin_html = f" · Δ {margin_pct}%"
                # Build rescued badge if this cluster was force-embedded
                rescued_badge = ""
                if is_rescued:
                    rescued_badge = (
                        ' <span style="background-color: #FFA726; color: black; '
                        'padding: 2px 6px; border-radius: 4px; font-size: 0.75em; '
                        'margin-left: 4px;" title="Force-embedded via quality rescue">🔧 RESCUED</span>'
                    )
                st.markdown(
                    f'<span style="background-color: {conf_color}; color: white; '
                    f'padding: 3px 8px; border-radius: 4px; font-size: 0.9em; font-weight: bold;">'
                    f'{conf_level.upper()}</span>{rescued_badge} {cast_badge} → **{cast_name}** '
                    f'<span style="font-size: 0.75em; color: #888;">({source_label}){margin_html}</span>',
                    unsafe_allow_html=True,
                )

                # Alternative suggestions
                alt_suggestions = entry["all_suggestions"][1:3]
                if alt_suggestions:
                    alt_text = " · ".join([
                        f"{alt.get('name', 'Unknown')} ({int(alt.get('similarity', 0) * 100)}%)"
                        for alt in alt_suggestions
                    ])
                    st.caption(f"Also: {alt_text}")
            else:
                # No suggestions available - missing centroids or reference embeddings
                st.markdown(
                    '<span style="background-color: #424242; color: white; '
                    'padding: 3px 8px; border-radius: 4px; font-size: 0.9em;">⚫ NO DATA</span> '
                    '<span style="font-size: 0.85em;">No centroids or reference faces available</span>',
                    unsafe_allow_html=True,
                )
                st.caption("Assign manually or re-run clustering")

        with action_col:
            # Action buttons - stacked vertically
            # Only show Accept button if there are suggestions
            if has_suggestions and cast_id:
                if st.button("✓ Accept", key=f"sp_accept_{cluster_id}", use_container_width=True):
                    # Find or create person for this cast member
                    people_resp = _fetch_people_cached(show_slug)
                    people = people_resp.get("people", []) if people_resp else []
                    target_person = next(
                        (p for p in people if p.get("cast_id") == cast_id),
                        None,
                    )
                    target_person_id = target_person.get("person_id") if target_person else None

                    payload = {
                        "strategy": "manual",
                        "cluster_ids": [cluster_id],
                        "target_person_id": target_person_id,
                        "cast_id": cast_id,
                    }
                    # ATOMIC: Call API first, only remove suggestion if successful
                    result = _api_post(f"/episodes/{ep_id}/clusters/group", payload)
                    if result.ok and result.data and result.data.get("status") == "success":
                        # SUCCESS: Track for undo, then remove from suggestions
                        _push_undo_action(ep_id, UndoAction(
                            action_type="assign",
                            ep_id=ep_id,
                            cluster_id=cluster_id,
                            cast_id=cast_id,
                            cast_name=cast_name,
                        ))
                        st.toast(f"Assigned to {cast_name}")
                        if cluster_id in cast_suggestions:
                            del cast_suggestions[cluster_id]
                        st.rerun()
                    else:
                        # FAILURE: Keep suggestion visible, show detailed error
                        error_detail = result.error_message if result.error else "Unknown error"
                        st.error(f"⚠️ Assignment failed: {error_detail}")
                        LOGGER.error(f"[{ep_id}] Failed to accept suggestion for cluster {cluster_id}: {error_detail}")

            if st.button("View", key=f"sp_view_{cluster_id}", use_container_width=True):
                # Smart navigation based on cluster structure
                st.session_state["facebank_ep"] = ep_id
                st.session_state.pop("facebank_query_applied", None)
                if "person" in st.query_params:
                    del st.query_params["person"]

                # Check track count to decide navigation level
                if tracks > 1 or len(track_list) > 1:
                    # Multiple tracks - show tracks view
                    st.session_state["facebank_view"] = "cluster_tracks"
                    st.session_state["selected_identity"] = cluster_id
                    st.session_state["selected_person"] = None
                    st.session_state["selected_track"] = None
                    st.query_params["view"] = "cluster_tracks"
                    st.query_params["cluster"] = cluster_id
                    st.query_params["ep_id"] = ep_id
                    if "track" in st.query_params:
                        del st.query_params["track"]
                elif len(track_list) == 1:
                    # Single track - go directly to frames view
                    track_id = track_list[0].get("track_id")
                    st.session_state["facebank_view"] = "track"
                    st.session_state["selected_identity"] = cluster_id
                    st.session_state["selected_track"] = track_id
                    st.session_state["selected_person"] = None
                    st.query_params["view"] = "frames"
                    st.query_params["cluster"] = cluster_id
                    st.query_params["track"] = str(track_id)
                    st.query_params["ep_id"] = ep_id
                    # Auto-enable "Show skipped faces" for low-quality clusters (no embeddings)
                    if not has_suggestions:
                        st.query_params["low_quality"] = "true"
                else:
                    # Fallback to cluster view
                    st.session_state["facebank_view"] = "cluster_tracks"
                    st.session_state["selected_identity"] = cluster_id
                    st.session_state["selected_person"] = None
                    st.session_state["selected_track"] = None
                    st.query_params["view"] = "cluster_tracks"
                    st.query_params["cluster"] = cluster_id
                    st.query_params["ep_id"] = ep_id
                    if "track" in st.query_params:
                        del st.query_params["track"]

                helpers.try_switch_page("pages/3_Faces_Review.py")

            if st.button("✗ Skip", key=f"sp_dismiss_{cluster_id}", use_container_width=True):
                dismissed.add(cluster_id)
                _save_dismissed_to_api([cluster_id])  # Persist to disk
                _push_undo_action(ep_id, UndoAction(
                    action_type="dismiss",
                    ep_id=ep_id,
                    cluster_id=cluster_id,
                ))
                st.rerun()


# Helper function for rendering person rows (used in grouped sections below)
def render_person_row(person_entry: Dict[str, Any]) -> None:
    """Render a single person suggestion row."""
    person_id = person_entry["person_id"]
    person_name = person_entry["name"]
    episode_clusters = person_entry["episode_clusters"]
    best_suggestion = person_entry["best_suggestion"]
    all_suggestions = person_entry["all_suggestions"]
    faces = person_entry["faces"]
    tracks = person_entry["tracks"]
    person_thumb_urls = person_entry["thumb_urls"]
    quality_metrics = person_entry.get("quality_metrics", {})
    has_suggestions = not person_entry.get("no_suggestions", False)

    with st.container(border=True):
        thumb_col, info_col, action_col = st.columns([3, 3, 1])

        with thumb_col:
            _render_carousel_thumbnails(person_thumb_urls, f"person_{person_id}")

        with info_col:
            outlier_risk = quality_metrics.get("outlier_risk", {})
            risk_badge_html = ""
            if outlier_risk and len(episode_clusters) > 1:
                risk_badge_html = _render_outlier_risk_badge(outlier_risk)
            st.markdown(f"**{person_name}** `{person_id}` {risk_badge_html}", unsafe_allow_html=True)
            cluster_count = len(episode_clusters)
            cluster_str = f"{cluster_count} cluster{'s' if cluster_count != 1 else ''}"
            counts_str = _format_counts(faces, tracks)
            st.caption(f"{cluster_str} · {counts_str}" if counts_str != "No data available" else cluster_str)

            if quality_metrics and cluster_count > 1:
                metrics_parts = []
                if "cohesion_avg" in quality_metrics:
                    coh_avg = int(quality_metrics["cohesion_avg"] * 100)
                    coh_min = int(quality_metrics.get("cohesion_min", 0) * 100)
                    coh_max = int(quality_metrics.get("cohesion_max", 0) * 100)
                    if coh_min != coh_max:
                        metrics_parts.append(f"Cohesion: {coh_min}–{coh_max}% (avg {coh_avg}%)")
                    else:
                        metrics_parts.append(f"Cohesion: {coh_avg}%")
                if "sim_min" in quality_metrics and "sim_max" in quality_metrics:
                    sim_min = int(quality_metrics["sim_min"] * 100)
                    sim_max = int(quality_metrics["sim_max"] * 100)
                    sim_spread = int(quality_metrics.get("sim_spread", 0) * 100)
                    if sim_min != sim_max:
                        metrics_parts.append(f"Cast Match: {sim_min}–{sim_max}%")
                    if sim_spread >= 20:
                        metrics_parts.append("⚠️ High variance")
                if "track_sim_min" in quality_metrics:
                    track_min = int(quality_metrics["track_sim_min"] * 100)
                    if track_min < 60:
                        metrics_parts.append(f"⚠️ Track consistency: {track_min}% min")
                if metrics_parts:
                    metrics_html = ' <span style="color: #888; font-size: 0.75em;">(' + " · ".join(metrics_parts) + ')</span>'
                    st.markdown(metrics_html, unsafe_allow_html=True)

            if best_suggestion:
                cast_id = best_suggestion.get("cast_id")
                cast_name = best_suggestion.get("name") or cast_options.get(cast_id, cast_id)
                similarity = best_suggestion.get("similarity", 0)
                source = best_suggestion.get("source", "facebank")

                cast_badge = render_similarity_badge(similarity, SimilarityType.CAST, show_label=True)
                conf_level = _get_confidence_level(similarity, face_count=faces, track_count=len(episode_clusters))
                conf_colors = {"high": "#4CAF50", "medium": "#FF9800", "low": "#F44336"}
                conf_color = conf_colors.get(conf_level, "#9E9E9E")
                margin_html = ""
                if len(all_suggestions) > 1:
                    runner_up = all_suggestions[1].get("similarity", 0) or 0
                    margin_pct = int(max(similarity - runner_up, 0) * 100)
                    if margin_pct == 0:
                        margin_html = " · ⚠️ Tie (ambiguous)"
                    else:
                        margin_html = f" · Δ {margin_pct}%"
                st.markdown(
                    f'<span style="background-color: {conf_color}; color: white; '
                    f'padding: 3px 8px; border-radius: 4px; font-size: 0.9em; font-weight: bold;">'
                    f'{conf_level.upper()}</span> {cast_badge} → **{cast_name}** '
                    f'<span style="font-size: 0.75em; color: #888;">({source}){margin_html}</span>',
                    unsafe_allow_html=True,
                )
                alt_suggestions = all_suggestions[1:3]
                if alt_suggestions:
                    alt_text = " · ".join([
                        f"{alt.get('name', 'Unknown')} ({int(alt.get('similarity', 0) * 100)}%)"
                        for alt in alt_suggestions
                    ])
                    st.caption(f"Also: {alt_text}")
            else:
                st.caption("💡 Click 'Refresh Suggestions' to find cast matches")

        with action_col:
            if best_suggestion:
                cast_id = best_suggestion.get("cast_id")
                cast_name = best_suggestion.get("name") or cast_options.get(cast_id, cast_id)

                if st.button("✓ Link", key=f"sp_link_person_{person_id}", use_container_width=True):
                    payload = {"cast_id": cast_id}
                    result = _api_patch(f"/shows/{show_slug}/people/{person_id}", payload)
                    if result.ok:
                        st.toast(f"Linked {person_name} to {cast_name}")
                        people_cache_key = f"people_cache:{show_slug}"
                        if people_cache_key in st.session_state:
                            del st.session_state[people_cache_key]
                        st.rerun()
                    else:
                        st.error(f"⚠️ Link failed: {result.error_message}")
                        LOGGER.error(f"[{ep_id}] Failed to link person {person_id}: {result.error_message}")

            if st.button("View", key=f"sp_view_person_{person_id}", use_container_width=True):
                # Smart navigation based on person structure
                st.session_state["facebank_ep"] = ep_id
                st.session_state.pop("facebank_query_applied", None)

                cluster_count = len(episode_clusters)
                if cluster_count > 1:
                    # Multiple clusters - show person clusters view
                    st.session_state["facebank_view"] = "person_clusters"
                    st.session_state["selected_person"] = person_id
                    st.session_state["selected_identity"] = None
                    st.session_state["selected_track"] = None
                    st.query_params["view"] = "person_clusters"
                    st.query_params["person"] = person_id
                    st.query_params["ep_id"] = ep_id
                    if "cluster" in st.query_params:
                        del st.query_params["cluster"]
                    if "track" in st.query_params:
                        del st.query_params["track"]
                elif cluster_count == 1:
                    # Single cluster - check track count
                    first_cluster_id = episode_clusters[0]
                    cluster_data = cluster_lookup.get(first_cluster_id, {})
                    track_list = cluster_data.get("tracks", [])

                    if len(track_list) > 1:
                        # Multiple tracks - show tracks view for this cluster
                        st.session_state["facebank_view"] = "cluster_tracks"
                        st.session_state["selected_identity"] = first_cluster_id
                        st.session_state["selected_person"] = None
                        st.session_state["selected_track"] = None
                        st.query_params["view"] = "cluster_tracks"
                        st.query_params["cluster"] = first_cluster_id
                        st.query_params["ep_id"] = ep_id
                        if "person" in st.query_params:
                            del st.query_params["person"]
                        if "track" in st.query_params:
                            del st.query_params["track"]
                    elif len(track_list) == 1:
                        # Single track - go directly to frames
                        track_id = track_list[0].get("track_id")
                        st.session_state["facebank_view"] = "track"
                        st.session_state["selected_identity"] = first_cluster_id
                        st.session_state["selected_track"] = track_id
                        st.session_state["selected_person"] = None
                        st.query_params["view"] = "frames"
                        st.query_params["cluster"] = first_cluster_id
                        st.query_params["track"] = str(track_id)
                        st.query_params["ep_id"] = ep_id
                        if "person" in st.query_params:
                            del st.query_params["person"]
                        # Auto-enable "Show skipped faces" for low-quality clusters (no embeddings)
                        if not has_suggestions:
                            st.query_params["low_quality"] = "true"
                    else:
                        # Fallback to person clusters view
                        st.session_state["facebank_view"] = "person_clusters"
                        st.session_state["selected_person"] = person_id
                        st.query_params["view"] = "person_clusters"
                        st.query_params["person"] = person_id
                        st.query_params["ep_id"] = ep_id
                else:
                    # No clusters (shouldn't happen) - fallback
                    st.session_state["facebank_view"] = "person_clusters"
                    st.session_state["selected_person"] = person_id
                    st.query_params["view"] = "person_clusters"
                    st.query_params["person"] = person_id
                    st.query_params["ep_id"] = ep_id

                helpers.try_switch_page("pages/3_Faces_Review.py")

            if st.button("✗ Skip", key=f"sp_dismiss_person_{person_id}", use_container_width=True):
                suggestion_id = f"person:{person_id}"
                dismissed.add(suggestion_id)
                _save_dismissed_to_api([suggestion_id])
                _push_undo_action(ep_id, UndoAction(
                    action_type="dismiss",
                    ep_id=ep_id,
                    person_id=person_id,
                ))
                st.rerun()


def _group_by_cast_id(entries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group cluster entries by their suggested cast_id.

    Returns dict mapping cast_id -> list of entries suggesting that cast member.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        cast_id = entry["suggestion"].get("cast_id")
        if not cast_id:
            # Put entries without cast_id in a special group
            cast_id = "__unknown__"
        if cast_id not in groups:
            groups[cast_id] = []
        groups[cast_id].append(entry)
    return groups


def render_grouped_cast_row(
    cast_id: str,
    cast_name: str,
    entries: List[Dict[str, Any]],
    confidence_level: str,
) -> None:
    """Render a grouped row for multiple clusters suggesting the same cast member.

    Args:
        cast_id: The cast member ID
        cast_name: Display name for the cast member
        entries: List of cluster entries suggesting this cast member
        confidence_level: "auto", "high", "medium", "low", or "rest"
    """
    # Aggregate data from all clusters
    all_thumb_urls: List[str] = []
    all_cluster_ids: List[str] = []
    total_faces = 0
    total_tracks = 0
    similarities: List[float] = []
    sources: set = set()

    for entry in entries:
        all_cluster_ids.append(entry["cluster_id"])
        all_thumb_urls.extend(entry.get("thumb_urls", []))
        total_faces += entry.get("faces", 0)
        total_tracks += entry.get("tracks", 0)
        sim = entry["suggestion"].get("similarity", 0)
        if sim > 0:
            similarities.append(sim)
        source = entry["suggestion"].get("source", "facebank")
        sources.add(source)

    # Calculate aggregate stats
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    max_similarity = max(similarities) if similarities else 0
    min_similarity = min(similarities) if similarities else 0
    cluster_count = len(entries)

    # Build source label
    source_label = "/".join(sorted(sources)) if sources else "facebank"

    # Color for confidence level
    conf_colors = {
        "auto": "#2196F3",   # Blue for auto
        "high": "#4CAF50",   # Green
        "medium": "#FF9800", # Orange
        "low": "#F44336",    # Red
        "rest": "#9E9E9E",   # Gray
    }
    conf_color = conf_colors.get(confidence_level, "#9E9E9E")

    # Generate unique key for this group
    group_key = f"cast_{cast_id}_{confidence_level}"

    with st.container(border=True):
        thumb_col, info_col, action_col = st.columns([3, 3, 1])

        with thumb_col:
            # Use carousel for all thumbnails from all clusters
            _render_carousel_thumbnails(all_thumb_urls, group_key)

        with info_col:
            # Show cast member name prominently
            st.markdown(f"### → **{cast_name}**")

            # Show aggregate stats
            st.caption(
                f"{cluster_count} cluster{'s' if cluster_count != 1 else ''} · "
                f"{_format_counts(total_faces, total_tracks)}"
            )

            # Show similarity range
            if min_similarity != max_similarity:
                sim_range = f"{int(min_similarity * 100)}–{int(max_similarity * 100)}%"
            else:
                sim_range = f"{int(max_similarity * 100)}%"

            cast_badge = render_similarity_badge(avg_similarity, SimilarityType.CAST, show_label=True)
            st.markdown(
                f'<span style="background-color: {conf_color}; color: white; '
                f'padding: 3px 8px; border-radius: 4px; font-size: 0.9em; font-weight: bold;">'
                f'{confidence_level.upper()}</span> {cast_badge} '
                f'<span style="font-size: 0.75em; color: #888;">({source_label}) Range: {sim_range}</span>',
                unsafe_allow_html=True,
            )

            # Show cluster IDs in collapsed view
            if cluster_count > 1:
                with st.expander(f"Cluster IDs ({cluster_count})", expanded=False):
                    for cid in all_cluster_ids:
                        st.text(cid)

        with action_col:
            # Accept All button - assigns all clusters to this cast member
            if st.button(f"✓ Accept All ({cluster_count})", key=f"grp_accept_{group_key}", use_container_width=True):
                accepted = 0
                people_resp = _fetch_people_cached(show_slug)
                people = people_resp.get("people", []) if people_resp else []
                target_person = next((p for p in people if p.get("cast_id") == cast_id), None)
                target_person_id = target_person.get("person_id") if target_person else None

                for entry in entries:
                    cluster_id = entry["cluster_id"]
                    payload = {
                        "strategy": "manual",
                        "cluster_ids": [cluster_id],
                        "target_person_id": target_person_id,
                        "cast_id": cast_id,
                    }
                    result = _api_post(f"/episodes/{ep_id}/clusters/group", payload, timeout=10)
                    if result.ok:
                        accepted += 1
                        if cluster_id in cast_suggestions:
                            del cast_suggestions[cluster_id]

                if accepted > 0:
                    st.toast(f"Assigned {accepted} cluster(s) to {cast_name}")
                    st.rerun()

            if st.button("View", key=f"grp_view_{group_key}", use_container_width=True):
                # Smart navigation: pick the most appropriate level
                # Multiple clusters → cluster list view
                # Single cluster with multiple tracks → tracks view
                # Single cluster with single track → frames view
                st.session_state["facebank_ep"] = ep_id
                st.session_state.pop("facebank_query_applied", None)

                if cluster_count > 1:
                    # Multiple clusters - show cluster list (use first cluster as anchor)
                    first_cluster = all_cluster_ids[0]
                    st.session_state["facebank_view"] = "cluster_tracks"
                    st.session_state["selected_identity"] = first_cluster
                    st.session_state["selected_person"] = None
                    st.session_state["selected_track"] = None
                    st.query_params["view"] = "cluster_tracks"
                    st.query_params["cluster"] = first_cluster
                    st.query_params["ep_id"] = ep_id
                elif cluster_count == 1:
                    # Single cluster - check track count
                    first_cluster = all_cluster_ids[0]
                    entry = entries[0]
                    track_count = entry.get("tracks", 0)
                    track_list = entry.get("cluster_data", {}).get("tracks", [])

                    if track_count > 1 or len(track_list) > 1:
                        # Multiple tracks - show tracks view
                        st.session_state["facebank_view"] = "cluster_tracks"
                        st.session_state["selected_identity"] = first_cluster
                        st.session_state["selected_person"] = None
                        st.session_state["selected_track"] = None
                        st.query_params["view"] = "cluster_tracks"
                        st.query_params["cluster"] = first_cluster
                        st.query_params["ep_id"] = ep_id
                    elif len(track_list) == 1:
                        # Single track - go directly to frames view
                        track_id = track_list[0].get("track_id")
                        st.session_state["facebank_view"] = "track"
                        st.session_state["selected_identity"] = first_cluster
                        st.session_state["selected_track"] = track_id
                        st.session_state["selected_person"] = None
                        st.query_params["view"] = "frames"
                        st.query_params["cluster"] = first_cluster
                        st.query_params["track"] = str(track_id)
                        st.query_params["ep_id"] = ep_id
                    else:
                        # Fallback to cluster view
                        st.session_state["facebank_view"] = "cluster_tracks"
                        st.session_state["selected_identity"] = first_cluster
                        st.query_params["view"] = "cluster_tracks"
                        st.query_params["cluster"] = first_cluster
                        st.query_params["ep_id"] = ep_id

                helpers.try_switch_page("pages/3_Faces_Review.py")

            if st.button(f"✗ Skip All ({cluster_count})", key=f"grp_dismiss_{group_key}", use_container_width=True):
                for cid in all_cluster_ids:
                    dismissed.add(cid)
                _save_dismissed_to_api(all_cluster_ids)
                st.rerun()


def render_confidence_section(
    section_title: str,
    section_caption: str | None,
    cluster_entries: List[Dict[str, Any]],
    person_entries: List[Dict[str, Any]],
    confidence_level: str,
) -> int:
    """Render a confidence section with clusters grouped by cast member.

    Args:
        section_title: Markdown title for the section
        section_caption: Optional caption below title
        cluster_entries: List of cluster suggestion entries
        person_entries: List of person suggestion entries
        confidence_level: "auto", "high", "medium", "low", or "rest"

    Returns:
        Number of items rendered (for index tracking)
    """
    if not cluster_entries and not person_entries:
        return 0

    st.markdown(section_title)
    if section_caption:
        st.caption(section_caption)

    # Group clusters by cast_id
    cast_groups = _group_by_cast_id(cluster_entries)

    # Render each cast member group
    items_rendered = 0
    for cast_id, entries in sorted(cast_groups.items(), key=lambda x: -max(e["suggestion"].get("similarity", 0) for e in x[1])):
        if cast_id == "__unknown__":
            # Render unknown entries individually
            for entry in entries:
                render_suggestion_row(entry, items_rendered)
                items_rendered += 1
        else:
            cast_name = entries[0]["suggestion"].get("name") or cast_options.get(cast_id, cast_id)
            render_grouped_cast_row(cast_id, cast_name, entries, confidence_level)
            items_rendered += len(entries)

    # Render person entries (these are already aggregated by person)
    for person_entry in person_entries:
        render_person_row(person_entry)

    return items_rendered


# --- NEEDS-ASSIGNMENT SECTION (Grouped by Confidence) ---
if suggestion_entries or person_suggestion_entries:
    st.markdown("### 🔍 Needs Cast Assignment")
    st.caption("Clusters and auto-people grouped by confidence level: AUTO ≥95%, HIGH 65-94%, MEDIUM 35-64%, LOW 15-34%, REST <15%")

    # Group cluster suggestions by confidence level
    def _get_entry_confidence(entry: Dict[str, Any]) -> str:
        sim = entry["suggestion"].get("similarity", 0)
        faces = entry.get("faces", 0)
        tracks = entry.get("tracks", 0)
        return _get_confidence_level(sim, face_count=faces, track_count=tracks)

    def _get_person_confidence(person_entry: Dict[str, Any]) -> str:
        sim = person_entry["best_suggestion"].get("similarity", 0)
        faces = person_entry.get("faces", 0)
        clusters = len(person_entry.get("episode_clusters", []))
        return _get_confidence_level(sim, face_count=faces, track_count=clusters)

    # Group all entries by CAST MEMBER first, then place group in tier based on BEST similarity
    # This ensures all clusters suggesting the same cast member appear together
    no_suggestion_clusters = [e for e in suggestion_entries if e.get("no_suggestions")]
    with_suggestions = [e for e in suggestion_entries if not e.get("no_suggestions")]

    # Group clusters by cast_id
    cast_member_groups: Dict[str, List[Dict[str, Any]]] = {}
    unknown_clusters: List[Dict[str, Any]] = []
    for entry in with_suggestions:
        cast_id = entry["suggestion"].get("cast_id")
        if cast_id:
            if cast_id not in cast_member_groups:
                cast_member_groups[cast_id] = []
            cast_member_groups[cast_id].append(entry)
        else:
            unknown_clusters.append(entry)

    # Place each cast member group in tier based on BEST similarity in the group
    auto_clusters: List[Dict[str, Any]] = []
    high_clusters: List[Dict[str, Any]] = []
    medium_clusters: List[Dict[str, Any]] = []
    low_clusters: List[Dict[str, Any]] = []
    rest_clusters: List[Dict[str, Any]] = []

    for cast_id, entries in cast_member_groups.items():
        # Find best similarity in the group
        best_sim = max(e["suggestion"].get("similarity", 0) for e in entries)
        # Use best entry's face/track counts for singleton penalty
        best_entry = max(entries, key=lambda e: e["suggestion"].get("similarity", 0))
        conf_level = _get_confidence_level(
            best_sim,
            face_count=best_entry.get("faces", 0),
            track_count=best_entry.get("tracks", 0),
        )
        # Place ALL entries for this cast member in the same tier
        if conf_level == "auto":
            auto_clusters.extend(entries)
        elif conf_level == "high":
            high_clusters.extend(entries)
        elif conf_level == "medium":
            medium_clusters.extend(entries)
        elif conf_level == "low":
            low_clusters.extend(entries)
        else:
            rest_clusters.extend(entries)

    # Unknown clusters go in their individual tiers
    for entry in unknown_clusters:
        conf_level = _get_entry_confidence(entry)
        if conf_level == "auto":
            auto_clusters.append(entry)
        elif conf_level == "high":
            high_clusters.append(entry)
        elif conf_level == "medium":
            medium_clusters.append(entry)
        elif conf_level == "low":
            low_clusters.append(entry)
        else:
            rest_clusters.append(entry)

    # Filter out persons without suggestions - they'll be rendered separately
    no_suggestion_persons = [p for p in person_suggestion_entries if p.get("no_suggestions")]
    persons_with_suggestions = [p for p in person_suggestion_entries if not p.get("no_suggestions")]

    auto_persons = [p for p in persons_with_suggestions if _get_person_confidence(p) == "auto"]
    high_persons = [p for p in persons_with_suggestions if _get_person_confidence(p) == "high"]
    medium_persons = [p for p in persons_with_suggestions if _get_person_confidence(p) == "medium"]
    low_persons = [p for p in persons_with_suggestions if _get_person_confidence(p) == "low"]
    rest_persons = [p for p in persons_with_suggestions if _get_person_confidence(p) == "rest"]

    # Get threshold labels for section headers
    thresholds = _fetch_config_thresholds()
    auto_thresh = int(thresholds.get("cast_auto_assign", 0.95) * 100)
    high_thresh = int(thresholds.get("cast_high", 0.65) * 100)
    med_thresh = int(thresholds.get("cast_medium", 0.35) * 100)
    low_thresh = int(thresholds.get("cast_low", 0.15) * 100)

    # Render each confidence section with grouping by cast member
    render_confidence_section(
        f"#### 🟢 Auto-Accept (≥{auto_thresh}%)",
        "Very high confidence - can be auto-assigned",
        auto_clusters,
        auto_persons,
        "auto",
    )

    render_confidence_section(
        f"#### ✅ High Confidence ({high_thresh}–{auto_thresh - 1}%)",
        None,
        high_clusters,
        high_persons,
        "high",
    )

    render_confidence_section(
        f"#### ⚠️ Medium Confidence ({med_thresh}–{high_thresh - 1}%)",
        None,
        medium_clusters,
        medium_persons,
        "medium",
    )

    render_confidence_section(
        f"#### ❓ Low Confidence ({low_thresh}–{med_thresh - 1}%)",
        "Weak matches - review carefully before accepting",
        low_clusters,
        low_persons,
        "low",
    )

    render_confidence_section(
        f"#### 🔴 Rest of Tracks (<{low_thresh}%)",
        "Very weak matches - these likely need manual review or are different people",
        rest_clusters,
        rest_persons,
        "rest",
    )

    # --- QUALITY-ONLY CLUSTERS SECTION (Low-Quality / No Embeddings) ---
    if quality_only_clusters:
        st.markdown(f"#### 🔧 Low-Quality / No Embeddings ({len(quality_only_clusters)})")

        # Explain what this section is and provide rescue options
        with st.expander("ℹ️ What are quality-only clusters?", expanded=False):
            st.markdown("""
**These clusters have faces that were all marked as too blurry or low-quality for embedding.**

Unlike "No Suggestions" clusters, these were **intentionally filtered** by the quality gate to prevent
unreliable matches. They can potentially be "rescued" if the blurriness threshold was too aggressive.

**Temporal Hints:**
When you see `⏱️ Nearby: Name1, Name2`, those cast members appear in frames close to this track.
This can help you guess who the blurry person might be, even without embeddings.

**Options:**
1. **Force Embed (Rescue)** - Re-embed faces with a lower quality threshold (`inclusive` profile).
   This may produce less reliable matches but allows the cluster to participate in suggestions.
2. **View in Faces Review** - Opens the track with skipped faces visible, so you can manually assign
   or review the quality.

**Quality Profiles (Laplacian variance thresholds):**
- `strict` (100.0) — Only crisp faces, maximum embedding quality
- `balanced` (50.0) — Default trade-off between quality and coverage
- `inclusive` (15.0) — Rescue mode for borderline blurry faces
- `bypass` (0.0) — No filtering, all faces pass (desperate rescue only)

**Note:** Rescued clusters will show a `🔧 RESCUED` badge in suggestions to indicate lower confidence.
            """)

        # Render each quality-only cluster
        for idx, qc in enumerate(quality_only_clusters):
            cluster_id = qc.get("cluster_id", f"qc_{idx}")
            track_ids = qc.get("track_ids", [])
            total_faces = qc.get("total_faces", 0)
            skipped_faces = qc.get("skipped_faces", 0)
            blurry_count = qc.get("blurry_count", 0)
            reason_summary = qc.get("reason_summary", "All faces skipped due to quality")
            person_id = qc.get("person_id")
            frame_range = qc.get("frame_range")
            temporal_suggestions = qc.get("temporal_suggestions", [])

            with st.container():
                cols = st.columns([0.6, 0.2, 0.2])

                with cols[0]:
                    # Cluster info
                    st.markdown(
                        f'<span style="background-color: #FFA726; color: black; '
                        f'padding: 3px 8px; border-radius: 4px; font-size: 0.9em;">🔧 QUALITY ONLY</span> '
                        f'<span style="font-size: 0.85em;">{reason_summary}</span>',
                        unsafe_allow_html=True,
                    )
                    tracks_str = f"{len(track_ids)} track{'s' if len(track_ids) != 1 else ''}"
                    faces_str = f"{total_faces} face{'s' if total_faces != 1 else ''}"
                    blurry_str = f"{blurry_count} blurry" if blurry_count else ""
                    detail_parts = [tracks_str, faces_str]
                    if blurry_str:
                        detail_parts.append(blurry_str)
                    if frame_range:
                        detail_parts.append(f"frames {frame_range[0]}-{frame_range[1]}")
                    st.caption(f"Cluster {cluster_id} · " + " · ".join(detail_parts))

                    # Display temporal suggestions (nearby cast members)
                    if temporal_suggestions:
                        nearby_names = [ts.get("cast_name", "Unknown") for ts in temporal_suggestions[:3]]
                        st.markdown(
                            f'<span style="font-size: 0.8em; color: #9E9E9E;">⏱️ Nearby: '
                            f'{", ".join(nearby_names)}</span>',
                            unsafe_allow_html=True,
                        )

                with cols[1]:
                    # Force Embed button
                    first_track = track_ids[0] if track_ids else None
                    if first_track is not None:
                        force_key = f"force_embed_{cluster_id}_{idx}"
                        if st.button("🔧 Force Embed", key=force_key, help="Re-embed with lower quality threshold"):
                            try:
                                result = _api_post(
                                    f"/episodes/{ep_id}/tracks/{first_track}/force_embed",
                                    {"quality_profile": "inclusive", "recompute_centroid": True},
                                    timeout=30,
                                )
                                if result.ok:
                                    st.success(f"Track {first_track} rescued! Refresh to see suggestions.")
                                    # Remove from quality_only_clusters in session state
                                    updated = [c for c in quality_only_clusters if c.get("cluster_id") != cluster_id]
                                    st.session_state[quality_only_key] = updated
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(f"Failed: {result.error}")
                            except Exception as e:
                                st.error(f"Error: {e}")

                with cols[2]:
                    # View in Faces Review button
                    if first_track is not None:
                        view_key = f"view_quality_{cluster_id}_{idx}"
                        if st.button("👁️ View", key=view_key, help="Open in Faces Review with skipped faces visible"):
                            faces_url = f"/3_Faces_Review?ep_id={ep_id}&cluster_id={cluster_id}&low_quality=true"
                            st.markdown(f'<meta http-equiv="refresh" content="0; url={faces_url}">', unsafe_allow_html=True)

                st.markdown("---")

    # --- NO SUGGESTIONS SECTION (Missing Centroids/Embeddings) ---
    if no_suggestion_clusters or no_suggestion_persons:
        total_no_sugg = len(no_suggestion_clusters) + len(no_suggestion_persons)
        st.markdown(f"#### ⚫ No Suggestions Available ({total_no_sugg})")

        # Explain why and provide actionable steps
        with st.expander("ℹ️ Why are these missing suggestions?", expanded=False):
            st.markdown("""
**These entities need suggestions but the system couldn't generate any. Common causes:**

1. **Blurry/Low-quality faces** - All faces in the cluster were marked as too blurry for reliable embedding.
   - *Why:* Low-quality faces produce unreliable embeddings that could cause incorrect matches.
   - *Fix:* Re-run detection at a different frame stride, or manually assign via Faces Review.

2. **Missing cluster centroids** - Clusters don't have computed centroids for comparison.
   - *Fix:* Go to **Faces Review → Cluster Cleanup → Standard** to regenerate embeddings and centroids.

3. **No assigned clusters to compare against** - No clusters in this episode are linked to cast members yet.
   - *Fix:* Manually assign a few clusters first (via Faces Review), then refresh suggestions.

4. **Empty facebank** - Cast members don't have reference face embeddings.
   - *Fix:* Add reference faces via **Show Management → Cast → Add Reference Faces**.

**Workaround:** Click "View" to open in Faces Review and manually assign to a cast member.
            """)

        # Render persons without suggestions first (unnamed persons)
        if no_suggestion_persons:
            st.caption(f"**Unnamed Persons** ({len(no_suggestion_persons)}) — need cast assignment")
            for person_entry in no_suggestion_persons:
                render_person_row(person_entry)

        # Render clusters without suggestions
        if no_suggestion_clusters:
            if no_suggestion_persons:
                st.markdown("---")
            st.caption(f"**Standalone Clusters** ({len(no_suggestion_clusters)})")
            for idx, entry in enumerate(no_suggestion_clusters):
                render_suggestion_row(entry, idx)

# Clear all dismissed button at bottom
if dismissed:
    st.markdown("---")
    if st.button("Clear All Dismissed", key="clear_dismissed_bottom", use_container_width=True):
        _clear_dismissed_via_api()  # Persist to disk
        st.session_state[dismissed_key] = set()
        st.rerun()

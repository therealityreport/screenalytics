from __future__ import annotations

import datetime
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from concurrent.futures import ThreadPoolExecutor
import math
import numpy as np

import requests
import streamlit as st

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except Exception:  # pragma: no cover - fallback for test shims
    add_script_run_ctx = None  # type: ignore[assignment]
    get_script_run_ctx = lambda: None  # type: ignore[assignment]

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402
from similarity_badges import (  # noqa: E402
    SimilarityType,
    render_similarity_badge,
    get_similarity_key_data,
    SIMILARITY_COLORS,
    TRACK_SORT_OPTIONS,
    CLUSTER_SORT_OPTIONS,
    UNASSIGNED_CLUSTER_SORT_OPTIONS,
    PERSON_SORT_OPTIONS,
    CAST_TRACKS_SORT_OPTIONS,
    sort_tracks,
    sort_clusters,
    sort_people,
    # Quality indicators (Feature 10)
    QualityIndicator,
    render_quality_indicator,
    render_cluster_quality_badges,
    get_cluster_quality_indicators,
)
from track_frame_utils import (  # noqa: E402
    best_track_frame_idx,
    coerce_int,
    quality_score,
    scope_track_frames,
    track_faces_debug,
)
import session_manager  # noqa: E402 - Job state management for Celery


# View name mapping for headers and URLs
VIEW_NAMES = {
    "people": ("🎬 Cast Members", "cast"),
    "person_clusters": ("👤 Person View", "person"),
    "cast_tracks": ("🎭 Cast Tracks View", "cast-tracks"),
    "cluster_tracks": ("📦 Cluster View", "cluster"),
    "track": ("🖼️ Frames View", "frames"),
}


def _render_view_header(view_state: str) -> None:
    """Render the current view header at the top of the page."""
    display_name, _ = VIEW_NAMES.get(view_state, ("Unknown View", "unknown"))
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(33, 150, 243, 0.25), rgba(156, 39, 176, 0.25));
            border: 1px solid rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            padding: 12px 20px;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 12px;
        ">
            <span style="font-size: 24px; font-weight: 600; color: #000;">{display_name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


MOCKED_STREAMLIT = "unittest.mock" in type(st).__module__
if MOCKED_STREAMLIT:
    class _Ctx:
        def __enter__(self):  # noqa: D401
            return st
        def __exit__(self, exc_type, exc, tb):  # noqa: D401
            return False

    st.session_state = {}
    st.sidebar = st
    st.title = lambda *a, **k: None
    st.expander = lambda *a, **k: _Ctx()
    st.columns = lambda n=1, *a, **k: tuple(st for _ in range(int(n) if n else 0))
    st.table = st.warning = st.error = st.info = st.success = st.markdown = st.caption = st.text = (
        lambda *a, **k: None
    )
    st.selectbox = lambda label, options, **k: options[0] if options else None
    st.slider = lambda *a, **k: 1
    st.checkbox = lambda *a, **k: False
    st.button = lambda *a, **k: False
    st.text_input = lambda *a, **k: ""
    st.progress = lambda *a, **k: type("P", (), {"progress": lambda self, x: None, "empty": lambda self: None})()
    st.empty = lambda *a, **k: type("E", (), {"text": lambda self, msg=None: None, "error": lambda self, msg=None: None})()
    st.toast = lambda *a, **k: None
    st.container = lambda *a, **k: _Ctx()
    st.stop = lambda *a, **k: None
    st.cache_data = lambda *a, **k: (lambda func: func)
    st.query_params = {}
    helpers.api_get = lambda *a, **k: {}
    helpers.api_post = lambda *a, **k: {}
    helpers.api_delete = lambda *a, **k: {}
    helpers.get_ep_id = lambda: "test-ep"
    helpers.detector_is_face_only = lambda ep_id, detect_status=None: True

cfg = helpers.init_page("Faces & Tracks")
st.title("Faces & Tracks Review")

# Global episode selector in sidebar to ensure ep_id is set for this page
sidebar_ep_id = None
try:
    sidebar_ep_id = helpers.render_sidebar_episode_selector()
except Exception:
    # Sidebar selector is best-effort; fallback to existing session/query ep_id
    sidebar_ep_id = None

# Similarity Scores Color Key (native Streamlit layout to avoid raw HTML)
with st.expander("📊 Similarity Scores Guide", expanded=False):
    st.markdown("### Similarity Types")

    def _render_similarity_card(color: str, title: str, description: str, details: List[str]) -> None:
        st.markdown(
            f"""
            <div style="
                display: flex;
                align-items: flex-start;
                gap: 10px;
                margin-bottom: 14px;
                background: rgba(255,255,255,0.03);
                border-radius: 8px;
                padding: 8px 12px;
            ">
                <div style="
                    width: 20px;
                    height: 20px;
                    border-radius: 50%;
                    background: {color};
                    flex-shrink: 0;
                    margin-top: 4px;
                    border: 1px solid rgba(255,255,255,0.15);
                "></div>
                <div>
                    <strong>{title}</strong><br/>
                    <span style="opacity:0.8;">{description}</span>
                    <ul style="margin:6px 0 0 16px; padding:0; opacity:0.7;">
                        {''.join(f'<li>{item}</li>' for item in details)}
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Row 1: Identity Similarity (Blue) and Cast Similarity (Purple)
    try:
        col1, col2 = st.columns(2)
    except Exception:
        col1 = col2 = st
    with col1:
        _render_similarity_card(
            SIMILARITY_COLORS[SimilarityType.IDENTITY].strong,
            "Identity Similarity",
            "How similar clusters are for AUTO-GENERATED PEOPLE.",
            ["≥ 75%: Strong match", "60–74%: Good match", "< 60%: Weak match"],
        )
    with col2:
        _render_similarity_card(
            SIMILARITY_COLORS[SimilarityType.CAST].strong,
            "Cast Similarity",
            "How similar clusters are for CAST MEMBERS (facebank).",
            ["≥ 68%: Auto-assigns to cast", "50–67%: Requires review", "< 50%: Weak match"],
        )

    # Row 2: Track Similarity (Orange) and Frame Similarity (Light Orange)
    try:
        col3, col4 = st.columns(2)
    except Exception:
        col3 = col4 = st
    with col3:
        _render_similarity_card(
            SIMILARITY_COLORS[SimilarityType.TRACK].strong,
            "Track Similarity",
            "How similar FRAMES within a TRACK are to each other.",
            ["≥ 85%: Strong consistency", "70–84%: Good consistency", "< 70%: Weak consistency"],
        )
    with col4:
        _render_similarity_card(
            SIMILARITY_COLORS[SimilarityType.FRAME].strong,
            "Frame Similarity",
            "How similar a specific frame is to rest of frames in track.",
            ["≥ 80%: Strong match", "65–79%: Good match", "< 65%: Potential outlier"],
        )

    # Row 3: Cluster Similarity (Green) and Cast Track Score (Teal)
    try:
        col5, col6 = st.columns(2)
    except Exception:
        col5 = col6 = st
    with col5:
        _render_similarity_card(
            SIMILARITY_COLORS[SimilarityType.CLUSTER].strong,
            "Cluster Similarity",
            "How cohesive/similar all tracks in a cluster are.",
            ["≥ 80%: Tight cluster", "60–79%: Moderate cohesion", "< 60%: Loose cluster"],
        )
    with col6:
        _render_similarity_card(
            SIMILARITY_COLORS[SimilarityType.CAST_TRACK].strong,
            "Cast Track Score",
            "How similar a track is to other tracks assigned to same cast/identity.",
            ["≥ 70%: Strong match", "55–69%: Possible match", "< 55%: Weak match"],
        )

    # Row 4: Quality Score (Green - separate from similarity)
    try:
        col7, col8 = st.columns(2)
    except Exception:
        col7 = col8 = st
    with col7:
        _render_similarity_card(
            "#4CAF50",
            "Quality Score",
            "Detection confidence + sharpness + face area. Appears as `Q: XX%` badge.",
            ["≥ 85%: High quality", "60–84%: Medium quality", "< 60%: Low quality"],
        )
    with col8:
        st.empty()

    st.markdown("---")
    st.markdown("### Quality Indicators")
    quality_df = {
        "Badge": [
            "Q: 85%+",
            "Q: 60-84%",
            "Q: < 60%",
            "ID: 75%+",
            "ID: 60-74%",
            "ID: < 60%",
        ],
        "Meaning": [
            "High quality (sharp, complete face, good detection)",
            "Medium quality (acceptable for most uses)",
            "Low quality (partial face, blurry, or low confidence)",
            "Strong identity match to track",
            "Good identity match",
            "Weak identity match (may be wrong person)",
        ],
    }
    st.table(quality_df)

    st.markdown("### Frame Badges")
    st.markdown(
        """
        - **★ BEST QUALITY (green)** – Complete face, high quality, good ID match
        - **⚠ BEST AVAILABLE (orange)** – Partial/low-quality, best available frame
        - **Partial (orange pill)** – Edge-clipped or incomplete face

        📚 **Full guide:** `docs/similarity-scores-guide.md`
        """,
    )


# Inject thumbnail CSS
helpers.inject_thumb_css()

# Pagination/sampling thresholds - configurable via environment variables
MAX_TRACKS_PER_ROW = int(os.environ.get("SCREENALYTICS_MAX_TRACKS_PER_ROW", "6"))
TRACK_SAMPLE_LONG_THRESHOLD = int(os.environ.get("SCREENALYTICS_TRACK_SAMPLE_LONG_THRESHOLD", "10000"))
TRACK_SAMPLE_HIGH_THRESHOLD = int(os.environ.get("SCREENALYTICS_TRACK_SAMPLE_HIGH_THRESHOLD", "20000"))
TRACK_PAGE_BASE_SIZE = int(os.environ.get("SCREENALYTICS_TRACK_PAGE_BASE_SIZE", "50"))
TRACK_PAGE_MEDIUM_SIZE = int(os.environ.get("SCREENALYTICS_TRACK_PAGE_MEDIUM_SIZE", "75"))
TRACK_PAGE_MAX_SIZE = int(os.environ.get("SCREENALYTICS_TRACK_PAGE_MAX_SIZE", "100"))
TRACK_MEDIA_BATCH_LIMIT = int(os.environ.get("SCREENALYTICS_TRACK_MEDIA_BATCH_LIMIT", "12"))
_CAST_CAROUSEL_CACHE_KEY = "cast_carousel_cache"
_CAST_PEOPLE_CACHE_KEY = "cast_carousel_people_cache"
_TRACK_MEDIA_CACHE_KEY = "track_media_cache"
_TRACK_MEDIA_MAX_ENTRIES = 50  # Maximum cached tracks to prevent memory bloat


def _cast_carousel_cache() -> Dict[str, Any]:
    return st.session_state.setdefault(_CAST_CAROUSEL_CACHE_KEY, {})


def _cast_people_cache() -> Dict[str, Any]:
    return st.session_state.setdefault(_CAST_PEOPLE_CACHE_KEY, {})


def _track_media_cache() -> Dict[str, Dict[str, Any]]:
    return st.session_state.setdefault(_TRACK_MEDIA_CACHE_KEY, {})


def _evict_oldest_cache_entries(cache: Dict[str, Dict[str, Any]], max_entries: int) -> None:
    """Evict oldest entries when cache exceeds max size (LRU-style based on access time)."""
    if len(cache) <= max_entries:
        return
    # Sort by access_time (oldest first), remove excess
    entries_with_time = [
        (k, v.get("access_time", 0)) for k, v in cache.items()
    ]
    entries_with_time.sort(key=lambda x: x[1])
    to_remove = len(cache) - max_entries
    for key, _ in entries_with_time[:to_remove]:
        cache.pop(key, None)


def _track_media_state(ep_id: str, track_id: int, sample: int = 1) -> Dict[str, Any]:
    """Get or create cache state for track media, keyed by ep_id, track_id, AND sample rate."""
    import time
    cache = _track_media_cache()
    key = f"{ep_id}::{track_id}::s{sample}"
    if key not in cache:
        # Evict old entries before adding new one
        _evict_oldest_cache_entries(cache, _TRACK_MEDIA_MAX_ENTRIES - 1)
        cache[key] = {
            "items": [],
            "cursor": None,
            "initialized": False,
            "sample": sample,
            "batch_limit": TRACK_MEDIA_BATCH_LIMIT,
            "access_time": time.time(),
        }
    else:
        # Update access time for LRU tracking
        cache[key]["access_time"] = time.time()
    return cache[key]


def _reset_track_media_state(ep_id: str, track_id: int) -> None:
    """Clear all cache entries for this track (any sample rate)."""
    cache = _track_media_cache()
    # Remove all keys matching this ep_id and track_id (any sample rate)
    keys_to_remove = [k for k in cache if k.startswith(f"{ep_id}::{track_id}::")]
    for k in keys_to_remove:
        cache.pop(k, None)


def _suggest_track_sample(frame_count: int | None) -> int:
    frames = int(frame_count or 0)
    if frames >= TRACK_SAMPLE_HIGH_THRESHOLD:
        return 3
    if frames >= TRACK_SAMPLE_LONG_THRESHOLD:
        return 2
    return 1


def _recommended_page_size(sample: int, total_sampled: int | None) -> int:
    total = int(total_sampled or 0)
    if sample >= 3 or total >= TRACK_SAMPLE_HIGH_THRESHOLD:
        return TRACK_PAGE_MAX_SIZE
    if sample >= 2 or total >= TRACK_SAMPLE_LONG_THRESHOLD:
        return TRACK_PAGE_MEDIUM_SIZE
    return TRACK_PAGE_BASE_SIZE


def _show_local_fallback_banner(api_response: Dict[str, Any] | None) -> None:
    """Show a warning banner if local files are being used instead of S3."""
    if not api_response:
        return
    local_fallbacks = api_response.get("_local_fallbacks", [])
    if local_fallbacks:
        with st.container():
            st.warning(
                f"⚠️ **USING LOCAL FILES** - The following media are being served locally instead of from S3. "
                f"This may cause stale/incorrect thumbnails:"
            )
            # Show first 5 paths, then count of remaining
            shown_paths = local_fallbacks[:5]
            for path in shown_paths:
                st.caption(f"• `{path}`")
            remaining = len(local_fallbacks) - len(shown_paths)
            if remaining > 0:
                st.caption(f"• ... and {remaining} more local file(s)")


def _extract_s3_key_from_url(url: str) -> str:
    """Extract S3 key from presigned URL, handling multiple URL formats.

    Handles formats like:
    - https://bucket.s3.amazonaws.com/key?presign_params
    - https://bucket.s3.region.amazonaws.com/key?presign_params
    - https://s3.region.amazonaws.com/bucket/key?presign_params
    - Plain S3 keys (passthrough)
    - Local file paths (passthrough)

    Returns the original URL if extraction fails, allowing fallback handling.
    """
    if not url or not isinstance(url, str):
        return url or ""

    # Already a plain key or local path - return as-is
    if not url.startswith(("http://", "https://")):
        return url

    # Strip query string first
    base_url = url.split("?")[0] if "?" in url else url

    try:
        # Format 1: bucket.s3.amazonaws.com/key or bucket.s3.region.amazonaws.com/key
        if ".s3." in base_url and "amazonaws.com/" in base_url:
            # Find the path after amazonaws.com/
            idx = base_url.find("amazonaws.com/")
            if idx != -1:
                s3_key = base_url[idx + len("amazonaws.com/"):]
                if s3_key:
                    return s3_key

        # Format 2: s3.amazonaws.com/bucket/key (virtual-host style)
        if base_url.startswith("https://s3.") and "amazonaws.com/" in base_url:
            idx = base_url.find("amazonaws.com/")
            if idx != -1:
                # Everything after amazonaws.com/ is bucket/key
                bucket_and_key = base_url[idx + len("amazonaws.com/"):]
                if "/" in bucket_and_key:
                    # Skip bucket name, return just the key
                    return bucket_and_key.split("/", 1)[1]
                return bucket_and_key
    except Exception:
        # On any parsing error, return original URL for fallback handling
        pass

    # Return original URL if no S3 pattern matched
    return url


def _safe_api_get(path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    try:
        return helpers.api_get(path, params=params)
    except requests.RequestException as exc:
        base = (cfg or {}).get("api_base") if isinstance(cfg, dict) else ""
        try:
            st.error(helpers.describe_error(f"{base}{path}", exc))
        except Exception as display_exc:
            # Fallback if error display fails
            st.error(f"API error: {path} - {exc}")
        return None


@st.cache_data(ttl=15)  # Reduced from 60s - frequently mutated by assignments
def _fetch_identities_cached(ep_id: str) -> Dict[str, Any] | None:
    return _safe_api_get(f"/episodes/{ep_id}/identities")


@st.cache_data(ttl=15)  # Reduced from 60s - frequently mutated by assignments
def _fetch_people_cached(show_slug: str | None) -> Dict[str, Any] | None:
    if not show_slug:
        return None
    return _safe_api_get(f"/shows/{show_slug}/people")


@st.cache_data(ttl=15)  # Reduced from 60s - frequently mutated by assignments
def _fetch_cast_cached(show_slug: str | None, season_label: str | None = None) -> Dict[str, Any] | None:
    if not show_slug:
        return None
    params: Dict[str, Any] = {"include_featured": "1"}
    if season_label:
        params["season"] = season_label
    return _safe_api_get(f"/shows/{show_slug}/cast", params=params)


@st.cache_data(ttl=60)
def _fetch_track_detail_cached(ep_id: str, track_id: int) -> Dict[str, Any] | None:
    return _safe_api_get(f"/episodes/{ep_id}/tracks/{track_id}")


@st.cache_data(ttl=60)
def _fetch_cluster_track_reps_cached(ep_id: str, cluster_id: str) -> Dict[str, Any] | None:
    """Cached fetch of cluster track representatives for faster navigation."""
    return _safe_api_get(f"/episodes/{ep_id}/clusters/{cluster_id}/track_reps")


def _get_best_crop_from_clusters(
    ep_id: str,
    cluster_ids: List[str],
) -> Optional[str]:
    """Find the best quality crop URL from all tracks across the given clusters.

    Iterates through clusters and their tracks to find the single best-quality
    crop to use as a featured thumbnail when no seeded thumbnail is available.

    Args:
        ep_id: Episode identifier
        cluster_ids: List of cluster IDs to search through

    Returns:
        URL of the best quality crop, or None if no valid crop found
    """
    if not cluster_ids:
        return None

    best_crop_url: Optional[str] = None
    best_quality_score: float = -1.0

    for cluster_id in cluster_ids:
        track_reps = _fetch_cluster_track_reps_cached(ep_id, cluster_id)
        if not track_reps:
            continue

        tracks = track_reps.get("tracks", [])
        for track in tracks:
            # Get quality score - nested in quality dict
            quality = track.get("quality", {})
            if isinstance(quality, dict):
                score = quality.get("score", 0.0) or 0.0
            else:
                score = 0.0

            crop_url = track.get("crop_url")
            if not crop_url:
                continue

            # Track with higher quality score wins
            if score > best_quality_score:
                best_quality_score = score
                best_crop_url = crop_url

    return best_crop_url


def _invalidate_assignment_caches() -> None:
    """Clear all caches affected by assignment operations.

    Call this immediately after any assignment/merge/create person operation
    to ensure the UI shows fresh data.
    """
    _fetch_identities_cached.clear()
    _fetch_people_cached.clear()
    _fetch_cast_cached.clear()
    _fetch_cluster_track_reps_cached.clear()
    # Clear session state caches if they exist
    for key in list(st.session_state.keys()):
        if key.startswith("cast_carousel_cache") or key.startswith("_thumb_result_cache"):
            del st.session_state[key]


def _persist_and_refresh_cast_suggestions(
    ep_id: str,
) -> tuple[Dict[str, List[Dict[str, Any]]], Optional[int]]:
    """Persist assignments and recompute cast suggestions from latest embeddings."""
    _invalidate_assignment_caches()
    st.session_state.pop(f"cast_suggestions:{ep_id}", None)
    st.session_state.pop(f"dismissed_suggestions:{ep_id}", None)

    saved_count: Optional[int] = None
    save_resp = _api_post(f"/episodes/{ep_id}/save_assignments", {})
    if isinstance(save_resp, dict):
        raw_saved = save_resp.get("saved_count")
        if isinstance(raw_saved, int):
            saved_count = raw_saved

    cache_buster = int(time.time() * 1000)
    suggestions_resp = _safe_api_get(f"/episodes/{ep_id}/cast_suggestions", params={"_t": cache_buster})
    suggestions_map: Dict[str, List[Dict[str, Any]]] = {}
    if suggestions_resp:
        for entry in suggestions_resp.get("suggestions", []):
            cid = entry.get("cluster_id")
            if not cid:
                continue
            suggestions_map[cid] = entry.get("cast_suggestions", []) or []

    if suggestions_map:
        st.session_state[f"cast_suggestions:{ep_id}"] = suggestions_map

    return suggestions_map, saved_count


def _fetch_tracks_meta(ep_id: str, track_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Batch fetch track metadata when possible; fallback to cached per-track fetch."""
    unique_ids = sorted({tid for tid in track_ids if tid is not None})
    if not unique_ids:
        return {}

    # Try batch endpoint first (silently falls back to per-track fetches if unavailable)
    # Use direct requests call to avoid _safe_api_get showing error for optional endpoint
    try:
        api_base = st.session_state.get("api_base") or os.environ.get("API_BASE", "http://127.0.0.1:8000")
        ids_param = ",".join(str(tid) for tid in unique_ids)
        resp = requests.get(
            f"{api_base}/episodes/{ep_id}/tracks",
            params={"ids": ids_param, "fields": "id,track_id,faces_count,frames"},
            timeout=30,
        )
        if resp.status_code == 200:
            batch_resp = resp.json()
            if batch_resp and isinstance(batch_resp, dict):
                items = batch_resp.get("tracks") or batch_resp.get("items") or []
                meta: Dict[int, Dict[str, Any]] = {}
                for item in items:
                    tid = coerce_int(item.get("track_id") or item.get("id"))
                    if tid is not None:
                        meta[int(tid)] = item
                if meta:
                    return meta
    except Exception:
        # Batch endpoint not available or failed - fall back to per-track fetches
        pass

    # Fallback: parallel per-track fetches (cached) to avoid sequential N+1
    meta: Dict[int, Dict[str, Any]] = {}
    max_workers = min(8, len(unique_ids)) or 1
    ctx = get_script_run_ctx() if get_script_run_ctx else None

    def _submit(pool: ThreadPoolExecutor, func, *args):
        if ctx and add_script_run_ctx:
            def _wrapped():
                add_script_run_ctx(threading.current_thread(), ctx)
                return func(*args)
            return pool.submit(_wrapped)
        return pool.submit(func, *args)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {_submit(pool, _fetch_track_detail_cached, ep_id, tid): tid for tid in unique_ids}
        for future, tid in future_map.items():
            try:
                data = future.result()
                if data:
                    meta[tid] = data
            except Exception:
                continue
    return meta


def _prefetch_adjacent_clusters(
    ep_id: str,
    current_cluster_id: str,
    all_cluster_ids: List[str],
) -> None:
    """Warm cache for adjacent clusters in background.

    When viewing a cluster, prefetch the previous and next clusters' track_reps
    data so navigation feels instant.
    """
    if not all_cluster_ids or current_cluster_id not in all_cluster_ids:
        return

    try:
        idx = all_cluster_ids.index(current_cluster_id)
    except ValueError:
        return

    adjacent_ids: List[str] = []
    if idx > 0:
        adjacent_ids.append(all_cluster_ids[idx - 1])
    if idx < len(all_cluster_ids) - 1:
        adjacent_ids.append(all_cluster_ids[idx + 1])

    if not adjacent_ids:
        return

    # Fire-and-forget prefetch using ThreadPoolExecutor (warms st.cache_data)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for cluster_id in adjacent_ids:
            pool.submit(_fetch_cluster_track_reps_cached, ep_id, cluster_id)


def _api_post(path: str, payload: Dict[str, Any] | None = None, *, timeout: float = 60.0) -> Dict[str, Any] | None:
    try:
        return helpers.api_post(path, payload or {}, timeout=timeout)
    except requests.RequestException as exc:
        base = (cfg or {}).get("api_base") if isinstance(cfg, dict) else ""
        try:
            st.error(helpers.describe_error(f"{base}{path}", exc))
        except Exception:
            pass
        return None


def _api_delete(path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    base = st.session_state.get("api_base")
    if not base:
        st.error("API base URL missing; re-run init_page().")
        return None
    try:
        resp = requests.delete(f"{base}{path}", json=payload or {}, timeout=60)
        resp.raise_for_status()
        return resp.json() if resp.content else {}
    except requests.RequestException as exc:
        st.error(helpers.describe_error(f"{base}{path}", exc))
        return None


def _episode_show_slug(ep_id: str) -> str | None:
    parsed = helpers.parse_ep_id(ep_id) or {}
    show = parsed.get("show")
    if not show:
        return None
    return str(show).lower()


def _roster_cache() -> Dict[str, List[str]]:
    return st.session_state.setdefault("show_roster_names", {})


def _fetch_roster_names(show: str | None) -> List[str]:
    """Fetch all cast names from both roster.json and cast.json (facebank)."""
    if not show:
        return []
    cache = _roster_cache()
    if show in cache:
        return cache[show]

    # Fetch roster names (from roster.json)
    roster_payload = _safe_api_get(f"/shows/{show}/cast_names")
    roster_names = roster_payload.get("names", []) if roster_payload else []

    # Fetch cast member names (from cast.json / facebank)
    cast_payload = _safe_api_get(f"/shows/{show}/cast")
    cast_members = cast_payload.get("cast", []) if cast_payload else []
    cast_names = [m.get("name") for m in cast_members if m.get("name")]

    # Merge and deduplicate (case-insensitive, preserving original case)
    seen_lower = set()
    combined_names = []
    for name in roster_names + cast_names:
        name_lower = name.lower()
        if name_lower not in seen_lower:
            seen_lower.add(name_lower)
            combined_names.append(name)

    # Sort alphabetically for easier selection
    combined_names.sort()

    cache[show] = combined_names
    return combined_names


def _refresh_roster_names(show: str | None) -> None:
    if not show:
        return
    _roster_cache().pop(show, None)


def _name_choice_widget(
    *,
    label: str,
    key_prefix: str,
    roster_names: List[str],
    current_name: str = "",
    text_label: str = "New name",
) -> str:
    names = roster_names[:]
    if current_name and current_name not in names:
        names = [current_name, *names]
    options = ["<Add new name…>", *names]
    default_idx = options.index(current_name) if current_name in names else 0
    choice = st.selectbox(label, options, index=default_idx, key=f"{key_prefix}_select")
    if choice == options[0]:
        return st.text_input(text_label, value=current_name, key=f"{key_prefix}_input").strip()
    return choice.strip()


def _save_identity_name(ep_id: str, identity_id: str, name: str, show: str | None) -> None:
    cleaned = name.strip()
    if not cleaned:
        st.warning("Provide a non-empty name before saving.")
        return
    payload: Dict[str, Any] = {"name": cleaned}
    if show:
        payload["show"] = show
    resp = _api_post(f"/episodes/{ep_id}/identities/{identity_id}/name", payload)
    if resp is None:
        return
    st.toast(f"Saved name '{cleaned}' for {identity_id}")
    _refresh_roster_names(show)
    st.rerun()


def _assign_track_name(ep_id: str, track_id: int, name: str, show: str | None, cast_id: str | None = None) -> None:
    cleaned = name.strip()
    if not cleaned:
        st.warning("Provide a non-empty name before saving.")
        return
    payload: Dict[str, Any] = {"name": cleaned}
    if show:
        payload["show"] = show
    if cast_id:
        payload["cast_id"] = cast_id
    resp = _api_post(f"/episodes/{ep_id}/tracks/{track_id}/name", payload)
    if resp is None:
        return
    split_flag = resp.get("split")
    suffix = " (moved into a new cluster)" if split_flag else ""
    st.toast(f"Saved name '{cleaned}' for track {track_id}{suffix}")
    new_identity_id = resp.get("identity_id")
    if new_identity_id:
        st.session_state["selected_identity"] = new_identity_id
    _refresh_roster_names(show)
    st.rerun()


def _bulk_assign_tracks(
    ep_id: str, track_ids: List[int], name: str, show: str | None, cast_id: str | None = None,
    identity_id: str | None = None,
) -> None:
    """Bulk assign multiple tracks to a cast member."""
    cleaned = name.strip()
    if not cleaned:
        st.warning("Provide a non-empty name before saving.")
        return
    if not track_ids:
        st.warning("No tracks selected.")
        return
    payload: Dict[str, Any] = {"track_ids": track_ids, "name": cleaned}
    if show:
        payload["show"] = show
    if cast_id:
        payload["cast_id"] = cast_id
    with st.spinner(f"Assigning {len(track_ids)} track(s) to '{cleaned}'..."):
        resp = _api_post(f"/episodes/{ep_id}/tracks/bulk_assign", payload)
    if resp is None:
        return
    _invalidate_assignment_caches()
    assigned = resp.get("assigned", 0)
    failed = resp.get("failed", 0)
    if assigned > 0:
        st.toast(f"Assigned {assigned} track(s) to '{cleaned}'")
    if failed > 0:
        st.warning(f"{failed} track(s) failed to assign. Check logs.")
    # Clear bulk selection state for all identity keys
    for key in list(st.session_state.keys()):
        if key.startswith("bulk_track_sel::"):
            st.session_state[key] = set()
    _refresh_roster_names(show)
    st.rerun()


def _create_and_assign_to_new_cast(
    ep_id: str, track_id: int, cast_name: str, show: str | None
) -> None:
    """Create a new cast member on the show's roster and assign the track to them."""
    if not show:
        st.error("Show slug is required to create a new cast member.")
        return
    # Step 1: Create the cast member via API
    cast_resp = _api_post(f"/shows/{show}/cast", {"name": cast_name})
    if cast_resp is None:
        st.error(f"Failed to create cast member '{cast_name}'.")
        return
    new_cast_id = cast_resp.get("cast_id")
    if not new_cast_id:
        st.error("Cast member created but no cast_id returned.")
        return
    st.toast(f"Created cast member '{cast_name}'")
    # Step 2: Assign the track to the new cast member (creates new cluster)
    _assign_track_name(ep_id, track_id, cast_name, show, new_cast_id)


def _move_frames_api(
    ep_id: str,
    track_id: int,
    frame_ids: List[int],
    target_identity_id: str | None,
    new_identity_name: str | None,
    show: str | None,
) -> None:
    payload: Dict[str, Any] = {"frame_ids": frame_ids}
    if target_identity_id:
        payload["target_identity_id"] = target_identity_id
    if new_identity_name:
        payload["new_identity_name"] = new_identity_name
    if show:
        payload["show_id"] = show
    resp = _api_post(f"/episodes/{ep_id}/tracks/{track_id}/frames/move", payload)
    if resp is None:
        st.error("Failed to move frames - API returned no response")
        return
    moved = resp.get("moved") or len(frame_ids)
    name = resp.get("target_name") or resp.get("target_identity_id") or target_identity_id or "target identity"
    st.toast(f"Moved {moved} frame(s) to {name}")
    _refresh_roster_names(show)
    # Only clear selection after successful move
    st.session_state.setdefault("track_frame_selection", {}).pop(track_id, None)
    st.rerun()


def _delete_frames_api(ep_id: str, track_id: int, frame_ids: List[int], delete_assets: bool = True) -> None:
    payload = {"frame_ids": frame_ids, "delete_assets": delete_assets}
    resp = _api_delete(f"/episodes/{ep_id}/tracks/{track_id}/frames", payload)
    if resp is None:
        st.error("Failed to delete frames - API returned no response")
        return
    deleted = resp.get("deleted") or len(frame_ids)
    st.toast(f"Deleted {deleted} frame(s)")
    # Only clear selection after successful delete
    st.session_state.setdefault("track_frame_selection", {}).pop(track_id, None)
    st.rerun()


def _track_episode_from_s3(item: Dict[str, Any]) -> None:
    ep_id = str(item.get("ep_id", "")).lower()
    ep_meta = helpers.parse_ep_id(ep_id) or {}
    show_slug = item.get("show") or ep_meta.get("show")
    season = item.get("season") or ep_meta.get("season")
    episode = item.get("episode") or ep_meta.get("episode")
    if not (ep_id and show_slug and season is not None and episode is not None):
        st.error("Unable to derive show/season/episode from the selected S3 key.")
        return
    payload = {
        "ep_id": ep_id,
        "show_slug": str(show_slug).lower(),
        "season": int(season),
        "episode": int(episode),
    }
    resp = _api_post("/episodes/upsert_by_id", payload)
    if resp is None:
        return
    st.success(f"Tracked episode {resp['ep_id']}.")
    helpers.set_ep_id(resp["ep_id"])


def _identity_name_controls(
    *,
    ep_id: str,
    identity: Dict[str, Any],
    show_slug: str | None,
    roster_names: List[str],
    prefix: str,
) -> None:
    if not show_slug:
        st.info("Show slug missing; unable to assign roster names.")
        return
    current_name = identity.get("name") or ""
    resolved = _name_choice_widget(
        label="Assign name",
        key_prefix=f"{prefix}_{identity['identity_id']}",
        roster_names=roster_names,
        current_name=current_name,
    )
    disabled = not resolved or resolved == current_name
    if st.button("Save name", key=f"{prefix}_save_{identity['identity_id']}", disabled=disabled):
        _save_identity_name(ep_id, identity["identity_id"], resolved, show_slug)


def _initialize_state(ep_id: str) -> None:
    if st.session_state.get("facebank_ep") != ep_id:
        old_ep_id = st.session_state.get("facebank_ep")
        st.session_state["facebank_ep"] = ep_id
        st.session_state["facebank_view"] = "people"
        st.session_state["selected_person"] = None
        st.session_state["selected_identity"] = None
        st.session_state["selected_track"] = None
        st.session_state.pop("facebank_query_applied", None)

        # Clear stale cast suggestions from previous episode
        if old_ep_id:
            st.session_state.pop(f"cast_suggestions:{old_ep_id}", None)

        # Clear pagination keys from previous episode (track_page_{old_ep_id}_* keys)
        if old_ep_id:
            prefix = f"track_page_{old_ep_id}_"
            keys_to_remove = [k for k in st.session_state.keys() if k.startswith(prefix)]
            for k in keys_to_remove:
                st.session_state.pop(k, None)


def _hydrate_view_from_query(ep_id: str) -> None:
    """Allow deep links to jump directly into person/cluster/track views."""
    if st.session_state.get("facebank_query_applied") == ep_id:
        return

    raw_params = getattr(st, "query_params", {}) or {}
    params: Dict[str, List[str]] = {}
    for key, value in raw_params.items():
        if value is None:
            continue
        if isinstance(value, list):
            params[key] = value
        else:
            params[key] = [value]
    if not params:
        return

    def _single(key: str) -> str | None:
        values = params.get(key)
        return values[0] if values else None

    target_view = _single("view")
    person_id = _single("person_id") or _single("person")
    identity_id = _single("identity_id") or _single("cluster_id") or _single("cluster")
    track_id = coerce_int(_single("track_id")) or coerce_int(_single("track"))

    # Normalize view
    if target_view not in {"people", "person_clusters", "cluster_tracks", "track"}:
        target_view = None

    if person_id or identity_id or track_id is not None or target_view:
        # Prefer explicit view; otherwise infer from provided ids
        inferred_view = target_view
        if not inferred_view:
            if track_id is not None:
                inferred_view = "track"
            elif identity_id:
                inferred_view = "cluster_tracks"
            elif person_id:
                inferred_view = "person_clusters"
        _set_view(
            inferred_view or "people",
            person_id=person_id,
            identity_id=identity_id,
            track_id=track_id,
        )
        st.session_state["facebank_query_applied"] = ep_id


def _set_view(
    view: str,
    *,
    person_id: str | None = None,
    identity_id: str | None = None,
    track_id: int | None = None,
) -> None:
    """Update view state and URL query params. Streamlit will auto-rerun after callback completes."""
    st.session_state["facebank_view"] = view
    st.session_state["selected_person"] = person_id
    st.session_state["selected_identity"] = identity_id
    st.session_state["selected_track"] = track_id

    # Auto-load frames when navigating to track view
    if view == "track" and track_id is not None:
        ep_id = st.session_state.get("facebank_ep")
        if ep_id:
            frames_load_key = f"load_frames_{ep_id}_{track_id}"
            st.session_state[frames_load_key] = True

    # Update URL with view name for easy reference
    _, view_slug = VIEW_NAMES.get(view, ("Unknown", "unknown"))
    try:
        st.query_params["view"] = view_slug
        if person_id:
            st.query_params["person"] = person_id
        elif "person" in st.query_params:
            del st.query_params["person"]
        if identity_id:
            st.query_params["cluster"] = identity_id
        elif "cluster" in st.query_params:
            del st.query_params["cluster"]
        if track_id is not None:
            st.query_params["track"] = str(track_id)
        elif "track" in st.query_params:
            del st.query_params["track"]
    except Exception:
        pass  # Ignore query param errors in mocked environments


def _episode_header(ep_id: str) -> Dict[str, Any] | None:
    detail = _safe_api_get(f"/episodes/{ep_id}")
    if not detail:
        return None
    cols = st.columns([2, 1, 1])
    with cols[0]:
        st.markdown(
            f"**Episode:** `{ep_id}` · Show `{detail['show_slug']}` · S{detail['season_number']:02d}E{detail['episode_number']:02d}"
        )
        st.caption(f"Detector: {helpers.tracks_detector_label(ep_id)}")
    with cols[1]:
        st.caption(f"S3 v2: `{detail['s3']['v2_key']}`")
    with cols[2]:
        st.caption(f"Local video: {'✅' if detail['local']['exists'] else '❌'}")
    if not detail["local"]["exists"]:
        if st.button("Mirror from S3", key="facebank_mirror"):
            if _api_post(f"/episodes/{ep_id}/mirror"):
                st.success("Mirror complete.")
                st.rerun()
    action_cols = st.columns([1, 1, 1])
    action_cols[0].button(
        "Open Episode Detail",
        key="facebank_open_detail",
        on_click=lambda: helpers.try_switch_page("pages/2_Episode_Detail.py"),
    )
    with action_cols[1]:
        # Enhancement #4: Selective Cleanup Actions + Enhancement #3: Preview
        with st.popover("🧹 Cluster Cleanup", help="Select which cleanup actions to run"):
            # Enhancement #3: Show cleanup preview first
            preview_resp = _safe_api_get(f"/episodes/{ep_id}/cleanup_preview")
            if preview_resp and preview_resp.get("preview"):
                preview = preview_resp["preview"]
                warning_level = preview.get("warning_level", "low")
                warnings = preview.get("warnings", [])

                # Show warning banner based on level
                if warning_level == "high":
                    st.error("⚠️ **High Impact Warning**")
                elif warning_level == "medium":
                    st.warning("⚡ **Medium Impact Warning**")

                # Show preview stats
                st.caption(
                    f"📊 {preview.get('total_clusters', 0)} clusters "
                    f"({preview.get('assigned_clusters', 0)} assigned, "
                    f"{preview.get('unassigned_clusters', 0)} unassigned)"
                )
                if preview.get("manual_assignments_count", 0) > 0:
                    st.caption(f"🔒 {preview.get('manual_assignments_count')} manually assigned cluster(s)")
                if preview.get("potential_merges", 0) > 0:
                    st.caption(f"🔄 {preview.get('potential_merges')} potential merge(s)")

                # Show warnings
                for warning in warnings:
                    st.info(warning)

                st.markdown("---")

            st.markdown("**Select cleanup actions:**")

            # Define actions with risk levels
            cleanup_actions = {
                "split_tracks": {
                    "label": "Fix tracking issues (split_tracks)",
                    "help": "Use when: A track contains multiple different people (identity switch mid-track). Splits incorrectly merged tracks. Low risk - usually beneficial.",
                    "default": True,
                    "risk": "low",
                },
                "reembed": {
                    "label": "Regenerate embeddings (reembed)",
                    "help": "Use when: Face quality has changed or embeddings seem outdated. Recalculates face embeddings. Low risk - just regenerates vectors.",
                    "default": True,
                    "risk": "low",
                },
                "recluster": {
                    "label": "Re-cluster faces (recluster)",
                    "help": "Use when: Starting fresh after major changes. ⚠️ HIGH RISK: Regenerates identities.json, may undo manual splits and assignments.",
                    "default": False,
                    "risk": "high",
                },
                "group_clusters": {
                    "label": "Auto-group clusters (group_clusters)",
                    "help": "Use when: You have unassigned clusters that need to be matched to people. Groups similar clusters into people. Medium risk - respects seed matching.",
                    "default": True,
                    "risk": "medium",
                },
            }

            selected_actions = []
            for action_key, action_info in cleanup_actions.items():
                risk_badge = ""
                if action_info["risk"] == "high":
                    risk_badge = " ⚠️"
                elif action_info["risk"] == "medium":
                    risk_badge = " ⚡"

                checked = st.checkbox(
                    f"{action_info['label']}{risk_badge}",
                    value=action_info["default"],
                    key=f"cleanup_action_{action_key}",
                    help=action_info["help"],
                )
                if checked:
                    selected_actions.append(action_key)

            # Enhancement #7: Show backup/restore info
            backups_resp = _safe_api_get(f"/episodes/{ep_id}/backups")
            backups = backups_resp.get("backups", []) if backups_resp else []
            if backups:
                latest = backups[0].get("backup_id", "")
                st.caption(f"💾 Last backup: {latest[-20:] if len(latest) > 20 else latest}")
                if st.button("↩️ Undo Last Cleanup", key="restore_backup_btn", help="Restore to previous state"):
                    restore_resp = _api_post(f"/episodes/{ep_id}/restore/{latest}", {})
                    if restore_resp and restore_resp.get("files_restored", 0) > 0:
                        st.success("✓ Restored from backup!")
                        st.rerun()

            st.markdown("---")
            if st.button("Run Selected Cleanup", key="facebank_cleanup_button", type="primary"):
                if not selected_actions:
                    st.warning("No cleanup actions selected.")
                else:
                    # Enhancement #7: Auto-backup before cleanup
                    _api_post(f"/episodes/{ep_id}/backup", {})
                    payload = helpers.default_cleanup_payload(ep_id)
                    payload["actions"] = selected_actions
                    with st.spinner(f"Running cleanup ({', '.join(selected_actions)})…"):
                        summary, error_message = helpers.run_job_with_progress(
                            ep_id,
                            "/jobs/episode_cleanup_async",
                            payload,
                            requested_device=helpers.DEFAULT_DEVICE,
                            async_endpoint="/jobs/episode_cleanup_async",
                            requested_detector=helpers.DEFAULT_DETECTOR,
                            requested_tracker=helpers.DEFAULT_TRACKER,
                            use_async_only=True,
                        )
                    if error_message:
                        st.error(error_message)
                    else:
                        report = summary or {}
                        if isinstance(report.get("summary"), dict):
                            report = report["summary"]
                        details: List[str] = []
                        tb = helpers.coerce_int(report.get("tracks_before"))
                        ta = helpers.coerce_int(report.get("tracks_after"))
                        cbefore = helpers.coerce_int(report.get("clusters_before"))
                        cafter = helpers.coerce_int(report.get("clusters_after"))
                        faces_after = helpers.coerce_int(report.get("faces_after"))
                        if tb is not None and ta is not None:
                            details.append(f"tracks {helpers.format_count(tb) or tb} → {helpers.format_count(ta) or ta}")
                        if cbefore is not None and cafter is not None:
                            details.append(
                                f"clusters {helpers.format_count(cbefore) or cbefore} → {helpers.format_count(cafter) or cafter}"
                            )
    with action_cols[2]:
        # Enhancement #8: Auto-link option
        auto_link_enabled = st.checkbox(
            "🔗 Auto-assign",
            value=True,
            key="auto_link_checkbox",
            help="Automatically assign clusters to cast members when facebank similarity ≥85%",
        )
        # Row of action buttons
        btn_row = st.columns([1, 1, 1])
        with btn_row[0]:
            refresh_clicked = st.button(
                "🔄 Refresh Values",
                key="facebank_refresh_similarity_button",
                type="primary",
                help="Recompute similarity scores and refresh suggestions",
            )
        with btn_row[1]:
            if st.button(
                "💾 Save Progress",
                key="facebank_save_progress_top",
                help="Save all current assignments to disk",
            ):
                with st.spinner("Saving assignments and refreshing suggestions..."):
                    suggestions_map, saved_count = _persist_and_refresh_cast_suggestions(ep_id)
                if saved_count is not None:
                    st.toast(f"✅ Saved {saved_count} assignment(s)")
                elif suggestions_map:
                    st.toast("✅ Assignments saved")
                else:
                    st.error("Failed to save assignments - see logs for details")
        with btn_row[2]:
            if st.button(
                "💡 Smart Suggestions",
                key="open_smart_suggestions",
                help="Review and accept cast suggestions for unassigned clusters and auto-clustered people",
            ):
                success = False
                with st.spinner("Recomputing Smart Suggestions from latest assignments..."):
                    suggestions_map, saved_count = _persist_and_refresh_cast_suggestions(ep_id)
                    success = bool(suggestions_map) or saved_count is not None
                    if suggestions_map:
                        st.toast(f"Refreshed suggestions for {len(suggestions_map)} cluster(s)")
                    elif saved_count is not None:
                        st.toast("Assignments saved; no new suggestions found")
                    else:
                        st.error("Failed to refresh Smart Suggestions")
                if success:
                    helpers.try_switch_page("pages/3_Smart_Suggestions.py")

        # Recovery button row
        recovery_row = st.columns([1, 2])
        with recovery_row[0]:
            recover_clicked = st.button(
                "🔧 Recover Noise Tracks",
                key="recover_noise_tracks",
                help="Find adjacent frames for single-frame tracks and expand them (±8 frames, similarity ≥70%)",
            )

    # Progress area below the buttons
    refresh_progress_area = st.empty()
    recovery_progress_area = st.empty()

    # Handle recovery button click
    if recover_clicked:
        with recovery_progress_area.container():
            with st.spinner("Recovering single-frame tracks..."):
                resp = _api_post(f"/episodes/{ep_id}/recover_noise_tracks", {})
                if resp:
                    tracks_analyzed = resp.get("tracks_analyzed", 0)
                    tracks_expanded = resp.get("tracks_expanded", 0)
                    faces_merged = resp.get("faces_merged", 0)

                    if tracks_expanded > 0:
                        st.success(
                            f"Recovered {tracks_expanded} track(s) by merging {faces_merged} adjacent face(s). "
                            f"({tracks_analyzed} single-frame tracks analyzed)"
                        )
                        # Show details
                        for detail in resp.get("details", [])[:5]:
                            track_id = detail.get("track_id")
                            added = detail.get("added_frames", [])
                            st.write(f"  • Track {track_id}: +{len(added)} frame(s)")
                        st.rerun()
                    else:
                        st.info(
                            f"No recoverable tracks found. "
                            f"({tracks_analyzed} single-frame tracks analyzed, none had similar faces in adjacent frames)"
                        )
                else:
                    st.error("Failed to recover noise tracks. Check logs.")

    if refresh_clicked:
        with refresh_progress_area.container():
            with st.status("Refreshing similarity values...", expanded=True) as status:
                # Step 1: Trigger similarity index refresh for all identities
                st.write("🔄 Computing similarity scores...")
                refresh_resp = _api_post(f"/episodes/{ep_id}/refresh_similarity", {})

                if not refresh_resp or refresh_resp.get("status") != "success":
                    status.update(label="❌ Refresh failed", state="error")
                    st.error("Failed to refresh similarity values. Check logs.")
                else:
                    auto_linked_count = 0
                    log_entries = []

                    # Show refresh stats
                    tracks_processed = refresh_resp.get("tracks_processed", 0)
                    centroids_computed = refresh_resp.get("centroids_computed", 0)
                    st.write(f"✓ Processed {tracks_processed} tracks")
                    st.write(f"✓ Computed {centroids_computed} cluster centroids")
                    log_entries.append(f"Tracks: {tracks_processed}, Centroids: {centroids_computed}")

                    # Enhancement #8: Auto-link high confidence matches
                    if auto_link_enabled:
                        st.write("🔗 Auto-assigning high confidence matches...")
                        auto_link_resp = _api_post(f"/episodes/{ep_id}/auto_link_cast", {})
                        if auto_link_resp and auto_link_resp.get("auto_assigned", 0) > 0:
                            auto_linked_count = auto_link_resp["auto_assigned"]
                            assignments = auto_link_resp.get("assignments", [])
                            st.write(f"✓ Auto-assigned {auto_linked_count} cluster(s)")
                            for asn in assignments[:5]:  # Show first 5
                                st.write(
                                    f"  • {asn.get('cluster_id')} → "
                                    f"{asn.get('cast_name')} ({int(asn.get('similarity', 0) * 100)}%)"
                                )
                            if len(assignments) > 5:
                                st.write(f"  ... and {len(assignments) - 5} more")
                            log_entries.append(f"Auto-assigned: {auto_linked_count}")
                        else:
                            st.write("✓ No high-confidence matches to auto-assign")

                    # Step 2: Refresh cluster suggestions based on new similarity values
                    st.write("📊 Refreshing cluster suggestions...")
                    suggestions_resp = _safe_api_get(f"/episodes/{ep_id}/cluster_suggestions_from_assigned")

                    # Step 3: Fetch cast suggestions from facebank (Enhancement #1)
                    st.write("🎭 Fetching cast suggestions from facebank...")
                    cast_suggestions_resp = _safe_api_get(f"/episodes/{ep_id}/cast_suggestions")
                    if cast_suggestions_resp and cast_suggestions_resp.get("suggestions"):
                        # Store in session state for display
                        st.session_state[f"cast_suggestions:{ep_id}"] = {
                            sugg["cluster_id"]: sugg.get("cast_suggestions", [])
                            for sugg in cast_suggestions_resp.get("suggestions", [])
                        }
                        num_suggestions = len(cast_suggestions_resp.get("suggestions", []))
                        st.write(f"✓ Found {num_suggestions} cluster(s) with cast suggestions")
                        log_entries.append(f"Cast suggestions: {num_suggestions}")

                    # Update status
                    if auto_linked_count > 0:
                        status.update(label=f"✅ Refreshed & auto-assigned {auto_linked_count} cluster(s)", state="complete")
                    else:
                        status.update(label="✅ Refresh complete!", state="complete")

                    st.rerun()

    return detail


def _episode_people(ep_id: str) -> tuple[str | None, List[Dict[str, Any]]]:
    meta = helpers.parse_ep_id(ep_id)
    if not meta:
        return None, []
    show_slug = str(meta.get("show") or "").lower()
    if not show_slug:
        return None, []
    people_resp = _fetch_people_cached(show_slug)
    people = people_resp.get("people", []) if people_resp else []
    return show_slug, people


def _episode_cluster_ids(person: Dict[str, Any], ep_id: str) -> List[str]:
    cluster_ids: List[str] = []
    for raw in person.get("cluster_ids", []) or []:
        if not isinstance(raw, str):
            continue
        if raw.startswith(f"{ep_id}:"):
            cluster_ids.append(raw.split(":", 1)[1] if ":" in raw else raw)
    return cluster_ids


def _cluster_counts(cluster_meta: Dict[str, Any] | None) -> tuple[int, int]:
    """Return (tracks, faces) for a cluster with fallbacks when counts are missing."""
    meta = cluster_meta or {}
    counts = meta.get("counts") or {}
    tracks = coerce_int(counts.get("tracks")) if isinstance(counts, dict) else None
    faces = coerce_int(counts.get("faces")) if isinstance(counts, dict) else None
    track_list = meta.get("tracks") or []
    tracks = tracks if tracks is not None else len(track_list)
    if faces is None:
        faces = sum(coerce_int(track.get("faces")) or 0 for track in track_list)
    return int(tracks or 0), int(faces or 0)


def _episode_person_counts(
    episode_clusters: List[str],
    cluster_lookup: Dict[str, Dict[str, Any]],
) -> tuple[int, int]:
    """Aggregate tracks and faces across all episode clusters for sorting."""
    total_tracks = 0
    total_faces = 0
    for cluster_id in episode_clusters:
        tracks, faces = _cluster_counts(cluster_lookup.get(cluster_id))
        total_tracks += tracks
        total_faces += faces
    return total_tracks, total_faces


def _load_cluster_centroids(ep_id: str) -> Dict[str, Any]:
    """Load cluster centroids (with per-cluster cohesion) from manifests."""
    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
    path = data_root / "manifests" / ep_id / "cluster_centroids.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    centroids = payload.get("centroids") if isinstance(payload, dict) else None
    return centroids if isinstance(centroids, dict) else {}


def _episode_person_cohesion(
    ep_id: str,
    episode_clusters: List[str],
    cluster_lookup: Dict[str, Dict[str, Any]],
    cluster_centroids: Dict[str, Any] | None = None,
) -> float | None:
    """Estimate identity cohesion using cluster centroids across all clusters.

    We compute a weighted centroid across all clusters for the person, then score
    how similar each cluster centroid is to that combined centroid. This punishes
    mixed identities (different people grouped together) even when each cluster
    is internally perfect (cohesion=1 for single-track clusters).
    """
    centroids = cluster_centroids or _load_cluster_centroids(ep_id)
    vectors: list[np.ndarray] = []
    weights: list[float] = []

    for cluster_id in episode_clusters:
        cent_data = centroids.get(cluster_id) if isinstance(centroids, dict) else None
        centroid = cent_data.get("centroid") if isinstance(cent_data, dict) else None
        if centroid:
            try:
                vec = np.array(centroid, dtype=np.float32)
                if np.linalg.norm(vec) == 0:
                    continue
                vectors.append(vec)
                faces = cluster_lookup.get(cluster_id, {}).get("counts", {}).get("faces", 1) or 1
                weights.append(float(faces))
            except Exception:
                continue

    # Fallback to average of cluster-level cohesion when no centroids available
    if not vectors:
        weighted_sum = 0.0
        total_weight = 0
        for cluster_id in episode_clusters:
            cluster_data = cluster_lookup.get(cluster_id, {})
            cohesion = cluster_data.get("cohesion")
            if cohesion is not None:
                faces = cluster_data.get("counts", {}).get("faces", 1) or 1
                weighted_sum += cohesion * faces
                total_weight += faces
        if total_weight > 0:
            return weighted_sum / total_weight
        return None

    if len(vectors) == 1:
        # Single-cluster person: fall back to that cluster's cohesion if present
        only_cluster = episode_clusters[0]
        cluster_cohesion = cluster_lookup.get(only_cluster, {}).get("cohesion")
        return float(cluster_cohesion) if cluster_cohesion is not None else 1.0

    weights_arr = np.array(weights, dtype=np.float32)
    weights_arr = weights_arr / max(weights_arr.sum(), 1e-6)
    stacked = np.vstack(vectors)
    combined = np.average(stacked, axis=0, weights=weights_arr)
    combined_norm = combined / max(np.linalg.norm(combined), 1e-6)

    sims = []
    for vec in stacked:
        norm_vec = vec / max(np.linalg.norm(vec), 1e-6)
        sims.append(float(np.dot(norm_vec, combined_norm)))

    return float(np.mean(sims)) if sims else None


def _clusters_by_identity(
    cluster_payload: Dict[str, Any] | None,
) -> Dict[str, Dict[str, Any]]:
    clusters = cluster_payload.get("clusters", []) if cluster_payload else []
    lookup: Dict[str, Dict[str, Any]] = {}
    for entry in clusters:
        identity_id = entry.get("identity_id")
        if identity_id:
            lookup[str(identity_id)] = entry
    return lookup


def _fetch_track_media(
    ep_id: str,
    track_id: int,
    *,
    sample: int = 1,
    limit: int = TRACK_MEDIA_BATCH_LIMIT,
    cursor: str | None = None,
) -> tuple[List[Dict[str, Any]], str | None]:
    """Fetch track media using face-metadata-backed URLs for correct track scoping.

    Uses /frames endpoint instead of /crops to ensure we get track-specific URLs
    from face metadata, which correctly identifies which crop belongs to which track
    even when multiple tracks share the same frame.
    """
    # Parse cursor for page-based pagination
    page = 1
    if cursor:
        try:
            page = int(cursor)
        except (TypeError, ValueError):
            page = 1

    # Use /frames endpoint which provides face-metadata-backed URLs
    # This ensures correct track-specific crops even for shared frames
    params: Dict[str, Any] = {
        "sample": int(sample),
        "page": page,
        "page_size": int(limit),
    }
    payload = _safe_api_get(f"/episodes/{ep_id}/tracks/{track_id}/frames", params=params) or {}
    items = payload.get("items", []) if isinstance(payload, dict) else []
    total = payload.get("total", 0)
    current_page = payload.get("page", 1)
    page_size = payload.get("page_size", limit)

    # Determine next cursor (next page number) if there are more items
    next_cursor: str | None = None
    if total > current_page * page_size:
        next_cursor = str(current_page + 1)

    normalized: List[Dict[str, Any]] = []
    for item in items:
        # Get track-specific URL from face metadata (media_url/thumbnail_url)
        # These URLs are resolved from crop_rel_path which is track-specific
        # and fall back to thumb_rel_path if crop unavailable
        faces = item.get("faces", [])
        # Find the face for this specific track WITH a valid URL
        face_for_track = None
        url = None
        for face in faces if isinstance(faces, list) else []:
            face_tid = face.get("track_id")
            try:
                face_tid_int = int(face_tid) if face_tid is not None else None
            except (TypeError, ValueError):
                face_tid_int = None
            if face_tid_int == track_id:
                # Check if this face has a usable URL before accepting it
                face_url = face.get("media_url") or face.get("thumbnail_url")
                if face_url:
                    face_for_track = face
                    url = face_url
                    break
                # If no URL, keep looking for another face with the same track_id that has a URL
                elif face_for_track is None:
                    face_for_track = face  # Remember first match even without URL

        # If we found a face but no URL from it, try face-specific URL one more time
        if face_for_track and not url:
            url = face_for_track.get("media_url") or face_for_track.get("thumbnail_url")

        # Fall back to item-level URL only if no track-specific face found
        if not url and not face_for_track:
            url = item.get("media_url") or item.get("thumbnail_url")

        if not url:
            continue

        resolved = helpers.resolve_thumb(url)
        # Track resolution failures for potential warning display
        resolution_failed = resolved is None
        normalized.append(
            {
                "url": resolved or url,  # Use original URL if resolution fails
                "frame_idx": item.get("frame_idx"),
                "track_id": track_id,
                "s3_key": item.get("s3_key"),  # Preserve S3 key for debugging
                "resolution_failed": resolution_failed,  # Track for warning display
            }
        )
    return normalized, next_cursor


def _fetch_track_frames(
    ep_id: str,
    track_id: int,
    *,
    sample: int,
    page: int,
    page_size: int,
) -> Dict[str, Any]:
    params = {
        "sample": int(sample),
        "page": int(page),
        "page_size": int(page_size),
    }
    payload = _safe_api_get(f"/episodes/{ep_id}/tracks/{track_id}/frames", params=params)
    if isinstance(payload, list):
        return {"items": payload}
    if isinstance(payload, dict):
        return payload
    return {}


def _render_track_media_section(ep_id: str, track_id: int, *, sample: int) -> None:
    """Show cached crops with lazy pagination for the active track."""
    # Cache key now includes sample rate, so we get the correct state directly
    state = _track_media_state(ep_id, track_id, sample)
    if not state.get("initialized"):
        batch_limit = TRACK_MEDIA_BATCH_LIMIT
        items, cursor = _fetch_track_media(
            ep_id,
            track_id,
            sample=sample,
            limit=batch_limit,
        )
        state.update(
            {
                "items": items,
                "cursor": cursor,
                "initialized": True,
                "sample": sample,
                "batch_limit": batch_limit,
            }
        )

    items = state.get("items", [])
    cursor = state.get("cursor")
    batch_limit = int(state.get("batch_limit", TRACK_MEDIA_BATCH_LIMIT))
    # Defensive filter: ignore crops that somehow belong to another track
    if items:
        scoped_items = [item for item in items if item.get("track_id") in (None, track_id)]
        dropped = len(items) - len(scoped_items)
        if dropped:
            st.warning(f"Ignoring {dropped} crop(s) not matching track {track_id}.")
        items = scoped_items
        state["items"] = items

        # Check for URL resolution failures and show warning
        resolution_failures = sum(1 for item in items if item.get("resolution_failed"))
        if resolution_failures > 0:
            st.warning(
                f"⚠️ {resolution_failures} image(s) could not be resolved from S3. "
                f"Using fallback URLs - some images may not display correctly."
            )

    st.markdown("#### Track crops preview")
    header_cols = st.columns([3, 1])
    loaded_label = f"{len(items)} crop{'s' if len(items) != 1 else ''} loaded · batch size {batch_limit}"
    header_cols[0].caption(loaded_label)
    with header_cols[1]:
        if st.button("Refresh track crops", key=f"track_media_refresh_{track_id}"):
            _reset_track_media_state(ep_id, track_id)
            st.rerun()

    if items:
        cols_per_row = min(len(items), 6) or 1
        for row_start in range(0, len(items), cols_per_row):
            row_items = items[row_start : row_start + cols_per_row]
            row_cols = st.columns(len(row_items))
            for idx, item in enumerate(row_items):
                with row_cols[idx]:
                    frame_idx = item.get("frame_idx")
                    caption = f"Frame {frame_idx}" if frame_idx is not None else f"Track {track_id}"
                    url = item.get("url")
                    thumb_markup = helpers.thumb_html(url, alt=caption, hide_if_missing=False)
                    if thumb_markup:
                        st.markdown(thumb_markup, unsafe_allow_html=True)
                    else:
                        st.caption("Crop unavailable.")
                    st.caption(caption)
    else:
        st.info("No crops available for this track.")

    if cursor:
        if st.button("Load more crops", key=f"track_media_more_{track_id}", use_container_width=True):
            more_items, next_cursor = _fetch_track_media(
                ep_id,
                track_id,
                sample=sample,
                limit=batch_limit,
                cursor=cursor,
            )
            if more_items:
                state["items"].extend(more_items)
            state["cursor"] = next_cursor
            state["sample"] = sample
            state["batch_limit"] = batch_limit
            st.rerun()


# ============================================================================
# CLUSTER COMPARISON MODE (Feature 3)
# ============================================================================

def _render_cluster_comparison_mode(
    ep_id: str,
    show_id: str,
    cluster_lookup: Dict[str, Dict[str, Any]],
) -> None:
    """Render the Cluster Comparison Mode interface.

    Allows selecting 2-3 clusters for side-by-side visual comparison,
    shows similarity scores, and provides one-click merge functionality.
    """
    # Session state keys for comparison
    comparison_key = f"comparison_clusters:{ep_id}"
    comparison_mode_key = f"comparison_mode:{ep_id}"

    # Initialize comparison state
    if comparison_key not in st.session_state:
        st.session_state[comparison_key] = []

    selected_clusters = st.session_state[comparison_key]
    comparison_active = st.session_state.get(comparison_mode_key, False)

    # Sidebar toggle for comparison mode
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔍 Cluster Comparison")

        if comparison_active:
            st.info(f"**Selection mode active** - {len(selected_clusters)}/3 clusters selected")

            if selected_clusters:
                st.caption("Selected:")
                for i, cid in enumerate(selected_clusters):
                    cols = st.columns([3, 1])
                    with cols[0]:
                        cluster_data = cluster_lookup.get(cid, {})
                        faces = cluster_data.get("counts", {}).get("faces", 0)
                        st.text(f"{i+1}. {cid[:8]}... ({faces} faces)")
                    with cols[1]:
                        if st.button("✗", key=f"rm_cmp_{cid}", help="Remove"):
                            selected_clusters.remove(cid)
                            st.rerun()

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Compare", key="do_compare",
                            disabled=len(selected_clusters) < 2,
                            use_container_width=True):
                    st.session_state[f"show_comparison:{ep_id}"] = True
                    st.rerun()
            with col2:
                if st.button("Cancel", key="cancel_compare", use_container_width=True):
                    st.session_state[comparison_mode_key] = False
                    st.session_state[comparison_key] = []
                    st.session_state.pop(f"show_comparison:{ep_id}", None)
                    st.rerun()
        else:
            if st.button("Start Comparison", key="start_compare", use_container_width=True):
                st.session_state[comparison_mode_key] = True
                st.session_state[comparison_key] = []
                st.rerun()
            st.caption("Select 2-3 clusters to compare side-by-side")

    # Show comparison view if active
    if st.session_state.get(f"show_comparison:{ep_id}") and len(selected_clusters) >= 2:
        _render_comparison_view(ep_id, show_id, selected_clusters, cluster_lookup)


def _render_comparison_view(
    ep_id: str,
    show_id: str,
    cluster_ids: List[str],
    cluster_lookup: Dict[str, Dict[str, Any]],
) -> None:
    """Render the side-by-side comparison view for selected clusters."""
    st.markdown("---")
    st.markdown("## 🔍 Cluster Comparison")

    # Create columns for each cluster
    num_clusters = len(cluster_ids)
    cols = st.columns(num_clusters)

    cluster_data_list = []

    for i, (col, cluster_id) in enumerate(zip(cols, cluster_ids)):
        with col:
            cluster_data = cluster_lookup.get(cluster_id, {})
            cluster_data_list.append(cluster_data)

            # Header
            st.markdown(f"**Cluster {i+1}**")
            st.code(cluster_id, language=None)

            # Stats
            counts = cluster_data.get("counts", {})
            faces = counts.get("faces", 0)
            tracks = counts.get("tracks", 0)
            cohesion = cluster_data.get("cohesion")

            st.metric("Faces", faces)
            st.metric("Tracks", tracks)
            if cohesion is not None:
                coh_pct = int(cohesion * 100)
                st.metric("Cohesion", f"{coh_pct}%")

            # Assignment status
            person_id = cluster_data.get("person_id")
            if person_id:
                person_name = cluster_data.get("person_name", "Unknown")
                st.success(f"Assigned: {person_name}")
            else:
                st.warning("Unassigned")

            # Fetch and show thumbnails
            track_list = cluster_data.get("tracks", [])
            if track_list:
                st.markdown("**Sample faces:**")
                # Show up to 4 thumbnails per cluster
                thumb_cols = st.columns(2)
                shown = 0
                for track in track_list[:4]:
                    url = track.get("thumbnail_url") or track.get("media_url")
                    if url:
                        resolved = resolve_thumbnail_url(url)
                        if resolved:
                            with thumb_cols[shown % 2]:
                                st.image(resolved, width=80)
                            shown += 1
                            if shown >= 4:
                                break

    # Similarity analysis between clusters
    st.markdown("---")
    st.markdown("### Similarity Analysis")

    # Calculate pairwise similarities (if API available)
    similarity_results = []
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            cid1, cid2 = cluster_ids[i], cluster_ids[j]
            # Try to get similarity from API
            sim_resp = _safe_api_get(
                f"/episodes/{ep_id}/cluster_similarity",
                params={"cluster_a": cid1, "cluster_b": cid2}
            )
            if sim_resp and "similarity" in sim_resp:
                similarity = sim_resp["similarity"]
            else:
                # Fallback: estimate from cohesion overlap
                coh1 = cluster_data_list[i].get("cohesion", 0) or 0
                coh2 = cluster_data_list[j].get("cohesion", 0) or 0
                similarity = (coh1 + coh2) / 2 * 0.8  # Rough estimate

            similarity_results.append({
                "pair": f"Cluster {i+1} ↔ Cluster {j+1}",
                "cluster_ids": (cid1, cid2),
                "similarity": similarity,
            })

    # Display similarity results
    for result in similarity_results:
        sim_pct = int(result["similarity"] * 100)
        if sim_pct >= 70:
            color = "#4CAF50"  # Green - likely same person
            label = "High - Likely same person"
        elif sim_pct >= 50:
            color = "#FF9800"  # Orange - uncertain
            label = "Medium - Review recommended"
        else:
            color = "#F44336"  # Red - likely different
            label = "Low - Likely different people"

        st.markdown(
            f'**{result["pair"]}**: '
            f'<span style="background-color: {color}; color: white; '
            f'padding: 2px 8px; border-radius: 4px; font-weight: bold;">'
            f'{sim_pct}%</span> {label}',
            unsafe_allow_html=True
        )

    # Merge buttons
    st.markdown("---")
    st.markdown("### Actions")

    action_cols = st.columns(len(cluster_ids))

    # Determine if any clusters can be merged (need at least one unassigned or same person)
    person_ids = [cluster_lookup.get(cid, {}).get("person_id") for cid in cluster_ids]
    unique_persons = set(p for p in person_ids if p)
    all_unassigned = all(p is None for p in person_ids)
    same_person = len(unique_persons) <= 1

    if len(cluster_ids) == 2:
        cid1, cid2 = cluster_ids

        if same_person or all_unassigned:
            if st.button("🔗 Merge Clusters", key="merge_comparison", use_container_width=True):
                # Merge the two clusters
                payload = {
                    "strategy": "merge",
                    "cluster_ids": [cid1, cid2],
                }
                resp = _api_post(f"/episodes/{ep_id}/clusters/group", payload)
                if resp and resp.get("status") == "success":
                    st.success("✓ Clusters merged successfully!")
                    # Clear comparison state
                    st.session_state[f"comparison_clusters:{ep_id}"] = []
                    st.session_state[f"show_comparison:{ep_id}"] = False
                    st.session_state[f"comparison_mode:{ep_id}"] = False
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Failed to merge clusters")
        else:
            st.warning("Cannot merge: Clusters are assigned to different people")

    elif len(cluster_ids) == 3:
        if same_person or all_unassigned:
            if st.button("🔗 Merge All 3 Clusters", key="merge_all_comparison", use_container_width=True):
                payload = {
                    "strategy": "merge",
                    "cluster_ids": cluster_ids,
                }
                resp = _api_post(f"/episodes/{ep_id}/clusters/group", payload)
                if resp and resp.get("status") == "success":
                    st.success("✓ All clusters merged successfully!")
                    st.session_state[f"comparison_clusters:{ep_id}"] = []
                    st.session_state[f"show_comparison:{ep_id}"] = False
                    st.session_state[f"comparison_mode:{ep_id}"] = False
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Failed to merge clusters")
        else:
            st.warning("Cannot merge: Clusters are assigned to different people")

    # Close comparison view button
    if st.button("Close Comparison", key="close_comparison", use_container_width=True):
        st.session_state[f"show_comparison:{ep_id}"] = False
        st.rerun()


def _add_cluster_to_comparison(ep_id: str, cluster_id: str) -> bool:
    """Add a cluster to the comparison selection.

    Returns True if added successfully, False if already at max (3) or already selected.
    """
    comparison_key = f"comparison_clusters:{ep_id}"
    if comparison_key not in st.session_state:
        st.session_state[comparison_key] = []

    selected = st.session_state[comparison_key]

    if cluster_id in selected:
        return False  # Already selected

    if len(selected) >= 3:
        return False  # Max 3 clusters

    selected.append(cluster_id)
    return True


def _render_cast_carousel(
    ep_id: str,
    show_id: str,
) -> None:
    """Render featured cast members carousel at the top - ONLY shows cast with clusters in this episode."""
    if not show_id:
        return
    show_key = str(show_id).lower()
    refresh_flag = f"cast_carousel_refresh::{show_key}"
    cache = _cast_carousel_cache()
    people_cache = _cast_people_cache()
    if st.session_state.pop(refresh_flag, False):
        cache.pop(show_key, None)
        people_cache.pop(show_key, None)

    cast_api_resp = cache.get(show_key)
    if cast_api_resp is None:
        cast_api_resp = _safe_api_get(f"/shows/{show_id}/cast")
        if cast_api_resp:
            cache[show_key] = cast_api_resp
    if not cast_api_resp:
        return

    cast_members = cast_api_resp.get("cast", [])
    if not cast_members:
        return

    # Get people data to check who has clusters
    people_resp = people_cache.get(show_key)
    if people_resp is None:
        people_resp = _fetch_people_cached(show_id)
        if people_resp:
            people_cache[show_key] = people_resp
    people = people_resp.get("people", []) if people_resp else []
    # Build lookup by cast_id, filtering out None/empty/whitespace-only cast_ids
    people_by_cast_id = {
        p.get("cast_id"): p
        for p in people
        if p.get("cast_id") and isinstance(p.get("cast_id"), str) and p.get("cast_id").strip()
    }

    # Filter to only cast members with clusters in this episode
    cast_with_clusters = []
    for cast in cast_members:
        cast_id = cast.get("cast_id")
        person = people_by_cast_id.get(cast_id)
        if person:
            episode_clusters = _episode_cluster_ids(person, ep_id)
            if len(episode_clusters) > 0:
                cast_with_clusters.append((cast, person, episode_clusters))

    # Don't show carousel if no cast members have clusters in this episode
    if not cast_with_clusters:
        return

    header_cols = st.columns([3, 1])
    with header_cols[0]:
        st.markdown("### 🎬 Cast Lineup")
        st.caption("Cast members with clusters in this episode")
    with header_cols[1]:
        if st.button(
            "Refresh cast list",
            key=f"refresh_cast_{show_key}",
            use_container_width=True,
        ):
            st.session_state[refresh_flag] = True
            st.rerun()

    # Create horizontal carousel (max 5 per row)
    max_cols_per_row = min(len(cast_with_clusters), 5)

    for row_start in range(0, len(cast_with_clusters), max_cols_per_row):
        row_items = cast_with_clusters[row_start : row_start + max_cols_per_row]
        # Use actual row item count to avoid empty columns on last row
        cols = st.columns(len(row_items))

        for idx, (cast, person, episode_clusters) in enumerate(row_items):
            with cols[idx]:
                cast_id = cast.get("cast_id")
                name = cast.get("name", "(unnamed)")

                # Get facebank featured image
                facebank_resp = _safe_api_get(f"/cast/{cast_id}/facebank?show_id={show_id}")
                featured_url = None
                if facebank_resp and facebank_resp.get("featured_seed"):
                    featured_seed = facebank_resp["featured_seed"]
                    featured_url = featured_seed.get("display_url")

                # Display featured image
                if featured_url:
                    thumb_markup = helpers.thumb_html(featured_url, alt=name, hide_if_missing=False)
                    st.markdown(thumb_markup, unsafe_allow_html=True)
                else:
                    st.markdown("_No featured image_")

                # Name
                st.markdown(f"**{name}**")

                # Show cluster count (always > 0 due to filtering)
                cluster_count = len(episode_clusters)
                st.caption(f"✓ {cluster_count} cluster{'s' if cluster_count != 1 else ''}")

                # View detections button
                if st.button("View", key=f"carousel_view_{cast_id}", use_container_width=True):
                    st.session_state["filter_cast_id"] = cast_id
                    st.session_state["filter_cast_name"] = name
                    st.rerun()

    st.markdown("---")


def _render_cast_gallery(
    ep_id: str,
    cast_cards: List[Dict[str, Any]],
    cluster_lookup: Dict[str, Dict[str, Any]],
    cluster_centroids: Dict[str, Any] | None = None,
) -> None:
    """Render cast members as a consistent grid with fixed column widths."""
    if not cast_cards:
        return

    # Always use 5 columns per row for consistent layout
    cols_per_row = 5

    for row_start in range(0, len(cast_cards), cols_per_row):
        row_members = cast_cards[row_start : row_start + cols_per_row]
        # Always create 5 columns, even if row has fewer items (ensures consistent widths)
        cols = st.columns(cols_per_row)

        for idx, card in enumerate(row_members):
            with cols[idx]:
                cast_info = card.get("cast") or {}
                person = card.get("person") or {}
                cast_id = cast_info.get("cast_id") or person.get("cast_id")
                person_id = person.get("person_id")
                episode_clusters = card.get("episode_clusters", [])

                name = cast_info.get("name") or person.get("name") or "(unnamed)"
                aliases = cast_info.get("aliases") or person.get("aliases") or []

                featured_source = card.get("featured_thumbnail") or person.get("rep_crop")
                featured_thumb = helpers.resolve_thumb(featured_source)
                if featured_thumb:
                    thumb_markup = helpers.thumb_html(featured_thumb, alt=name, hide_if_missing=False)
                    st.markdown(thumb_markup, unsafe_allow_html=True)
                else:
                    st.markdown("_No featured image_")

                st.markdown(f"**{name}**")

                if aliases:
                    alias_text = ", ".join(f"`{a}`" for a in aliases[:2])
                    if len(aliases) > 2:
                        alias_text += f" +{len(aliases) - 2}"
                    st.caption(alias_text)

                total_tracks = 0
                total_faces = 0
                avg_cohesion: float | None = None

                if person_id and episode_clusters:
                    total_tracks, total_faces = _episode_person_counts(episode_clusters, cluster_lookup)
                    avg_cohesion = _episode_person_cohesion(
                        ep_id, episode_clusters, cluster_lookup, cluster_centroids
                    )

                st.caption(f"**{len(episode_clusters)}** clusters · **{total_tracks}** tracks · **{total_faces}** frames")

                if avg_cohesion is not None:
                    badge = render_similarity_badge(avg_cohesion, SimilarityType.IDENTITY)
                    st.markdown(f"Identity Similarity {badge}", unsafe_allow_html=True)

                if person_id and episode_clusters:
                    if st.button(
                        "View",
                        key=f"view_cast_{cast_id}_{person_id}",
                        use_container_width=True,
                    ):
                        _set_view("person_clusters", person_id=person_id)
                        st.rerun()
                elif person_id:
                    st.caption("No clusters yet")
                else:
                    st.caption("Not linked to a person record yet")


def _render_unassigned_cluster_card(
    ep_id: str,
    show_id: str,
    cluster_id: str,
    suggestion: Optional[Dict[str, Any]],
    cast_options: Dict[str, str],
    cluster_lookup: Dict[str, Dict[str, Any]],
    cast_suggestions: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Render an unassigned cluster card with suggestion and assignment UI.

    Args:
        cast_suggestions: Optional list of cast member suggestions from facebank with format:
            [{"cast_id": "...", "name": "Kyle", "similarity": 0.87, "confidence": "high"}, ...]
    """
    cluster_meta = cluster_lookup.get(cluster_id, {})
    counts = cluster_meta.get("counts", {})
    original_tracks_count = counts.get("tracks", 0)
    original_faces_count = counts.get("faces", 0)
    track_list = cluster_meta.get("tracks", [])

    # Filter out tracks with only 1 frame (likely noise/false positives)
    original_track_list = track_list
    track_list = [t for t in track_list if t.get("faces", 0) > 1]
    filtered_tracks_count = len(original_track_list) - len(track_list)

    # Recalculate counts after filtering
    tracks_count = len(track_list)
    faces_count = sum(t.get("faces", 0) for t in track_list)

    # Show filtered-out cluster with explanation instead of silently skipping
    if not track_list or tracks_count == 0:
        with st.container(border=True):
            st.markdown(f"### 🔍 Cluster `{cluster_id}`")
            st.warning(
                f"⚠️ This cluster has no reviewable tracks. "
                f"({filtered_tracks_count} single-frame track(s) were filtered as noise.)"
            )
            st.caption(
                "Smart Suggestions skip this cluster. Delete it to clear the noise or rerun detection "
                "if you believe frames are missing."
            )
            col1, col2 = st.columns([2, 1])
            with col2:
                if st.button("Delete", key=f"delete_empty_{cluster_id}", type="secondary"):
                    resp = _api_delete(f"/episodes/{ep_id}/identities/{cluster_id}")
                    if resp:
                        st.success(f"Deleted cluster {cluster_id}")
                        st.rerun()
        return

    # Get suggested person if available
    suggested_person_id = suggestion.get("suggested_person_id") if suggestion else None
    suggested_distance = suggestion.get("distance") if suggestion else None
    suggested_cast_id = None
    suggested_cast_name = None
    suggested_person = None

    if suggested_person_id:
        # Find the person for the suggested person_id
        people_resp = _fetch_people_cached(show_id)
        if people_resp:
            people = people_resp.get("people", [])
            suggested_person = next((p for p in people if p.get("person_id") == suggested_person_id), None)
            if suggested_person:
                suggested_cast_id = suggested_person.get("cast_id")
                # Use person name first, fallback to cast name if no person name
                suggested_cast_name = suggested_person.get("name")
                if not suggested_cast_name and suggested_cast_id:
                    suggested_cast_name = cast_options.get(suggested_cast_id)
                # If still no name, use person_id
                if not suggested_cast_name:
                    suggested_cast_name = f"Person {suggested_person_id}"

    # Build cluster data for quality indicator analysis
    cluster_cohesion = cluster_meta.get("cohesion")
    cluster_quality_data = {
        "cohesion": cluster_cohesion,
        "faces": faces_count,
        "tracks": tracks_count,
        # Get confidence from cast suggestions if available
        "confidence": cast_suggestions[0].get("similarity") if cast_suggestions else None,
    }

    with st.container(border=True):
        # Header with cluster info
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
        with col1:
            st.markdown(f"### 🔍 Cluster `{cluster_id}`")
            caption_parts = [f"{tracks_count} track(s) · {faces_count} frame(s)"]
            if filtered_tracks_count > 0:
                caption_parts.append(f"({filtered_tracks_count} single-frame filtered)")
            st.caption(" ".join(caption_parts))
            # Show similarity badge - use cluster cohesion for multi-track, track similarity for single-track
            similarity_value = None
            similarity_label = None
            if tracks_count > 1 and cluster_cohesion is not None:
                # Multi-track cluster: show cluster cohesion (how similar tracks are to each other)
                similarity_value = cluster_cohesion
                similarity_label = "Cluster Similarity"
            elif tracks_count == 1 and track_list:
                # Single-track cluster: show track similarity (how similar frames are within the track)
                track_sim = track_list[0].get("similarity")
                if track_sim is not None:
                    similarity_value = track_sim
                    similarity_label = "Track Similarity"
            if similarity_value is not None and similarity_label:
                sim_pct = int(similarity_value * 100)
                sim_color = "#4CAF50" if similarity_value >= 0.7 else "#FF9800" if similarity_value >= 0.5 else "#F44336"
                st.markdown(
                    f'<span style="background-color: {sim_color}; color: white; '
                    f'padding: 2px 6px; border-radius: 3px; font-size: 0.75em;">{similarity_label}: {sim_pct}%</span>',
                    unsafe_allow_html=True
                )
            # Quality indicators (Feature 10)
            quality_badges = render_cluster_quality_badges(cluster_quality_data, compact=True, max_badges=2)
            if quality_badges:
                st.markdown(quality_badges, unsafe_allow_html=True)
        with col2:
            # View cluster button
            if st.button("View", key=f"view_unassigned_{cluster_id}"):
                _set_view("cluster_tracks", identity_id=cluster_id)
                st.rerun()
        with col3:
            # "Suggest for Me" button (Enhancement #6)
            if st.button("💡 Suggest", key=f"suggest_me_{cluster_id}", help="Find matching cast members"):
                with st.spinner("Finding matches..."):
                    suggest_resp = _safe_api_get(f"/episodes/{ep_id}/clusters/{cluster_id}/suggest_cast")
                    if suggest_resp and suggest_resp.get("suggestions"):
                        # Store suggestions in session state
                        st.session_state[f"cast_suggestions:{ep_id}"] = st.session_state.get(
                            f"cast_suggestions:{ep_id}", {}
                        )
                        st.session_state[f"cast_suggestions:{ep_id}"][cluster_id] = suggest_resp["suggestions"]
                        st.toast(f"Found {len(suggest_resp['suggestions'])} suggestion(s)!")
                        st.rerun()
                    else:
                        message = suggest_resp.get("message", "No matches found") if suggest_resp else "API error"
                        st.warning(message)
        with col4:
            # Compare button (Feature 3)
            comparison_mode_active = st.session_state.get(f"comparison_mode:{ep_id}", False)
            comparison_clusters = st.session_state.get(f"comparison_clusters:{ep_id}", [])
            is_selected = cluster_id in comparison_clusters
            if comparison_mode_active:
                if is_selected:
                    if st.button("✓ Selected", key=f"cmp_sel_{cluster_id}", type="primary"):
                        comparison_clusters.remove(cluster_id)
                        st.rerun()
                elif len(comparison_clusters) < 3:
                    if st.button("🔍 Compare", key=f"cmp_add_{cluster_id}"):
                        if _add_cluster_to_comparison(ep_id, cluster_id):
                            st.rerun()
                else:
                    st.button("🔍 Compare", key=f"cmp_dis_{cluster_id}", disabled=True, help="Max 3 clusters")
        with col5:
            # Delete cluster button
            if st.button("Delete", key=f"delete_unassigned_{cluster_id}", type="secondary"):
                resp = _api_delete(f"/episodes/{ep_id}/identities/{cluster_id}")
                if resp:
                    st.success(f"Deleted cluster {cluster_id}")
                    st.rerun()
                else:
                    st.error("Failed to delete cluster")

        # Display one representative frame from each track in the cluster with scrollable carousel
        if track_list:
            num_tracks = len(track_list)
            max_visible = 6  # Show up to 6 tracks at a time

            # Track pagination state for this cluster (include ep_id to avoid collisions)
            page_key = f"track_page_{ep_id}_{cluster_id}"
            if page_key not in st.session_state:
                st.session_state[page_key] = 0

            current_page = st.session_state[page_key]
            total_pages = (num_tracks + max_visible - 1) // max_visible

            # Navigation controls if more than max_visible tracks
            if total_pages > 1:
                col_left, col_center, col_right = st.columns([1, 6, 1])
                with col_left:
                    if st.button("◀", key=f"prev_{ep_id}_{cluster_id}", disabled=current_page == 0):
                        st.session_state[page_key] = max(0, current_page - 1)
                        st.rerun()
                with col_center:
                    st.caption(
                        f"Showing tracks {current_page * max_visible + 1}-{min((current_page + 1) * max_visible, num_tracks)} of {num_tracks}"
                    )
                with col_right:
                    if st.button(
                        "▶",
                        key=f"next_{ep_id}_{cluster_id}",
                        disabled=current_page >= total_pages - 1,
                    ):
                        st.session_state[page_key] = min(total_pages - 1, current_page + 1)
                        st.rerun()

            # Display current page of tracks in a single row
            start_idx = current_page * max_visible
            end_idx = min(start_idx + max_visible, num_tracks)
            visible_tracks = track_list[start_idx:end_idx]

            # Create single-row columns for the visible tracks
            if visible_tracks:
                cols = st.columns(len(visible_tracks))
                for idx, track in enumerate(visible_tracks):
                    with cols[idx]:
                        thumb_url = track.get("rep_thumb_url")
                        track_id = track.get("track_id")
                        track_faces = track.get("faces", 0)

                        if thumb_url:
                            # Extract S3 key from presigned URL to generate fresh presign
                            thumb_src = _extract_s3_key_from_url(thumb_url)
                            thumb_markup = helpers.thumb_html(
                                thumb_src,
                                alt=f"Track {track_id}",
                                hide_if_missing=False,
                            )
                            st.markdown(thumb_markup, unsafe_allow_html=True)
                            # Show track info with similarity score if available
                            track_sim = track.get("similarity")
                            if track_sim is not None:
                                sim_pct = int(track_sim * 100)
                                st.caption(f"Track {track_id} · {track_faces} faces · {sim_pct}% sim")
                            else:
                                st.caption(f"Track {track_id} · {track_faces} faces")

        # Show cast suggestions from facebank (Enhancement #1) if available
        if cast_suggestions:
            st.markdown("**🎯 Cast Suggestions from Facebank:**")
            for idx, cast_sugg in enumerate(cast_suggestions[:3]):
                sugg_cast_id = cast_sugg.get("cast_id")
                sugg_name = cast_sugg.get("name", sugg_cast_id)
                sugg_sim = cast_sugg.get("similarity", 0)
                sugg_confidence = cast_sugg.get("confidence", "low")

                # Confidence badge colors
                confidence_colors = {
                    "high": "#4CAF50",  # Green
                    "medium": "#FF9800",  # Orange
                    "low": "#F44336",  # Red
                }
                badge_color = confidence_colors.get(sugg_confidence, "#9E9E9E")
                sim_pct = int(sugg_sim * 100)

                sugg_col1, sugg_col2 = st.columns([5, 1])
                with sugg_col1:
                    st.markdown(
                        f'<span style="background-color: {badge_color}; color: white; padding: 2px 8px; '
                        f'border-radius: 4px; font-size: 0.85em; font-weight: bold; margin-right: 8px;">'
                        f'{sugg_confidence.upper()}</span> **{sugg_name}** ({sim_pct}% similarity)',
                        unsafe_allow_html=True,
                    )
                with sugg_col2:
                    if st.button("✓", key=f"cast_sugg_assign_{cluster_id}_{idx}", help=f"Assign to {sugg_name}"):
                        # Find or create person for this cast member
                        people_resp = _fetch_people_cached(show_id)
                        people = people_resp.get("people", []) if people_resp else []
                        target_person = next(
                            (p for p in people if p.get("cast_id") == sugg_cast_id),
                            None,
                        )
                        target_person_id = target_person.get("person_id") if target_person else None

                        payload = {
                            "strategy": "manual",
                            "cluster_ids": [cluster_id],
                            "target_person_id": target_person_id,
                            "cast_id": sugg_cast_id,
                        }
                        resp = _api_post(f"/episodes/{ep_id}/clusters/group", payload)
                        if resp and resp.get("status") == "success":
                            st.success(f"Assigned cluster to {sugg_name}!")
                            st.rerun()
                        else:
                            st.error("Failed to assign cluster. Check logs.")
            st.markdown("---")

        # Show person-based suggestion if available (existing logic)
        if suggested_cast_id and suggested_cast_name:
            similarity_pct = int((1 - suggested_distance) * 100) if suggested_distance is not None else 0
            # Enhancement #5: Confidence scoring
            if similarity_pct >= 80:
                confidence = "HIGH"
                badge_color = "#4CAF50"  # Green
            elif similarity_pct >= 65:
                confidence = "MEDIUM"
                badge_color = "#FF9800"  # Orange
            else:
                confidence = "LOW"
                badge_color = "#F44336"  # Red

            sugg_col1, sugg_col2 = st.columns([5, 1])
            with sugg_col1:
                st.markdown(
                    f'✨ <span style="background-color: {badge_color}; color: white; padding: 2px 8px; '
                    f'border-radius: 4px; font-size: 0.85em; font-weight: bold; margin-right: 8px;">'
                    f'{confidence}</span> Suggested (from assigned clusters): **{suggested_cast_name}** '
                    f'({similarity_pct}% similarity)',
                    unsafe_allow_html=True,
                )
            with sugg_col2:
                # Wrap in form to prevent double-click requirement
                # Use custom CSS to remove form border
                st.markdown(
                    """
                <style>
                [data-testid="stForm"] {
                    border: 0px !important;
                    padding: 0px !important;
                }
                </style>
                """,
                    unsafe_allow_html=True,
                )

                with st.form(key=f"quick_assign_form_{cluster_id}"):
                    submit_quick_assign = st.form_submit_button(
                        "✓",
                        help="Accept suggestion and assign",
                        type="primary",
                        use_container_width=True,
                    )

                if submit_quick_assign:
                    # Use the suggested person_id directly
                    payload = {
                        "strategy": "manual",
                        "cluster_ids": [cluster_id],
                        "target_person_id": suggested_person_id,
                        "cast_id": suggested_cast_id,  # May be None, that's ok
                    }
                    resp = _api_post(f"/episodes/{ep_id}/clusters/group", payload)
                    if resp and resp.get("status") == "success":
                        st.success(f"Assigned cluster to {suggested_cast_name}!")
                        st.rerun()
                    else:
                        st.error("Failed to assign cluster. Check logs.")

        st.markdown("**Assign this cluster to:**")

        # Assignment options
        assign_choice = st.radio(
            "Assignment type",
            ["Existing cast member", "New person"],
            key=f"assign_type_unassigned_{cluster_id}",
            horizontal=True,
        )

        if assign_choice == "Existing cast member":
            if cast_options:
                # Determine default index
                default_index = 0
                if suggested_cast_id and suggested_cast_id in cast_options:
                    default_index = list(cast_options.keys()).index(suggested_cast_id)

                # Use a form to ensure selectbox and button states are synchronized
                with st.form(key=f"assign_form_unassigned_{cluster_id}"):
                    selected_cast_id = st.selectbox(
                        "Select cast member",
                        options=list(cast_options.keys()),
                        format_func=lambda cid: cast_options[cid],
                        index=default_index,
                        key=f"cast_select_unassigned_{cluster_id}",
                    )

                    submit_assign = st.form_submit_button("Assign Cluster")

                    if submit_assign:
                        with st.spinner("Assigning cluster..."):
                            # First, find or get the person_id for this cast member
                            people_resp = _fetch_people_cached(show_id)
                            people = people_resp.get("people", []) if people_resp else []
                            target_person = next(
                                (p for p in people if p.get("cast_id") == selected_cast_id),
                                None,
                            )

                            if target_person:
                                target_person_id = target_person.get("person_id")
                            else:
                                # Person doesn't exist yet - the API will create one
                                target_person_id = None

                            # Call manual grouping API
                            payload = {
                                "strategy": "manual",
                                "cluster_ids": [cluster_id],
                                "target_person_id": target_person_id,
                                "cast_id": selected_cast_id,  # Include cast_id for linking
                            }
                            resp = _api_post(f"/episodes/{ep_id}/clusters/group", payload)
                            if resp and resp.get("status") == "success":
                                st.success(f"Assigned cluster to {cast_options[selected_cast_id]}!")
                                st.rerun()
                            else:
                                st.error("Failed to assign cluster. Check logs.")
            else:
                st.warning("No cast members available. Import cast first.")

        else:  # New person
            new_name = st.text_input(
                "Person name",
                key=f"new_name_unassigned_{cluster_id}",
                placeholder="Enter name...",
            )
            if st.button(
                "Create person",
                key=f"create_person_btn_{cluster_id}",
                disabled=not new_name,
            ):
                with st.spinner("Creating person..."):
                    payload = {
                        "strategy": "manual",
                        "cluster_ids": [cluster_id],
                        "name": new_name,
                    }
                    resp = _api_post(f"/episodes/{ep_id}/clusters/group", payload)
                    if resp and resp.get("status") == "success":
                        st.success(f"Created person '{new_name}' with this cluster!")
                        st.rerun()
                    else:
                        st.error("Failed to create person. Check logs.")


def _render_auto_person_card(
    ep_id: str,
    show_id: str,
    person: Dict[str, Any],
    episode_clusters: List[str],
    cast_options: Dict[str, str],
    suggestions_by_cluster: Dict[str, Dict[str, Any]] | None = None,
    cast_suggestions_by_cluster: Dict[str, List[Dict[str, Any]]] | None = None,
    avg_cohesion: float | None = None,
) -> None:
    """Render an auto-detected person card with detailed clusters and bulk assignment."""
    person_id = str(person.get("person_id") or "")
    name = person.get("name") or "(unnamed)"
    aliases = person.get("aliases") or []
    total_clusters = len(person.get("cluster_ids", []) or [])

    # Get suggested cast member if available (from cluster-to-cluster comparison)
    suggested_person_id = None
    suggested_distance = None
    if episode_clusters and suggestions_by_cluster:
        first_cluster = episode_clusters[0].split(":")[-1]
        suggestion = suggestions_by_cluster.get(first_cluster)
        if suggestion:
            suggested_person_id = suggestion.get("suggested_person_id")
            suggested_distance = suggestion.get("distance")

    # Get best cast suggestion from facebank (more reliable)
    best_cast_suggestion = None
    if episode_clusters and cast_suggestions_by_cluster:
        first_cluster = episode_clusters[0].split(":")[-1]
        cast_suggestions = cast_suggestions_by_cluster.get(first_cluster, [])
        if cast_suggestions:
            best_cast_suggestion = cast_suggestions[0]  # Already sorted by similarity

    with st.container(border=True):
        # Name
        st.markdown(f"### 👤 {name}")

        # Show aliases if present
        if aliases:
            alias_text = ", ".join(f"`{a}`" for a in aliases)
            st.caption(f"Aliases: {alias_text}")

        # Metrics line
        metrics_text = f"ID: {person_id} · {total_clusters} cluster(s) overall · {len(episode_clusters)} in this episode"
        st.caption(metrics_text)

        # Identity Similarity row + View button
        sim_cols = st.columns([3, 1])
        with sim_cols[0]:
            if avg_cohesion is not None:
                cohesion_badge = render_similarity_badge(avg_cohesion, SimilarityType.IDENTITY)
                st.markdown(f"**Identity Similarity:** {cohesion_badge}", unsafe_allow_html=True)
        with sim_cols[1]:
            if person_id and episode_clusters:
                if st.button(
                    "View All Clusters",
                    key=f"view_all_clusters_{person_id}",
                    use_container_width=True,
                ):
                    _set_view("person_clusters", person_id=person_id)
                    st.rerun()

        # Archive button for auto-clustered people (those without cast_id)
        if not person.get("cast_id"):
            if st.button(f"🗃️ Archive {name}", key=f"delete_person_{person_id}", type="secondary"):
                archive_success = False
                try:
                    resp = helpers.api_delete(f"/shows/{show_id}/people/{person_id}")
                    st.success(f"Archived {name} ({person_id})")
                    archive_success = True
                except Exception as exc:
                    st.error(f"Failed to archive person: {exc}")
                if archive_success:
                    st.rerun()

    # --- CLUSTERS CAROUSEL ---
    # Fetch clusters summary to show thumbnails
    clusters_summary = _safe_api_get(f"/episodes/{ep_id}/people/{person_id}/clusters_summary")
    if clusters_summary and clusters_summary.get("clusters"):
        clusters = clusters_summary.get("clusters", [])
        st.markdown(f"**Clusters in this episode:** ({len(clusters)})")

        # Render clusters in a grid (3 per row)
        cols_per_row = 3
        for row_start in range(0, len(clusters), cols_per_row):
            row_clusters = clusters[row_start : row_start + cols_per_row]
            row_cols = st.columns(cols_per_row)

            for idx, cluster in enumerate(row_clusters):
                with row_cols[idx]:
                    with st.container(border=True):
                        cluster_id = cluster.get("cluster_id")
                        cohesion = cluster.get("cohesion")
                        track_reps = cluster.get("track_reps", [])

                        # Show first track rep as cluster thumbnail
                        if track_reps:
                            first_track = track_reps[0]
                            crop_url = first_track.get("crop_url")
                            resolved = helpers.resolve_thumb(crop_url)
                            thumb_markup = helpers.thumb_html(
                                resolved,
                                alt=f"Cluster {cluster_id}",
                                hide_if_missing=False,
                            )
                            st.markdown(thumb_markup, unsafe_allow_html=True)

                        # Show cluster ID and cohesion badge
                        cohesion_badge = render_similarity_badge(cohesion, SimilarityType.CLUSTER) if cohesion else ""
                        st.markdown(f"**{cluster_id}** {cohesion_badge}", unsafe_allow_html=True)
                        st.caption(f"{cluster.get('tracks', 0)} tracks · {cluster.get('faces', 0)} frames")

                        # View and Delete cluster buttons
                        btn_cols = st.columns([1, 1])
                        with btn_cols[0]:
                            if st.button("View", key=f"view_cluster_{person_id}_{cluster_id}"):
                                _set_view(
                                    "cluster_tracks",
                                    person_id=person_id,
                                    identity_id=cluster_id,
                                )
                                st.rerun()
                        with btn_cols[1]:
                            if st.button(
                                "Delete",
                                key=f"delete_cluster_{person_id}_{cluster_id}",
                                type="secondary",
                            ):
                                resp = _api_delete(f"/episodes/{ep_id}/identities/{cluster_id}")
                                if resp:
                                    st.success(f"Deleted cluster {cluster_id}")
                                    st.rerun()
                                else:
                                    st.error("Failed to delete cluster")

                        # Per-cluster assignment to cast
                        if cast_options:
                            suggestion = (suggestions_by_cluster or {}).get(cluster_id)
                            suggested_cast_id = suggestion.get("cast_id") if suggestion else None
                            cast_ids = list(cast_options.keys())
                            placeholder_opts = [""] + cast_ids
                            with st.form(key=f"assign_cluster_form_{person_id}_{cluster_id}"):
                                selected_cast_id = st.selectbox(
                                    "Assign to cast member",
                                    options=placeholder_opts,
                                    format_func=lambda cid: cast_options.get(cid, "Select cast member…") if cid else "Select cast member…",
                                    index=0,
                                    key=f"assign_cast_select_{person_id}_{cluster_id}",
                                )
                                submitted = st.form_submit_button("Assign cluster", use_container_width=True)
                                if submitted:
                                    if not selected_cast_id:
                                        st.error("Select a cast member before assigning.")
                                    elif _assign_cluster_to_cast(ep_id, show_id, cluster_id, selected_cast_id):
                                        st.success(f"Assigned to {cast_options[selected_cast_id]}")
                                        st.rerun()

    # --- ASSIGN ALL CLUSTERS SECTION ---
    # Cast suggestion box for unnamed people with facebank matches (unless dismissed)
    dismiss_key = f"dismissed_sugg:{person_id}"
    suggestion_dismissed = st.session_state.get(dismiss_key, False)
    if (
        best_cast_suggestion
        and not person.get("cast_id")
        and not person.get("name")
        and not suggestion_dismissed
    ):
        sugg_cast_id = best_cast_suggestion.get("cast_id")
        sugg_name = best_cast_suggestion.get("name", "Unknown")
        sugg_sim = best_cast_suggestion.get("similarity", 0)
        sugg_confidence = best_cast_suggestion.get("confidence", "low")

        # Confidence badge colors
        confidence_colors = {"high": "#4CAF50", "medium": "#FF9800", "low": "#F44336"}
        conf_color = confidence_colors.get(sugg_confidence, "#F44336")
        sim_pct = int(sugg_sim * 100)

        with st.container(border=True):
            st.markdown(
                f"**💡 Suggested Match:** {sugg_name} "
                f'<span style="background-color: {conf_color}; color: white; padding: 2px 8px; '
                f'border-radius: 4px; font-size: 0.85em; font-weight: bold;">{sim_pct}%</span>',
                unsafe_allow_html=True,
            )
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("✅ Accept", key=f"accept_sugg_{person_id}", use_container_width=True, type="primary"):
                    if not show_id:
                        st.error("Cannot assign: show_id is missing.")
                    else:
                        # Assign all clusters to the suggested cast member
                        with st.spinner(f"Assigning {len(episode_clusters)} cluster(s) to {sugg_name}..."):
                            result = _bulk_assign_clusters(
                                ep_id, show_id, person_id, sugg_cast_id, episode_clusters, sugg_name
                            )
                        if result:
                            st.toast(f"✅ Assigned to {sugg_name}!")
                            st.rerun()
            with btn_col2:
                if st.button("❌ Decline", key=f"decline_sugg_{person_id}", use_container_width=True):
                    st.session_state[dismiss_key] = True
                    st.rerun()

    # Bulk assignment for unnamed people (only show when multiple clusters)
    if not person.get("cast_id") and not person.get("name") and len(episode_clusters) > 1:
        st.markdown("**Assign all clusters to:**")

        # Assignment options
        assign_choice = st.radio(
            "Assignment type",
            ["Existing cast member", "New person"],
            key=f"assign_type_{person_id}",
            horizontal=True,
        )

        if assign_choice == "Existing cast member":
            if cast_options:
                # Find suggested cast_id if we have a person_id suggestion
                suggested_cast_id = None
                if suggested_person_id:
                    # Find the cast_id for this person
                    people_resp = _fetch_people_cached(show_id)
                    if people_resp:
                        people_list = people_resp.get("people", [])
                        suggested_person = next(
                            (p for p in people_list if p.get("person_id") == suggested_person_id),
                            None,
                        )
                        if suggested_person:
                            suggested_cast_id = suggested_person.get("cast_id")

                # Use a form to ensure selectbox and button states are synchronized
                with st.form(key=f"assign_form_{person_id}"):
                    cast_ids = list(cast_options.keys())
                    placeholder_options = [""] + cast_ids
                    selected_cast_id = st.selectbox(
                        "Select cast member",
                        options=placeholder_options,
                        format_func=lambda pid: cast_options.get(pid, "Select cast member…") if pid else "Select cast member…",
                        index=0,
                        key=f"cast_select_{person_id}",
                    )

                    # Show suggestion info if available (Enhancement #5: with confidence)
                    if (
                        suggested_cast_id
                        and suggested_cast_id == selected_cast_id
                        and suggested_distance is not None
                    ):
                        similarity_pct = int((1 - suggested_distance) * 100)
                        if similarity_pct >= 80:
                            conf_label, conf_color = "HIGH", "#4CAF50"
                        elif similarity_pct >= 65:
                            conf_label, conf_color = "MEDIUM", "#FF9800"
                        else:
                            conf_label, conf_color = "LOW", "#F44336"
                        st.markdown(
                            f'<span style="background-color: {conf_color}; color: white; padding: 1px 6px; '
                            f'border-radius: 3px; font-size: 0.75em; font-weight: bold;">{conf_label}</span> '
                            f'✨ Suggested match ({similarity_pct}%)',
                            unsafe_allow_html=True,
                        )

                    submit_assign = st.form_submit_button("Assign Cluster", use_container_width=True)

                    if submit_assign:
                        if not selected_cast_id:
                            st.error("Select a cast member before assigning.")
                        else:
                            st.toast(f"Assigning {len(episode_clusters)} cluster(s)...")
                            result = _bulk_assign_clusters(
                                ep_id,
                                show_id,
                                person_id,
                                selected_cast_id,
                                episode_clusters,
                            )
                            if result:
                                st.success(
                                    f"Assigned {len(episode_clusters)} clusters to {cast_options[selected_cast_id]}"
                                )
                                st.rerun()
            else:
                st.info("No cast members available. Create one first in the Cast page.")
        else:
            new_name = st.text_input(
                "New person name",
                key=f"new_name_{person_id}",
                placeholder="Enter name...",
            )
            if new_name and st.button("Create & Assign", key=f"create_assign_btn_{person_id}"):
                # Assign all clusters with this name
                with st.spinner(f"Creating '{new_name}' and assigning {len(episode_clusters)} cluster(s)..."):
                    result = _bulk_assign_to_new_person(ep_id, show_id, person_id, new_name, episode_clusters)
                if result:
                    st.success(f"Created '{new_name}' and assigned {len(episode_clusters)} clusters")
                    st.rerun()


def _assign_cluster_to_cast(ep_id: str, show_id: str, cluster_id: str, cast_id: str) -> bool:
    """Assign a single cluster to a cast-linked person, creating the person if needed."""
    try:
        people_resp = _fetch_people_cached(show_id)
        people = people_resp.get("people", []) if people_resp else []
        target_person = next((p for p in people if p.get("cast_id") == cast_id), None)
        target_person_id = target_person.get("person_id") if target_person else None

        if not target_person_id:
            cast_resp = _fetch_cast_cached(show_id)
            cast_members = cast_resp.get("cast", []) if cast_resp else []
            cast_member = next((cm for cm in cast_members if cm.get("cast_id") == cast_id), None)
            if not cast_member:
                st.error(f"Cast member {cast_id} not found.")
                return False
            create_payload = {
                "name": cast_member.get("name"),
                "cast_id": cast_id,
                "aliases": cast_member.get("aliases", []),
            }
            new_person = _api_post(f"/shows/{show_id}/people", create_payload)
            if not new_person or not isinstance(new_person, dict) or "person_id" not in new_person:
                st.error("Failed to create person for this cast member.")
                return False
            target_person_id = new_person["person_id"]

        payload = {
            "strategy": "manual",
            "cluster_ids": [cluster_id],
            "target_person_id": target_person_id,
            "cast_id": cast_id,
        }
        resp = _api_post(f"/episodes/{ep_id}/clusters/group", payload)
        if resp and resp.get("status") == "success":
            _invalidate_assignment_caches()
            return True
        st.error("Failed to assign cluster. Check logs.")
        return False
    except Exception as exc:
        st.error(f"Failed to assign cluster: {exc}")
        return False


def _bulk_assign_clusters(
    ep_id: str,
    show_id: str,
    source_person_id: str,
    target_cast_id: str,
    cluster_ids: List[str],
    target_name: str | None = None,  # Optional: for display/logging purposes
) -> bool:
    """Assign all clusters from source person to a cast member.

    Args:
        target_name: Optional display name for the target cast member (used in messages).
    """
    # Use session state for debug flag, with env var as fallback for initial value
    if "debug_assign_cluster" not in st.session_state:
        st.session_state["debug_assign_cluster"] = os.getenv("SCREENALYTICS_DEBUG_ASSIGN_CLUSTER") == "1"
    debug_assign = st.session_state.get("debug_assign_cluster", False)

    def _debug(msg: str, payload: Any | None = None) -> None:
        if not debug_assign:
            return
        if payload is None:
            st.info(f"DEBUG Assign Cluster: {msg}")
        else:
            st.info(f"DEBUG Assign Cluster: {msg} -> {payload}")

    try:
        # Find or create a person record for this cast_id via API
        people_resp = _fetch_people_cached(show_id)
        people = people_resp.get("people", []) if people_resp else []
        target_person = next((p for p in people if p.get("cast_id") == target_cast_id), None)
        _debug(
            "resolved target person",
            {
                "existing": bool(target_person),
                "source_person_id": source_person_id,
                "cluster_ids": cluster_ids,
                "cast_id": target_cast_id,
            },
        )

        if not target_person:
            # Fetch cast member details to get the name
            cast_resp = _fetch_cast_cached(show_id)
            cast_members = cast_resp.get("cast", []) if cast_resp else []
            cast_member = next((cm for cm in cast_members if cm.get("cast_id") == target_cast_id), None)

            if not cast_member:
                st.error(f"Cast member {target_cast_id} not found")
                _debug("cast lookup failed", {"cast_id": target_cast_id})
                return False

            # Create a new person record linked to this cast member via API
            create_payload = {
                "name": cast_member.get("name"),
                "cast_id": target_cast_id,
                "aliases": cast_member.get("aliases", []),
            }
            _debug("creating person", {"payload": create_payload})
            target_person = _api_post(f"/shows/{show_id}/people", create_payload)
            _debug("person create response", target_person)
            if target_person is None:
                st.error("Failed to create person for this cast member. Please check logs.")
                return False
            if not isinstance(target_person, dict) or "person_id" not in target_person:
                st.error("Person creation returned unexpected data (missing person_id).")
                _debug("person create missing person_id", target_person)
                return False

        # Merge source person into target person via API
        merge_payload = {
            "source_person_id": source_person_id,
            "target_person_id": target_person["person_id"],
        }
        expected_cluster_count = len(cluster_ids)
        target_cluster_count_before = len(target_person.get("cluster_ids") or [])

        _debug("merging people", merge_payload)
        _debug(
            "before merge",
            {
                "target_clusters_before": target_cluster_count_before,
                "source_clusters": expected_cluster_count,
                "expected_after": target_cluster_count_before + expected_cluster_count,
            },
        )

        result = _api_post(f"/shows/{show_id}/people/merge", merge_payload)
        _debug("merge response", result)
        if result is None:
            st.error("Failed to assign clusters to the target person. See logs for details.")
            return False

        # CRITICAL: Verify the merge actually worked by re-fetching the target person
        _debug("verifying merge success", {"target_person_id": target_person["person_id"]})
        verify_resp = _safe_api_get(f"/shows/{show_id}/people/{target_person['person_id']}")
        if not verify_resp:
            st.error("Merge API returned success, but failed to verify. Target person not found.")
            return False

        actual_cluster_count = len(verify_resp.get("cluster_ids") or [])
        expected_final_count = target_cluster_count_before + expected_cluster_count

        _debug(
            "after merge",
            {
                "actual_clusters": actual_cluster_count,
                "expected_clusters": expected_final_count,
                "discrepancy": expected_final_count - actual_cluster_count,
            },
        )

        if actual_cluster_count != expected_final_count:
            lost_count = expected_final_count - actual_cluster_count
            _debug("merge verification failed", {"lost_clusters": lost_count})

            # Attempt recovery: try to recreate source person with lost clusters
            recovery_attempted = False
            recovery_success = False
            if cluster_ids:
                _debug("attempting recovery", {"cluster_ids": cluster_ids})
                try:
                    # Get clusters that are NOT in the target person
                    target_cluster_ids = set(verify_resp.get("cluster_ids") or [])
                    lost_cluster_ids = [cid for cid in cluster_ids if cid not in target_cluster_ids]

                    if lost_cluster_ids:
                        # Try to create a recovery person with the lost clusters
                        recovery_payload = {
                            "name": f"RECOVERY_{source_person_id}",
                            "cluster_ids": lost_cluster_ids,
                            "aliases": [f"Lost clusters from merge attempt at {datetime.datetime.now().isoformat()}"],
                        }
                        recovery_person = _api_post(f"/shows/{show_id}/people", recovery_payload)
                        recovery_attempted = True
                        if recovery_person and recovery_person.get("person_id"):
                            recovery_success = True
                            _debug("recovery successful", {"recovery_person": recovery_person})
                except Exception as recovery_exc:
                    _debug("recovery failed", {"error": str(recovery_exc)})

            if recovery_success:
                st.warning(
                    f"⚠️ Merge partially failed - {lost_count} cluster(s) were not transferred.\n\n"
                    f"A recovery person was created with name `RECOVERY_{source_person_id}` "
                    f"containing the lost clusters. Please review and re-assign manually."
                )
            else:
                st.error(
                    f"❌ Merge verification FAILED!\n\n"
                    f"Expected {expected_final_count} clusters after merge "
                    f"(target had {target_cluster_count_before}, adding {expected_cluster_count}), "
                    f"but target now has {actual_cluster_count} clusters.\n\n"
                    f"**{lost_count} cluster(s) may have been LOST during the merge!**\n\n"
                    f"{'Recovery was attempted but failed. ' if recovery_attempted else ''}"
                    f"Lost cluster IDs: {', '.join(cluster_ids[:5])}{'...' if len(cluster_ids) > 5 else ''}\n\n"
                    f"Check API logs for merge_people errors and restore from backup if needed."
                )
            return False

        _debug("merge verified", {"clusters_transferred": expected_cluster_count})
        _invalidate_assignment_caches()
        return True
    except Exception as exc:
        if debug_assign:
            st.exception(exc)
        else:
            st.error(f"Unexpected error during Assign Cluster: {exc}")
        return False


def _bulk_assign_to_new_person(
    ep_id: str,
    show_id: str,
    source_person_id: str,
    new_name: str,
    cluster_ids: List[str],
) -> bool:
    """Create a new person and assign all clusters to them."""
    try:
        # Check if person with this name exists via API
        people_resp = _fetch_people_cached(show_id)
        people = people_resp.get("people", []) if people_resp else []

        # Simple name matching (case-insensitive)
        existing = next((p for p in people if p.get("name", "").lower() == new_name.lower()), None)

        if existing:
            # Merge into existing person
            merge_payload = {
                "source_person_id": source_person_id,
                "target_person_id": existing["person_id"],
            }
            result = _api_post(f"/shows/{show_id}/people/merge", merge_payload)
            if result is not None:
                _invalidate_assignment_caches()
            return result is not None

        # Get source person data
        source = _safe_api_get(f"/shows/{show_id}/people/{source_person_id}")
        if not source:
            st.error(f"Source person {source_person_id} not found")
            return False

        # Create new person via API
        create_payload = {
            "name": new_name,
            "cluster_ids": source.get("cluster_ids", []),
            "aliases": [],
        }
        new_person = _api_post(f"/shows/{show_id}/people", create_payload)
        if not new_person:
            st.error("Failed to create new person")
            return False

        # Delete old person via API
        delete_result = _api_delete(f"/shows/{show_id}/people/{source_person_id}")

        _invalidate_assignment_caches()
        return True
    except Exception as exc:
        st.error(f"Failed to create person: {exc}")
        return False


def _render_people_view(
    ep_id: str,
    show_id: str | None,
    people: List[Dict[str, Any]],
    cluster_lookup: Dict[str, Dict[str, Any]],
    identity_index: Dict[str, Dict[str, Any]],
    season_label: str | None,
    *,
    cast_api_resp: Dict[str, Any] | None = None,
) -> None:
    # Note: Cast Members header is rendered inline with the count in the section below
    if not show_id:
        st.error("Unable to determine show for this episode.")
        return

    # Check for cast filter
    filter_cast_id = st.session_state.get("filter_cast_id")
    filter_cast_name = st.session_state.get("filter_cast_name")
    if filter_cast_id and filter_cast_name:
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.info(f"🔍 Filtering by cast: **{filter_cast_name}**")
            with col2:
                if st.button("Clear Filter", key="clear_cast_filter"):
                    st.session_state.pop("filter_cast_id", None)
                    st.session_state.pop("filter_cast_name", None)
                    st.rerun()

    # Load cluster centroids once for all identity cohesion calculations
    cluster_centroids = _load_cluster_centroids(ep_id)

    cast_resp = cast_api_resp if cast_api_resp is not None else _fetch_cast_cached(show_id, season_label)
    raw_cast_entries = cast_resp.get("cast", []) if cast_resp else []
    # Build lookup by cast_id, filtering out None/empty/whitespace-only cast_ids
    people_by_cast_id = {
        p.get("cast_id"): p
        for p in people
        if p.get("cast_id") and isinstance(p.get("cast_id"), str) and p.get("cast_id").strip()
    }

    deduped_cast_entries: List[Dict[str, Any]] = []
    seen_cast_ids: set[str] = set()
    seen_names: set[str] = set()
    for entry in raw_cast_entries:
        cast_id = entry.get("cast_id") or ""
        name = (entry.get("name") or "").strip()
        norm_name = name.lower()
        if cast_id and cast_id in seen_cast_ids:
            continue
        if not cast_id and norm_name and norm_name in seen_names:
            continue
        if cast_id:
            seen_cast_ids.add(cast_id)
        if norm_name:
            seen_names.add(norm_name)
        deduped_cast_entries.append(entry)

    # Build cast gallery cards by joining cast roster with people metadata
    cast_gallery_cards: List[Dict[str, Any]] = []
    for cast_entry in deduped_cast_entries:
        cast_id = cast_entry.get("cast_id")
        if not cast_id:
            continue
        if filter_cast_id and str(cast_id) != str(filter_cast_id):
            continue
        person = people_by_cast_id.get(cast_id)
        episode_clusters = _episode_cluster_ids(person, ep_id) if person else []
        if not person or not episode_clusters:
            continue
        featured_thumb = cast_entry.get("featured_thumbnail_url")
        if not featured_thumb and person:
            featured_thumb = person.get("rep_crop")
        # Fallback: select best quality crop from episode tracks if no seeded thumbnail
        if not featured_thumb and episode_clusters:
            featured_thumb = _get_best_crop_from_clusters(ep_id, episode_clusters)
        cast_gallery_cards.append(
            {
                "cast": cast_entry,
                "person": person,
                "episode_clusters": episode_clusters,
                "featured_thumbnail": featured_thumb,
            }
        )

    if not people:
        if cast_gallery_cards:
            st.markdown(f"### 🎬 Cast Members ({len(cast_gallery_cards)})")
            st.caption(f"Show-level cast members for {show_id}")
            _render_cast_gallery(ep_id, cast_gallery_cards, cluster_lookup, cluster_centroids)
        st.info("No people found for this show. Run 'Group Clusters (auto)' to create people.")
        return

    # Include legacy cast/person pairs (absent from cast.json) when they have clusters
    seen_cast_ids = {card.get("cast", {}).get("cast_id") for card in cast_gallery_cards}
    for person in people:
        cast_id = person.get("cast_id")
        if not cast_id or cast_id in seen_cast_ids:
            continue
        if filter_cast_id and str(cast_id) != str(filter_cast_id):
            continue
        episode_clusters = _episode_cluster_ids(person, ep_id)
        if not episode_clusters:
            continue
        # Use rep_crop if available, otherwise find best quality crop from tracks
        featured_thumb = person.get("rep_crop")
        if not featured_thumb:
            featured_thumb = _get_best_crop_from_clusters(ep_id, episode_clusters)
        cast_gallery_cards.append(
            {
                "cast": {
                    "cast_id": cast_id,
                    "name": person.get("name") or "(unnamed)",
                    "aliases": person.get("aliases") or [],
                },
                "person": person,
                "episode_clusters": episode_clusters,
                "featured_thumbnail": featured_thumb,
            }
        )
        seen_cast_ids.add(cast_id)

    # Separate people without cast_id but with clusters (auto people)
    episode_auto_people: List[Dict[str, Any]] = []
    for person in people:
        if person.get("cast_id"):
            continue
        if filter_cast_id and str(person.get("person_id") or "") != str(filter_cast_id):
            continue
        episode_clusters = _episode_cluster_ids(person, ep_id)
        if episode_clusters:
            total_tracks, total_faces = _episode_person_counts(episode_clusters, cluster_lookup)
            avg_cohesion = _episode_person_cohesion(ep_id, episode_clusters, cluster_lookup, cluster_centroids)
            episode_auto_people.append(
                {
                    "person": person,
                    "episode_clusters": episode_clusters,
                    "counts": {
                        "clusters": len(episode_clusters),
                        "tracks": total_tracks,
                        "faces": total_faces,
                    },
                    "avg_cohesion": avg_cohesion,
                }
            )

    # ALSO find unassigned clusters (clusters without person_id) to show as suggestions
    # Note: identity_id is just "id_XXXX" without episode prefix. The episode context
    # comes from the fact that we're querying /episodes/{ep_id}/identities
    unassigned_clusters: List[str] = []
    for ident in identity_index.values():
        ident_id = ident.get("identity_id", "")
        # Check if this identity has no person_id (unassigned)
        if not ident.get("person_id"):
            unassigned_clusters.append(ident_id)

    # Sort cast members (for gallery) by name
    cast_gallery_cards.sort(key=lambda card: (card.get("cast", {}).get("name") or "").lower())

    # --- CAST MEMBERS SECTION ---
    if cast_gallery_cards:
        # Styled collapsible header for Cast Members
        cast_header_html = f"""
        <div style="background: linear-gradient(135deg, rgba(33, 150, 243, 0.25), rgba(156, 39, 176, 0.25));
                    border: 1px solid rgba(0, 0, 0, 0.1); border-radius: 8px;
                    padding: 12px 20px; margin-bottom: 8px;
                    display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 24px; font-weight: 600;">🎬 Cast Members ({len(cast_gallery_cards)})</span>
        </div>
        """
        st.markdown(cast_header_html, unsafe_allow_html=True)

        with st.expander("Show/Hide", expanded=True):
            # Load centroids once for identity cohesion calculations
            cast_cluster_centroids = _load_cluster_centroids(ep_id)
            _render_cast_gallery(ep_id, cast_gallery_cards, cluster_lookup, cast_cluster_centroids)

    # --- ARCHIVED ITEMS VIEWER ---
    # Show archived items for this show (deleted people, clusters, tracks)
    show_id_for_archive = ep_id.split("-")[0].upper() if "-" in ep_id else ep_id.upper()
    archive_stats = _safe_api_get(f"/archive/shows/{show_id_for_archive}/stats")
    total_archived = archive_stats.get("total_archived", 0) if archive_stats else 0

    if total_archived > 0:
        with st.expander(f"🗃️ View Archived Items ({total_archived})", expanded=False):
            st.caption(
                "Archived items are deleted faces that will be auto-matched in future episodes. "
                "If the same face appears again, it can be automatically archived."
            )

            # Fetch archived items
            archived_resp = _safe_api_get(
                f"/archive/shows/{show_id_for_archive}",
                params={"limit": 50, "episode_id": ep_id},
            )
            archived_items = archived_resp.get("items", []) if archived_resp else []
            counts = archived_resp.get("counts", {}) if archived_resp else {}

            # Show counts
            stat_cols = st.columns(4)
            stat_cols[0].metric("People", counts.get("people", 0))
            stat_cols[1].metric("Clusters", counts.get("clusters", 0))
            stat_cols[2].metric("Tracks", counts.get("tracks", 0))
            stat_cols[3].metric("Total", total_archived)

            if archived_items:
                st.markdown("##### Recent Archived Items")
                for item in archived_items[:20]:  # Show first 20
                    item_type = item.get("type", "unknown")
                    archive_id = item.get("archive_id", "")
                    name = item.get("name") or item.get("original_id") or archive_id[:12]
                    archived_at = item.get("archived_at", "")[:10]  # Date only
                    reason = item.get("reason", "deleted")
                    rep_crop_url = item.get("rep_crop_url")

                    item_cols = st.columns([1, 3, 2, 1])
                    with item_cols[0]:
                        if rep_crop_url:
                            st.image(rep_crop_url, width=50)
                        else:
                            st.markdown("👤")
                    with item_cols[1]:
                        type_icon = {"person": "👤", "cluster": "🎯", "track": "🔗"}.get(item_type, "📦")
                        st.markdown(f"**{type_icon} {name}**")
                        st.caption(f"{item_type} · {reason} · {archived_at}")
                    with item_cols[2]:
                        if item.get("episode_id"):
                            st.caption(f"From: {item['episode_id']}")
                    with item_cols[3]:
                        if st.button("🗑️", key=f"perm_delete_{archive_id}", help="Permanently delete"):
                            delete_resp = helpers.api_delete(f"/archive/shows/{show_id_for_archive}/{archive_id}")
                            if delete_resp:
                                st.success("Permanently deleted")
                                st.rerun()
            else:
                st.info("No archived items for this episode yet.")

    # --- EPISODE AUTO-PEOPLE SECTION ---
    if episode_auto_people:
        st.markdown("---")

        # Header with sort dropdown
        header_cols = st.columns([3, 1])
        with header_cols[0]:
            st.markdown(f"### 👥 Episode Auto-Clustered People ({len(episode_auto_people)})")
            st.caption(f"People auto-detected in episode {ep_id}")
        with header_cols[1]:
            people_sort = st.selectbox(
                "Sort by:",
                PERSON_SORT_OPTIONS,
                key=f"sort_auto_people_{ep_id}",
                label_visibility="collapsed",
            )

        # Apply sorting using centralized function
        sort_people(episode_auto_people, people_sort)

        # Build options: map cast_id to name
        cast_options = {
            cm.get("cast_id"): cm.get("name") for cm in deduped_cast_entries if cm.get("cast_id") and cm.get("name")
        }

        # Fetch suggestions once (used by cards)
        suggestions_by_cluster = {}
        suggestions_resp = _safe_api_get(f"/episodes/{ep_id}/cluster_suggestions")
        if suggestions_resp:
            for suggestion in suggestions_resp.get("suggestions", []):
                cid = suggestion.get("cluster_id")
                if cid:
                    suggestions_by_cluster[cid] = suggestion

        # Get cast suggestions from facebank (from session state or fetch)
        cast_suggestions_by_cluster = st.session_state.get(f"cast_suggestions:{ep_id}", {})

        # Paginate auto-people to avoid rendering large lists at once
        page_size = 30
        total = len(episode_auto_people)
        total_pages = max(math.ceil(total / page_size), 1)
        page_key = f"auto_people_page_{ep_id}"
        current_page = int(st.session_state.get(page_key, 0))
        current_page = max(0, min(current_page, total_pages - 1))
        start = current_page * page_size
        end = min(start + page_size, total)

        if total_pages > 1:
            nav_cols = st.columns([1, 2, 1])
            with nav_cols[0]:
                if st.button("← Prev", key=f"auto_people_prev_{ep_id}", disabled=current_page == 0):
                    st.session_state[page_key] = current_page - 1
                    st.rerun()
            with nav_cols[1]:
                st.caption(f"Page {current_page + 1} of {total_pages} • Showing {start + 1}-{end} of {total}")
            with nav_cols[2]:
                if st.button("Next →", key=f"auto_people_next_{ep_id}", disabled=current_page >= total_pages - 1):
                    st.session_state[page_key] = current_page + 1
                    st.rerun()

        for entry in episode_auto_people[start:end]:
            person = entry.get("person", {})
            episode_clusters = entry.get("episode_clusters", [])
            avg_cohesion = entry.get("avg_cohesion")
            _render_auto_person_card(
                ep_id,
                show_id,
                person,
                episode_clusters,
                cast_options,
                suggestions_by_cluster,
                cast_suggestions_by_cluster,
                avg_cohesion=avg_cohesion,
            )

    # --- UNASSIGNED CLUSTERS (SUGGESTIONS) SECTION ---
    if unassigned_clusters:
        st.markdown("---")
        header_col1, header_col2, header_col3 = st.columns([3, 1, 1])
        with header_col1:
            st.markdown(f"### 🔍 Unassigned Clusters - Review Suggestions ({len(unassigned_clusters)})")
            st.caption("Clusters detected but not yet assigned to cast members. Review and assign manually.")
        with header_col2:
            if st.button(
                "💾 Save Progress",
                key=f"save_progress_{ep_id}",
                help="Save all current assignments",
            ):
                with st.spinner("Saving progress..."):
                    # Trigger a grouping with manual strategy to persist assignments
                    # This ensures all current assignments are written to people.json and identities.json
                    save_resp = _api_post(f"/episodes/{ep_id}/save_assignments", {})
                    if save_resp and save_resp.get("status") == "success":
                        st.success(f"✅ Saved {save_resp.get('saved_count', 0)} assignment(s)!")
                    else:
                        st.error("Failed to save progress. Check logs.")
        with header_col3:
            if st.button("🔄 Refresh Suggestions", key=f"refresh_suggestions_{ep_id}"):
                with st.spinner("Refreshing suggestions..."):
                    # Use the new endpoint that compares against assigned clusters
                    suggestions_resp = _safe_api_get(f"/episodes/{ep_id}/cluster_suggestions_from_assigned")
                    if suggestions_resp:
                        st.success("Suggestions refreshed!")
                    st.rerun()

        # Build options: map cast_id to name
        cast_options = {
            cm.get("cast_id"): cm.get("name") for cm in deduped_cast_entries if cm.get("cast_id") and cm.get("name")
        }

        # Fetch suggestions from API - now comparing against assigned clusters in this episode
        suggestions_resp = _safe_api_get(f"/episodes/{ep_id}/cluster_suggestions_from_assigned")
        suggestions_by_cluster = {}
        if suggestions_resp:
            # Build set of person_ids for ALL cast members (including those without clusters in this episode)
            # This allows suggestions to point to cast members who don't yet have clusters in this episode
            cast_person_ids = set()

            # Include all people with cast_id from the people list
            for person in people:
                if person.get("cast_id") and person.get("person_id"):
                    cast_person_ids.add(person.get("person_id"))

            # Filter suggestions to only those pointing to cast members
            for suggestion in suggestions_resp.get("suggestions", []):
                cluster_id = suggestion.get("cluster_id")
                suggested_person_id = suggestion.get("suggested_person_id")
                if cluster_id and suggested_person_id in cast_person_ids:
                    suggestions_by_cluster[cluster_id] = suggestion

        # Get cast suggestions from facebank (Enhancement #1) from session state
        cast_suggestions_by_cluster = st.session_state.get(f"cast_suggestions:{ep_id}", {})

        # Sort control for unassigned clusters
        sort_cols = st.columns([3, 1])
        with sort_cols[1]:
            unassigned_sort = st.selectbox(
                "Sort by:",
                UNASSIGNED_CLUSTER_SORT_OPTIONS,
                key=f"sort_unassigned_{ep_id}",
                label_visibility="collapsed",
            )

        # Build cluster dicts for sorting
        cluster_dicts = []
        for cid in unassigned_clusters:
            cluster_info = cluster_lookup.get(cid, {})
            counts = cluster_info.get("counts", {})
            cluster_dicts.append({
                "cluster_id": cid,
                "tracks": counts.get("tracks", 0),
                "faces": counts.get("faces", 0),
                "cohesion": cluster_info.get("cohesion"),
            })

        # Apply sorting using centralized function
        sort_clusters(cluster_dicts, unassigned_sort, cast_suggestions=cast_suggestions_by_cluster)

        # Render each unassigned cluster as a suggestion card
        for cluster_data in cluster_dicts:
            cluster_id = cluster_data["cluster_id"]
            _render_unassigned_cluster_card(
                ep_id,
                show_id,
                cluster_id,
                suggestions_by_cluster.get(cluster_id),
                cast_options,
                cluster_lookup,
                cast_suggestions=cast_suggestions_by_cluster.get(cluster_id),
            )

    # Show message if filtering but nothing found
    if filter_cast_id and not cast_gallery_cards and not episode_auto_people:
        st.warning(f"{filter_cast_name or filter_cast_id} has no clusters in episode {ep_id}.")

    # Show message if no people at all
    if not cast_gallery_cards and not episode_auto_people and not unassigned_clusters and not filter_cast_id:
        st.info("No people with clusters in this episode yet. Run 'Group Clusters (auto)' to create people.")


def _render_person_clusters(
    ep_id: str,
    person_id: str,
    people_lookup: Dict[str, Dict[str, Any]],
    cluster_lookup: Dict[str, Dict[str, Any]],
    identities_payload: Dict[str, Any],
    show_slug: str | None,
    roster_names: List[str],
    cast_options: Dict[str, str] | None = None,
) -> None:
    _render_view_header("person_clusters")
    st.button(
        "← Back to people",
        key="facebank_back_people",
        on_click=lambda: _set_view("people"),
    )
    person = people_lookup.get(person_id)
    if not person:
        st.warning("Selected person not found. Returning to people list.")
        _set_view("people")
        st.rerun()
    name = person.get("name") or "(unnamed)"
    total_clusters = len(person.get("cluster_ids", []) or [])
    episode_clusters = _episode_cluster_ids(person, ep_id)
    st.subheader(f"👤 {name}")
    featured_crop = helpers.resolve_thumb(person.get("rep_crop"))
    if featured_crop:
        st.image(featured_crop, caption="Featured image", width=220)

    if not episode_clusters:
        st.info("No clusters assigned to this person in this episode yet.")
        return

    # Fetch clusters summary to get all tracks across all clusters
    clusters_summary = _safe_api_get(f"/episodes/{ep_id}/people/{person_id}/clusters_summary")
    if not clusters_summary:
        st.error("Failed to load cluster data.")
        return

    # Show local fallback banner if any local files are being used
    _show_local_fallback_banner(clusters_summary)

    # Collect all tracks from all clusters
    # NOTE: clusters_summary already includes track_reps data, no need for separate API calls
    all_tracks = []
    total_tracks = 0
    total_faces = 0
    all_frame_embeddings = []  # Collect all frame embeddings for person-level similarity

    clusters_list = clusters_summary.get("clusters", [])
    for cluster_data in clusters_list:
        total_tracks += cluster_data.get("tracks", 0)
        total_faces += cluster_data.get("faces", 0)

        # Use track_reps data already included in clusters_summary response
        cluster_id = cluster_data.get("cluster_id")
        track_reps = cluster_data.get("track_reps", [])
        for track in track_reps:
            track["cluster_id"] = cluster_id  # Tag with source cluster
            all_tracks.append(track)

    # Pre-fetch identities data for potential person-level similarity scoring
    # (actual computation happens during frame rendering if needed)
    _fetch_identities_cached(ep_id)

    st.caption(f"{len(episode_clusters)} clusters · {total_tracks} tracks · {total_faces} frames")

    # View All Tracks button - compact grid view for outlier detection
    if total_tracks > 0:
        if st.button(
            f"🎭 View All {total_tracks} Tracks (Outlier Detection)",
            key=f"view_all_tracks_{person_id}",
            help="Grid view with one crop per track. Sort by Cast Track Score to find misassigned tracks.",
        ):
            _set_view("cast_tracks", person_id=person_id)
            st.rerun()

    if not all_tracks:
        st.info("No tracks found for this person.")
        return

    def _parse_track_int(track_id_val: Any) -> int | None:
        if isinstance(track_id_val, str) and track_id_val.startswith("track_"):
            track_id_val = track_id_val.replace("track_", "")
        try:
            return int(track_id_val)
        except (TypeError, ValueError):
            return None

    # Precompute track IDs and fetch metadata in batch (cached)
    track_ids: List[int] = []
    for track in all_tracks:
        track_int = _parse_track_int(track.get("track_id"))
        track["track_int"] = track_int
        if track_int is not None:
            track_ids.append(track_int)
    track_meta_map = _fetch_tracks_meta(ep_id, track_ids)

    def _track_meta(track_int: int | None) -> Dict[str, Any]:
        if track_int is None:
            return {}
        meta = track_meta_map.get(track_int)
        if meta is None:
            meta = _fetch_track_detail_cached(ep_id, track_int) or {}
            if meta:
                track_meta_map[track_int] = meta
        return meta or {}

    # Sorting options
    sort_cols = st.columns([3, 1])
    with sort_cols[0]:
        st.markdown(f"**All {len(all_tracks)} Tracks**")
    with sort_cols[1]:
        sort_option = st.selectbox(
            "Sort by:",
            TRACK_SORT_OPTIONS,
            key=f"sort_tracks_{person_id}",
            label_visibility="collapsed",
        )

    # Apply sorting using centralized function with track metadata getter
    sort_tracks(all_tracks, sort_option, track_meta_getter=_track_meta)

    # Render each track as one row showing up to 6 frames
    for track in all_tracks:
        track_id_str = track.get("track_id", "")
        similarity = track.get("similarity")
        cluster_id = track.get("cluster_id", "unknown")
        track_id_int = track.get("track_int")

        # Parse track ID
        track_num = track_id_str
        if isinstance(track_id_str, str) and track_id_str.startswith("track_"):
            track_num = track_id_str.replace("track_", "")

        # Fetch all frames for this track (cached/batch)
        track_meta = _track_meta(track_id_int)
        frames = track_meta.get("frames", []) if track_meta else []
        if not frames and track_id_int is not None:
            track_meta = _track_meta(track_id_int)  # ensure cache retry
            frames = track_meta.get("frames", []) if track_meta else []

        # Sort frames by similarity (lowest first)
        frames_sorted = sorted(
            frames,
            key=lambda f: (f.get("similarity") if f.get("similarity") is not None else 999.0),
        )

        # Take up to 6 frames
        visible_frames = frames_sorted[:6]

        with st.container(border=True):
            # Track header
            badge_html = render_similarity_badge(similarity, SimilarityType.TRACK)
            st.markdown(
                f"**Track {track_num}** {badge_html} · Cluster `{cluster_id}` · {len(frames)} frames",
                unsafe_allow_html=True,
            )

            # Display frames in a single row
            if visible_frames:
                cols = st.columns(len(visible_frames))
                for idx, frame in enumerate(visible_frames):
                    with cols[idx]:
                        # Frames have thumbnail_url or media_url, not crop_url
                        crop_url = frame.get("thumbnail_url") or frame.get("media_url")
                        frame_sim = frame.get("similarity")
                        frame_idx = frame.get("frame_idx", idx)

                        resolved = helpers.resolve_thumb(crop_url)
                        thumb_markup = helpers.thumb_html(resolved, alt=f"Frame {frame_idx}", hide_if_missing=False)
                        st.markdown(thumb_markup, unsafe_allow_html=True)

                        # Show frame similarity badge
                        frame_badge = render_similarity_badge(frame_sim, SimilarityType.FRAME)
                        st.caption(f"F{frame_idx} {frame_badge}", unsafe_allow_html=True)

            # Actions
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                if track_id_int is not None and st.button(
                    "View all frames",
                    key=f"view_track_frames_{person_id}_{track_id_int}",
                ):
                    _set_view(
                        "track",
                        person_id=person_id,
                        identity_id=cluster_id,
                        track_id=track_id_int,
                    )
            with col2:
                if track_id_int is not None and cast_options:
                    # Re-assign track to different cast member
                    with st.popover("🔄 Re-assign Track", use_container_width=True):
                        st.markdown("**Re-assign to Cast Member:**")
                        reassign_key = f"reassign_cast_{person_id}_{track_id_int}"
                        cast_ids = list(cast_options.keys())
                        # Filter out current person's cast_id if they have one
                        current_person = people_lookup.get(person_id, {})
                        current_cast_id = current_person.get("cast_id")
                        available_cast = [cid for cid in cast_ids if cid != current_cast_id]
                        if available_cast:
                            selected_cast_id = st.selectbox(
                                "Select cast member",
                                options=[""] + available_cast,
                                format_func=lambda cid: cast_options.get(cid, "Select...") if cid else "Select cast member...",
                                key=reassign_key,
                                label_visibility="collapsed",
                            )
                            if selected_cast_id and st.button(
                                f"Assign to {cast_options.get(selected_cast_id)}",
                                key=f"reassign_btn_{person_id}_{track_id_int}",
                                type="primary",
                                use_container_width=True,
                            ):
                                # Re-assign via API
                                _assign_track_name(ep_id, track_id_int, cast_options.get(selected_cast_id), show_slug, selected_cast_id)
                        else:
                            st.caption("No other cast members available")
            with col3:
                if track_id_int is not None and st.button(
                    "🗃️ Archive",
                    key=f"delete_track_{person_id}_{track_id_int}",
                    type="secondary",
                ):
                    _archive_track(ep_id, track_id_int)


def _render_cast_all_tracks(
    ep_id: str,
    person_id: str,
    people_lookup: Dict[str, Dict[str, Any]],
    cluster_lookup: Dict[str, Dict[str, Any]],
    identities_payload: Dict[str, Any],
    show_slug: str | None,
) -> None:
    """Render all tracks for a cast member/person as a grid with one crop per track.

    This view is optimized for outlier detection - shows one representative crop per track
    with sorting by Cast Track Score and Track Similarity to find misassigned tracks.
    """
    _render_view_header("cast_tracks")
    st.button(
        "← Back to clusters",
        key="cast_tracks_back_clusters",
        on_click=lambda: _set_view("person_clusters", person_id=person_id),
    )

    person = people_lookup.get(person_id)
    if not person:
        st.warning("Selected person not found. Returning to people list.")
        _set_view("people")
        st.rerun()

    name = person.get("name") or "(unnamed)"
    cast_id = person.get("cast_id")
    episode_clusters = _episode_cluster_ids(person, ep_id)

    st.subheader(f"🎭 All Tracks for {name}")

    if not episode_clusters:
        st.info("No clusters assigned to this person in this episode yet.")
        return

    # Fetch clusters summary to get all tracks across all clusters
    clusters_summary = _safe_api_get(f"/episodes/{ep_id}/people/{person_id}/clusters_summary")
    if not clusters_summary:
        st.error("Failed to load cluster data.")
        return

    # Show local fallback banner if any local files are being used
    _show_local_fallback_banner(clusters_summary)

    # Collect all tracks from all clusters with their embeddings for cross-track scoring
    all_tracks = []
    all_track_embeddings = []  # For computing cast track scores
    track_to_embedding_idx = {}  # Map track index -> embedding index
    total_faces = 0

    clusters_list = clusters_summary.get("clusters", [])
    for cluster_data in clusters_list:
        total_faces += cluster_data.get("faces", 0)
        cluster_id = cluster_data.get("cluster_id")
        track_reps = cluster_data.get("track_reps", [])

        for track in track_reps:
            track["cluster_id"] = cluster_id
            track_idx = len(all_tracks)
            all_tracks.append(track)

            # Collect embedding for cast track score computation
            embedding = track.get("embedding")
            if embedding:
                track_to_embedding_idx[track_idx] = len(all_track_embeddings)
                all_track_embeddings.append(embedding)

    if not all_tracks:
        st.info("No tracks found for this person.")
        return

    # Compute cast_track_score for each track (similarity to other tracks in this person)
    # This measures how well a track "fits" with the other tracks assigned to this cast member
    if len(all_track_embeddings) > 1:
        import numpy as np
        try:
            embeddings_array = np.array(all_track_embeddings)
            # Compute pairwise cosine similarities
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            normalized = embeddings_array / (norms + 1e-10)
            similarity_matrix = normalized @ normalized.T

            # For each track with an embedding, compute average similarity to all other tracks
            for track_idx, emb_idx in track_to_embedding_idx.items():
                row = similarity_matrix[emb_idx]
                # Exclude self-similarity
                other_sims = [row[j] for j in range(len(row)) if j != emb_idx]
                if other_sims:
                    all_tracks[track_idx]["cast_track_score"] = float(np.mean(other_sims))
        except Exception:
            pass  # If computation fails, cast_track_score will be None

    # Parse track IDs
    def _parse_track_int(track_id_val: Any) -> int | None:
        if isinstance(track_id_val, str) and track_id_val.startswith("track_"):
            track_id_val = track_id_val.replace("track_", "")
        try:
            return int(track_id_val)
        except (TypeError, ValueError):
            return None

    for track in all_tracks:
        track["track_int"] = _parse_track_int(track.get("track_id"))

    # Precompute track metadata
    track_ids = [t["track_int"] for t in all_tracks if t.get("track_int") is not None]
    track_meta_map = _fetch_tracks_meta(ep_id, track_ids)

    def _track_meta(track_int: int | None) -> Dict[str, Any]:
        if track_int is None:
            return {}
        meta = track_meta_map.get(track_int)
        if meta is None:
            meta = _fetch_track_detail_cached(ep_id, track_int) or {}
            if meta:
                track_meta_map[track_int] = meta
        return meta or {}

    # Add frame counts to tracks for sorting
    for track in all_tracks:
        meta = _track_meta(track.get("track_int"))
        frame_count = meta.get("faces_count") or meta.get("frames_count") or len(meta.get("frames", []) or [])
        track["_frame_count"] = int(frame_count) if frame_count else 0

    # Sort controls
    st.markdown(f"**{len(all_tracks)} Tracks** · {total_faces} frames total")

    sort_cols = st.columns([2, 2, 1])
    with sort_cols[0]:
        sort_option = st.selectbox(
            "Sort by:",
            CAST_TRACKS_SORT_OPTIONS,
            key=f"cast_tracks_sort_{person_id}",
            label_visibility="collapsed",
        )
    with sort_cols[1]:
        cols_per_row = st.selectbox(
            "Columns:",
            [4, 5, 6, 8],
            index=1,
            key=f"cast_tracks_cols_{person_id}",
            label_visibility="collapsed",
            format_func=lambda x: f"{x} per row",
        )

    # Apply sorting
    sort_tracks(all_tracks, sort_option, track_meta_getter=_track_meta)

    # Render tracks as grid - one crop per track
    for row_start in range(0, len(all_tracks), cols_per_row):
        row_tracks = all_tracks[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for idx, track in enumerate(row_tracks):
            with cols[idx]:
                track_id_str = track.get("track_id", "")
                track_int = track.get("track_int")
                cluster_id = track.get("cluster_id", "unknown")
                track_sim = track.get("similarity")
                cast_track_score = track.get("cast_track_score")
                frame_count = track.get("_frame_count", 0)

                # Get representative crop (API may return crop_url, rep_crop, or thumbnail_url)
                rep_crop = track.get("crop_url") or track.get("rep_crop") or track.get("thumbnail_url")
                resolved = helpers.resolve_thumb(rep_crop)

                # Display thumbnail
                thumb_markup = helpers.thumb_html(resolved, alt=f"Track {track_id_str}", hide_if_missing=False)
                st.markdown(thumb_markup, unsafe_allow_html=True)

                # Track number
                track_num = track_id_str
                if isinstance(track_id_str, str) and track_id_str.startswith("track_"):
                    track_num = track_id_str.replace("track_", "")
                st.markdown(f"**Track {track_num}**")

                # Badges - show both Track Similarity and Cast Track Score
                track_badge = render_similarity_badge(track_sim, SimilarityType.TRACK)
                cast_badge = render_similarity_badge(cast_track_score, SimilarityType.CAST_TRACK) if cast_track_score is not None else '<span style="color: #888;">N/A</span>'
                st.markdown(f"TRK {track_badge} · MATCH {cast_badge}", unsafe_allow_html=True)

                st.caption(f"{frame_count} frames · `{cluster_id[:8]}...`")

                # View frames button
                if track_int is not None:
                    if st.button("View Frames", key=f"cast_tracks_view_{person_id}_{track_int}", use_container_width=True):
                        _set_view(
                            "track",
                            person_id=person_id,
                            identity_id=cluster_id,
                            track_id=track_int,
                        )
                        st.rerun()


def _render_cluster_tracks(
    ep_id: str,
    identity_id: str,
    cluster_lookup: Dict[str, Dict[str, Any]],
    identity_index: Dict[str, Dict[str, Any]],
    show_slug: str | None,
    roster_names: List[str],
    person_id: str | None,
) -> None:
    _render_view_header("cluster_tracks")
    st.button(
        "← Back to clusters",
        key="facebank_back_person_clusters",
        on_click=lambda: _set_view("person_clusters", person_id=person_id),
    )

    # Fetch track representatives (cached for 60s for faster navigation)
    track_reps_data = _fetch_cluster_track_reps_cached(ep_id, identity_id)
    if not track_reps_data:
        st.error("Failed to load track representatives.")
        return

    # Prefetch adjacent clusters for faster navigation
    person_cluster_ids = [
        ident_id
        for ident_id, meta in identity_index.items()
        if meta.get("person_id") == person_id
    ]
    _prefetch_adjacent_clusters(ep_id, identity_id, person_cluster_ids)

    # Show local fallback banner if any local files are being used
    _show_local_fallback_banner(track_reps_data)

    people_payload = _fetch_people_cached(show_slug) if show_slug else None
    people = people_payload.get("people", []) if people_payload else []
    people_lookup = {p.get("person_id"): p for p in people if p.get("person_id")}
    cast_payload = _fetch_cast_cached(show_slug) if show_slug else None
    cast_lookup = {c.get("cast_id"): c for c in (cast_payload.get("cast") if cast_payload else []) if c.get("cast_id")}

    identity_meta = identity_index.get(identity_id, {})
    display_name = identity_meta.get("name")
    label = identity_meta.get("label")

    tracks_count = track_reps_data.get("total_tracks", 0)
    cohesion = track_reps_data.get("cohesion")
    track_reps = track_reps_data.get("tracks", [])

    # Header: Cluster ID
    st.subheader(f"Cluster {identity_id}")

    # Show assigned person info
    cluster_person_id = identity_meta.get("person_id")
    if display_name:
        st.caption(f"👤 Assigned to: **{display_name}**")
    elif cluster_person_id:
        # Look up person's name from registry
        person_record = people_lookup.get(cluster_person_id, {})
        person_name = person_record.get("name")
        if person_name:
            st.caption(f"👤 Assigned to: **{person_name}**")
        else:
            st.caption(f"👤 Assigned to: `{cluster_person_id}`")

    # Cohesion badge (with color) directly under the ID
    if cohesion is not None:
        cohesion_badge = render_similarity_badge(cohesion, SimilarityType.CLUSTER)
        st.markdown(f"**Cluster Cohesion:** {cohesion_badge}", unsafe_allow_html=True)

    # Tracks count
    st.caption(f"**{tracks_count}** track(s)")

    _identity_name_controls(
        ep_id=ep_id,
        identity={
            "identity_id": identity_id,
            "name": display_name,
            "label": label,
        },
        show_slug=show_slug,
        roster_names=roster_names,
        prefix=f"cluster_tracks_{identity_id}",
    )

    # Display all track representatives with similarity scores
    if not track_reps:
        st.info("No track representatives available.")
        return

    # Track sorting controls
    track_sort_cols = st.columns([3, 1])
    with track_sort_cols[0]:
        st.markdown(f"**All {len(track_reps)} Track(s) with Similarity Scores:**")
    with track_sort_cols[1]:
        cluster_track_sort = st.selectbox(
            "Sort by:",
            TRACK_SORT_OPTIONS,
            key=f"sort_cluster_tracks_{identity_id}",
            label_visibility="collapsed",
        )

    # Apply sorting
    sort_tracks(track_reps, cluster_track_sort)
    prev_identity_key = "cast_select_active_identity"
    if st.session_state.get(prev_identity_key) != identity_id:
        for key in list(st.session_state.keys()):
            if key.startswith(f"cluster_move_select_{identity_id}_"):
                st.session_state.pop(key, None)
        st.session_state[prev_identity_key] = identity_id
    cast_entries_sorted = sorted(
        [(cast_entry.get("name") or cast_id, cast_id) for cast_id, cast_entry in cast_lookup.items()],
        key=lambda opt: opt[0].lower(),
    )
    move_options: List[tuple[str, str]] = [("Select Cast Member", "")] + cast_entries_sorted
    move_options.append(("➕ Add New Cast Member", "__new_cast__"))

    # --- Bulk Selection UI ---
    bulk_sel_key = f"bulk_track_sel::{identity_id}"
    if bulk_sel_key not in st.session_state:
        st.session_state[bulk_sel_key] = set()
    selected_tracks: set = st.session_state[bulk_sel_key]

    # Collect all valid track IDs for select all
    all_track_ids: List[int] = []
    for tr in track_reps:
        tid_str = tr.get("track_id", "")
        if isinstance(tid_str, str) and tid_str.startswith("track_"):
            tid_str = tid_str.replace("track_", "")
        try:
            all_track_ids.append(int(tid_str))
        except (TypeError, ValueError):
            pass

    # --- Quick Assign Entire Cluster ---
    with st.container(border=True):
        st.markdown("### 🎯 Assign Entire Cluster")
        st.caption(f"Assign all {len(all_track_ids)} track(s) in this cluster to a cast member")

        assign_all_col1, assign_all_col2 = st.columns([2, 1])
        with assign_all_col1:
            assign_all_cast_key = f"assign_all_cast_select_{identity_id}"
            assign_all_choice = st.selectbox(
                "Cast member",
                move_options,
                format_func=lambda opt: opt[0],
                index=0,
                key=assign_all_cast_key,
                label_visibility="collapsed",
            )
            assign_all_cast_id = assign_all_choice[1] if assign_all_choice else None
            assign_all_cast_name = assign_all_choice[0] if assign_all_choice else None

            if assign_all_cast_id == "__new_cast__":
                new_all_cast_key = f"new_all_cast_name_{identity_id}"
                new_all_name = st.text_input(
                    "New cast member name",
                    key=new_all_cast_key,
                    placeholder="Enter cast member name",
                )
                if new_all_name and new_all_name.strip():
                    if st.button(
                        f"Create & Assign All {len(all_track_ids)} Track(s)",
                        key=f"create_assign_all_{identity_id}",
                        type="primary",
                        use_container_width=True,
                    ):
                        cast_resp = _api_post(f"/shows/{show_slug}/cast", {"name": new_all_name.strip()})
                        if cast_resp and cast_resp.get("cast_id"):
                            new_cast_id = cast_resp.get("cast_id")
                            st.toast(f"Created cast member '{new_all_name.strip()}'")
                            _bulk_assign_tracks(ep_id, all_track_ids, new_all_name.strip(), show_slug, new_cast_id)
                        else:
                            st.error("Failed to create cast member.")

        with assign_all_col2:
            if assign_all_cast_id and assign_all_cast_id != "__new_cast__" and assign_all_cast_name:
                if st.button(
                    f"Assign All {len(all_track_ids)}",
                    key=f"assign_all_btn_{identity_id}",
                    type="primary",
                    use_container_width=True,
                ):
                    _bulk_assign_tracks(ep_id, all_track_ids, assign_all_cast_name, show_slug, assign_all_cast_id)

    # --- Selective Bulk Assignment (expandable) ---
    with st.expander("📦 Select specific tracks to assign", expanded=False):
        sel_col1, sel_col2, sel_col3 = st.columns([1, 1, 2])
        with sel_col1:
            if st.button("☑️ Select All", key=f"bulk_select_all_{identity_id}", use_container_width=True):
                st.session_state[bulk_sel_key] = set(all_track_ids)
                st.rerun()
        with sel_col2:
            if st.button("☐ Deselect All", key=f"bulk_deselect_all_{identity_id}", use_container_width=True):
                st.session_state[bulk_sel_key] = set()
                st.rerun()
        with sel_col3:
            st.caption(f"**{len(selected_tracks)}** of {len(all_track_ids)} tracks selected")

        if selected_tracks:
            st.markdown("---")
            assign_col1, assign_col2 = st.columns([2, 1])
            with assign_col1:
                bulk_cast_key = f"bulk_cast_select_{identity_id}"
                bulk_choice = st.selectbox(
                    "Assign selected tracks to",
                    move_options,
                    format_func=lambda opt: opt[0],
                    index=0,
                    key=bulk_cast_key,
                )
                bulk_cast_id = bulk_choice[1] if bulk_choice else None
                bulk_cast_name = bulk_choice[0] if bulk_choice else None

                if bulk_cast_id == "__new_cast__":
                    new_bulk_cast_key = f"new_bulk_cast_name_{identity_id}"
                    new_bulk_name = st.text_input(
                        "New cast member name",
                        key=new_bulk_cast_key,
                        placeholder="Enter cast member name",
                    )
                    if new_bulk_name and new_bulk_name.strip():
                        if st.button(
                            f"Create & Assign {len(selected_tracks)} Track(s)",
                            key=f"bulk_create_assign_{identity_id}",
                            type="primary",
                        ):
                            cast_resp = _api_post(f"/shows/{show_slug}/cast", {"name": new_bulk_name.strip()})
                            if cast_resp and cast_resp.get("cast_id"):
                                new_cast_id = cast_resp.get("cast_id")
                                st.toast(f"Created cast member '{new_bulk_name.strip()}'")
                                _bulk_assign_tracks(
                                    ep_id, list(selected_tracks), new_bulk_name.strip(), show_slug, new_cast_id
                                )
                            else:
                                st.error("Failed to create cast member.")

            with assign_col2:
                if bulk_cast_id and bulk_cast_id != "__new_cast__" and bulk_cast_name:
                    if st.button(
                        f"Assign {len(selected_tracks)} Track(s)",
                        key=f"bulk_assign_btn_{identity_id}",
                        type="primary",
                        use_container_width=True,
                    ):
                        _bulk_assign_tracks(ep_id, list(selected_tracks), bulk_cast_name, show_slug, bulk_cast_id)
    st.markdown("---")

    # Sort tracks by similarity (lowest to highest) - worst matches first for easier review
    sorted_track_reps = sorted(
        track_reps,
        key=lambda t: t.get("similarity") if t.get("similarity") is not None else 999.0,
    )

    # Render tracks in grid
    for row_start in range(0, len(sorted_track_reps), MAX_TRACKS_PER_ROW):
        row_tracks = sorted_track_reps[row_start : row_start + MAX_TRACKS_PER_ROW]
        cols = st.columns(len(row_tracks))

        for idx, track_rep in enumerate(row_tracks):
            with cols[idx]:
                track_id_str = track_rep.get("track_id", "")
                similarity = track_rep.get("similarity")
                crop_url = track_rep.get("crop_url")
                checkbox_key = f"track_bulk::{identity_id}::{track_id_str}"

                # Parse track ID for display and operations
                if isinstance(track_id_str, str) and track_id_str.startswith("track_"):
                    track_num = track_id_str.replace("track_", "")
                    try:
                        track_id_int = int(track_num)
                    except (TypeError, ValueError):
                        track_id_int = None
                else:
                    track_num = track_id_str
                    try:
                        track_id_int = int(track_id_str)
                    except (TypeError, ValueError):
                        track_id_int = None

                # Display thumbnail
                resolved = helpers.resolve_thumb(crop_url)
                thumb_markup = helpers.thumb_html(resolved, alt=f"Track {track_num}", hide_if_missing=False)
                st.markdown(thumb_markup, unsafe_allow_html=True)

                # Checkbox for bulk selection
                if track_id_int is not None:
                    is_selected = track_id_int in selected_tracks
                    if st.checkbox(
                        f"Select",
                        value=is_selected,
                        key=checkbox_key,
                        help="Select for bulk assignment",
                    ):
                        if track_id_int not in selected_tracks:
                            selected_tracks.add(track_id_int)
                    else:
                        selected_tracks.discard(track_id_int)

                # Display track ID and similarity badge
                badge_html = render_similarity_badge(similarity, SimilarityType.TRACK)
                st.markdown(f"Track {track_num} {badge_html}", unsafe_allow_html=True)

                # Actions
                if track_id_int is not None:
                    if st.button(
                        "View frames",
                        key=f"view_track_{identity_id}_{track_id_int}",
                    ):
                        _set_view(
                            "track",
                            person_id=person_id,
                            identity_id=identity_id,
                            track_id=track_id_int,
                        )

                    if move_options:
                        cast_select_key = f"cluster_move_select_{identity_id}_{track_id_int}"
                        prev_value = st.session_state.get(cast_select_key)
                        default_index = 0
                        if prev_value and isinstance(prev_value, (list, tuple)) and len(prev_value) >= 2:
                            prev_cast = prev_value[1]
                            for idx_opt, opt in enumerate(move_options):
                                if opt[1] == prev_cast:
                                    default_index = idx_opt
                                    break
                        choice = st.selectbox(
                            "Move to",
                            move_options,
                            format_func=lambda opt: opt[0],
                            index=default_index,
                            key=cast_select_key,
                        )
                        cast_choice = choice[1] if choice else None
                        cast_name = choice[0] if choice else None
                        if cast_choice == "__new_cast__":
                            new_cast_key = f"new_cast_name_{identity_id}_{track_id_int}"
                            new_name = st.text_input(
                                "New cast member name",
                                key=new_cast_key,
                                placeholder="Enter cast member name",
                            )
                            if new_name and new_name.strip():
                                if st.button(
                                    "Create & Assign",
                                    key=f"create_assign_cast_{identity_id}_{track_id_int}",
                                ):
                                    _create_and_assign_to_new_cast(
                                        ep_id, track_id_int, new_name.strip(), show_slug
                                    )
                        elif cast_choice and cast_name:
                            # Create a NEW cluster for this track under the cast member
                            # (using assign_track_name splits track into new cluster with cast association)
                            if st.button(
                                "Assign",
                                key=f"cluster_move_btn_{identity_id}_{track_id_int}",
                            ):
                                _assign_track_name(ep_id, track_id_int, cast_name, show_slug, cast_choice)

                    if st.button(
                        "🗃️ Archive",
                        key=f"cluster_delete_btn_{identity_id}_{track_id_int}",
                        type="secondary",
                    ):
                        _archive_track(ep_id, track_id_int)

    # --- Export to Facebank (at the bottom) ---
    person_id_for_export = identity_meta.get("person_id")
    if person_id_for_export:
        st.markdown("---")
        with st.container(border=True):
            st.markdown("### 💾 Export to Facebank")
            st.caption(
                "Export high-quality seed frames to permanent facebank for cross-episode similarity matching. "
                f"This will save the best frames (up to 20) for person **{display_name or person_id_for_export}**."
            )

            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("✅ Requirements: Detection score ≥0.75 · Sharpness ≥15 · Similarity ≥0.70")
            with col2:
                if st.button(
                    "💾 Export Seeds", key=f"export_seeds_{identity_id}", use_container_width=True, type="primary"
                ):
                    with st.spinner(f"Selecting and exporting seeds for {display_name or person_id_for_export}..."):
                        export_resp = _api_post(f"/episodes/{ep_id}/identities/{identity_id}/export_seeds", {})
                        if export_resp and export_resp.get("status") == "success":
                            seeds_count = export_resp.get("seeds_exported", 0)
                            seeds_path = export_resp.get("seeds_path", "")
                            st.success(
                                f"✅ Exported {seeds_count} high-quality seeds to facebank!\n\n" f"Path: `{seeds_path}`"
                            )
                            st.info("💡 Tip: Run similarity refresh to update cross-episode matching.")
                        else:
                            st.error("Failed to export seeds. Check logs for details.")


def _render_track_view(ep_id: str, track_id: int, identities_payload: Dict[str, Any]) -> None:
    _render_view_header("track")
    st.button(
        "← Back to tracks",
        key="facebank_back_tracks",
        on_click=lambda: _set_view(
            "cluster_tracks",
            person_id=st.session_state.get("selected_person"),
            identity_id=st.session_state.get("selected_identity"),
        ),
    )
    st.markdown(f"### Track {track_id}")
    identities = identities_payload.get("identities", [])
    identity_lookup = {identity.get("identity_id"): identity for identity in identities}
    current_identity = st.session_state.get("selected_identity")

    # Show assigned person/cluster info
    show_slug = _episode_show_slug(ep_id)
    if current_identity:
        identity_data = identity_lookup.get(current_identity, {})
        assigned_name = identity_data.get("name")
        person_id = identity_data.get("person_id")
        if assigned_name:
            st.caption(f"👤 Assigned to: **{assigned_name}**")
        elif person_id:
            # Look up person's name from registry
            people_payload = _fetch_people_cached(show_slug) if show_slug else None
            people = people_payload.get("people", []) if people_payload else []
            people_lookup = {p.get("person_id"): p for p in people if p.get("person_id")}
            person_record = people_lookup.get(person_id, {})
            person_name = person_record.get("name")
            if person_name:
                st.caption(f"👤 Assigned to: **{person_name}**")
            else:
                st.caption(f"👤 Assigned to: `{person_id}`")
        else:
            st.caption(f"📦 Cluster: `{current_identity}`")
    roster_names = _fetch_roster_names(show_slug)
    sample_key = f"track_sample_{ep_id}_{track_id}"
    sample_seeded = sample_key in st.session_state
    if not sample_seeded:
        st.session_state[sample_key] = 1
    sample_prev_key = f"{sample_key}::prev"
    prev_sample = st.session_state.get(sample_prev_key, st.session_state[sample_key])
    sample_col, page_col, info_col = st.columns([1, 1, 2])
    with sample_col:
        sample = int(
            st.slider(
                "Sample every N crops",
                min_value=1,
                max_value=20,
                value=st.session_state[sample_key],
                key=sample_key,
            )
        )
    sample_changed = sample != prev_sample
    st.session_state[sample_prev_key] = sample
    page_key = f"track_page_{ep_id}_{track_id}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    page_size_key = f"track_page_size_{ep_id}_{track_id}"
    if page_size_key not in st.session_state:
        st.session_state[page_size_key] = _recommended_page_size(st.session_state[sample_key], None)
    page_size = int(st.session_state[page_size_key])
    if sample_changed:
        st.session_state[page_key] = 1
        _reset_track_media_state(ep_id, track_id)
    # Read page value for widget default BEFORE rendering widget
    widget_default_page = int(st.session_state[page_key])
    with page_col:
        st.number_input(
            "Page",
            min_value=1,
            value=widget_default_page,
            step=1,
            key=page_key,
        )
    # Read final page value AFTER widget has potentially updated session state
    page = int(st.session_state[page_key])

    # Lazy loading: defer frame fetch until user requests it
    frames_load_key = f"load_frames_{ep_id}_{track_id}"
    if not st.session_state.get(frames_load_key, False):
        from ui_helpers import track_skeleton_html

        st.markdown(track_skeleton_html(12), unsafe_allow_html=True)
        if st.button("Load Track Frames", key=f"load_btn_{track_id}", type="primary"):
            st.session_state[frames_load_key] = True
            st.rerun()
        st.caption("Click to load frame details for this track")
        return

    frames_payload = _fetch_track_frames(ep_id, track_id, sample=sample, page=page, page_size=page_size)
    frames = frames_payload.get("items", [])
    mismatched_frames = [frame for frame in frames if frame.get("track_id") not in (None, track_id)]
    if mismatched_frames:
        frames = [frame for frame in frames if frame.get("track_id") in (None, track_id)]
    frames, missing_faces = scope_track_frames(frames, track_id)
    debug_frames = os.getenv("SCREENALYTICS_DEBUG_TRACK_FRAMES") == "1" or st.session_state.get(
        "debug_track_frames"
    )
    if debug_frames:
        st.write("DEBUG raw track frames (first 5 items)")
        st.write(frames_payload.get("items", [])[:5])
        st.write(f"DEBUG scoped frames for track {track_id}")
        st.write(frames[:5])
        for line in track_faces_debug(frames, track_id):
            st.write(line)
    best_frame_idx = best_track_frame_idx(frames, track_id, frames_payload.get("best_frame_idx"))
    total_sampled = int(frames_payload.get("total") or 0)
    # Preserve zero: only use total_sampled if total_frames is None
    total_frames_raw = frames_payload.get("total_frames")
    total_frames = int(total_frames_raw) if total_frames_raw is not None else total_sampled
    max_page = max(1, math.ceil(total_sampled / page_size)) if total_sampled else 1

    candidate_frames = total_frames or total_sampled
    recommended_sample = _suggest_track_sample(candidate_frames)
    if not sample_seeded and sample == 1 and recommended_sample > 1:
        st.session_state[sample_key] = recommended_sample
        st.session_state[sample_prev_key] = recommended_sample
        st.session_state[page_key] = 1
        _reset_track_media_state(ep_id, track_id)
        st.rerun()

    recommended_page_size = _recommended_page_size(sample, total_sampled)
    if page_size != recommended_page_size:
        st.session_state[page_size_key] = recommended_page_size
        st.session_state[page_key] = 1
        st.rerun()
    page_size = int(st.session_state[page_size_key])
    with info_col:
        st.caption(f"Auto page size: up to {page_size} sampled frames per page (adaptive)")

    if page > max_page:
        st.session_state[page_key] = max_page
        st.rerun()

    # Reorder frames to show best-quality frame first (if present in current page)
    if best_frame_idx is not None and frames:
        best_frame = None
        best_frame_position = None
        for i, frame in enumerate(frames):
            if frame.get("frame_idx") == best_frame_idx:
                best_frame = frame
                best_frame_position = i
                break

        if best_frame is not None and best_frame_position is not None and best_frame_position > 0:
            # Move best frame to the front
            frames = [best_frame] + frames[:best_frame_position] + frames[best_frame_position + 1 :]
    nav_cols = st.columns([1, 1, 3])
    with nav_cols[0]:
        if st.button("Prev page", key=f"track_prev_{track_id}", disabled=page <= 1):
            st.session_state[page_key] = max(1, page - 1)
            st.rerun()
    with nav_cols[1]:
        if st.button("Next page", key=f"track_next_{track_id}", disabled=page >= max_page):
            st.session_state[page_key] = min(max_page, page + 1)
            st.rerun()
    shown = len(frames)
    summary = (
        f"Frames shown: {shown} / {total_sampled or 0} (page {page}/{max_page}) · "
        f"Sample every {sample} · up to {page_size} per page (auto)"
    )
    if total_frames:
        summary += f" · Faces tracked: {total_frames}"
    nav_cols[2].caption(summary)
    if mismatched_frames:
        st.warning(f"Dropped {len(mismatched_frames)} frame(s) that were tagged to a different track.")
    if missing_faces:
        missing_labels = ", ".join(str(idx) for idx in sorted({m for m in missing_faces if m is not None}))
        st.warning(
            f"No faces found for track {track_id} in frame(s): {missing_labels}. Dropped mismatched faces."
        )

    # Carousel preview of all frames (4:5 aspect ratio with arrows)
    if frames:
        carousel_frames = []
        for frame_meta in frames:
            # Get thumbnail URL: prefer face-level, fall back to frame-level
            thumb_url = None
            similarity = None
            raw_faces = frame_meta.get("faces") if isinstance(frame_meta.get("faces"), list) else []
            faces_for_track = [face for face in raw_faces if coerce_int(face.get("track_id")) == track_id]
            if faces_for_track:
                best_face = faces_for_track[0]
                thumb_url = best_face.get("media_url") or best_face.get("thumbnail_url")
                similarity = best_face.get("similarity")
            # Fall back to frame-level URL (this is what the grid uses)
            if not thumb_url:
                thumb_url = frame_meta.get("thumbnail_url") or frame_meta.get("media_url")
            if not similarity:
                similarity = frame_meta.get("similarity")
            if thumb_url:
                carousel_frames.append({
                    "crop_url": helpers.resolve_thumb(thumb_url),
                    "frame_idx": frame_meta.get("frame_idx"),
                    "similarity": similarity,
                })
        if carousel_frames:
            from ui_helpers import track_carousel_html
            import streamlit.components.v1 as components
            # Use components.html for JavaScript to work
            carousel_html = track_carousel_html(track_id, carousel_frames)
            # Height: 150px thumb + 16px margin top + 8px margin bot + 8px padding + 20px info = ~220px
            components.html(carousel_html, height=230, scrolling=False)
        elif debug_frames:
            st.warning(f"Carousel: No frames with valid URLs found for track {track_id}")

    _render_track_media_section(ep_id, track_id, sample=sample)
    if current_identity:
        assign_container = st.container(border=True)
        with assign_container:
            st.markdown(f"**Assign Track {track_id} to Cast Name**")
            identity_meta = identity_lookup.get(current_identity, {})
            current_name = identity_meta.get("name") or ""
            if not show_slug:
                st.info("Show slug missing; unable to assign roster names.")
            else:
                # Fetch cast roster to get cast_ids
                cast_payload = _safe_api_get(f"/shows/{show_slug}/cast")
                cast_members = cast_payload.get("cast", []) if cast_payload else []
                # Build name → cast_id mapping
                cast_id_by_name = {cm.get("name"): cm.get("cast_id") for cm in cast_members if cm.get("name")}

                resolved = _name_choice_widget(
                    label="Cast name",
                    key_prefix=f"track_assign_{current_identity}_{track_id}",
                    roster_names=roster_names,
                    current_name=current_name,
                )
                disabled = not resolved or resolved == current_name
                if st.button(
                    "Save cast name",
                    key=f"track_assign_save_{current_identity}_{track_id}",
                    disabled=disabled,
                ):
                    # Look up cast_id for the resolved name
                    resolved_cast_id = cast_id_by_name.get(resolved)
                    _assign_track_name(ep_id, track_id, resolved, show_slug, resolved_cast_id)
    integrity = _safe_api_get(f"/episodes/{ep_id}/tracks/{track_id}/integrity")
    if integrity and not integrity.get("ok"):
        st.warning(
            "Crops on disk are missing for this track. Faces manifest"
            f"={integrity.get('faces_manifest', 0)} · crops={integrity.get('crops_files', 0)}"
        )
    action_cols = st.columns([1.0, 1.0, 1.0])
    with action_cols[0]:
        targets = [ident["identity_id"] for ident in identities if ident["identity_id"] != current_identity]
        if targets:
            move_select_key = f"track_view_move_{ep_id}_{track_id}_{current_identity or 'none'}"
            target_choice = st.selectbox(
                "Move entire track",
                targets,
                key=move_select_key,
            )
            if st.button("Move track", key=f"{move_select_key}_btn"):
                _move_track(ep_id, track_id, target_choice)
    with action_cols[1]:
        if st.button("Remove from identity", key=f"track_view_remove_{track_id}"):
            _move_track(ep_id, track_id, None)
    with action_cols[2]:
        if st.button("🗃️ Archive track", key=f"track_view_delete_{track_id}"):
            _archive_track(ep_id, track_id)

    selection_store: Dict[int, set[int]] = st.session_state.setdefault("track_frame_selection", {})
    track_selection = selection_store.setdefault(track_id, set())
    selected_frames: List[int] = []
    if frames:
        st.caption("Select frames to move or delete.")
        cols_per_row = 6
        for row_start in range(0, len(frames), cols_per_row):
            row_frames = frames[row_start : row_start + cols_per_row]
            row_cols = st.columns(len(row_frames))
            for idx, frame_meta in enumerate(row_frames):
                raw_faces = frame_meta.get("faces") if isinstance(frame_meta.get("faces"), list) else []
                faces_for_track = [face for face in raw_faces if coerce_int(face.get("track_id")) == track_id]
                invalid_faces = [face for face in raw_faces if coerce_int(face.get("track_id")) not in (track_id,)]
                frame_idx = frame_meta.get("frame_idx")

                # CRITICAL: Never display frames with faces from other tracks
                # This prevents showing the wrong person's thumbnail
                if invalid_faces:
                    if debug_frames:
                        st.warning(
                            f"Skipping frame {frame_idx} with mismatched track faces: "
                            f"{[coerce_int(f.get('track_id')) for f in invalid_faces]}"
                        )
                    continue

                if not faces_for_track:
                    if debug_frames:
                        st.warning(f"Skipping frame {frame_idx}: no faces for track {track_id}")
                    continue

                # Best face MUST be from this track (sorted by quality in scope_track_frames)
                best_face = faces_for_track[0]

                # Sanity check: ensure best_face actually belongs to this track
                best_face_track_id = coerce_int(best_face.get("track_id"))
                if best_face_track_id != track_id:
                    if debug_frames:
                        st.error(
                            f"BUG: best_face has track_id={best_face_track_id} but expected {track_id} "
                            f"for frame {frame_idx}. Skipping."
                        )
                    continue
                face_id = best_face.get("face_id")
                try:
                    frame_idx_int = int(frame_idx)
                except (TypeError, ValueError):
                    frame_idx_int = None

                # Get thumbnail URL from the specific face, NOT from frame-level data
                # Face-level URLs ensure we always show the correct person for this track
                thumb_url = best_face.get("media_url") or best_face.get("thumbnail_url")

                # Fallback to frame_meta only if face has no URLs (shouldn't happen after API fix)
                if not thumb_url:
                    thumb_url = frame_meta.get("media_url") or frame_meta.get("thumbnail_url")
                    if debug_frames and thumb_url:
                        st.warning(
                            f"Frame {frame_idx}: using frame-level URL (face-level URL missing). "
                            f"This may show wrong person if multiple tracks in frame."
                        )
                skip_reason = best_face.get("skip") or frame_meta.get("skip")
                with row_cols[idx]:
                    caption = f"Frame {frame_idx}" if frame_idx is not None else (face_id or "frame")
                    resolved_thumb = helpers.resolve_thumb(thumb_url)
                    thumb_markup = helpers.thumb_html(resolved_thumb, alt=caption, hide_if_missing=True)
                    if thumb_markup:
                        st.markdown(thumb_markup, unsafe_allow_html=True)
                    else:
                        st.caption("Crop unavailable.")
                    st.caption(caption)
                    face_track_id = coerce_int(best_face.get("track_id")) or coerce_int(frame_meta.get("track_id")) or track_id
                    st.caption(f":grey[Track: {face_track_id}]")
                    other_tracks = frame_meta.get("other_tracks") or []
                    if isinstance(other_tracks, list):
                        scoped_tracks = [str(t) for t in other_tracks if t is not None]
                        if scoped_tracks:
                            st.caption(f":grey[Other track(s) in this frame: {', '.join(scoped_tracks)}]")
                    # Show best-quality frame badge
                    if frame_idx == best_frame_idx:
                        st.markdown(
                            '<span style="background-color: #4CAF50; color: white; padding: 2px 6px; '
                            'border-radius: 3px; font-size: 0.8em; font-weight: bold;">★ BEST QUALITY</span>',
                            unsafe_allow_html=True,
                        )
                    # Show similarity badge if available
                    similarity = best_face.get("similarity") if isinstance(best_face, dict) else None
                    if similarity is None:
                        similarity = frame_meta.get("similarity")
                    if similarity is not None:
                        similarity_badge = render_similarity_badge(similarity, SimilarityType.FRAME)
                        st.markdown(similarity_badge, unsafe_allow_html=True)
                        # Show quality score if available
                        quality = best_face.get("quality") or frame_meta.get("quality")
                        if quality and isinstance(quality, dict):
                            quality_score_value = quality.get("score")
                            if quality_score_value is not None:
                                quality_pct = int(quality_score_value * 100)
                                if quality_score_value >= 0.85:
                                    quality_color = "#2E7D32"
                                elif quality_score_value >= 0.60:
                                    quality_color = "#81C784"
                                else:
                                    quality_color = "#EF5350"
                                st.markdown(
                                    f'<span style="background-color: {quality_color}; color: white; padding: 2px 6px; '
                                    f'border-radius: 3px; font-size: 0.75em;">Q: {quality_pct}%</span>',
                                    unsafe_allow_html=True,
                                )
                    if skip_reason:
                        st.markdown(f":red[⚠ invalid crop] {skip_reason}")
                    if frame_idx_int is None:
                        continue
                    key = f"face_select_{track_id}_{frame_idx_int}"
                    checked = st.checkbox(
                        "Select",
                        value=frame_idx_int in track_selection,
                        key=key,
                    )
                    if checked:
                        track_selection.add(frame_idx_int)
                    else:
                        track_selection.discard(frame_idx_int)
                    # Full-frame overlay button
                    overlay_key = f"ff_overlay_{ep_id}_{frame_idx_int}_{track_id}"
                    if st.button("🖼️ FF Overlay", key=overlay_key, help="Generate full-frame overlay with all face bboxes"):
                        with st.spinner("Generating overlay..."):
                            overlay_resp = _api_post(
                                f"/episodes/{ep_id}/frames/{frame_idx_int}/overlay",
                                {"highlight_track_id": track_id},
                            )
                            if overlay_resp and overlay_resp.get("url"):
                                st.session_state[f"overlay_url_{ep_id}_{frame_idx_int}"] = overlay_resp.get("url")
                                st.success(f"Overlay created: {len(overlay_resp.get('tracks', []))} track(s)")
                            else:
                                st.error("Failed to generate overlay")
                    # Show overlay link if already generated
                    overlay_url = st.session_state.get(f"overlay_url_{ep_id}_{frame_idx_int}")
                    if overlay_url:
                        st.markdown(f"[View Overlay]({overlay_url})", unsafe_allow_html=True)
        selected_frames = sorted(track_selection)
        st.caption(f"{len(selected_frames)} frame(s) selected.")
    else:
        if total_sampled:
            st.info("No frames on this page. Try a smaller page number or lower sampling.")
        else:
            st.info("No frames recorded for this track yet.")

    if selected_frames:
        identity_values = [None] + [ident["identity_id"] for ident in identities]
        identity_labels = ["Create new identity"] + [
            f"{ident['identity_id']} · {(ident.get('name') or ident.get('label') or ident['identity_id'])}"
            for ident in identities
        ]
        identity_idx = st.selectbox(
            "Send selected frames to identity",
            list(range(len(identity_values))),
            format_func=lambda idx: identity_labels[idx],
            key=f"track_identity_choice_{track_id}",
        )
        target_identity_id = identity_values[identity_idx]
        target_name = _name_choice_widget(
            label="Or assign roster name",
            key_prefix=f"track_reassign_{track_id}",
            roster_names=roster_names,
            current_name="",
            text_label="New roster name",
        )
        move_disabled = not (target_identity_id or target_name)
        action_cols = st.columns([1, 1])
        if action_cols[0].button(
            "Move selected",
            key=f"track_move_selected_{track_id}",
            disabled=move_disabled,
        ):
            _move_frames_api(
                ep_id,
                track_id,
                selected_frames,
                target_identity_id,
                target_name,
                show_slug,
            )
        if action_cols[1].button("Delete selected", key=f"track_delete_selected_{track_id}", type="secondary"):
            _delete_frames_api(ep_id, track_id, selected_frames)


def _rename_identity(ep_id: str, identity_id: str, label: str) -> None:
    endpoint = f"/identities/{ep_id}/rename"
    payload = {"identity_id": identity_id, "new_label": label}
    if _api_post(endpoint, payload):
        st.success("Identity renamed.")
        st.rerun()


def _delete_identity(ep_id: str, identity_id: str) -> None:
    endpoint = f"/episodes/{ep_id}/identities/{identity_id}"
    if _api_delete(endpoint):
        st.success("Identity deleted.")
        st.rerun()


def _api_merge(ep_id: str, source_id: str, target_id: str) -> None:
    endpoint = f"/identities/{ep_id}/merge"
    if _api_post(endpoint, {"source_id": source_id, "target_id": target_id}):
        st.success("Identities merged.")
        st.rerun()


def _move_track(ep_id: str, track_id: int, target_identity_id: str | None) -> None:
    endpoint = f"/identities/{ep_id}/move_track"
    payload = {"track_id": track_id, "target_identity_id": target_identity_id}
    resp = _api_post(endpoint, payload)
    if resp:
        st.success("Track assigned.")
        st.rerun()
    else:
        st.error("Failed to assign track. Check logs.")


def _archive_track(ep_id: str, track_id: int) -> None:
    """Archive a track before removing it from the identity."""
    # Get track details for archiving
    track_detail = _fetch_track_detail_cached(ep_id, track_id)
    show_id = ep_id.split("-")[0].upper() if "-" in ep_id else ep_id.upper()
    identity_id = st.session_state.get("selected_identity")

    # Prepare archive data
    archive_payload = {
        "episode_id": ep_id,
        "track_id": track_id,
        "reason": "user_archived",
        "cluster_id": identity_id,
    }

    if track_detail:
        archive_payload["frame_count"] = track_detail.get("face_count", 0)
        archive_payload["rep_crop_url"] = track_detail.get("rep_thumb_url") or track_detail.get("rep_media_url")
        # Centroid would need to be fetched from track_reps if available
        centroid = track_detail.get("centroid")
        if centroid:
            archive_payload["centroid"] = centroid

    # Archive via API (fire and forget - don't block on archive success)
    try:
        _api_post(f"/archive/shows/{show_id}/tracks", archive_payload)
    except Exception:
        pass  # Archive failure shouldn't block deletion

    # Now delete the track
    if _api_post(f"/identities/{ep_id}/drop_track", {"track_id": track_id}):
        # Clear selection state for this track
        st.session_state.get("track_frame_selection", {}).pop(track_id, None)
        # Navigate back within the person/cluster context
        st.toast("Track archived.")
        identity_id = st.session_state.get("selected_identity")
        person_id = st.session_state.get("selected_person")
        if identity_id:
            _set_view("cluster_tracks", person_id=person_id, identity_id=identity_id)
        else:
            _set_view("people")


def _delete_track(ep_id: str, track_id: int) -> None:
    """Legacy function - now archives track before deletion."""
    _archive_track(ep_id, track_id)


def _delete_frame(ep_id: str, track_id: int, frame_idx: int, delete_assets: bool) -> None:
    payload = {
        "track_id": track_id,
        "frame_idx": frame_idx,
        "delete_assets": delete_assets,
    }
    if _api_post(f"/identities/{ep_id}/drop_frame", payload):
        # Stay on track view - don't navigate away
        st.success("Frame removed.")
        st.rerun()


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
_initialize_state(ep_id)

# Check for active Celery jobs and render status (Phase 2 - non-blocking jobs)
_api_base = st.session_state.get("api_base")
if _api_base and session_manager.render_job_status(_api_base):
    # Job is still running - schedule auto-refresh
    import time as _time
    _time.sleep(2)
    st.rerun()

_hydrate_view_from_query(ep_id)
episode_detail = _episode_header(ep_id)
if not episode_detail:
    st.stop()
if not helpers.detector_is_face_only(ep_id):
    st.warning(
        "Tracks were generated with a legacy detector. Rerun detect/track with RetinaFace or YOLOv8-face for best results."
    )

# Get show_slug early - needed for facebank grouping strategy
show_slug = _episode_show_slug(ep_id)
if not show_slug:
    st.error(f"Could not determine show slug from episode ID: {ep_id}")
    st.stop()

strategy_labels = {
    "Auto (episode regroup + show match)": "auto",
    "Use existing face bank": "facebank",
}
selected_label = st.selectbox(
    "Grouping strategy",
    options=list(strategy_labels.keys()),
    key=f"{ep_id}::faces_review_grouping_strategy",
)
if selected_label not in strategy_labels:
    selected_label = list(strategy_labels.keys())[0]
selected_strategy = strategy_labels[selected_label]
facebank_ready = True
if selected_strategy == "facebank":
    confirm = st.text_input(
        "Confirm show slug before regrouping",
        value="",
        key=f"{ep_id}::facebank_show_confirm",
        help="Protect against grouping the wrong show",
        placeholder=show_slug or "",
    ).strip()
    expected_slug = (show_slug or "").lower()
    facebank_ready = bool(expected_slug and confirm.lower() == expected_slug)
    if not facebank_ready:
        st.info(f"Enter '{show_slug}' to enable facebank regrouping.")

button_label = "Group Clusters (facebank)" if selected_strategy == "facebank" else "Group Clusters (auto)"
caption_text = (
    "Auto-clusters within episode, assigns people, and computes cross-episode matches"
    if selected_strategy == "auto"
    else "Uses existing facebank seeds to align clusters to known cast members"
)

# Enhancement #2: Manual Assignment Protection toggle
protect_manual = False
facebank_first = True  # Default to True for better accuracy
if selected_strategy == "auto":
    with st.expander("⚙️ Advanced Clustering Options", expanded=False):
        protect_manual = st.checkbox(
            "🔒 Protect manual assignments",
            value=True,
            key="protect_manual_checkbox",
            help="Don't merge clusters that have been manually assigned to different cast members",
        )
        facebank_first = st.checkbox(
            "🎯 Facebank-first matching",
            value=True,
            key="facebank_first_checkbox",
            help="Try matching clusters to cast member facebank seeds before using people prototypes (recommended)",
        )
        st.caption(
            "Facebank-first matching provides more accurate results when cast members have "
            "uploaded reference images. It matches clusters directly to known faces."
        )

busy_flag = f"{ep_id}::group_clusters_busy"
if st.session_state.get(busy_flag):
    st.info("Cluster grouping already running…")
if st.button(
    button_label,
    key="group_clusters_action",
    type="primary",
    disabled=(selected_strategy == "facebank" and not facebank_ready) or st.session_state.get(busy_flag, False),
):
    payload = {
        "strategy": selected_strategy,
        "protect_manual": protect_manual,
        "facebank_first": facebank_first,
    }
    st.session_state[busy_flag] = True

    try:
        if selected_strategy == "auto":
            # Show detailed progress for auto clustering
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            log_expander = st.expander("📋 Detailed Progress Log", expanded=True)
            with log_expander:
                log_placeholder = st.empty()

            status_text.text("🚀 Starting auto-clustering…")
            result = None
            error = None

            # Collect log messages with timestamps
            log_messages = []

            def _add_log(msg: str):
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                log_messages.append(f"[{timestamp}] {msg}")
                log_placeholder.code("\n".join(log_messages), language="text")

            _add_log(f"Starting auto-cluster for {ep_id} (auto regroup + show match)…")

            try:
                # Grouping can take >60s; allow longer timeout
                status_text.text("⏳ Contacting API…")
                progress_bar.progress(0.05)
                _add_log(f"POST /episodes/{ep_id}/clusters/group (strategy=auto)")

                # Capture API base before thread execution (session_state not accessible in threads)
                api_base = st.session_state.get("api_base")
                if not api_base:
                    raise RuntimeError("API base URL not configured. Please reload the page.")

                # Thread-safe API call function
                def _thread_safe_api_post():
                    url = f"{api_base}/episodes/{ep_id}/clusters/group"
                    resp = requests.post(url, json=payload, timeout=300)
                    resp.raise_for_status()
                    return resp.json()

                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(_thread_safe_api_post)
                    start_time = time.time()
                    last_heartbeat = 0
                    fallback_progress = 0.1
                    last_progress_entry = 0

                    while future.running():
                        elapsed = int(time.time() - start_time)
                        # Monotonic placeholder progress up to 90% until real data arrives
                        fallback_progress = min(fallback_progress + 0.01, 0.9)
                        current_progress = fallback_progress
                        new_entries = False

                        status_text.text(f"⏳ Auto-cluster running… {elapsed}s elapsed")

                        # Poll backend progress to show real-time steps
                        try:
                            progress_resp = requests.get(
                                f"{api_base}/episodes/{ep_id}/clusters/group/progress",
                                timeout=5,
                            )
                            if progress_resp.status_code == 200:
                                progress_data = progress_resp.json()
                                entries = progress_data.get("entries", []) if isinstance(progress_data, dict) else []
                                for entry in entries[last_progress_entry:]:
                                    new_entries = True
                                    pct = entry.get("progress")
                                    step = entry.get("step") or "working"
                                    msg = entry.get("message") or ""
                                    total_clusters = entry.get("total_clusters")
                                    processed_clusters = entry.get("processed_clusters")
                                    assigned_clusters = entry.get("assigned_clusters")
                                    merged_clusters = entry.get("merged_clusters")
                                    new_people = entry.get("new_people")
                                    percent_complete = None
                                    if total_clusters and processed_clusters is not None:
                                        try:
                                            percent_complete = int(min(max(float(processed_clusters) / float(total_clusters), 0.0), 1.0) * 100)
                                        except Exception:
                                            percent_complete = None
                                    if pct is not None and percent_complete is None:
                                        percent_complete = int(float(pct) * 100)
                                    if pct is not None:
                                        current_progress = max(current_progress, min(float(pct), 0.99))
                                    elif percent_complete is not None:
                                        current_progress = max(current_progress, min(percent_complete / 100, 0.99))

                                    # Build detailed progress message
                                    parts = []
                                    if total_clusters and processed_clusters is not None:
                                        parts.append(f"Processed {int(processed_clusters)}/{int(total_clusters)} clusters")
                                    if assigned_clusters is not None:
                                        parts.append(f"assigned {int(assigned_clusters)}")
                                    if merged_clusters:
                                        parts.append(f"merged {int(merged_clusters)}")
                                    if new_people:
                                        parts.append(f"new people {int(new_people)}")
                                    if not parts:
                                        parts.append(msg or step)
                                    if percent_complete is not None:
                                        progress_str = f"[{percent_complete:3d}%]"
                                    else:
                                        progress_str = "[ --%]"
                                    elapsed_suffix = f" ({elapsed}s elapsed)" if elapsed else ""
                                    _add_log(f"{progress_str} {step}: {'; '.join(parts)}{elapsed_suffix}")
                                last_progress_entry = len(entries)
                        except Exception:
                            pass

                        # Log heartbeat only if no new progress entries arrived
                        if not new_entries and elapsed // 10 > last_heartbeat:
                            last_heartbeat = elapsed // 10
                            _add_log(f"⏳ Waiting for cluster progress… ({elapsed}s elapsed)")

                        progress_bar.progress(current_progress)

                        time.sleep(0.5)

                    result = future.result()
                    total_elapsed = int(time.time() - start_time)
                    _add_log(f"✓ API returned response after {total_elapsed}s")
            except Exception as exc:
                error = exc

            if error:
                progress_bar.empty()
                err_msg = f"❌ Auto-cluster failed: {error}"
                status_text.error(err_msg)
                _add_log(err_msg)
            elif result is None:
                progress_bar.empty()
                err_msg = "❌ Auto-cluster failed: API returned no response. Check backend logs."
                status_text.error(err_msg)
                _add_log(err_msg)
            else:
                error_msg = None
                if isinstance(result, dict):
                    error_msg = result.get("error") or result.get("detail")
                    if isinstance(error_msg, dict):
                        error_msg = error_msg.get("message") or str(error_msg)
                    status_value = str(result.get("status", "")).lower()
                    if status_value and status_value not in {"success", "ok"} and not error_msg:
                        error_msg = f"Unexpected status: {status_value}"
                if error_msg:
                    progress_bar.empty()
                    status_text.error(f"❌ Auto-cluster failed: {error_msg}")
                    _add_log(f"❌ Auto-cluster failed: {error_msg}")
                else:
                    # Show progress log if available from API
                    progress_log = result.get("progress_log", [])
                    if progress_log:
                        _add_log(f"Processing {len(progress_log)} progress step(s) from API…")
                        for entry in progress_log:
                            progress = entry.get("progress", 0.0)
                            message = entry.get("message", "")
                            step = entry.get("step", "")
                            progress_bar.progress(min(progress, 1.0))
                            status_text.text(f"{step or 'working'} – {message}")
                            _add_log(f"[{int(progress*100):3d}%] {step}: {message}")
                            time.sleep(0.05)  # Brief pause to show progression visually
                    else:
                        _add_log("⚠️ No progress_log returned from API (completed without detailed progress)")

                    progress_bar.progress(1.0)

                    # Show detailed summary - check both possible locations for log
                    # API returns result -> log, but also could be at top level
                    log_data = result.get("result", {}).get("log", {}) or result.get("log", {})
                    steps = log_data.get("steps", []) if isinstance(log_data, dict) else []
                    cleared = next(
                        (s.get("cleared_count", 0) for s in steps if s.get("step") == "clear_assignments"),
                        0,
                    )
                    centroids = next(
                        (s.get("centroids_count", 0) for s in steps if s.get("step") == "compute_centroids"),
                        0,
                    )
                    merged_groups = next(
                        (s.get("merged_count", 0) for s in steps if s.get("step") == "group_within_episode"),
                        0,
                    )
                    assigned_clusters = next(
                        (s.get("assigned_count", 0) for s in steps if s.get("step") == "group_across_episodes"),
                        0,
                    )
                    new_people = next(
                        (s.get("new_people_count", 0) for s in steps if s.get("step") == "group_across_episodes"),
                        0,
                    )
                    merged_cluster_count = next(
                        (s.get("merged_clusters", 0) for s in steps if s.get("step") == "apply_within_groups"),
                        0,
                    )
                    pruned_people = next(
                        (s.get("pruned_people", 0) for s in steps if s.get("step") == "apply_within_groups"),
                        0,
                    )

                    assignment_entries = (
                        result.get("assignments")
                        or result.get("result", {}).get("assignments")
                        or result.get("across_episodes", {}).get("assigned")
                        or []
                    )
                    if isinstance(assignment_entries, dict):
                        assignment_entries = assignment_entries.get("assigned", [])
                    unique_people_assigned = {
                        entry.get("person_id") for entry in assignment_entries if entry.get("person_id")
                    }
                    # Fallback counts when log is missing
                    if not steps and isinstance(result, dict):
                        centroids = centroids or len((result.get("result", {}) or {}).get("centroids", {}) or [])
                        within = (result.get("result", {}) or {}).get("within_episode", {}) or {}
                        merged_groups = merged_groups or within.get("merged_count", 0)
                        assigned_clusters = assigned_clusters or len(assignment_entries or [])
                        merged_cluster_count = merged_cluster_count or within.get("merged_count", 0)
                        pruned_people = pruned_people or result.get("result", {}).get("across_episodes", {}).get(
                            "pruned_people_count", 0
                        )

                    # Add detailed summary to log
                    _add_log("=" * 50)
                    _add_log("DETAILED SUMMARY:")
                    for step in steps:
                        step_name = step.get("step", "")
                        step_status = step.get("status", "")
                        _add_log(f"✓ {step_name}: {step_status}")
                        if step_name == "clear_assignments":
                            _add_log(f"  → Cleared {step.get('cleared_count', 0)} stale assignments")
                        elif step_name == "compute_centroids":
                            _add_log(f"  → Computed {step.get('centroids_count', 0)} centroids")
                        elif step_name == "group_within_episode":
                            _add_log(f"  → Merged {step.get('merged_count', 0)} cluster group(s)")
                        elif step_name == "group_across_episodes":
                            _add_log(
                                f"  → Assigned {step.get('assigned_count', 0)} cluster(s); created "
                                f"{step.get('new_people_count', 0)} new people"
                            )
                        elif step_name == "apply_within_groups":
                            _add_log(
                                f"  → Applied {step.get('groups', 0)} group(s); "
                                f"merged {step.get('merged_clusters', 0)} cluster(s); "
                                f"pruned {step.get('pruned_people', 0)} empty people"
                            )
                    _add_log("=" * 50)

                    status_text.success(
                        f"✅ Clustering complete!\n"
                        f"• Cleared {cleared} stale assignment(s)\n"
                        f"• Computed {centroids} centroid(s)\n"
                        f"• Merged {merged_groups} cluster group(s)\n"
                        f"• Assigned {assigned_clusters} cluster(s) to {len(unique_people_assigned)} people "
                        f"(created {new_people} new)\n"
                        f"• Merged {merged_cluster_count} cluster(s) after regrouping "
                        f"(pruned {pruned_people} empty auto people)\n\n"
                        "Check Episode Auto-Clustered People below to review and assign."
                    )
                    progress_bar.empty()
                    st.session_state[busy_flag] = False
                    st.rerun()
        else:
            # Facebank regrouping with basic progress + error handling
            with st.spinner("Running cluster grouping..."):
                result = _api_post(f"/episodes/{ep_id}/clusters/group", payload, timeout=300)
            if not result:
                st.error("Facebank regroup failed: empty response from API.")
            elif isinstance(result, dict) and (result.get("error") or result.get("detail")):
                err_msg = result.get("error") or result.get("detail")
                st.error(f"Facebank regroup failed: {err_msg}")
            else:
                matched = result.get("result", {}).get("matched_clusters", 0) if isinstance(result, dict) else 0
                st.success(f"Facebank regroup complete! {matched} clusters matched to seeds.")
                st.session_state[busy_flag] = False
                st.rerun()
    finally:
        st.session_state[busy_flag] = False
st.caption(caption_text)

view_state = st.session_state.get("facebank_view", "people")
ep_meta = helpers.parse_ep_id(ep_id) or {}
season_label: str | None = None
season_value = ep_meta.get("season")
if isinstance(season_value, int):
    season_label = f"S{season_value:02d}"

ctx = get_script_run_ctx() if get_script_run_ctx else None


def _submit_with_ctx(pool: ThreadPoolExecutor, func, *args):
    if ctx and add_script_run_ctx:
        def _wrapped():
            add_script_run_ctx(threading.current_thread(), ctx)
            return func(*args)
        return pool.submit(_wrapped)
    return pool.submit(func, *args)

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {
        "identities": _submit_with_ctx(executor, _fetch_identities_cached, ep_id),
        "people": _submit_with_ctx(executor, _fetch_people_cached, show_slug),
        "cast": _submit_with_ctx(executor, _fetch_cast_cached, show_slug, season_label),
        "cluster_tracks": _submit_with_ctx(executor, _safe_api_get, f"/episodes/{ep_id}/cluster_tracks"),
    }
    identities_payload = futures["identities"].result()
    people_resp = futures["people"].result()
    cast_api_resp = futures["cast"].result()
    cluster_payload = futures["cluster_tracks"].result() or {"clusters": []}

if not identities_payload:
    st.stop()

# Show local fallback banner if any local files are being used
_show_local_fallback_banner(cluster_payload)

cluster_lookup = _clusters_by_identity(cluster_payload)
identities = identities_payload.get("identities", [])
identity_index = {ident["identity_id"]: ident for ident in identities}
# show_slug already defined above - no need to recompute
roster_names = _fetch_roster_names(show_slug)
people = people_resp.get("people", []) if people_resp else []
show_id = show_slug
people_lookup = {str(person.get("person_id") or ""): person for person in people}

selected_person = st.session_state.get("selected_person")
selected_identity = st.session_state.get("selected_identity")
selected_track = st.session_state.get("selected_track")

# Build cast_options for Smart Suggestions
cast_options = {}
if cast_api_resp:
    cast_members = cast_api_resp.get("cast", [])
    cast_options = {
        cm.get("cast_id"): cm.get("name")
        for cm in cast_members
        if cm.get("cast_id") and cm.get("name")
    }

if view_state == "track" and selected_track is not None:
    _render_track_view(ep_id, selected_track, identities_payload)
elif view_state == "cluster_tracks" and selected_identity:
    _render_cluster_tracks(
        ep_id,
        selected_identity,
        cluster_lookup,
        identity_index,
        show_slug,
        roster_names,
        selected_person,
    )
elif view_state == "person_clusters" and selected_person:
    _render_person_clusters(
        ep_id,
        selected_person,
        people_lookup,
        cluster_lookup,
        identities_payload,
        show_slug,
        roster_names,
        cast_options=cast_options,
    )
elif view_state == "cast_tracks" and selected_person:
    _render_cast_all_tracks(
        ep_id,
        selected_person,
        people_lookup,
        cluster_lookup,
        identities_payload,
        show_slug,
    )
else:
    _render_people_view(
        ep_id,
        show_id,
        people,
        cluster_lookup,
        identity_index,
        season_label,
        cast_api_resp=cast_api_resp,
    )


# Render Cluster Comparison Mode in sidebar (Feature 3)
_render_cluster_comparison_mode(ep_id, show_id, cluster_lookup)

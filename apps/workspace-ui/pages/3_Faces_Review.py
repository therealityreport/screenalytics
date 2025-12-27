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
from py_screenalytics import run_layout  # noqa: E402
import faces_review_artifacts  # noqa: E402
import faces_review_run_scoped  # noqa: E402
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
    # Enhanced rendering functions (Nov 2024)
    render_cluster_range_badge,
    render_quality_breakdown_badge,
    render_cast_rank_badge,
    render_track_with_dropout,
    render_outlier_severity_badge,
    render_ambiguity_badge,
    render_isolation_badge,
    render_confidence_trend_badge,
    render_temporal_badge,
)
from metrics_strip import (  # noqa: E402
    MetricData,
    render_metrics_strip,
    render_metrics_strip_inline,
    build_cluster_metrics,
    build_track_metrics,
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
# Operator docs entry points; env overrides allow pointing at an internal wiki.
FACES_REVIEW_GUIDE_URL = os.environ.get(
    "SCREENALYTICS_FACES_REVIEW_GUIDE_URL",
    "/Faces_Review_Docs",  # Local page with full metrics documentation
)
FACES_REVIEW_GUIDE_FALLBACK = os.environ.get(
    "SCREENALYTICS_FACES_REVIEW_GUIDE_FALLBACK",
    "https://github.com/therealityreport/screenalytics/blob/main/docs/ops/faces_review_guide.md",
)


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
    st.columns = lambda n=1, *a, **k: tuple(
        st for _ in range(len(n) if isinstance(n, (list, tuple)) else int(n) if n else 0)
    )
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
helpers.render_page_header("workspace-ui:3_Faces_Review", "Faces & Tracks Review")

# Current run_id scope for this page (required for run-scoped mutations + artifact reads).
_CURRENT_RUN_ID: str | None = None
_LEGACY_PEOPLE_FALLBACK: bool = False
help_cols = st.columns([3, 1])
with help_cols[0]:
    st.caption("Need the playbook? Open the full Faces Review guide for flow, job settings, and safety tips.")
with help_cols[1]:
    link_button = getattr(st, "link_button", None)
    label = "📖 Faces Review Guide"
    if callable(link_button):
        link_button(label, FACES_REVIEW_GUIDE_URL, help="Full walkthrough of flows, job settings, and guardrails.")
    else:
        st.markdown(f"[{label}]({FACES_REVIEW_GUIDE_URL})")
if FACES_REVIEW_GUIDE_FALLBACK and FACES_REVIEW_GUIDE_FALLBACK != FACES_REVIEW_GUIDE_URL:
    st.caption(f"[Docs fallback]({FACES_REVIEW_GUIDE_FALLBACK})")

# Global episode selector in sidebar to ensure ep_id is set for this page
sidebar_ep_id = None
try:
    sidebar_ep_id = helpers.render_sidebar_episode_selector()
except Exception:
    # Sidebar selector is best-effort; fallback to existing session/query ep_id
    sidebar_ep_id = None

# Similarity Scores Color Key (native Streamlit layout to avoid raw HTML)
with st.expander("📊 Similarity Scores Guide", expanded=False):
    st.markdown("### Core Similarity Types")

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
            ["≥ 75%: Strong match", "70–74%: Good match", "< 70%: Needs review"],
        )
    with col2:
        _render_similarity_card(
            SIMILARITY_COLORS[SimilarityType.CAST].strong,
            "Cast Similarity",
            "How similar clusters are for CAST MEMBERS (facebank).",
            ["≥ 68%: Auto-assigns to cast", "50–67%: Requires review", "< 50%: Weak match"],
        )

    # Row 2: Track Similarity (Orange) and Cluster Cohesion (Green)
    try:
        col3, col4 = st.columns(2)
    except Exception:
        col3 = col4 = st
    with col3:
        _render_similarity_card(
            SIMILARITY_COLORS[SimilarityType.TRACK].strong,
            "Track Similarity",
            "How consistent FRAMES within a TRACK are. Shows excluded frames.",
            ["≥ 85%: Strong consistency", "70–84%: Good consistency", "< 70%: Weak"],
        )
    with col4:
        _render_similarity_card(
            SIMILARITY_COLORS[SimilarityType.CLUSTER].strong,
            "Cluster Cohesion",
            "How cohesive tracks in a cluster are. Shows min-max range.",
            ["≥ 80%: Tight cluster", "60–79%: Moderate", "< 60%: Loose cluster"],
        )

    # Row 3: Person Cohesion (Teal) and Quality Score (Green)
    try:
        col5, col6 = st.columns(2)
    except Exception:
        col5 = col6 = st
    with col5:
        _render_similarity_card(
            SIMILARITY_COLORS[SimilarityType.PERSON_COHESION].strong,
            "Person Cohesion",
            "How well a track fits with other tracks of same person.",
            ["≥ 70%: Strong fit", "50–69%: Good fit", "< 50%: Poor (review)"],
        )
    with col6:
        _render_similarity_card(
            "#4CAF50",
            "Quality Score",
            "Detection + sharpness + area. Hover for Det/Sharp/Area breakdown.",
            ["≥ 85%: High (green)", "60–84%: Medium (amber)", "< 60%: Low (red)"],
        )

    # NEW METRICS SECTION (Nov 2024)
    st.markdown("---")
    st.markdown(
        "### New Metrics "
        '<span style="background:#E91E63;color:white;padding:2px 6px;border-radius:3px;'
        'font-size:10px;font-weight:bold;margin-left:6px;">NOV 2024</span>',
        unsafe_allow_html=True,
    )


def _render_similarity_badge(similarity: float | None) -> str:
    if similarity is None:
        return ""
    pct = int(round(similarity * 100))
    if similarity >= 0.75:
        color = "green"
    elif similarity >= 0.60:
        color = "orange"
    else:
        color = "red"
    return f'<span style="color:{color}; font-weight:600;">{pct}%</span>'

    # Row 5: Temporal Consistency and Ambiguity Score
    try:
        col9, col10 = st.columns(2)
    except Exception:
        col9 = col10 = st
    with col9:
        _render_similarity_card(
            SIMILARITY_COLORS[SimilarityType.TEMPORAL].strong,
            "Temporal Consistency",
            "How consistent a person looks across time in episode.",
            ["≥ 80%: Consistent", "60–79%: Variable", "< 60%: Significant changes"],
        )
    with col10:
        _render_similarity_card(
            SIMILARITY_COLORS[SimilarityType.AMBIGUITY].weak,  # Red for risky
            "Ambiguity Score",
            "Gap between 1st and 2nd best match. LOW = risky.",
            ["≥ 15%: Clear winner", "8–14%: OK", "< 8%: Risky (review!)"],
        )

    # Row 6: Cluster Isolation and Confidence Trend
    try:
        col11, col12 = st.columns(2)
    except Exception:
        col11 = col12 = st
    with col11:
        _render_similarity_card(
            SIMILARITY_COLORS[SimilarityType.ISOLATION].strong,
            "Cluster Isolation",
            "Distance to nearest cluster. LOW = merge candidate.",
            ["≥ 40%: Well isolated", "25–39%: Moderate", "< 25%: Close (merge?)"],
        )
    with col12:
        _render_similarity_card(
            SIMILARITY_COLORS[SimilarityType.CONFIDENCE_TREND].strong,
            "Confidence Trend",
            "Is assignment confidence improving or degrading?",
            ["↑ Improving (green)", "→ Stable (gray)", "↓ Degrading (red)"],
        )

    st.markdown("---")
    st.markdown("### Enhanced Badge Features")
    enhanced_df = {
        "Badge": [
            "CAST: 68% (#1 of 5)",
            "CLU: 72% (58-89%)",
            "TRK: 85% (3 excl)",
            "Q: 82%",
            "OUTLIER: 45% ⚠️",
            "AMB: Risky (3%)",
            "ISO: Close",
            "↑ Improving",
        ],
        "Enhancement": [
            "Shows rank among all cast suggestions",
            "Shows min-max range for cluster cohesion",
            "Shows frames excluded from centroid",
            "Hover to see Det/Sharp/Area breakdown",
            "Only shown on frames below threshold",
            "Gap to 2nd best match - risky assignments",
            "Merge candidate indicator",
            "Confidence trend over time",
        ],
    }
    st.table(enhanced_df)

    st.markdown("### Quality Indicators")
    quality_df = {
        "Badge": [
            "Q: 85%+",
            "Q: 60-84%",
            "Q: < 60%",
            "ID: 75%+",
            "ID: 70-74%",
            "ID: < 70%",
        ],
        "Meaning": [
            "High quality (sharp, complete face, good detection)",
            "Medium quality (acceptable for most uses)",
            "Low quality (partial face, blurry, or low confidence)",
            "Strong identity match to track",
            "Good identity match",
            "Needs review (threshold raised from 60%)",
        ],
    }
    st.table(quality_df)

    st.markdown("### Frame Badges")
    st.markdown(
        """
        - **★ BEST QUALITY (green)** – Complete face, high quality, good ID match
        - **⚠ BEST AVAILABLE (orange)** – Partial/low-quality, best available frame
        - **Partial (orange pill)** – Edge-clipped or incomplete face
        - **OUTLIER: XX% ⚠️** – Frame differs significantly from track (only shown on outliers)

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


def _track_media_state(
    ep_id: str, track_id: int, sample: int = 1, cache_suffix: str = ""
) -> Dict[str, Any]:
    """Get or create cache state for track media, keyed by ep_id, track_id, sample rate, and optional suffix."""
    import time
    cache = _track_media_cache()
    key = f"{ep_id}::{track_id}::s{sample}{cache_suffix}"
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


def _render_sync_to_s3_button(ep_id: str) -> None:
    """Render a button to sync local thumbnails/crops to S3 when images aren't loading."""
    sync_key = f"sync_s3_expanded:{ep_id}"
    with st.expander("🔧 Thumbnails not loading?", expanded=st.session_state.get(sync_key, False)):
        st.caption(
            "If thumbnails are showing as placeholders, local files may need to be synced to S3. "
            "Click below to upload any missing thumbnails and crops."
        )
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("📤 Sync to S3", key=f"sync_thumbs_s3_{ep_id}", use_container_width=True):
                st.session_state[sync_key] = True
                with st.spinner("Syncing thumbnails and crops to S3..."):
                    try:
                        api_base = st.session_state.get("api_base", "http://localhost:8000")
                        params = {"run_id": _CURRENT_RUN_ID} if _CURRENT_RUN_ID else None
                        resp = requests.post(
                            f"{api_base}/episodes/{ep_id}/sync_thumbnails_to_s3",
                            params=params,
                            timeout=120,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            uploaded = data.get("uploaded_thumbs", 0) + data.get("uploaded_crops", 0)
                            if uploaded > 0:
                                st.success(f"✅ Uploaded {uploaded} file(s) to S3!")
                                st.toast(f"Synced {uploaded} thumbnails/crops to S3")
                                st.rerun()
                            else:
                                st.info("All thumbnails already in S3 (or no local files found).")
                            if data.get("total_errors", 0) > 0:
                                st.warning(f"⚠️ {data['total_errors']} upload error(s) - check API logs")
                        else:
                            st.error(f"Sync failed: {resp.status_code} - {resp.text[:200]}")
                    except requests.Timeout:
                        st.error("Sync timed out - try again or check S3 connectivity")
                    except Exception as e:
                        st.error(f"Sync failed: {e}")


def _ensure_faces_review_artifacts(ep_id: str, run_id: str) -> faces_review_artifacts.RunArtifactHydration:
    return faces_review_artifacts.ensure_run_artifacts_local(
        ep_id,
        run_id,
        faces_review_artifacts.FACES_REVIEW_REQUIRED_ARTIFACTS,
        optional=faces_review_artifacts.FACES_REVIEW_OPTIONAL_ARTIFACTS,
    )


def _render_missing_run_artifacts(
    ep_id: str,
    run_id: str,
    check: faces_review_artifacts.RunArtifactHydration,
) -> None:
    if not check.missing_required:
        return
    if not check.storage_enabled:
        st.warning(
            "Run artifacts for this attempt are not present locally and hydration_from_s3 is false; "
            "cannot display clusters or thumbnails."
        )
    else:
        st.warning("Run artifacts are missing locally and could not be hydrated from S3.")
    st.caption(f"Run: `{ep_id}` / `{run_id}`")
    st.caption("Missing required files:")
    for rel_path in check.missing_required:
        st.caption(f"• `{rel_path}`")
    if check.missing_optional:
        st.caption("Missing optional files (metrics/quality may be incomplete):")
        for rel_path in check.missing_optional:
            st.caption(f"• `{rel_path}`")


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
    merged_params: Dict[str, Any] = dict(params or {})
    if _CURRENT_RUN_ID:
        merged_params.setdefault("run_id", _CURRENT_RUN_ID)
    try:
        return helpers.api_get(path, params=(merged_params or None))
    except requests.RequestException as exc:
        base = (cfg or {}).get("api_base") if isinstance(cfg, dict) else ""
        try:
            st.error(helpers.describe_error(f"{base}{path}", exc))
        except Exception as display_exc:
            # Fallback if error display fails
            st.error(f"API error: {path} - {exc}")
        return None


def _fetch_faces_review_bundle(
    ep_id: str,
    *,
    filter_cast_id: str | None = None,
    include_archived: bool = False,
) -> Dict[str, Any] | None:
    params: Dict[str, Any] = {}
    if filter_cast_id:
        params["filter_cast_id"] = filter_cast_id
    if include_archived:
        params["include_archived"] = "1"
    return _safe_api_get(f"/episodes/{ep_id}/faces_review_bundle", params=params or None)


def _run_scope_token() -> str:
    return _CURRENT_RUN_ID or "legacy"


def _cast_suggestions_cache_key(ep_id: str) -> str:
    return f"cast_suggestions:{ep_id}:{_run_scope_token()}"


def _dismissed_suggestions_cache_key(ep_id: str) -> str:
    return f"dismissed_suggestions:{ep_id}:{_run_scope_token()}"


def _improve_faces_state_key(ep_id: str, suffix: str) -> str:
    return f"{ep_id}::{_run_scope_token()}::improve_faces::{suffix}"


@st.cache_data(ttl=15)  # Reduced from 60s - frequently mutated by assignments
def _fetch_identities_cached(ep_id: str, run_id: str | None) -> Dict[str, Any] | None:
    params = {"run_id": run_id} if run_id else None
    return _safe_api_get(f"/episodes/{ep_id}/identities", params=params)


@st.cache_data(ttl=15)  # Reduced from 60s - frequently mutated by assignments
def _fetch_people_cached(show_slug: str | None) -> Dict[str, Any] | None:
    if not show_slug:
        return None
    return _safe_api_get(f"/shows/{show_slug}/people")


@st.cache_data(ttl=15)  # Unified auto-people + unassigned clusters
def _fetch_unlinked_entities(ep_id: str) -> Dict[str, Any] | None:
    return _safe_api_get(f"/episodes/{ep_id}/unlinked_entities")


@st.cache_data(ttl=15)
def _fetch_archived_ids(ep_id: str, run_id: str | None = None) -> Dict[str, set]:
    """Fetch archived cluster/track ids for an episode so UI can hide them."""
    parsed = helpers.parse_ep_id(ep_id) or {}
    show_value = parsed.get("show")
    if show_value:
        show_id = str(show_value).upper()
    else:
        show_id = ep_id.split("-")[0].upper() if "-" in ep_id else ep_id.upper()
    resp = _safe_api_get(
        f"/archive/shows/{show_id}",
        params={"episode_id": ep_id, "limit": 500},
    )
    clusters: set[str] = set()
    tracks: set[int] = set()
    if resp:
        items = resp.get("items", []) or []
        for item in items:
            item_type = item.get("type")
            if item_type == "cluster":
                cid = item.get("cluster_id") or item.get("identity_id") or item.get("original_id")
                if cid:
                    clusters.add(str(cid))
            elif item_type == "track":
                tid = item.get("track_id") or item.get("original_id")
                try:
                    if tid is not None:
                        tracks.add(int(tid))
                except (TypeError, ValueError):
                    continue
    return {"clusters": clusters, "tracks": tracks}


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
def _fetch_cluster_track_reps_cached(
    ep_id: str, cluster_id: str, frames_per_track: int = 0
) -> Dict[str, Any] | None:
    """Cached fetch of cluster track representatives for faster navigation.

    Args:
        frames_per_track: Number of sample frames to include per track (0=none, default).
                          Set to 10 for row-based display with frame scrolling.
    """
    params = {}
    if frames_per_track > 0:
        params["frames_per_track"] = frames_per_track
    return _safe_api_get(f"/episodes/{ep_id}/clusters/{cluster_id}/track_reps", params=params)


@st.cache_data(ttl=60)
def _fetch_cluster_metrics_cached(ep_id: str, cluster_id: str) -> Dict[str, Any] | None:
    """Cached fetch of cluster metrics (cohesion, isolation, ambiguity, temporal, quality)."""
    return _safe_api_get(f"/episodes/{ep_id}/clusters/{cluster_id}/metrics")


@st.cache_data(ttl=60)
def _fetch_track_metrics_cached(ep_id: str, track_id: int) -> Dict[str, Any] | None:
    """Cached fetch of track metrics (similarity, quality, person cohesion)."""
    return _safe_api_get(f"/episodes/{ep_id}/tracks/{track_id}/metrics")


def _coerce_track_int(val: Any) -> int | None:
    """Parse track ids in either int or track_123 string form."""
    if isinstance(val, str) and val.startswith("track_"):
        val = val.replace("track_", "")
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _start_improve_faces(ep_id: str, *, force: bool = False) -> bool:
    """Fetch initial suggestions and activate the Improve Faces modal."""
    resp = _safe_api_get(f"/episodes/{ep_id}/face_review/initial_unassigned_suggestions")
    if not resp:
        st.error("Failed to load Improve Faces suggestions.")
        return False

    suggestions = resp.get("suggestions", []) if isinstance(resp, dict) else []
    initial_done = bool(resp.get("initial_pass_done")) if isinstance(resp, dict) else False

    if not suggestions or initial_done:
        if force:
            st.info("No Improve Faces suggestions right now.")
        st.session_state.pop(_improve_faces_state_key(ep_id, "active"), None)
        st.session_state.pop(_improve_faces_state_key(ep_id, "suggestions"), None)
        st.session_state.pop(_improve_faces_state_key(ep_id, "index"), None)
        return False

    st.session_state[_improve_faces_state_key(ep_id, "active")] = True
    st.session_state[_improve_faces_state_key(ep_id, "suggestions")] = suggestions
    st.session_state[_improve_faces_state_key(ep_id, "index")] = 0
    st.session_state.pop(_improve_faces_state_key(ep_id, "trigger"), None)
    st.rerun()
    return True


def _render_improve_faces_modal(ep_id: str) -> None:
    """Render Improve Faces dialog if active."""
    if not st.session_state.get(_improve_faces_state_key(ep_id, "active")):
        return

    suggestions = st.session_state.get(_improve_faces_state_key(ep_id, "suggestions"), []) or []
    idx = st.session_state.get(_improve_faces_state_key(ep_id, "index"), 0) or 0

    @st.dialog("Improve Face Clustering", width="large")
    def _dialog():
        suggestions_local = st.session_state.get(_improve_faces_state_key(ep_id, "suggestions"), []) or []
        current_idx = st.session_state.get(_improve_faces_state_key(ep_id, "index"), 0) or 0

        def _render_thumb(url: str | None) -> None:
            """Render face crop filling the column width."""
            if not url:
                st.markdown("*No image available*")
                return
            st.image(url, use_container_width=True)

        if not suggestions_local or current_idx >= len(suggestions_local):
            st.success("All suggestions reviewed!")
            st.markdown("Click **Go to Faces Review** to continue assigning faces to cast members.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Go to Faces Review", type="primary", use_container_width=True):
                    st.session_state.pop(_improve_faces_state_key(ep_id, "active"), None)
                    st.session_state.pop(_improve_faces_state_key(ep_id, "suggestions"), None)
                    st.session_state.pop(_improve_faces_state_key(ep_id, "index"), None)
                    st.rerun()
            with col2:
                if st.button("Close", use_container_width=True):
                    st.session_state.pop(_improve_faces_state_key(ep_id, "active"), None)
                    st.session_state.pop(_improve_faces_state_key(ep_id, "suggestions"), None)
                    st.session_state.pop(_improve_faces_state_key(ep_id, "index"), None)
                    st.rerun()
            return

        suggestion = suggestions_local[current_idx]
        cluster_a = suggestion.get("cluster_a", {}) if isinstance(suggestion, dict) else {}
        cluster_b = suggestion.get("cluster_b", {}) if isinstance(suggestion, dict) else {}
        similarity = suggestion.get("similarity", 0)

        st.markdown(f"**Are they the same person?** — {current_idx + 1} of {len(suggestions_local)}")
        st.progress((current_idx + 1) / len(suggestions_local))

        img_col1, img_col2 = st.columns(2)
        with img_col1:
            crop_url_a = cluster_a.get("crop_url")
            resolved_a = helpers.resolve_thumb(crop_url_a) if crop_url_a else None
            _render_thumb(resolved_a)
            st.caption(f"Cluster: {cluster_a.get('id', '?')}")
            st.caption(f"Tracks: {cluster_a.get('tracks', 0)} · Faces: {cluster_a.get('faces', 0)}")

        with img_col2:
            crop_url_b = cluster_b.get("crop_url")
            resolved_b = helpers.resolve_thumb(crop_url_b) if crop_url_b else None
            _render_thumb(resolved_b)
            st.caption(f"Cluster: {cluster_b.get('id', '?')}")
            st.caption(f"Tracks: {cluster_b.get('tracks', 0)} · Faces: {cluster_b.get('faces', 0)}")

        st.caption(f"Similarity: {similarity:.1%}")

        btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 1])

        def _advance():
            st.session_state[_improve_faces_state_key(ep_id, "index")] = current_idx + 1

        with btn_col1:
            if st.button("Yes", type="primary", use_container_width=True, key=f"improve_yes_{current_idx}"):
                exec_mode = helpers.get_execution_mode(ep_id)
                payload = {
                    "pair_type": "unassigned_unassigned",
                    "cluster_a_id": cluster_a.get("id"),
                    "cluster_b_id": cluster_b.get("id"),
                    "decision": "merge",
                    "execution_mode": "redis" if exec_mode != "local" else "local",
                }
                if not _api_post(f"/episodes/{ep_id}/face_review/decision/start", payload, timeout=60):
                    st.error("Failed to save merge decision.")
                _advance()

        with btn_col2:
            if st.button("No", use_container_width=True, key=f"improve_no_{current_idx}"):
                exec_mode = helpers.get_execution_mode(ep_id)
                payload = {
                    "pair_type": "unassigned_unassigned",
                    "cluster_a_id": cluster_a.get("id"),
                    "cluster_b_id": cluster_b.get("id"),
                    "decision": "reject",
                    "execution_mode": "redis" if exec_mode != "local" else "local",
                }
                if not _api_post(f"/episodes/{ep_id}/face_review/decision/start", payload, timeout=60):
                    st.error("Failed to save reject decision.")
                _advance()

        with btn_col3:
            if st.button("Skip All", use_container_width=True, key=f"improve_skip_{current_idx}"):
                st.session_state.pop(_improve_faces_state_key(ep_id, "active"), None)
                st.session_state.pop(_improve_faces_state_key(ep_id, "suggestions"), None)
                st.session_state.pop(_improve_faces_state_key(ep_id, "index"), None)
                st.rerun()

    _dialog()


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

            crop_url = (
                track.get("crop_url")
                or track.get("rep_thumb_url")
                or track.get("rep_media_url")
            )
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
    to ensure the UI shows fresh data. Also clears cross-page caches used by
    Smart Suggestions and Singletons Review pages.
    """
    _fetch_identities_cached.clear()
    _fetch_people_cached.clear()
    _fetch_cast_cached.clear()
    _fetch_cluster_track_reps_cached.clear()
    _fetch_unlinked_entities.clear()  # Critical: clears "Needs Assignment" cache

    # Clear session state caches if they exist (safe copy to avoid mutation during iteration)
    # Include cross-page cache keys for Smart Suggestions and Singletons Review
    cache_prefixes = (
        "cast_carousel_cache",
        "_thumb_result_cache",
        "cast_suggestions:",  # Smart Suggestions page
        "dismissed_suggestions:",  # Smart Suggestions page
        "dismissed_loaded:",  # Smart Suggestions page
        "people_cache:",  # Smart Suggestions page
    )
    keys_to_clear = [
        key for key in list(st.session_state.keys())
        if any(key.startswith(prefix) for prefix in cache_prefixes)
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)  # Use pop with default to avoid KeyError

    # Also clear Streamlit's built-in cache to ensure fresh data
    try:
        st.cache_data.clear()
    except Exception:
        pass  # Ignore if cache clearing fails


def _persist_and_refresh_cast_suggestions(
    ep_id: str,
) -> tuple[Dict[str, List[Dict[str, Any]]], Optional[int]]:
    """Persist assignments and recompute cast suggestions from latest embeddings."""
    _invalidate_assignment_caches()
    st.session_state.pop(_cast_suggestions_cache_key(ep_id), None)
    st.session_state.pop(_dismissed_suggestions_cache_key(ep_id), None)

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
        st.session_state[_cast_suggestions_cache_key(ep_id)] = suggestions_map

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
        params = {"ids": ids_param, "fields": "id,track_id,faces_count,frames"}
        if _CURRENT_RUN_ID:
            params["run_id"] = _CURRENT_RUN_ID
        resp = requests.get(
            f"{api_base}/episodes/{ep_id}/tracks",
            params=params,
            timeout=30,
        )
        if resp.status_code == 200:
            batch_resp = resp.json()
            if batch_resp and isinstance(batch_resp, dict):
                items = batch_resp.get("tracks") or batch_resp.get("items") or []
                # Validate items is actually a list to avoid iteration errors
                if not isinstance(items, list):
                    items = []
                meta: Dict[int, Dict[str, Any]] = {}
                for item in items:
                    if not isinstance(item, dict):
                        continue
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
        params = {"run_id": _CURRENT_RUN_ID} if _CURRENT_RUN_ID else None
        return helpers.api_post(path, payload or {}, timeout=timeout, params=params)
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
        params = {"run_id": _CURRENT_RUN_ID} if _CURRENT_RUN_ID else None
        resp = requests.delete(f"{base}{path}", params=params, json=payload or {}, timeout=60)
        resp.raise_for_status()
        return resp.json() if resp.content else {}
    except requests.RequestException as exc:
        st.error(helpers.describe_error(f"{base}{path}", exc))
        return None


def _trigger_run_stage_job(
    ep_id: str,
    stage: str,
    params: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    payload = {"params": params or {}}
    return _api_post(f"/episodes/{ep_id}/runs/{_CURRENT_RUN_ID}/jobs/{stage}", payload)


def _render_run_status_panel(ep_id: str, run_id: str | None, bundle: Dict[str, Any]) -> None:
    if not run_id:
        return
    run_state = bundle.get("run_state") or {}
    validation = bundle.get("validation") or {}
    if not run_state:
        st.info("Run status unavailable for this attempt.")
        return

    st.subheader("Run Status / Health")
    stages = run_state.get("stages") or {}
    artifacts = run_state.get("artifacts") or {}
    faces_artifacts = artifacts.get("faces") or {}
    stage_order = [
        ("detect_track", "Detect/Track"),
        ("faces_embed", "Faces Embed"),
        ("cluster", "Cluster"),
        ("screentime", "Screentime"),
        ("export", "Export"),
    ]

    def _stage_label(state_value: str | None) -> str:
        value = (state_value or "pending").lower()
        badge = {
            "pending": "⏳",
            "queued": "⏸",
            "running": "🚧",
            "done": "✅",
            "failed": "❌",
        }.get(value, "⏳")
        return f"{badge} {value}"

    for stage_key, stage_label in stage_order:
        entry = stages.get(stage_key) or {}
        state_value = entry.get("state")
        progress_value = entry.get("progress")
        last_error = entry.get("last_error")
        col_label, col_status, col_action = st.columns([2, 2, 2])
        with col_label:
            st.write(stage_label)
        with col_status:
            if isinstance(progress_value, (int, float)) and state_value in {"running", "queued"}:
                st.write(f"{_stage_label(state_value)} ({progress_value * 100:.0f}%)")
            else:
                st.write(_stage_label(state_value))
            if last_error and state_value == "failed":
                st.caption(str(last_error)[:200])
        with col_action:
            if state_value in {None, "pending", "failed"}:
                button_label = "Retry" if state_value == "failed" else "Run"
                if st.button(f"{button_label} {stage_label}", key=f"run_stage_{stage_key}"):
                    resp = _trigger_run_stage_job(ep_id, stage_key)
                    if resp and resp.get("status") in {"queued", "existing"}:
                        st.success(f"{stage_label} job queued.")
                        st.rerun()
                    else:
                        st.error("Failed to start job. See API logs.")

    faces_source = faces_artifacts.get("source") or "unknown"
    faces_manifest_key = faces_artifacts.get("manifest_key") or faces_artifacts.get("s3_key")
    faces_manifest_exists = faces_artifacts.get("manifest_exists")
    if faces_manifest_exists is None:
        faces_manifest_exists = faces_artifacts.get("exists")
    if faces_manifest_key:
        st.caption(
            f"Faces source: {faces_source} · manifest: {faces_manifest_key} "
            f"({'present' if faces_manifest_exists else 'missing'})"
        )

    summary = validation.get("summary") if isinstance(validation, dict) else None
    if isinstance(summary, dict):
        error_count = int(summary.get("error_count", 0) or 0)
        warning_count = int(summary.get("warning_count", 0) or 0)
        unclustered_tracks = int(summary.get("unclustered_tracks", 0) or 0)
        if error_count:
            st.error(f"Validator found {error_count} blocking issue(s).")
        elif warning_count:
            st.warning(f"Validator found {warning_count} warning(s).")
        else:
            st.success("Validator checks look good.")
        if unclustered_tracks:
            st.info(f"Unclustered tracks: {unclustered_tracks}")
    if validation:
        with st.expander("Validator details", expanded=False):
            errors = validation.get("errors", [])
            warnings = validation.get("warnings", [])
            if errors:
                st.write("Errors")
                for entry in errors:
                    st.write(f"- {entry.get('code')}: {entry.get('message')}")
            if warnings:
                st.write("Warnings")
                for entry in warnings:
                    st.write(f"- {entry.get('code')}: {entry.get('message')}")
            if isinstance(summary, dict):
                by_reason = summary.get("unclustered_tracks_by_reason") or {}
                sample_tracks = summary.get("unclustered_track_samples") or []
                if by_reason:
                    st.write("Unclustered breakdown")
                    st.write(by_reason)
                if sample_tracks:
                    st.write(f"Sample unclustered tracks: {', '.join(str(tid) for tid in sample_tracks)}")


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


def _focus_cast_members(cast_id: str | None, cast_name: str | None = None) -> None:
    """Switch to Cast Members view and focus on a cast member after assignment."""
    _set_view("people")
    if cast_id:
        st.session_state["filter_cast_id"] = cast_id
        st.session_state["filter_cast_name"] = cast_name or cast_id


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
    _invalidate_assignment_caches()  # Clear cached data so UI reflects changes immediately
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
    _invalidate_assignment_caches()  # Clear cached data so UI reflects changes immediately
    if cast_id:
        _focus_cast_members(cast_id, cleaned)
    st.rerun()


def _set_cluster_assignment(ep_id: str, cluster_id: str, cast_id: str | None) -> bool:
    payload = {
        "cluster_id": cluster_id,
        "cast_id": cast_id,
        "source": "manual",
    }
    resp = _api_post(f"/episodes/{ep_id}/assignments/cluster", payload)
    return bool(resp)


def _set_track_override(ep_id: str, track_id: int, cast_id: str | None) -> bool:
    payload = {
        "track_id": track_id,
        "cast_id": cast_id,
        "source": "manual",
    }
    resp = _api_post(f"/episodes/{ep_id}/assignments/track", payload)
    return bool(resp)


def _set_face_exclusion(
    ep_id: str,
    face_id: str,
    *,
    excluded: bool = True,
    reason: str | None = None,
    track_id: int | None = None,
) -> bool:
    payload = {
        "face_id": face_id,
        "excluded": excluded,
        "reason": reason,
        "source": "manual",
    }
    if track_id is not None:
        payload["track_id"] = track_id
    resp = _api_post(f"/episodes/{ep_id}/assignments/face_exclusion", payload)
    return bool(resp)


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
    skipped_locked = resp.get("skipped_locked", 0) or resp.get("skipped_locked_count", 0) or 0
    failed = resp.get("failed", 0)
    if assigned > 0:
        st.toast(f"Assigned {assigned} track(s) to '{cleaned}'")
    if skipped_locked > 0:
        st.warning(f"Skipped {skipped_locked} track(s) because their identities are locked.")
    if failed > 0:
        st.warning(f"{failed} track(s) failed to assign. Check logs.")
    # Clear bulk selection state for all identity keys
    for key in list(st.session_state.keys()):
        if key.startswith("bulk_track_sel::"):
            st.session_state[key] = set()
    _refresh_roster_names(show)
    if cast_id:
        _focus_cast_members(cast_id, cleaned)
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
    _invalidate_assignment_caches()  # Clear cached data so UI reflects changes immediately
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
    _invalidate_assignment_caches()  # Clear caches so UI reflects changes
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
    current_scope = _run_scope_token()
    if st.session_state.get("facebank_ep") != ep_id or st.session_state.get("facebank_run_scope") != current_scope:
        old_ep_id = st.session_state.get("facebank_ep")
        old_scope = st.session_state.get("facebank_run_scope") or "legacy"
        st.session_state["facebank_ep"] = ep_id
        st.session_state["facebank_run_scope"] = current_scope
        st.session_state["facebank_view"] = "people"
        st.session_state["selected_person"] = None
        st.session_state["selected_identity"] = None
        st.session_state["selected_track"] = None
        st.session_state.pop("facebank_query_applied", None)

        # Clear stale cast suggestions from previous episode
        if old_ep_id:
            st.session_state.pop(f"cast_suggestions:{old_ep_id}:{old_scope}", None)
            st.session_state.pop(f"dismissed_suggestions:{old_ep_id}:{old_scope}", None)

        # Clear pagination keys from previous episode (track_page_{old_ep_id}_* keys)
        if old_ep_id:
            prefix = f"track_page_{old_ep_id}_"
            keys_to_remove = [k for k in st.session_state.keys() if k.startswith(prefix)]
            for k in keys_to_remove:
                st.session_state.pop(k, None)

        # Clear run-scoped UI selections (bulk selection, etc.) when the scope changes.
        for key in list(st.session_state.keys()):
            if key.startswith("bulk_track_sel::"):
                st.session_state.pop(key, None)

        # Switching run_id scopes should not reuse cached API data.
        for cached in (
            _fetch_identities_cached,
            _fetch_unlinked_entities,
            _fetch_track_detail_cached,
            _fetch_cluster_track_reps_cached,
            _fetch_cluster_metrics_cached,
            _fetch_track_metrics_cached,
        ):
            try:
                cached.clear()  # type: ignore[attr-defined]
            except Exception:
                pass


def _hydrate_view_from_query(ep_id: str) -> None:
    """Allow deep links to jump directly into person/cluster/track views."""
    applied_token = st.session_state.get("facebank_query_applied")
    desired_token = f"{ep_id}::{_run_scope_token()}"
    if applied_token == desired_token:
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
    # low_quality param: auto-enable "Show skipped faces" for tracks from low-quality clusters
    low_quality = _single("low_quality") in ("true", "1", "yes")

    # Normalize view - accept both internal names and URL slugs
    # URL slugs from VIEW_NAMES: cast, person, cast-tracks, cluster, frames
    URL_SLUG_TO_VIEW = {
        "cast": "people",
        "person": "person_clusters",
        "cast-tracks": "cast_tracks",
        "cluster": "cluster_tracks",
        "frames": "track",
    }
    valid_views = {"people", "person_clusters", "cluster_tracks", "track", "cast_tracks"}
    if target_view in URL_SLUG_TO_VIEW:
        target_view = URL_SLUG_TO_VIEW[target_view]
    elif target_view not in valid_views:
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
            low_quality=low_quality,
        )
        st.session_state["facebank_query_applied"] = desired_token


def _set_view(
    view: str,
    *,
    person_id: str | None = None,
    identity_id: str | None = None,
    track_id: int | None = None,
    low_quality: bool = False,
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
            # Auto-enable "Show skipped faces" for low-quality clusters (all faces skipped)
            if low_quality:
                show_skipped_key = f"show_skipped_{ep_id}_{track_id}"
                st.session_state[show_skipped_key] = True
                st.session_state[f"{show_skipped_key}::prev"] = True

    # Update URL with view name for easy reference
    _, view_slug = VIEW_NAMES.get(view, ("Unknown", "unknown"))
    try:
        if _CURRENT_RUN_ID:
            st.query_params["run_id"] = _CURRENT_RUN_ID
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
        if _CURRENT_RUN_ID:
            st.caption(f"Run: `{_CURRENT_RUN_ID}`")
    with cols[1]:
        st.caption(f"S3 v2: `{detail['s3']['v2_key']}`")
    with cols[2]:
        st.caption(f"Local video: {'✅' if detail['local']['exists'] else '❌'}")
    if not detail["local"]["exists"]:
        if st.button("Mirror from S3", key="facebank_mirror"):
            if _api_post(f"/episodes/{ep_id}/mirror"):
                st.success("Mirror complete.")
                st.rerun()
    action_cols = st.columns([1, 2, 2])

    def _open_episode_detail() -> None:
        try:
            st.query_params["ep_id"] = ep_id
            if _CURRENT_RUN_ID:
                st.query_params["run_id"] = _CURRENT_RUN_ID
        except Exception:
            pass
        helpers.try_switch_page("pages/2_Episode_Detail.py")

    action_cols[0].button(
        "Open Episode Detail",
        key="facebank_open_detail",
        on_click=_open_episode_detail,
    )
    with action_cols[1]:
        # Row of action buttons: REFRESH and AUTO-ASSIGN
        btn_row = st.columns([1, 1])
        with btn_row[0]:
            refresh_clicked = st.button(
                "🔄 Refresh",
                key="facebank_refresh_similarity_button",
                type="primary",
                help="Recompute similarity scores, refresh suggestions, and save progress",
            )
        with btn_row[1]:
            auto_assign_clicked = st.button(
                "🔗 Auto-assign",
                key="facebank_auto_assign_button",
                help="Automatically assign clusters to cast members when facebank similarity ≥85%",
            )

        # Recovery controls
        with st.popover("🔧 Recover Noise Tracks", help="Expand single-frame tracks by finding similar faces"):
            st.markdown("**Recovery Settings:**")

            # Show last recovery timestamp if available
            last_recovery = st.session_state.get(f"last_recovery:{ep_id}")
            if last_recovery:
                try:
                    dt = datetime.datetime.fromisoformat(last_recovery)
                    st.caption(f"🕐 Last run: {dt.strftime('%b %d, %H:%M')}")
                except (ValueError, TypeError):
                    pass

            # Configurable settings
            frame_window = st.slider(
                "Frame Window (±)",
                min_value=1,
                max_value=30,
                value=st.session_state.get("recovery_frame_window", 8),
                key="recovery_frame_window_slider",
                help="Number of frames to search before/after each single-frame track",
            )
            st.session_state["recovery_frame_window"] = frame_window

            min_similarity = st.slider(
                "Min Similarity (%)",
                min_value=50,
                max_value=100,
                value=st.session_state.get("recovery_min_similarity", 70),
                key="recovery_min_similarity_slider",
                help="Minimum face similarity to merge (higher = stricter matching)",
            )
            st.session_state["recovery_min_similarity"] = min_similarity

            st.caption(f"Settings: ±{frame_window} frames, ≥{min_similarity}% similarity")

            # Preview button
            st.markdown("---")
            if st.button("👁️ Preview", key="recovery_preview_btn", help="See what would be recovered"):
                preview_resp = _safe_api_get(
                    f"/episodes/{ep_id}/recover_noise_tracks/preview"
                    f"?frame_window={frame_window}&min_similarity={min_similarity / 100:.2f}"
                )
                if preview_resp:
                    st.info("**Preview:**")
                    st.write(f"• Single-frame tracks: {preview_resp.get('single_frame_tracks', 0)}")
                    st.write(f"• Multi-frame tracks: {preview_resp.get('multi_frame_tracks', 0)}")
                    st.write(f"• Est. recoverable: ~{preview_resp.get('estimated_recoverable', 0)}")

            # Run button
            recover_clicked = st.button(
                "▶️ Run Recovery",
                key="recover_noise_tracks",
                type="primary",
                help="Find and merge similar faces from adjacent frames",
            )

            # Undo last recovery
            last_recovery_backup = st.session_state.get(f"last_recovery_backup:{ep_id}")
            if last_recovery_backup:
                st.markdown("---")
                if st.button("↩️ Undo Last Recovery", key="undo_recovery_btn", help="Restore to state before last recovery"):
                    restore_resp = _api_post(f"/episodes/{ep_id}/restore/{last_recovery_backup}", {})
                    if restore_resp and restore_resp.get("files_restored", 0) > 0:
                        _invalidate_assignment_caches()
                        st.session_state.pop(f"last_recovery_backup:{ep_id}", None)
                        st.success("✓ Restored from recovery backup!")
                        st.rerun()
                    else:
                        st.error("Failed to restore from recovery backup.")

            # Show recovery history
            history = st.session_state.get(f"recovery_history:{ep_id}", [])
            if history:
                with st.expander(f"📜 Recovery History ({len(history)})", expanded=False):
                    for entry in history[:5]:
                        try:
                            dt = datetime.datetime.fromisoformat(entry["timestamp"])
                            time_str = dt.strftime("%b %d, %H:%M")
                        except (ValueError, TypeError, KeyError):
                            time_str = "Unknown"
                        fw = entry.get("frame_window", 8)
                        ms = int(entry.get("min_similarity", 0.7) * 100)
                        expanded = entry.get("tracks_expanded", 0)
                        merged = entry.get("faces_merged", 0)
                        st.caption(f"**{time_str}**: ±{fw} frames, ≥{ms}%")
                        st.caption(f"  ↳ {expanded} tracks expanded, {merged} faces merged")

        # Cluster Cleanup popover (flattened to avoid nested columns)
        with st.popover("🧹 Cluster Cleanup", help="Select which cleanup actions to run"):
            st.info(
                "Cleanup focuses on **Needs Cast Assignment** items only. "
                "It fixes noisy tracks, regenerates embeddings for faces that have crops, and regroups unassigned clusters without touching named cast links. "
                "**Note:** Faces marked as 'blurry' cannot be re-embedded without re-running detection."
            )
            # Show last cleanup timestamp if available
            last_cleanup = st.session_state.get(f"last_cleanup:{ep_id}")
            if last_cleanup:
                try:
                    dt = datetime.datetime.fromisoformat(last_cleanup)
                    st.caption(f"🕐 Last cleanup: {dt.strftime('%b %d, %H:%M')}")
                except (ValueError, TypeError):
                    pass

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

                # Quick Cleanup Presets (two buttons in a single row to avoid deeper nesting)
                st.markdown("**Quick Presets:**")
                preset_col_left, preset_col_right = st.columns(2)
                with preset_col_left:
                    if st.button("🚀 Quick Fix", key="preset_quick", help="Low risk: Just fix tracking issues"):
                        st.session_state["cleanup_preset"] = "quick"
                        st.rerun()
                with preset_col_right:
                    if st.button("⚡ Standard", key="preset_standard", help="Recommended: Fix + reembed + regroup unassigned"):
                        st.session_state["cleanup_preset"] = "standard"
                        st.rerun()

                # Apply preset if set
                active_preset = st.session_state.get("cleanup_preset", "standard")
                preset_defaults = {
                    "quick": {"split_tracks": True, "reembed": False, "group_clusters": False},
                    "standard": {"split_tracks": True, "reembed": True, "group_clusters": True},
                }
                current_defaults = preset_defaults.get(active_preset, preset_defaults["standard"])

                st.markdown("---")
                st.markdown("**Select cleanup actions (unassigned-only):**")

                # Define actions with risk levels (defaults come from preset)
                cleanup_actions = {
                    "split_tracks": {
                        "label": "Fix tracking issues (split_tracks)",
                        "help": "Use when: A track contains multiple different people (identity switch mid-track). Splits incorrectly merged tracks. Low risk - usually beneficial.",
                        "default": current_defaults.get("split_tracks", True),
                        "risk": "low",
                        "est_time": "~30s",
                    },
                    "reembed": {
                        "label": "Regenerate embeddings (reembed)",
                        "help": "Use when: Face quality has changed or embeddings seem outdated. Recalculates face embeddings for unassigned clusters. Low risk - just regenerates vectors.",
                        "default": current_defaults.get("reembed", True),
                        "risk": "low",
                        "est_time": "~1-2min",
                    },
                    "group_clusters": {
                        "label": "Auto-group clusters (group_clusters)",
                        "help": "Use when: You have unassigned clusters that need to be matched to people. Groups similar unassigned clusters into draft people (remains Needs Cast Assignment until named).",
                        "default": current_defaults.get("group_clusters", True),
                        "risk": "medium",
                        "est_time": "~1min",
                    },
                }

                selected_actions = []
                for action_key, action_info in cleanup_actions.items():
                    risk_badge = ""
                    if action_info["risk"] == "high":
                        risk_badge = " ⚠️"
                    elif action_info["risk"] == "medium":
                        risk_badge = " ⚡"

                    est_time = action_info.get("est_time", "")
                    time_badge = f" ({est_time})" if est_time else ""

                    checked = st.checkbox(
                        f"{action_info['label']}{risk_badge}{time_badge}",
                        value=action_info["default"],
                        key=f"cleanup_action_{action_key}",
                        help=action_info["help"],
                    )
                    if checked:
                        selected_actions.append(action_key)

                # Show total estimated time
                if selected_actions:
                    time_map = {"split_tracks": 30, "reembed": 90, "group_clusters": 60}
                    total_seconds = sum(time_map.get(a, 0) for a in selected_actions)
                    if total_seconds >= 60:
                        est_str = f"~{total_seconds // 60}min {total_seconds % 60}s" if total_seconds % 60 else f"~{total_seconds // 60}min"
                    else:
                        est_str = f"~{total_seconds}s"
                    st.caption(f"⏱️ Estimated total time: {est_str}")

                # Protection for recently-edited identities
                recently_edited = st.session_state.get(f"recently_edited_identities:{ep_id}", {})
                num_recent = len(recently_edited)
                protect_recent = st.checkbox(
                    f"🛡️ Protect recently-edited ({num_recent})",
                    value=num_recent > 0,
                    key="protect_recent_edits",
                    help="Skip identities you've manually edited during cleanup to preserve your work",
                    disabled=num_recent == 0,
                )
                if num_recent > 0 and protect_recent:
                    st.caption(f"  ↳ {num_recent} {'identity' if num_recent == 1 else 'identities'} will be protected")

                # Enhancement #7: Show backup/restore info
                backups_resp = _safe_api_get(f"/episodes/{ep_id}/backups")
                backups = backups_resp.get("backups", []) if backups_resp else []
                if backups:
                    latest = backups[0].get("backup_id", "")
                    st.caption(f"💾 Last backup: {latest[-20:] if len(latest) > 20 else latest}")
                    if st.button("↩️ Undo Last Cleanup", key="restore_backup_btn", help="Restore to previous state"):
                        restore_resp = _api_post(f"/episodes/{ep_id}/restore/{latest}", {})
                        if restore_resp and restore_resp.get("files_restored", 0) > 0:
                            _invalidate_assignment_caches()  # Clear caches so UI reflects restored state
                            st.success("✓ Restored from backup!")
                            st.rerun()
                        else:
                            st.error("Failed to restore from backup. Check API logs.")

                # Show cleanup history
                history = st.session_state.get(f"cleanup_history:{ep_id}", [])
                if history:
                    with st.expander(f"📜 Cleanup History ({len(history)})", expanded=False):
                        for i, entry in enumerate(history[:5]):  # Show last 5
                            try:
                                dt = datetime.datetime.fromisoformat(entry["timestamp"])
                                time_str = dt.strftime("%b %d, %H:%M")
                            except (ValueError, TypeError, KeyError):
                                time_str = "Unknown"
                            actions_str = ", ".join(entry.get("actions", []))
                            details_str = " · ".join(entry.get("details", []))
                            st.caption(f"**{time_str}**: {actions_str}")
                            if details_str:
                                st.caption(f"  ↳ {details_str}")

                st.markdown("---")

                # Dry-run option
                dry_run_cols = st.columns([1, 1])
                with dry_run_cols[0]:
                    if st.button("👁️ Preview Changes", key="cleanup_dry_run", help="Show what would change without making changes"):
                        if not selected_actions:
                            st.warning("No cleanup actions selected.")
                        else:
                            with st.spinner("Analyzing potential changes..."):
                                # Get detailed preview
                                preview_detail = _safe_api_get(f"/episodes/{ep_id}/cleanup_preview")
                                if preview_detail and preview_detail.get("preview"):
                                    p = preview_detail["preview"]
                                    st.info("**Dry Run Results:**")
                                    st.write(f"• Current clusters: {p.get('total_clusters', 0)}")
                                    st.write(f"• Assigned: {p.get('assigned_clusters', 0)}")
                                    st.write(f"• Unassigned: {p.get('unassigned_clusters', 0)}")
                                    if "recluster" in selected_actions:
                                        st.warning("⚠️ Recluster selected - all cluster assignments may be reset!")
                                    if "split_tracks" in selected_actions:
                                        st.write("• split_tracks: May fix tracks with multiple identities")
                                    if "reembed" in selected_actions:
                                        st.write("• reembed: Will regenerate all face embeddings")
                                    if "group_clusters" in selected_actions:
                                        merges = p.get("potential_merges", 0)
                                        st.write(f"• group_clusters: ~{merges} potential cluster merge(s)")

                with dry_run_cols[1]:
                    pass  # Placeholder for layout

                if st.button("Run Selected Cleanup", key="facebank_cleanup_button", type="primary"):
                    if not selected_actions:
                        st.warning("No cleanup actions selected.")
                    else:
                        # Enhancement #7: Auto-backup before cleanup
                        backup_resp = _api_post(f"/episodes/{ep_id}/backup", {})
                        if not backup_resp:
                            st.error("Failed to create backup before cleanup. Aborting.")
                        else:
                            payload = helpers.default_cleanup_payload(ep_id)
                            payload["actions"] = selected_actions
                            # Add protected identity IDs if enabled
                            if protect_recent and recently_edited:
                                payload["protected_identity_ids"] = list(recently_edited.keys())
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
                                # Build summary of changes
                                details: List[str] = []
                                tb = helpers.coerce_int(report.get("tracks_before"))
                                ta = helpers.coerce_int(report.get("tracks_after"))
                                cbefore = helpers.coerce_int(report.get("clusters_before"))
                                cafter = helpers.coerce_int(report.get("clusters_after"))
                                faces_after = helpers.coerce_int(report.get("faces_after"))
                                if tb is not None and ta is not None:
                                    track_delta = ta - tb
                                    delta_str = f"+{track_delta}" if track_delta > 0 else str(track_delta)
                                    details.append(f"Tracks: {tb} → {ta} ({delta_str})")
                                if cbefore is not None and cafter is not None:
                                    cluster_delta = cafter - cbefore
                                    delta_str = f"+{cluster_delta}" if cluster_delta > 0 else str(cluster_delta)
                                    details.append(f"Clusters: {cbefore} → {cafter} ({delta_str})")
                                if faces_after is not None:
                                    details.append(f"Faces: {faces_after}")
                                # Display success message with details
                                _invalidate_assignment_caches()  # Clear caches so UI reflects changes
                                # Track last cleanup timestamp and add to history
                                now_iso = datetime.datetime.now().isoformat()
                                st.session_state[f"last_cleanup:{ep_id}"] = now_iso
                                # Add to cleanup history (keep last 10)
                                history_key = f"cleanup_history:{ep_id}"
                                history = st.session_state.get(history_key, [])
                                history.insert(0, {
                                    "timestamp": now_iso,
                                    "actions": selected_actions,
                                    "details": details,
                                    "backup_id": backup_resp.get("backup_id") if backup_resp else None,
                                })
                                st.session_state[history_key] = history[:10]  # Keep last 10
                                if details:
                                    st.success(f"✓ Cleanup complete! {' · '.join(details)}")
                                else:
                                    st.success("✓ Cleanup complete!")
                                st.rerun()

    # Progress area below the buttons
    # Progress area below the buttons
    refresh_progress_area = st.empty()
    recovery_progress_area = st.empty()

    # Handle recovery button click
    if recover_clicked:
        # Get configured settings
        frame_window = st.session_state.get("recovery_frame_window", 8)
        min_similarity = st.session_state.get("recovery_min_similarity", 70) / 100.0

        with recovery_progress_area.container():
            with st.status(f"Recovering noise tracks (±{frame_window} frames, ≥{int(min_similarity*100)}% sim)...", expanded=True) as status:
                # Step 0: Create backup before recovery (for undo capability)
                st.write("💾 Creating backup...")
                progress_bar = st.progress(5, text="Creating backup...")
                backup_resp = _api_post(f"/episodes/{ep_id}/backup", {})
                recovery_backup_id = backup_resp.get("backup_id") if backup_resp else None
                if recovery_backup_id:
                    st.session_state[f"last_recovery_backup:{ep_id}"] = recovery_backup_id
                    st.write(f"✓ Backup created: {recovery_backup_id[-15:]}")

                # Step 1: Call the recovery API with configured parameters
                st.write("🔍 Analyzing single-frame tracks...")
                progress_bar.progress(10, text="Loading face data...")

                resp = _api_post(
                    f"/episodes/{ep_id}/recover_noise_tracks?frame_window={frame_window}&min_similarity={min_similarity:.2f}",
                    {}
                )
                progress_bar.progress(60, text="Recovery analysis complete...")

                if not resp:
                    progress_bar.progress(100, text="Failed")
                    status.update(label="❌ Recovery failed", state="error")
                    st.error("Failed to recover noise tracks. Check API logs.")
                else:
                    tracks_analyzed = resp.get("tracks_analyzed", 0)
                    tracks_expanded = resp.get("tracks_expanded", 0)
                    faces_merged = resp.get("faces_merged", 0)
                    details = resp.get("details", [])

                    progress_bar.progress(80, text="Generating report...")

                    # Step 2: Show analysis results
                    st.write(f"📊 Analyzed **{tracks_analyzed}** single-frame tracks")

                    if tracks_expanded > 0:
                        st.write(f"✅ Expanded **{tracks_expanded}** track(s) with **{faces_merged}** adjacent face(s)")

                        # Step 3: Show detailed log of recovered tracks
                        st.write("📋 **Recovery Details:**")
                        for detail in details[:10]:  # Show up to 10 details
                            track_id = detail.get("track_id")
                            original_frame = detail.get("original_frame")
                            added_frames = detail.get("added_frames", [])

                            # Build frame info
                            frame_info = []
                            for af in added_frames[:5]:  # Show up to 5 added frames per track
                                frame_idx = af.get("frame_idx")
                                similarity = af.get("similarity", 0)
                                frame_info.append(f"frame {frame_idx} ({int(similarity * 100)}% sim)")

                            frames_str = ", ".join(frame_info)
                            if len(added_frames) > 5:
                                frames_str += f" +{len(added_frames) - 5} more"

                            st.write(f"  • Track {track_id} (frame {original_frame}): +{len(added_frames)} → {frames_str}")

                        if len(details) > 10:
                            st.write(f"  ... and {len(details) - 10} more tracks recovered")

                        # Show which clusters gained faces (group by from_track to find affected clusters)
                        affected_source_tracks = set()
                        for detail in details:
                            for af in detail.get("added_frames", []):
                                src = af.get("from_track")
                                if src is not None:
                                    affected_source_tracks.add(src)
                        if affected_source_tracks:
                            st.write(f"📦 Faces moved from **{len(affected_source_tracks)}** other track(s)")

                        progress_bar.progress(100, text="Complete!")
                        status.update(
                            label=f"✅ Recovered {tracks_expanded} track(s) with {faces_merged} face(s)",
                            state="complete",
                        )

                        # Track last recovery timestamp and add to history
                        now_iso = datetime.datetime.now().isoformat()
                        st.session_state[f"last_recovery:{ep_id}"] = now_iso

                        # Add to recovery history (keep last 10)
                        history_key = f"recovery_history:{ep_id}"
                        history = st.session_state.get(history_key, [])
                        history.insert(0, {
                            "timestamp": now_iso,
                            "frame_window": frame_window,
                            "min_similarity": min_similarity,
                            "tracks_expanded": tracks_expanded,
                            "faces_merged": faces_merged,
                            "backup_id": recovery_backup_id,
                        })
                        st.session_state[history_key] = history[:10]

                        # Clear caches and refresh UI with new data
                        _invalidate_assignment_caches()
                        st.rerun()
                    else:
                        progress_bar.progress(100, text="No recoverable tracks found")
                        st.write(f"ℹ️ No similar faces found in adjacent frames (±{frame_window} frames, ≥{int(min_similarity*100)}% similarity)")
                        status.update(
                            label="✅ Analysis complete - no recoverable tracks",
                            state="complete",
                        )

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
                    # Show refresh stats
                    tracks_processed = refresh_resp.get("tracks_processed", 0)
                    centroids_computed = refresh_resp.get("centroids_computed", 0)
                    st.write(f"✓ Processed {tracks_processed} tracks")
                    st.write(f"✓ Computed {centroids_computed} cluster centroids")

                    # Step 2: Refresh cluster suggestions based on new similarity values
                    st.write("📊 Refreshing cluster suggestions...")
                    suggestions_resp = _safe_api_get(f"/episodes/{ep_id}/cluster_suggestions_from_assigned")

                    # Step 3: Fetch cast suggestions from facebank (Enhancement #1)
                    st.write("🎭 Fetching cast suggestions from facebank...")
                    cast_suggestions_resp = _safe_api_get(f"/episodes/{ep_id}/cast_suggestions")
                    if cast_suggestions_resp and cast_suggestions_resp.get("suggestions"):
                        # Store in session state for display
                        st.session_state[_cast_suggestions_cache_key(ep_id)] = {
                            sugg["cluster_id"]: sugg.get("cast_suggestions", [])
                            for sugg in cast_suggestions_resp.get("suggestions", [])
                        }
                        num_suggestions = len(cast_suggestions_resp.get("suggestions", []))
                        st.write(f"✓ Found {num_suggestions} cluster(s) with cast suggestions")

                    # Update status
                    status.update(label="✅ Refresh complete!", state="complete")

                    st.rerun()

    if auto_assign_clicked:
        with refresh_progress_area.container():
            with st.status("Auto-assigning clusters to cast...", expanded=True) as status:
                st.write("🔗 Finding high confidence matches...")
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
                    status.update(label=f"✅ Auto-assigned {auto_linked_count} cluster(s)", state="complete")
                    st.rerun()
                else:
                    st.write("ℹ️ No high-confidence matches found to auto-assign")
                    status.update(label="✅ No matches to auto-assign", state="complete")

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
    run_id = _CURRENT_RUN_ID
    run_ids, legacy_ids = faces_review_run_scoped.split_cluster_ids(
        person.get("cluster_ids", []) or [],
        ep_id,
        run_id,
    )
    if run_id:
        if run_ids:
            return run_ids
        if _LEGACY_PEOPLE_FALLBACK:
            return legacy_ids
        return []
    return legacy_ids


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
    run_id = _CURRENT_RUN_ID
    if run_id:
        try:
            run_id_norm = run_layout.normalize_run_id(run_id)
            path = run_layout.run_root(ep_id, run_id_norm) / "cluster_centroids.json"
        except ValueError:
            path = Path("missing")
    else:
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
    include_skipped: bool = False,
) -> tuple[List[Dict[str, Any]], str | None]:
    """Fetch track media using face-metadata-backed URLs for correct track scoping.

    Uses /frames endpoint instead of /crops to ensure we get track-specific URLs
    from face metadata, which correctly identifies which crop belongs to which track
    even when multiple tracks share the same frame.
    """
    # Parse cursor for page-based pagination or start_after cursor
    page = 1
    start_after: str | None = None
    if cursor:
        try:
            page = int(cursor)
        except (TypeError, ValueError):
            start_after = str(cursor)

    # Use /frames endpoint which provides face-metadata-backed URLs
    # This ensures correct track-specific crops even for shared frames
    params: Dict[str, Any] = {
        "sample": int(sample),
        "page": page,
        "page_size": int(limit),
        "include_skipped": include_skipped,
    }
    if start_after:
        params["start_after"] = start_after
    payload = _safe_api_get(f"/episodes/{ep_id}/tracks/{track_id}/frames", params=params) or {}
    items = payload.get("items", []) if isinstance(payload, dict) else []
    total = payload.get("total", 0)
    current_page = payload.get("page", 1)
    page_size = payload.get("page_size", limit)
    next_start_after = payload.get("next_start_after") if isinstance(payload, dict) else None

    # Determine next cursor (next page number) if there are more items
    next_cursor: str | None = None
    if next_start_after:
        next_cursor = str(next_start_after)
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
    include_skipped: bool = False,
) -> Dict[str, Any]:
    params = {
        "sample": int(sample),
        "page": int(page),
        "page_size": int(page_size),
        "include_skipped": include_skipped,
    }
    payload = _safe_api_get(f"/episodes/{ep_id}/tracks/{track_id}/frames", params=params)
    if isinstance(payload, list):
        return {"items": payload}
    if isinstance(payload, dict):
        return payload
    return {}


def _render_track_media_section(
    ep_id: str, track_id: int, *, sample: int, include_skipped: bool = False
) -> None:
    """Show cached crops with lazy pagination for the active track."""
    # Cache key now includes sample rate and include_skipped, so we get the correct state
    # Append "_skipped" suffix when showing skipped faces
    cache_suffix = "_skipped" if include_skipped else ""
    state = _track_media_state(ep_id, track_id, sample, cache_suffix=cache_suffix)
    if not state.get("initialized"):
        batch_limit = TRACK_MEDIA_BATCH_LIMIT
        items, cursor = _fetch_track_media(
            ep_id,
            track_id,
            sample=sample,
            limit=batch_limit,
            include_skipped=include_skipped,
        )
        state.update(
            {
                "items": items,
                "cursor": cursor,
                "initialized": True,
                "sample": sample,
                "batch_limit": batch_limit,
                "include_skipped": include_skipped,
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
                include_skipped=state.get("include_skipped", False),
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
                        resolved = helpers.resolve_thumb(url)
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
                    _invalidate_assignment_caches()  # Clear caches so UI reflects changes
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
                    _invalidate_assignment_caches()  # Clear caches so UI reflects changes
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


def _render_cast_carousel(cast_cards: List[Dict[str, Any]]) -> None:
    """Render featured cast members carousel at the top (run-scoped inputs)."""
    if not cast_cards:
        return

    header_cols = st.columns([3, 1])
    with header_cols[0]:
        st.markdown("### 🎬 Cast Lineup")
        st.caption("Cast members with clusters in this episode")

    # Create horizontal carousel (max 5 per row)
    max_cols_per_row = min(len(cast_cards), 5)

    for row_start in range(0, len(cast_cards), max_cols_per_row):
        row_items = cast_cards[row_start : row_start + max_cols_per_row]
        # Use actual row item count to avoid empty columns on last row
        cols = st.columns(len(row_items))

        for idx, card in enumerate(row_items):
            with cols[idx]:
                cast_info = card.get("cast") or {}
                person = card.get("person") or {}
                cast_id = cast_info.get("cast_id") or person.get("cast_id")
                name = cast_info.get("name") or person.get("name") or "(unnamed)"
                episode_clusters = card.get("episode_clusters") or []
                featured_url = card.get("featured_thumbnail")

                # Display featured image
                if featured_url:
                    resolved = helpers.resolve_thumb(featured_url)
                    thumb_markup = helpers.thumb_html(resolved, alt=name, hide_if_missing=False)
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
    raw_tracks = cluster_meta.get("tracks")
    has_track_details = isinstance(raw_tracks, list)
    track_list = raw_tracks if has_track_details else []
    missing_track_details = not has_track_details and original_tracks_count > 0

    # For single-track clusters, show even single-frame tracks (with low-confidence badge)
    # For multi-track clusters, filter out single-frame tracks as noise UNLESS that would leave 0 tracks
    original_track_list = track_list
    is_single_track_cluster = len(track_list) == 1
    has_single_frame_track = any(t.get("faces", 0) <= 1 for t in track_list)

    if is_single_track_cluster:
        # Keep single-frame tracks for single-track clusters (show with low-confidence badge)
        filtered_tracks_count = 0
    else:
        # Filter out single-frame tracks for multi-track clusters
        multi_frame_tracks = [t for t in track_list if t.get("faces", 0) > 1]
        if multi_frame_tracks:
            # We have some multi-frame tracks, filter out single-frame ones
            track_list = multi_frame_tracks
            filtered_tracks_count = len(original_track_list) - len(track_list)
        else:
            # ALL tracks are single-frame — show them with a warning rather than hiding everything
            filtered_tracks_count = 0  # Don't filter, show all with warning

    # Recalculate counts after filtering
    tracks_count = len(track_list)
    faces_count = sum(t.get("faces", 0) for t in track_list)
    if missing_track_details:
        tracks_count = original_tracks_count
        faces_count = original_faces_count

    # Show filtered-out cluster with explanation instead of silently skipping
    if (not track_list or tracks_count == 0) and not missing_track_details:
        with st.container(border=True):
            st.markdown(f"### 🔍 Cluster `{cluster_id}`")
            # Distinguish between empty clusters (no tracks) vs filtered clusters (all single-frame)
            if not original_track_list:
                # Truly empty cluster - no tracks at all
                st.warning("⚠️ This cluster is empty (no tracks).")
                st.caption(
                    "Empty clusters occur when all tracks are moved/merged elsewhere. "
                    "Delete to clean up, or use 'Cleanup Empty Clusters' in the sidebar."
                )
            else:
                # Had tracks but all were filtered as single-frame noise
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
                        _invalidate_assignment_caches()  # Clear caches so UI reflects changes
                        st.success(f"Deleted cluster {cluster_id}")
                        st.rerun()
                    else:
                        st.error("Failed to delete cluster")
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
        col1, col2, col3, col4, col5, col6 = st.columns([3, 1, 1, 1, 1, 1])
        with col1:
            st.markdown(f"### 🔍 Cluster `{cluster_id}`")
            caption_parts = [f"{tracks_count} track(s) · {faces_count} frame(s)"]
            if filtered_tracks_count > 0:
                caption_parts.append(f"({filtered_tracks_count} single-frame filtered)")
            st.caption(" ".join(caption_parts))
            if missing_track_details:
                st.warning(
                    "Track details are unavailable for this cluster. Refresh or regenerate track reps to view frames."
                )
            # Show similarity badge - use cluster cohesion for multi-track, internal similarity for single-track
            similarity_value = None
            similarity_label = None
            similarity_type = None
            if tracks_count > 1 and cluster_cohesion is not None:
                # Multi-track cluster: show cluster cohesion (how similar tracks are to each other)
                similarity_value = cluster_cohesion
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
            if similarity_value is not None and similarity_label and similarity_type:
                badge = render_similarity_badge(similarity_value, similarity_type, show_label=True, custom_label=similarity_label)
                st.markdown(badge, unsafe_allow_html=True)
            # Low-confidence badge for single-frame single-track clusters
            if is_single_track_cluster and has_single_frame_track:
                st.markdown(
                    '<span style="background-color: #9E9E9E; color: white; '
                    'padding: 2px 6px; border-radius: 3px; font-size: 0.75em;">⚠️ Single-Frame</span>',
                    unsafe_allow_html=True
                )
            # Quality indicators (Feature 10)
            quality_badges = render_cluster_quality_badges(cluster_quality_data, compact=True, max_badges=2)
            if quality_badges:
                st.markdown(quality_badges, unsafe_allow_html=True)

            # New metrics strip (Nov 2024) - temporal, ambiguity, isolation + track metrics for single-track
            new_metrics = []

            # For single-track clusters, add track-level metrics first
            if is_single_track_cluster and track_list:
                track_data = track_list[0]
                track_id_val = track_data.get("track_id")
                if track_id_val:
                    track_metrics = _fetch_track_metrics_cached(ep_id, track_id_val)
                    if track_metrics:
                        # Track similarity (internal consistency)
                        if track_metrics.get("track_similarity") is not None:
                            new_metrics.append(MetricData(
                                metric_type="track",
                                value=track_metrics["track_similarity"],
                                excluded=track_metrics.get("excluded_frames"),
                            ))
                        # Person cohesion (how well track fits with assigned person)
                        if track_metrics.get("person_cohesion") is not None:
                            new_metrics.append(MetricData(
                                metric_type="person_cohesion",
                                value=track_metrics["person_cohesion"],
                            ))
                    # Also use track-level data from cluster lookup if API didn't return
                    elif track_data.get("internal_similarity") is not None:
                        new_metrics.append(MetricData(
                            metric_type="track",
                            value=track_data["internal_similarity"],
                            label="Track Consistency",
                        ))

            # Cluster-level metrics
            cluster_metrics = _fetch_cluster_metrics_cached(ep_id, cluster_id)
            if cluster_metrics:
                # Temporal consistency
                if cluster_metrics.get("temporal_consistency") is not None:
                    new_metrics.append(MetricData(
                        metric_type="temporal",
                        value=cluster_metrics["temporal_consistency"],
                    ))
                # Ambiguity (gap to 2nd best match)
                if cluster_metrics.get("ambiguity") is not None:
                    new_metrics.append(MetricData(
                        metric_type="ambiguity",
                        value=cluster_metrics["ambiguity"],
                        first_match=cluster_metrics.get("first_match"),
                        second_match=cluster_metrics.get("second_match"),
                    ))
                # Isolation (distance to nearest cluster)
                if cluster_metrics.get("isolation") is not None:
                    new_metrics.append(MetricData(
                        metric_type="isolation",
                        value=cluster_metrics["isolation"],
                    ))
            if new_metrics:
                render_metrics_strip(new_metrics, compact=True, strip_id=f"unassigned_{cluster_id}")
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
                        st.session_state[_cast_suggestions_cache_key(ep_id)] = st.session_state.get(
                            _cast_suggestions_cache_key(ep_id), {}
                        )
                        st.session_state[_cast_suggestions_cache_key(ep_id)][cluster_id] = suggest_resp["suggestions"]
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
                    _invalidate_assignment_caches()  # Clear caches so UI reflects changes
                    st.success(f"Deleted cluster {cluster_id}")
                    st.rerun()
                else:
                    st.error("Failed to delete cluster")
        with col6:
            # Copy debug info button
            debug_info_lines = [
                f"=== CLUSTER DEBUG INFO ===",
                f"cluster_id: {cluster_id}",
                f"episode_id: {ep_id}",
                f"show_id: {show_id}",
                f"",
                f"--- COUNTS ---",
                f"tracks: {tracks_count} (original: {original_tracks_count})",
                f"faces: {faces_count} (original: {original_faces_count})",
                f"filtered_single_frame_tracks: {filtered_tracks_count}",
                f"",
                f"--- CLUSTER META ---",
                f"cohesion: {cluster_meta.get('cohesion')}",
            ]
            # Add metrics if available
            cluster_metrics = _fetch_cluster_metrics_cached(ep_id, cluster_id)
            if cluster_metrics:
                debug_info_lines.append("")
                debug_info_lines.append("--- METRICS (from API) ---")
                debug_info_lines.append(f"temporal_consistency: {cluster_metrics.get('temporal_consistency')}")
                debug_info_lines.append(f"ambiguity: {cluster_metrics.get('ambiguity')}")
                debug_info_lines.append(f"isolation: {cluster_metrics.get('isolation')}")
                debug_info_lines.append(f"first_match: {cluster_metrics.get('first_match')}")
                debug_info_lines.append(f"second_match: {cluster_metrics.get('second_match')}")
                if cluster_metrics.get("error"):
                    debug_info_lines.append(f"error: {cluster_metrics.get('error')}")
            else:
                debug_info_lines.append("")
                debug_info_lines.append("--- METRICS ---")
                debug_info_lines.append("(no metrics returned from API)")
            # Add track details
            debug_info_lines.append("")
            debug_info_lines.append("--- TRACKS ---")
            debug_info_lines.append(f"is_single_track_cluster: {is_single_track_cluster}")
            for i, t in enumerate(track_list[:10]):  # Limit to 10 tracks
                debug_info_lines.append(
                    f"  [{i}] track_id={t.get('track_id')}, faces={t.get('faces')}, "
                    f"internal_sim={t.get('internal_similarity')}, sim={t.get('similarity')}"
                )
            if len(track_list) > 10:
                debug_info_lines.append(f"  ... and {len(track_list) - 10} more tracks")
            # For single-track clusters, include track-level API metrics
            if is_single_track_cluster and track_list:
                track_id_val = track_list[0].get("track_id")
                if track_id_val:
                    track_metrics = _fetch_track_metrics_cached(ep_id, track_id_val)
                    debug_info_lines.append("")
                    debug_info_lines.append("--- TRACK METRICS (single-track cluster) ---")
                    if track_metrics:
                        debug_info_lines.append(f"track_similarity: {track_metrics.get('track_similarity')}")
                        debug_info_lines.append(f"person_cohesion: {track_metrics.get('person_cohesion')}")
                        debug_info_lines.append(f"avg_quality: {track_metrics.get('avg_quality')}")
                        debug_info_lines.append(f"excluded_frames: {track_metrics.get('excluded_frames')}")
                        if track_metrics.get("error"):
                            debug_info_lines.append(f"error: {track_metrics.get('error')}")
                    else:
                        debug_info_lines.append("(no track metrics from API)")
            # Add suggestion info
            if suggestion:
                debug_info_lines.append("")
                debug_info_lines.append("--- SUGGESTION ---")
                debug_info_lines.append(f"suggested_person_id: {suggestion.get('suggested_person_id')}")
                debug_info_lines.append(f"distance: {suggestion.get('distance')}")
            if cast_suggestions:
                debug_info_lines.append("")
                debug_info_lines.append("--- CAST SUGGESTIONS ---")
                for cs in cast_suggestions[:5]:
                    debug_info_lines.append(
                        f"  {cs.get('name')}: sim={cs.get('similarity')}, conf={cs.get('confidence')}"
                    )
            debug_info = "\n".join(debug_info_lines)
            # Use st.code for copyable text with a popover
            with st.popover("📋", help="Copy debug info"):
                st.text_area("Debug Info", debug_info, height=300, key=f"debug_{cluster_id}")
                st.caption("Select all (Cmd+A) and copy (Cmd+C)")

        # Display one representative frame from each track in the cluster with scrollable carousel
        if track_list:
            num_tracks = len(track_list)
            max_visible = 12  # Show up to 12 tracks at a time

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

            # Create rows of 6 tracks each (prevents overlap with assignment controls)
            TRACKS_PER_ROW = 6
            if visible_tracks:
                for row_start in range(0, len(visible_tracks), TRACKS_PER_ROW):
                    row_tracks = visible_tracks[row_start:row_start + TRACKS_PER_ROW]
                    # Create exactly 6 columns per row for consistent layout
                    cols = st.columns(TRACKS_PER_ROW)
                    for idx, track in enumerate(row_tracks):
                        with cols[idx]:
                            thumb_url = track.get("rep_thumb_url")
                            track_id = track.get("track_id")
                            track_faces = track.get("faces", 0)
                            track_sim = track.get("similarity")
                            track_internal_sim = track.get("internal_similarity")
                            # Prefer explicit track similarity, else internal similarity as fallback
                            track_sim_value = track_sim if track_sim is not None else track_internal_sim
                            # Nov 2024: Enhanced with dropout indicator
                            track_badge = None
                            if track_sim_value is not None:
                                excluded_frames = track.get("excluded_frames", 0)
                                track_badge = render_track_with_dropout(track_sim_value, excluded_frames, track_faces)

                            # Show track info with similarity score if available
                            caption = f"Track {track_id} · {track_faces} faces"
                            if track_sim_value is not None:
                                sim_pct = int(track_sim_value * 100)
                                caption = f"Track {track_id} · {track_faces} faces · {sim_pct}% sim"

                            if thumb_url:
                                # Use presigned URL directly if already HTTPS, otherwise extract key
                                if thumb_url.startswith("https://"):
                                    thumb_src = thumb_url
                                else:
                                    thumb_src = _extract_s3_key_from_url(thumb_url)
                                thumb_markup = helpers.thumb_html(
                                    thumb_src,
                                    alt=f"Track {track_id}",
                                    hide_if_missing=False,
                                )
                                st.markdown(thumb_markup, unsafe_allow_html=True)
                                if track_badge:
                                    st.markdown(track_badge, unsafe_allow_html=True)
                            else:
                                # Show placeholder when no thumbnail
                                st.markdown(
                                    '<div style="width:147px;height:184px;background:#333;border-radius:6px;'
                                    'display:flex;align-items:center;justify-content:center;color:#666;">'
                                    '📷</div>',
                                    unsafe_allow_html=True,
                                )
                            st.caption(caption)

        # Visual separator between tracks and assignment controls
        st.markdown("---")

        # Show cast suggestions (Enhancement #1) if available
        if cast_suggestions:
            st.markdown("**🎯 Cast Suggestions:**")
            margin_pct = None
            top_sim = cast_suggestions[0].get("similarity") or 0

            # Filter suggestions: only show runners-up if they're close to top match
            # - If delta >= 10%, only show the top suggestion (confident match)
            # - If delta < 10%, show close runners-up (ambiguous, needs review)
            filtered_suggestions = [cast_suggestions[0]]
            if len(cast_suggestions) > 1:
                runner_up = cast_suggestions[1].get("similarity") or 0
                margin_pct = int((max(top_sim - runner_up, 0)) * 100)

                # Only show runners-up if margin is small (< 10%) AND runner is decent (>= 40%)
                if margin_pct < 10:
                    for sugg in cast_suggestions[1:3]:
                        sugg_sim = sugg.get("similarity") or 0
                        # Include if within 10% of top AND at least 40% similarity
                        if (top_sim - sugg_sim) < 0.10 and sugg_sim >= 0.40:
                            filtered_suggestions.append(sugg)

            for idx, cast_sugg in enumerate(filtered_suggestions):
                sugg_cast_id = cast_sugg.get("cast_id")
                sugg_name = cast_sugg.get("name", sugg_cast_id)
                sugg_sim = cast_sugg.get("similarity", 0)
                sugg_confidence = cast_sugg.get("confidence", "low")
                sugg_source = cast_sugg.get("source", "facebank")
                faces_used = cast_sugg.get("faces_used")

                # Nov 2024: Enhanced with rank context (use filtered list count)
                total_suggs = len(filtered_suggestions)
                sim_badge = render_cast_rank_badge(sugg_sim, rank=idx + 1, total_suggestions=total_suggs, cast_name=sugg_name)
                # Confidence buckets aligned to cast thresholds (68/50)
                conf_level = "high" if sugg_sim >= 0.68 else "medium" if sugg_sim >= 0.50 else "low"
                confidence_colors = {
                    "high": "#4CAF50",  # Green
                    "medium": "#FF9800",  # Orange
                    "low": "#F44336",  # Red
                }
                badge_color = confidence_colors.get(conf_level, "#9E9E9E")

                # Source label with extra info
                source_label = sugg_source
                if sugg_source == "frame" and faces_used:
                    source_label = f"frame ({faces_used} face{'s' if faces_used > 1 else ''})"

                sugg_col1, sugg_col2 = st.columns([5, 1])
                with sugg_col1:
                    st.markdown(
                        f'<span style="background-color: {badge_color}; color: white; padding: 2px 8px; '
                        f'border-radius: 4px; font-size: 0.85em; font-weight: bold; margin-right: 8px;">'
                        f'{conf_level.upper()}</span> {sim_badge} **{sugg_name}** '
                        f'<span style="font-size: 0.75em; color: #888;">via {source_label}</span>'
                        + (f" · Δ {margin_pct}%" if idx == 0 and margin_pct is not None else ""),
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
                            _invalidate_assignment_caches()
                            st.success(f"Assigned cluster to {sugg_name}!")
                            _focus_cast_members(sugg_cast_id, sugg_name)
                            st.rerun()
                        else:
                            st.error("Failed to assign cluster. Check logs.")
            st.markdown("---")

        # Show person-based suggestion if available (existing logic)
        if suggested_cast_id and suggested_cast_name:
            similarity_pct = int((1 - suggested_distance) * 100) if suggested_distance is not None else 0
            similarity_value = similarity_pct / 100.0 if similarity_pct else 0.0
            # Nov 2024: Enhanced with rank context (single suggestion = rank 1 of 1)
            cast_badge = render_cast_rank_badge(similarity_value, rank=1, total_suggestions=1, cast_name=suggested_cast_name)
            # Confidence scoring aligned to cast thresholds (68/50)
            if similarity_value >= 0.68:
                confidence = "HIGH"
                badge_color = "#4CAF50"  # Green
            elif similarity_value >= 0.50:
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
                    f'{confidence}</span> Suggested (from assigned clusters): {cast_badge} **{suggested_cast_name}**',
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
                        _invalidate_assignment_caches()
                        st.success(f"Assigned cluster to {suggested_cast_name}!")
                        _focus_cast_members(suggested_cast_id, suggested_cast_name)
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
                # Use a form to ensure selectbox and button states are synchronized
                with st.form(key=f"assign_form_unassigned_{cluster_id}"):
                    cast_choices = [""] + list(cast_options.keys())
                    selected_cast_id = st.selectbox(
                        "Select cast member",
                        options=cast_choices,
                        format_func=lambda cid: cast_options.get(cid, "Select cast member") if cid else "Select cast member",
                        index=0,
                        key=f"cast_select_unassigned_{cluster_id}",
                    )

                    submit_assign = st.form_submit_button("Assign Cluster")

                    if submit_assign:
                        if not selected_cast_id:
                            st.warning("Pick a cast member before assigning.")
                        else:
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
                                    _invalidate_assignment_caches()
                                    st.success(f"Assigned cluster to {cast_options.get(selected_cast_id, 'cast member')}!")
                                    _focus_cast_members(selected_cast_id, cast_options.get(selected_cast_id))
                                    st.rerun()
                                else:
                                    st.error("Failed to assign cluster. Check logs.")
            else:
                st.warning("No cast members available. Import cast first.")

        else:  # New person
            with st.form(key=f"create_person_form_{cluster_id}"):
                new_name = st.text_input(
                    "Person name",
                    key=f"new_name_unassigned_{cluster_id}",
                    placeholder="Enter name...",
                )
                submit_create = st.form_submit_button("Create person")

                if submit_create:
                    if not new_name:
                        st.warning("Enter a name before creating.")
                    else:
                        with st.spinner("Creating person..."):
                            payload = {
                                "strategy": "manual",
                                "cluster_ids": [cluster_id],
                                "name": new_name,
                            }
                            resp = _api_post(f"/episodes/{ep_id}/clusters/group", payload)
                            if resp and resp.get("status") == "success":
                                _invalidate_assignment_caches()
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
        # Get representative crop from person's clusters
        featured_crop = None
        if episode_clusters:
            # Extract cluster IDs without episode prefix
            cluster_ids_plain = [c.split(":")[-1] for c in episode_clusters]
            featured_crop = _get_best_crop_from_clusters(ep_id, cluster_ids_plain)

        # Two-column layout: image on left, details on right
        img_col, details_col = st.columns([1, 3])

        with img_col:
            if featured_crop:
                resolved = helpers.resolve_thumb(featured_crop)
                if resolved:
                    thumb_markup = helpers.thumb_html(resolved, alt=name, hide_if_missing=True)
                    if thumb_markup:
                        st.markdown(thumb_markup, unsafe_allow_html=True)
                    else:
                        st.markdown("👤", help="No image available")
                else:
                    st.markdown("👤", help="No image available")
            else:
                st.markdown("👤", help="No image available")

        with details_col:
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

        # --- SUGGESTED MATCH (INSIDE CONTAINER) ---
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

            st.markdown("---")
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

        # Archive button for auto-clustered people (those without cast_id)
        if not person.get("cast_id"):
            if st.button("🗃️ Archive", key=f"delete_person_{person_id}", type="secondary"):
                resp = _api_delete(f"/shows/{show_id}/people/{person_id}")
                if resp is not None:
                    display_name = name if name != "(unnamed)" else person_id
                    st.success(f"Archived {display_name}")
                    _invalidate_assignment_caches()
                    st.rerun()
                else:
                    st.error("Failed to archive person. Check API logs.")

        # --- CLUSTERS CAROUSEL (INSIDE CONTAINER) ---
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

                        # Show cluster ID and cohesion badge (Nov 2024: enhanced with range)
                        min_sim = cluster.get("min_similarity")
                        max_sim = cluster.get("max_similarity")
                        cohesion_badge = render_cluster_range_badge(cohesion, min_sim, max_sim) if cohesion else ""
                        st.markdown(f"**{cluster_id}** {cohesion_badge}", unsafe_allow_html=True)
                        st.caption(f"{cluster.get('tracks', 0)} tracks · {cluster.get('faces', 0)} frames")

                        # New metrics strip (Nov 2024) for this cluster
                        cluster_metrics = _fetch_cluster_metrics_cached(ep_id, cluster_id)
                        if cluster_metrics:
                            new_metrics = []
                            if cluster_metrics.get("temporal_consistency") is not None:
                                new_metrics.append(MetricData(
                                    metric_type="temporal",
                                    value=cluster_metrics["temporal_consistency"],
                                ))
                            if cluster_metrics.get("isolation") is not None:
                                new_metrics.append(MetricData(
                                    metric_type="isolation",
                                    value=cluster_metrics["isolation"],
                                ))
                            if new_metrics:
                                render_metrics_strip(new_metrics, compact=True, strip_id=f"person_{person_id}_{cluster_id}")

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
                                    _invalidate_assignment_caches()  # Clear caches so UI reflects changes
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
                                        st.success(f"Assigned to {cast_options.get(selected_cast_id, 'cast member')}")
                                        st.rerun()

    # --- ASSIGN ALL CLUSTERS SECTION ---
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
                                    f"Assigned {len(episode_clusters)} clusters to {cast_options.get(selected_cast_id, 'cast member')}"
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

        cast_name: str | None = None
        if not target_person_id:
            cast_resp = _fetch_cast_cached(show_id)
            cast_members = cast_resp.get("cast", []) if cast_resp else []
            cast_member = next((cm for cm in cast_members if cm.get("cast_id") == cast_id), None)
            if not cast_member:
                st.error(f"Cast member {cast_id} not found.")
                return False
            cast_name = cast_member.get("name")
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
        elif target_person:
            cast_name = target_person.get("name")

        payload = {
            "strategy": "manual",
            "cluster_ids": [cluster_id],
            "target_person_id": target_person_id,
            "cast_id": cast_id,
        }
        resp = _api_post(f"/episodes/{ep_id}/clusters/group", payload)
        if resp and resp.get("status") == "success":
            _invalidate_assignment_caches()
            _focus_cast_members(cast_id, cast_name)
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
        focus_name = target_name or (target_person.get("name") if target_person else None)

        # Check if source person actually exists in people.json
        source_person = next((p for p in people if p.get("person_id") == source_person_id), None)
        source_exists = source_person is not None

        _debug(
            "resolved persons",
            {
                "target_existing": bool(target_person),
                "source_person_id": source_person_id,
                "source_exists": source_exists,
                "cluster_ids": cluster_ids,
                "cast_id": target_cast_id,
            },
        )

        # If source person doesn't exist (stale reference), use direct cluster assignment
        if not source_exists:
            _debug("source person not found, using direct cluster assignment")
            # Clear stale cache and use direct cluster assignment
            _invalidate_assignment_caches()

            # Assign clusters directly to target cast member
            payload = {
                "strategy": "manual",
                "cluster_ids": cluster_ids,
                "target_person_id": target_person["person_id"] if target_person else None,
                "cast_id": target_cast_id,
            }
            _debug("direct assign payload", payload)
            resp = _api_post(f"/episodes/{ep_id}/clusters/group", payload)
            _debug("direct assign response", resp)

            if resp and resp.get("status") == "success":
                _invalidate_assignment_caches()
                _focus_cast_members(target_cast_id, focus_name or target_cast_id)
                return True
            st.error("Failed to assign clusters. Check logs for details.")
            return False

        if not target_person:
            # Fetch cast member details to get the name
            cast_resp = _fetch_cast_cached(show_id)
            cast_members = cast_resp.get("cast", []) if cast_resp else []
            cast_member = next((cm for cm in cast_members if cm.get("cast_id") == target_cast_id), None)

            if not cast_member:
                st.error(f"Cast member {target_cast_id} not found")
                _debug("cast lookup failed", {"cast_id": target_cast_id})
                return False
            if not focus_name:
                focus_name = cast_member.get("name")

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
        _focus_cast_members(target_cast_id, focus_name or target_cast_id)
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
    bundle: Dict[str, Any] | None = None,
) -> None:
    # Note: Cast Members header is rendered inline with the count in the section below
    if not show_id:
        st.error("Unable to determine show for this episode.")
        return

    bundle_payload = bundle or {}
    cast_gallery_cards = bundle_payload.get("cast_gallery_cards", []) or []
    cast_options = bundle_payload.get("cast_options", {}) or {}
    unlinked_entities = bundle_payload.get("unlinked_entities", []) or []
    _render_cast_carousel(cast_gallery_cards)

    # Build people lookup for quick access by person_id
    people_lookup = {str(person.get("person_id") or ""): person for person in people}

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

    if not people:
        if cast_gallery_cards:
            st.markdown(f"### 🎬 Cast Members ({len(cast_gallery_cards)})")
            st.caption(f"Show-level cast members for {show_id}")
            _render_cast_gallery(ep_id, cast_gallery_cards, cluster_lookup, cluster_centroids)
        st.info("No people found for this show. Run 'Group Clusters (auto)' to create people.")

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
    # Show archived items for THIS EPISODE only (not entire show)
    parsed_show = (helpers.parse_ep_id(ep_id) or {}).get("show")
    show_id_for_archive = str(parsed_show).upper() if parsed_show else (ep_id.split("-")[0].upper() if "-" in ep_id else ep_id.upper())

    # Fetch archived items filtered by episode_id
    archived_resp = _safe_api_get(
        f"/archive/shows/{show_id_for_archive}",
        params={"limit": 50, "episode_id": ep_id},
    )
    archived_items = archived_resp.get("items", []) if archived_resp else []
    counts = archived_resp.get("counts", {}) if archived_resp else {}

    # Use counts from filtered response (not show-wide stats)
    total_archived = counts.get("total", 0) or len(archived_items)

    if total_archived > 0:
        with st.expander(f"🗃️ View Archived Items ({total_archived})", expanded=False):
            st.caption(
                "Archived items are deleted faces from this episode. "
                "If the same face appears again, it can be automatically archived."
            )

            # Clear All button (clears only this episode's items)
            clear_col1, clear_col2 = st.columns([3, 1])
            with clear_col2:
                if st.button("🗑️ Clear All", key="clear_all_archived", type="secondary"):
                    # Clear only items from this episode
                    clear_resp = helpers.api_delete(
                        f"/archive/shows/{show_id_for_archive}/clear",
                        params={"episode_id": ep_id}
                    )
                    if clear_resp:
                        st.success(f"Cleared {clear_resp.get('deleted_count', 0)} archived items")
                        st.rerun()
                    else:
                        st.error("Failed to clear archive")

            # Show counts (already filtered by episode)
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
                    name = str(item.get("name") or item.get("original_id") or archive_id[:12])
                    archived_at = item.get("archived_at", "")[:10]  # Date only
                    reason = item.get("reason", "deleted")
                    rep_crop_url = item.get("rep_crop_url")

                    item_cols = st.columns([1, 3, 2, 1])
                    with item_cols[0]:
                        if rep_crop_url:
                            # Use resolve_thumb + thumb_html for safe image loading
                            resolved = helpers.resolve_thumb(rep_crop_url)
                            if resolved:
                                thumb_markup = helpers.thumb_html(resolved, alt=name, hide_if_missing=True)
                                if thumb_markup:
                                    st.markdown(thumb_markup, unsafe_allow_html=True)
                                else:
                                    st.markdown("👤")
                            else:
                                st.markdown("👤")
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

    # --- NEEDS CAST ASSIGNMENT (UNIFIED) ---

    # Load dismissed suggestions to filter out skipped items (shared with Smart Suggestions)
    dismissed_resp = _safe_api_get(f"/episodes/{ep_id}/dismissed_suggestions")
    dismissed_ids: set = set()
    if dismissed_resp and dismissed_resp.get("dismissed"):
        dismissed_ids = set(dismissed_resp.get("dismissed", []))

    # Filter out dismissed entities
    if dismissed_ids:
        filtered_entities = []
        for entity in unlinked_entities:
            cluster_ids = entity.get("cluster_ids", [])
            # Check if any cluster in this entity is dismissed
            if entity.get("entity_type") == "person":
                person_id = entity.get("entity_id")
                if f"person:{person_id}" in dismissed_ids:
                    continue
            # Filter out dismissed cluster_ids from the entity
            remaining_clusters = [cid for cid in cluster_ids if cid not in dismissed_ids]
            if not remaining_clusters:
                continue
            entity_copy = dict(entity)
            entity_copy["cluster_ids"] = remaining_clusters
            filtered_entities.append(entity_copy)
        unlinked_entities = filtered_entities

    # Suggestions: facebank (cached) and assigned-cluster similarity
    cast_suggestions_by_cluster = st.session_state.get(_cast_suggestions_cache_key(ep_id), {})
    assigned_suggestions_by_cluster: Dict[str, Dict[str, Any]] = {}
    assigned_suggestions_resp = _safe_api_get(f"/episodes/{ep_id}/cluster_suggestions_from_assigned")
    if assigned_suggestions_resp:
        cast_person_ids = {p.get("person_id") for p in people if p.get("person_id") and p.get("cast_id")}
        for suggestion in assigned_suggestions_resp.get("suggestions", []):
            cid = suggestion.get("cluster_id")
            suggested_person_id = suggestion.get("suggested_person_id")
            if cid and suggested_person_id in cast_person_ids:
                assigned_suggestions_by_cluster[cid] = suggestion

    # Cross-episode suggestions (used by auto-people cards)
    cross_suggestions_by_cluster: Dict[str, Dict[str, Any]] = {}
    cross_resp = _safe_api_get(f"/episodes/{ep_id}/cluster_suggestions")
    if cross_resp:
        for suggestion in cross_resp.get("suggestions", []):
            cid = suggestion.get("cluster_id")
            if cid:
                cross_suggestions_by_cluster[cid] = suggestion

    # Build separate queues: multi-cluster identities vs single-cluster entries
    multi_cluster_queue: List[Dict[str, Any]] = []  # Identities with >1 cluster
    single_cluster_queue: List[Dict[str, Any]] = []  # Standalone clusters OR identities with 1 cluster

    for entity in unlinked_entities:
        cluster_ids = [cid for cid in entity.get("cluster_ids", []) if cid]
        if not cluster_ids:
            continue

        if entity.get("entity_type") == "person":
            person_id = entity.get("entity_id")
            person = people_lookup.get(str(person_id)) or entity.get("person") or {"person_id": person_id}
            if filter_cast_id and str(person.get("cast_id") or "") != str(filter_cast_id):
                continue

            item = {
                "kind": "person",
                "person": person,
                "episode_clusters": cluster_ids,
                "avg_cohesion": entity.get("avg_cohesion"),
                "counts": {
                    "clusters": len(cluster_ids),
                    "tracks": entity.get("tracks", 0),
                    "faces": entity.get("faces", 0),
                },
            }

            # Multi-cluster identities go to main queue, single-cluster to secondary
            if len(cluster_ids) > 1:
                multi_cluster_queue.append(item)
            else:
                # Single-cluster identity - treat like a cluster but keep person context
                single_cluster_queue.append(item)
        else:
            # Single-cluster entity (standalone cluster without person)
            cluster_id = cluster_ids[0]
            cluster_info = cluster_lookup.get(cluster_id, {})
            counts = cluster_info.get("counts", {})
            track_list = cluster_info.get("tracks", [])

            # Calculate filtered track count (excluding single-frame tracks for multi-track clusters)
            original_tracks = counts.get("tracks", 0)
            if len(track_list) > 1:
                # Multi-track cluster: count only multi-frame tracks
                multi_frame_tracks = [t for t in track_list if t.get("faces", 0) > 1]
                filtered_tracks = len(multi_frame_tracks) if multi_frame_tracks else len(track_list)
            else:
                # Single-track cluster: keep as-is
                filtered_tracks = len(track_list) if track_list else original_tracks

            single_cluster_queue.append(
                {
                    "kind": "cluster",
                    "cluster_id": cluster_id,
                    "tracks": filtered_tracks,  # Use filtered count for sorting
                    "original_tracks": original_tracks,
                    "faces": counts.get("faces", 0),
                    "cohesion": cluster_info.get("cohesion"),
                }
            )

    # Sort both queues: largest first (faces then tracks)
    def sort_key(item):
        return (
            item.get("faces") or item.get("counts", {}).get("faces", 0),
            item.get("tracks") or item.get("counts", {}).get("tracks", 0),
        )

    multi_cluster_queue.sort(key=sort_key, reverse=True)
    single_cluster_queue.sort(key=sort_key, reverse=True)

    total_items = len(multi_cluster_queue) + len(single_cluster_queue)

    if total_items > 0:
        st.markdown("---")
        st.markdown(f"### 🔍 Needs Cast Assignment ({total_items})")

        # Sort options for unassigned clusters
        sort_col, analyze_col, autoassign_col, spacer_col = st.columns([1.5, 1, 1, 1.5])
        with sort_col:
            unassigned_sort_key = f"unassigned_sort:{ep_id}"
            unassigned_sort_option = st.selectbox(
                "Sort by",
                UNASSIGNED_CLUSTER_SORT_OPTIONS,
                index=0,  # Default: Face Count (High to Low)
                key=unassigned_sort_key,
                label_visibility="collapsed",
            )

            # Apply sorting based on selected option
            def unassigned_sort_key_fn(item):
                # Get faces - check direct key first, then counts dict
                faces = item.get("faces")
                if faces is None:
                    faces = item.get("counts", {}).get("faces", 0)
                # Get tracks - check direct key first, then counts dict
                tracks = item.get("tracks")
                if tracks is None:
                    tracks = item.get("counts", {}).get("tracks", 0)
                cohesion = item.get("cohesion") or item.get("avg_cohesion") or 0.0
                cluster_id = item.get("cluster_id") or ""
                # Get cast match score from suggestions if available
                cid = item.get("cluster_id") or (item.get("episode_clusters", [None])[0])
                cast_score = 0.0
                if cid and cid in cast_suggestions_by_cluster:
                    suggestions = cast_suggestions_by_cluster[cid]
                    # Handle both list and dict formats
                    if isinstance(suggestions, list) and suggestions:
                        cast_score = suggestions[0].get("similarity", 0.0)
                    elif isinstance(suggestions, dict):
                        cast_score = suggestions.get("similarity", 0.0)

                # Use negative values for descending, positive for ascending
                # No reverse flag needed - just return appropriate tuple
                if unassigned_sort_option == "Face Count (High to Low)":
                    return (-faces, -tracks)
                elif unassigned_sort_option == "Face Count (Low to High)":
                    return (faces, tracks)
                elif unassigned_sort_option == "Track Count (High to Low)":
                    return (-tracks, -faces)
                elif unassigned_sort_option == "Track Count (Low to High)":
                    return (tracks, faces)
                elif unassigned_sort_option == "Cast Match Score (High to Low)":
                    return (-cast_score, -faces)
                elif unassigned_sort_option == "Cast Match Score (Low to High)":
                    return (cast_score, faces)
                elif unassigned_sort_option == "Cluster Similarity (High to Low)":
                    return (-cohesion if cohesion else 0, -faces)
                elif unassigned_sort_option == "Cluster Similarity (Low to High)":
                    return (cohesion if cohesion else 999, faces)
                elif unassigned_sort_option == "Cluster ID (A-Z)":
                    return (cluster_id, 0)
                elif unassigned_sort_option == "Cluster ID (Z-A)":
                    # Reverse by using negative ord values
                    return tuple(-ord(c) for c in cluster_id) if cluster_id else (0,)
                return (-faces, -tracks)

            # Re-sort with user's chosen option (no reverse needed, handled in key)
            multi_cluster_queue.sort(key=unassigned_sort_key_fn)
            single_cluster_queue.sort(key=unassigned_sort_key_fn)

        # Analyze & Group Similar button
        with analyze_col:
            if st.button("🔬 Analyze & Group Similar", key=f"analyze_unassigned_{ep_id}", help="Group similar unassigned clusters and suggest cast members"):
                with st.spinner("Analyzing clusters..."):
                    analysis_resp = _safe_api_get(f"/episodes/{ep_id}/analyze_unassigned")
                    if analysis_resp and analysis_resp.get("status") == "success":
                        st.session_state[f"unassigned_analysis:{ep_id}"] = analysis_resp
                        st.rerun()
                    else:
                        msg = analysis_resp.get("message") if analysis_resp else "API error"
                        st.error(f"Analysis failed: {msg}")

        # Auto-assign all clusters to cast based on facebank
        with autoassign_col:
            if st.button(
                "⚡ Auto-Assign All",
                key=f"autoassign_all_{ep_id}",
                help="Automatically assign all unassigned clusters to cast members based on facebank matches",
                type="primary",
            ):
                with st.spinner("Auto-assigning clusters to cast..."):
                    autoassign_resp = _api_post(
                        f"/episodes/{ep_id}/clusters/group",
                        {
                            "strategy": "auto",
                            "protect_manual": True,
                            "facebank_first": True,
                            "skip_cast_assignment": False,
                        },
                    )
                    if autoassign_resp and autoassign_resp.get("status") == "success":
                        result = autoassign_resp.get("result", {})
                        assigned = len(result.get("assignments", {}).get("assigned", []))
                        st.success(f"Auto-assigned {assigned} cluster(s) to cast members!")
                        _invalidate_assignment_caches()
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        msg = autoassign_resp.get("error") if autoassign_resp else "API error"
                        st.error(f"Auto-assign failed: {msg}")

        # Show analysis results if available
        analysis_key = f"unassigned_analysis:{ep_id}"
        if analysis_key in st.session_state:
            analysis = st.session_state[analysis_key]
            groups = analysis.get("groups", [])
            singletons = analysis.get("singletons", [])
            summary = analysis.get("summary", {})

            with st.expander(f"📊 **Analysis Results** — {summary.get('group_count', 0)} groups found, {summary.get('singleton_clusters', 0)} singletons", expanded=True):
                # Clear button
                if st.button("✕ Clear Analysis", key=f"clear_analysis_{ep_id}"):
                    del st.session_state[analysis_key]
                    st.rerun()

                if groups:
                    st.markdown("#### 🔗 Similar Cluster Groups")
                    st.caption("These clusters look similar and may be the same person. Consider merging them.")
                    for grp in groups:
                        grp_id = grp.get("group_id", "")
                        cluster_ids = grp.get("clusters", [])
                        cast_recs = grp.get("cast_recommendations", [])
                        avg_sim = grp.get("avg_internal_similarity")

                        with st.container(border=True):
                            # Group header
                            sim_str = f" · {int(avg_sim * 100)}% similar" if avg_sim else ""
                            st.markdown(f"**{grp_id}** — {len(cluster_ids)} clusters{sim_str}")

                            # Cast recommendations
                            if cast_recs:
                                rec_parts = []
                                for rec in cast_recs[:3]:
                                    conf_emoji = "🟢" if rec.get("confidence") == "high" else "🟡" if rec.get("confidence") == "medium" else "🔴"
                                    rec_parts.append(f"{conf_emoji} {rec.get('name')} ({int(rec.get('similarity', 0) * 100)}%)")
                                st.markdown(f"**Suggested:** {' · '.join(rec_parts)}")
                            else:
                                st.caption("No cast matches found")

                            # Cluster details
                            cluster_details = grp.get("cluster_details", [])
                            details_str = ", ".join([
                                f"`{d.get('cluster_id')}` ({d.get('tracks', 0)} tracks)"
                                for d in cluster_details[:5]
                            ])
                            if len(cluster_details) > 5:
                                details_str += f" +{len(cluster_details) - 5} more"
                            st.caption(f"Clusters: {details_str}")

                if singletons:
                    st.markdown("#### 🔹 Unique Clusters (No Similar Matches)")
                    st.caption("These clusters don't closely match other unassigned clusters.")
                    singleton_info = []
                    for s in singletons[:10]:
                        cid = s.get("clusters", [""])[0]
                        recs = s.get("cast_recommendations", [])
                        if recs:
                            top_rec = recs[0]
                            singleton_info.append(f"`{cid}` → {top_rec.get('name')} ({int(top_rec.get('similarity', 0) * 100)}%)")
                        else:
                            singleton_info.append(f"`{cid}` → no matches")
                    st.markdown("  \n".join(singleton_info))
                    if len(singletons) > 10:
                        st.caption(f"... and {len(singletons) - 10} more singletons")

                if not groups and not singletons:
                    st.info(analysis.get("message", "No unassigned clusters to analyze"))

        # Section 1: Multi-cluster identities (unnamed people with >1 cluster)
        if multi_cluster_queue:
            st.markdown(f"**Grouped Identities** ({len(multi_cluster_queue)}) — multiple clusters linked together")
            for item in multi_cluster_queue:
                _render_auto_person_card(
                    ep_id,
                    show_id,
                    item.get("person", {}),
                    item.get("episode_clusters", []),
                    cast_options,
                    cross_suggestions_by_cluster,
                    cast_suggestions_by_cluster,
                    avg_cohesion=item.get("avg_cohesion"),
                )

        # Section 2: Single-cluster entries (standalone clusters OR single-cluster identities)
        if single_cluster_queue:
            if multi_cluster_queue:
                st.markdown("---")
            st.markdown(f"**Single Clusters** ({len(single_cluster_queue)}) — not yet grouped into identities")

            for item in single_cluster_queue:
                if item.get("kind") == "person":
                    # Single-cluster identity - render as cluster card but with person context
                    cluster_ids = item.get("episode_clusters", [])
                    cluster_id = cluster_ids[0] if cluster_ids else None
                    if cluster_id:
                        _render_unassigned_cluster_card(
                            ep_id,
                            show_id,
                            cluster_id,
                            assigned_suggestions_by_cluster.get(cluster_id),
                            cast_options,
                            cluster_lookup,
                            cast_suggestions=cast_suggestions_by_cluster.get(cluster_id),
                        )
                else:
                    cluster_id = item.get("cluster_id")
                    _render_unassigned_cluster_card(
                        ep_id,
                        show_id,
                        cluster_id,
                        assigned_suggestions_by_cluster.get(cluster_id),
                        cast_options,
                        cluster_lookup,
                        cast_suggestions=cast_suggestions_by_cluster.get(cluster_id),
                    )

    # Show message if filtering but nothing found
    if filter_cast_id and not cast_gallery_cards and total_items == 0:
        st.warning(f"{filter_cast_name or filter_cast_id} has no clusters in episode {ep_id}.")

    # Show message if no people at all
    if not cast_gallery_cards and total_items == 0 and not filter_cast_id:
        already_ran = st.session_state.get(f"{ep_id}::{_CURRENT_RUN_ID}::group_clusters_auto_ran", False)
        should_offer = faces_review_run_scoped.should_offer_group_clusters(
            people,
            ep_id,
            _CURRENT_RUN_ID,
            already_ran,
        )
        has_clusters = bool(identity_index) or bool(cluster_lookup)
        if should_offer and has_clusters:
            st.info(
                "Clusters exist for this attempt, but no grouped people were found. "
                "Run Group Clusters (auto) to create people for this run."
            )
            if st.button(
                "Run Group Clusters (auto)",
                key=f"group_clusters_auto:{ep_id}:{_CURRENT_RUN_ID}",
                use_container_width=True,
            ):
                with st.spinner("Grouping clusters..."):
                    payload = {
                        "strategy": "auto",
                        "protect_manual": True,
                        "facebank_first": True,
                        "skip_cast_assignment": False,
                        "execution_mode": "local",
                    }
                    resp = _api_post(f"/episodes/{ep_id}/clusters/group", payload, timeout=180.0)
                    if resp and resp.get("status") in {"success", "ok"}:
                        st.session_state[f"{ep_id}::{_CURRENT_RUN_ID}::group_clusters_auto_ran"] = True
                        _invalidate_assignment_caches()
                        st.success("Grouped clusters. Refreshing…")
                        st.rerun()
                    else:
                        st.error("Failed to group clusters. Check API logs for details.")
        else:
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
    _fetch_identities_cached(ep_id, _CURRENT_RUN_ID)

    st.caption(f"{len(episode_clusters)} clusters · {total_tracks} tracks · {total_faces} frames")

    # View All Tracks button - compact grid view for outlier detection
    if total_tracks > 0:
        if st.button(
            f"🎭 View All {total_tracks} Tracks (Outlier Detection)",
            key=f"view_all_tracks_{person_id}",
            help="Grid view with one crop per track. Sort by Person Cohesion to find misassigned tracks.",
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

    archived_tracks = st.session_state.get(f"{ep_id}::archived_tracks", set())
    if not isinstance(archived_tracks, set):
        archived_tracks = set(archived_tracks)
    if archived_tracks:
        all_tracks = [t for t in all_tracks if _coerce_track_int(t.get("track_int") or t.get("track_id")) not in archived_tracks]
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

    # Compute cast track scores (similarity of each track to others for this person)
    embeddings: list[np.ndarray] = []
    track_to_embedding_idx: dict[int, int] = {}
    for idx, track in enumerate(all_tracks):
        embedding = track.get("embedding")
        if embedding is None:
            meta = _track_meta(track.get("track_int"))
            embedding = meta.get("embedding") if isinstance(meta, dict) else None
        if embedding:
            try:
                embeddings.append(np.array(embedding))
                track_to_embedding_idx[idx] = len(embeddings) - 1
            except Exception:
                continue

    if len(embeddings) > 1:
        try:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / (norms + 1e-10)
            similarity_matrix = normalized @ normalized.T
            for track_idx, emb_idx in track_to_embedding_idx.items():
                row = similarity_matrix[emb_idx]
                other_sims = [row[j] for j in range(len(row)) if j != emb_idx]
                if other_sims:
                    all_tracks[track_idx]["cast_track_score"] = float(np.mean(other_sims))
        except Exception:
            pass

    cast_scores = [t.get("cast_track_score") for t in all_tracks if t.get("cast_track_score") is not None]
    avg_cast_score = float(np.mean(cast_scores)) if cast_scores else None
    low_cast_tracks = len([s for s in cast_scores if s is not None and s < 0.55])

    # --- Bulk Track Selection State ---
    bulk_sel_key = f"bulk_track_sel::person::{person_id}"
    if bulk_sel_key not in st.session_state:
        st.session_state[bulk_sel_key] = set()
    selected_track_ids: set[int] = st.session_state[bulk_sel_key]

    # Sorting options and select all toggle
    sort_cols = st.columns([2, 1, 1])
    with sort_cols[0]:
        st.markdown(f"**All {len(all_tracks)} Tracks**")
        if avg_cast_score is not None:
            st.caption(
                f"Avg Person Cohesion {render_similarity_badge(avg_cast_score, SimilarityType.PERSON_COHESION)}"
                f" · Low (<55%): {low_cast_tracks}",
                unsafe_allow_html=True,
            )
    with sort_cols[1]:
        # Select all / Deselect all
        all_track_ids_set = {t.get("track_int") for t in all_tracks if t.get("track_int") is not None}
        if selected_track_ids:
            if st.button("☐ Deselect All", key=f"deselect_all_{person_id}", use_container_width=True):
                st.session_state[bulk_sel_key] = set()
                st.rerun()
        else:
            if st.button("☑ Select All", key=f"select_all_{person_id}", use_container_width=True):
                st.session_state[bulk_sel_key] = all_track_ids_set
                st.rerun()
    with sort_cols[2]:
        sort_option = st.selectbox(
            "Sort by:",
            TRACK_SORT_OPTIONS,
            key=f"sort_tracks_{person_id}",
            label_visibility="collapsed",
        )

    # Apply sorting using centralized function with track metadata getter
    sort_tracks(all_tracks, sort_option, track_meta_getter=_track_meta)

    # --- Bulk Re-assign Section (shown when tracks are selected) ---
    if selected_track_ids and cast_options:
        with st.container(border=True):
            st.markdown(f"**📦 {len(selected_track_ids)} Track(s) Selected**")
            bulk_cols = st.columns([2, 2, 1])
            with bulk_cols[0]:
                # Filter out current person's cast_id
                current_person = people_lookup.get(person_id, {})
                current_cast_id = current_person.get("cast_id")
                available_cast = [cid for cid in cast_options.keys() if cid != current_cast_id]
                if available_cast:
                    bulk_cast_id = st.selectbox(
                        "Re-assign all selected to:",
                        options=[""] + available_cast,
                        format_func=lambda cid: cast_options.get(cid, "Select...") if cid else "Select cast member...",
                        key=f"bulk_reassign_cast_{person_id}",
                        label_visibility="collapsed",
                    )
                else:
                    bulk_cast_id = None
                    st.caption("No other cast members available")
            with bulk_cols[1]:
                if bulk_cast_id:
                    bulk_cast_name = cast_options.get(bulk_cast_id, "")
                    if st.button(
                        f"Re-assign {len(selected_track_ids)} Track(s)",
                        key=f"bulk_reassign_btn_{person_id}",
                        type="primary",
                        use_container_width=True,
                    ):
                        _bulk_assign_tracks(
                            ep_id,
                            list(selected_track_ids),
                            bulk_cast_name,
                            show_slug,
                            bulk_cast_id,
                        )
                        st.session_state[bulk_sel_key] = set()
            with bulk_cols[2]:
                if st.button("Clear Selection", key=f"clear_bulk_sel_{person_id}"):
                    st.session_state[bulk_sel_key] = set()
                    st.rerun()

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
            # Track header with selection checkbox
            header_cols = st.columns([0.5, 5])
            with header_cols[0]:
                if track_id_int is not None:
                    is_selected = track_id_int in selected_track_ids
                    if st.checkbox(
                        "Select",
                        value=is_selected,
                        key=f"sel_track_{person_id}_{track_id_int}",
                        label_visibility="collapsed",
                    ):
                        selected_track_ids.add(track_id_int)
                    else:
                        selected_track_ids.discard(track_id_int)
            with header_cols[1]:
                # Nov 2024: Enhanced with dropout indicator
                excluded_frames = track.get("excluded_frames") or track_meta.get("excluded_frames", 0) if track_meta else 0
                badge_html = render_track_with_dropout(similarity, excluded_frames, len(frames))
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
                            cast_name = cast_options.get(selected_cast_id)
                            if selected_cast_id and cast_name and st.button(
                                f"Assign to {cast_name}",
                                key=f"reassign_btn_{person_id}_{track_id_int}",
                                type="primary",
                                use_container_width=True,
                            ):
                                # Re-assign via API
                                _assign_track_name(ep_id, track_id_int, cast_name, show_slug, selected_cast_id)
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
    with sorting by Person Cohesion and Track Similarity to find misassigned tracks.
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
    archived_clusters = st.session_state.get(f"{ep_id}::archived_clusters", set())
    if not isinstance(archived_clusters, set):
        archived_clusters = set(archived_clusters)
    if archived_clusters:
        clusters_list = [c for c in clusters_list if c.get("cluster_id") not in archived_clusters]
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

    cast_scores = [t.get("cast_track_score") for t in all_tracks if t.get("cast_track_score") is not None]
    avg_cast_score = float(np.mean(cast_scores)) if cast_scores else None
    low_cast_tracks = len([s for s in cast_scores if s is not None and s < 0.55])

    # Sort controls
    st.markdown(f"**{len(all_tracks)} Tracks** · {total_faces} frames total")
    if avg_cast_score is not None:
        st.caption(
            f"Avg Person Cohesion {render_similarity_badge(avg_cast_score, SimilarityType.PERSON_COHESION)}"
            f" · Low (<55%): {low_cast_tracks}",
            unsafe_allow_html=True,
        )

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

                # Badges - show both Track Similarity and Person Cohesion (Nov 2024: enhanced with dropout)
                excluded_frames = track.get("excluded_frames", 0)
                track_badge = render_track_with_dropout(track_sim, excluded_frames, frame_count)
                cast_badge = render_similarity_badge(cast_track_score, SimilarityType.PERSON_COHESION) if cast_track_score is not None else '<span style="color: #888;">N/A</span>'
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

    # Fetch track representatives with sample frames (cached for 60s)
    track_reps_data = _fetch_cluster_track_reps_cached(ep_id, identity_id, frames_per_track=10)
    if not track_reps_data:
        st.error("Failed to load track representatives.")
        return

    # Fetch cast suggestions once to show cast similarity + ambiguity margin
    cast_suggestion = None
    cast_margin_pct: int | None = None
    suggestions_cache_key = _cast_suggestions_cache_key(ep_id)
    suggestions_map = st.session_state.get(suggestions_cache_key)
    if suggestions_map is None:
        suggestions_resp = _safe_api_get(f"/episodes/{ep_id}/cast_suggestions")
        suggestions_map = {}
        if suggestions_resp:
            for entry in suggestions_resp.get("suggestions", []):
                cid = entry.get("cluster_id")
                if not cid:
                    continue
                suggestions_map[cid] = entry.get("cast_suggestions", []) or []
        st.session_state[suggestions_cache_key] = suggestions_map
    cast_suggestions_for_cluster = suggestions_map.get(identity_id) if suggestions_map else []
    if cast_suggestions_for_cluster:
        cast_suggestion = cast_suggestions_for_cluster[0]
        if len(cast_suggestions_for_cluster) > 1:
            top = cast_suggestions_for_cluster[0].get("similarity") or 0
            runner = cast_suggestions_for_cluster[1].get("similarity") or 0
            cast_margin_pct = int(max(top - runner, 0) * 100)

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
    cluster_meta = cluster_lookup.get(identity_id, {}) if isinstance(cluster_lookup, dict) else {}
    assignment_meta = cluster_meta.get("assignment") if isinstance(cluster_meta, dict) else {}
    assigned_cast_id = None
    if isinstance(assignment_meta, dict):
        assigned_cast_id = assignment_meta.get("cast_id")
    if not assigned_cast_id:
        assigned_cast_id = cluster_meta.get("assigned_cast_id") if isinstance(cluster_meta, dict) else None

    track_assignment_lookup: Dict[int, Dict[str, Any]] = {}
    if isinstance(cluster_meta, dict):
        for track in cluster_meta.get("tracks", []) or []:
            if not isinstance(track, dict):
                continue
            tid_val = track.get("track_id") or track.get("track") or track.get("track_int")
            tid_int = _coerce_track_int(tid_val)
            if tid_int is None:
                continue
            track_assignment_lookup[tid_int] = track.get("assignment") if isinstance(track.get("assignment"), dict) else {}

    tracks_count = track_reps_data.get("total_tracks", 0)
    cohesion = track_reps_data.get("cohesion")
    track_reps = track_reps_data.get("tracks", [])

    # Header: Cluster ID
    st.subheader(f"Cluster {identity_id}")

    # Show assigned person info (prefer bundle assignment cast_id)
    if isinstance(assignment_meta, dict) and assignment_meta.get("unassigned"):
        st.caption(":orange[🚫 Cast unassigned (manual override)]")
    elif assigned_cast_id:
        cast_name = cast_lookup.get(assigned_cast_id, {}).get("name") if cast_lookup else None
        st.caption(f"🎭 Assigned cast: **{cast_name or assigned_cast_id}**")
        source_label = assignment_meta.get("source") if isinstance(assignment_meta, dict) else None
        if source_label:
            st.caption(f":gray[Assignment source: {source_label}]")
        if st.button("Unassign cluster", key=f"unassign_cluster_{identity_id}", type="secondary"):
            if _set_cluster_assignment(ep_id, identity_id, None):
                _invalidate_assignment_caches()
                st.success("Cluster unassigned.")
                st.rerun()

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

    # Cohesion badge (with color) directly under the ID (Nov 2024: enhanced with range)
    if cohesion is not None:
        min_sim = track_reps_data.get("min_similarity")
        max_sim = track_reps_data.get("max_similarity")
        cohesion_badge = render_cluster_range_badge(cohesion, min_sim, max_sim)
        st.markdown(f"**Cluster Cohesion:** {cohesion_badge}", unsafe_allow_html=True)

    if cast_suggestion:
        cast_sim = cast_suggestion.get("similarity") or 0.0
        cast_name = cast_suggestion.get("name") or cast_suggestion.get("cast_id") or "cast"
        # Nov 2024: Enhanced with rank context
        total_suggs = len(cast_suggestions_for_cluster) if cast_suggestions_for_cluster else 1
        cast_badge = render_cast_rank_badge(cast_sim, rank=1, total_suggestions=total_suggs, cast_name=cast_name)
        margin_html = f" · Δ {cast_margin_pct}%" if cast_margin_pct is not None else ""
        st.markdown(
            f"**Cast Similarity:** {cast_badge} · {cast_name}{margin_html}",
            unsafe_allow_html=True,
        )

    # Tracks count
    st.caption(f"**{tracks_count}** track(s)")

    # New metrics strip (Nov 2024) - temporal, ambiguity, isolation, quality
    cluster_metrics = _fetch_cluster_metrics_cached(ep_id, identity_id)
    if cluster_metrics:
        metrics = []
        if cluster_metrics.get("temporal_consistency") is not None:
            metrics.append(MetricData(
                metric_type="temporal",
                value=cluster_metrics["temporal_consistency"],
            ))
        if cluster_metrics.get("ambiguity") is not None:
            metrics.append(MetricData(
                metric_type="ambiguity",
                value=cluster_metrics["ambiguity"],
                first_match=cluster_metrics.get("first_match"),
                second_match=cluster_metrics.get("second_match"),
            ))
        if cluster_metrics.get("isolation") is not None:
            metrics.append(MetricData(
                metric_type="isolation",
                value=cluster_metrics["isolation"],
            ))
        if cluster_metrics.get("avg_quality") is not None:
            metrics.append(MetricData(
                metric_type="quality",
                value=cluster_metrics["avg_quality"],
                breakdown=cluster_metrics.get("quality_breakdown"),
            ))
        if metrics:
            render_metrics_strip(metrics, compact=False, strip_id=f"cluster_detail_{identity_id}")

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

    # track_reps is already sorted by user's selection via sort_tracks() above
    # Render each track as its own row with sample frames
    FRAMES_TO_SHOW = 12  # Number of frames visible at once

    for track_rep in track_reps:
        track_id_str = track_rep.get("track_id", "")
        similarity = track_rep.get("similarity")
        sample_frames = track_rep.get("sample_frames", [])
        frame_count = track_rep.get("frame_count", len(sample_frames))
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

        # Each track in its own bordered container
        with st.container(border=True):
            # Header row: checkbox, track info, actions
            header_cols = st.columns([0.5, 3, 2])

            with header_cols[0]:
                # Checkbox for bulk selection
                if track_id_int is not None:
                    is_selected = track_id_int in selected_tracks
                    if st.checkbox(
                        "Sel",
                        value=is_selected,
                        key=checkbox_key,
                        label_visibility="collapsed",
                        help="Select for bulk assignment",
                    ):
                        if track_id_int not in selected_tracks:
                            selected_tracks.add(track_id_int)
                    else:
                        selected_tracks.discard(track_id_int)

            with header_cols[1]:
                # Track ID and similarity badge
                excluded_frames = track_rep.get("excluded_frames", 0)
                badge_html = render_track_with_dropout(similarity, excluded_frames, frame_count)
                override_badge = ""
                assignment_meta = track_assignment_lookup.get(track_id_int or -1, {})
                if isinstance(assignment_meta, dict) and assignment_meta.get("assignment_type") == "track_override":
                    override_badge = " · ⚡ Override"
                st.markdown(
                    f"**Track {track_num}** {badge_html}{override_badge} · {frame_count} frames",
                    unsafe_allow_html=True
                )

            with header_cols[2]:
                # Action buttons in a row
                btn_cols = st.columns(3)
                with btn_cols[0]:
                    if track_id_int is not None and st.button(
                        "👁️ View",
                        key=f"view_track_{identity_id}_{track_id_int}",
                        use_container_width=True,
                    ):
                        _set_view(
                            "track",
                            person_id=person_id,
                            identity_id=identity_id,
                            track_id=track_id_int,
                        )
                with btn_cols[1]:
                    if track_id_int is not None and move_options:
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
                            "Move",
                            move_options,
                            format_func=lambda opt: opt[0],
                            index=default_index,
                            key=cast_select_key,
                            label_visibility="collapsed",
                        )
                        cast_choice = choice[1] if choice else None
                        cast_name = choice[0] if choice else None
                with btn_cols[2]:
                    if track_id_int is not None and st.button(
                        "🗃️",
                        key=f"cluster_delete_btn_{identity_id}_{track_id_int}",
                        help="Archive track",
                        use_container_width=True,
                    ):
                        _archive_track(ep_id, track_id_int)

            # Handle cast assignment (if selected from dropdown above)
            if track_id_int is not None and move_options:
                cast_select_key = f"cluster_move_select_{identity_id}_{track_id_int}"
                choice = st.session_state.get(cast_select_key)
                if choice and isinstance(choice, (list, tuple)) and len(choice) >= 2:
                    cast_choice = choice[1]
                    cast_name = choice[0]
                    if cast_choice == "__new_cast__":
                        new_cast_key = f"new_cast_name_{identity_id}_{track_id_int}"
                        ncol1, ncol2 = st.columns([2, 1])
                        with ncol1:
                            new_name = st.text_input(
                                "New name",
                                key=new_cast_key,
                                placeholder="Enter cast member name",
                                label_visibility="collapsed",
                            )
                        with ncol2:
                            if new_name and new_name.strip():
                                if st.button(
                                    "Create & Assign",
                                    key=f"create_assign_cast_{identity_id}_{track_id_int}",
                                    use_container_width=True,
                                ):
                                    _create_and_assign_to_new_cast(
                                        ep_id, track_id_int, new_name.strip(), show_slug
                                    )
                    elif cast_choice and cast_name and cast_name != "Select Cast Member":
                        if st.button(
                            f"Assign to {cast_name}",
                            key=f"cluster_move_btn_{identity_id}_{track_id_int}",
                        ):
                            _assign_track_name(ep_id, track_id_int, cast_name, show_slug, cast_choice)

            # Frame gallery with scrolling
            if sample_frames:
                # Session state for frame offset (for scrolling)
                offset_key = f"frame_offset::{identity_id}::{track_id_str}"
                if offset_key not in st.session_state:
                    st.session_state[offset_key] = 0
                offset = st.session_state[offset_key]

                total_sample = len(sample_frames)
                max_offset = max(0, total_sample - FRAMES_TO_SHOW)

                # Navigation row (only show if more frames than FRAMES_TO_SHOW)
                if total_sample > FRAMES_TO_SHOW:
                    nav_cols = st.columns([1, 6, 1])
                    with nav_cols[0]:
                        if st.button("◀", key=f"prev_{identity_id}_{track_id_str}", disabled=offset <= 0):
                            st.session_state[offset_key] = max(0, offset - FRAMES_TO_SHOW)
                            st.rerun()
                    with nav_cols[1]:
                        # Show position indicator
                        end_idx = min(offset + FRAMES_TO_SHOW, total_sample)
                        st.caption(f"Showing {offset + 1}-{end_idx} of {total_sample} sample frames")
                    with nav_cols[2]:
                        if st.button("▶", key=f"next_{identity_id}_{track_id_str}", disabled=offset >= max_offset):
                            st.session_state[offset_key] = min(max_offset, offset + FRAMES_TO_SHOW)
                            st.rerun()

                # Display frames
                visible_frames = sample_frames[offset : offset + FRAMES_TO_SHOW]
                frame_cols = st.columns(len(visible_frames)) if visible_frames else []

                for idx, frame_data in enumerate(visible_frames):
                    with frame_cols[idx]:
                        crop_url = frame_data.get("crop_url")
                        frame_idx = frame_data.get("frame_idx", "?")
                        resolved = helpers.resolve_thumb(crop_url)
                        thumb_markup = helpers.thumb_html(
                            resolved,
                            alt=f"Frame {frame_idx}",
                            hide_if_missing=False,
                        )
                        st.markdown(thumb_markup, unsafe_allow_html=True)
                        st.caption(f"F{frame_idx}")
            else:
                # Fallback: show single representative thumbnail
                crop_url = track_rep.get("crop_url")
                if crop_url:
                    resolved = helpers.resolve_thumb(crop_url)
                    thumb_markup = helpers.thumb_html(resolved, alt=f"Track {track_num}", hide_if_missing=False)
                    st.markdown(thumb_markup, unsafe_allow_html=True)
                else:
                    st.caption("No frames available")

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


def _render_track_view(
    ep_id: str,
    track_id: int,
    identities_payload: Dict[str, Any],
    cluster_lookup: Dict[str, Dict[str, Any]] | None = None,
    cast_options: Dict[str, str] | None = None,
    face_exclusions: Dict[str, Any] | None = None,
) -> None:
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
    cluster_lookup = cluster_lookup or {}
    face_exclusions = face_exclusions or {}

    # Show assigned person/cluster info
    show_slug = _episode_show_slug(ep_id)
    if current_identity:
        identity_data = identity_lookup.get(current_identity, {})
        cluster_meta = cluster_lookup.get(current_identity, {}) if isinstance(cluster_lookup, dict) else {}
        assignment_meta = cluster_meta.get("assignment") if isinstance(cluster_meta, dict) else {}
        assigned_cast_id = assignment_meta.get("cast_id") if isinstance(assignment_meta, dict) else None
        if not assigned_cast_id and isinstance(cluster_meta, dict):
            assigned_cast_id = cluster_meta.get("assigned_cast_id")
        track_assignment_meta: Dict[str, Any] = {}
        if isinstance(cluster_meta, dict):
            for track in cluster_meta.get("tracks", []) or []:
                if not isinstance(track, dict):
                    continue
                tid_val = track.get("track_id") or track.get("track") or track.get("track_int")
                if _coerce_track_int(tid_val) == track_id:
                    track_assignment_meta = track.get("assignment") if isinstance(track.get("assignment"), dict) else {}
                    break

        if isinstance(assignment_meta, dict) and assignment_meta.get("unassigned"):
            st.caption(":orange[🚫 Cast unassigned (manual override)]")
        elif assigned_cast_id:
            cast_name = cast_options.get(assigned_cast_id) if cast_options else None
            st.caption(f"🎭 Assigned cast: **{cast_name or assigned_cast_id}**")
            if track_assignment_meta.get("assignment_type") == "track_override":
                st.caption(":orange[⚡ Track override active]")
        else:
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

        if cast_options and track_id is not None:
            override_key = f"track_override_{current_identity}_{track_id}"
            with st.popover("Override cast for this track"):
                selected_cast_id = st.selectbox(
                    "Select cast member",
                    options=[""] + list(cast_options.keys()),
                    format_func=lambda cid: cast_options.get(cid, "Select...") if cid else "Select cast member...",
                    key=override_key,
                    label_visibility="collapsed",
                )
                if selected_cast_id and st.button(
                    "Apply override",
                    key=f"{override_key}_apply",
                    type="primary",
                    use_container_width=True,
                ):
                    if _set_track_override(ep_id, track_id, selected_cast_id):
                        _invalidate_assignment_caches()
                        st.success("Track override saved.")
                        st.rerun()
                if track_assignment_meta.get("assignment_type") == "track_override" and st.button(
                    "Clear override",
                    key=f"{override_key}_clear",
                    use_container_width=True,
                ):
                    if _set_track_override(ep_id, track_id, None):
                        _invalidate_assignment_caches()
                        st.success("Track override cleared.")
                        st.rerun()

    # Track metrics strip (Nov 2024)
    track_metrics = _fetch_track_metrics_cached(ep_id, track_id)
    if track_metrics:
        metrics = []
        # Track similarity (frame consistency within track)
        if track_metrics.get("track_similarity") is not None:
            metrics.append(MetricData(
                metric_type="track",
                value=track_metrics["track_similarity"],
                excluded=track_metrics.get("excluded_frames"),
            ))
        # Person cohesion (how well this track fits with person)
        if track_metrics.get("person_cohesion") is not None:
            metrics.append(MetricData(
                metric_type="person_cohesion",
                value=track_metrics["person_cohesion"],
            ))
        # Quality score
        if track_metrics.get("avg_quality") is not None:
            metrics.append(MetricData(
                metric_type="quality",
                value=track_metrics["avg_quality"],
                breakdown=track_metrics.get("quality_breakdown"),
            ))
        if metrics:
            render_metrics_strip(metrics, compact=False, strip_id=f"track_detail_{track_id}")

    roster_names = _fetch_roster_names(show_slug)
    sample_key = f"track_sample_{ep_id}_{track_id}"
    sample_seeded = sample_key in st.session_state
    if not sample_seeded:
        st.session_state[sample_key] = 1
    sample_prev_key = f"{sample_key}::prev"
    prev_sample = st.session_state.get(sample_prev_key, st.session_state[sample_key])
    sample_col, page_col, info_col = st.columns([1, 1, 2])
    with sample_col:
        # Note: Don't use both value= and key= with session state - causes Streamlit warning
        # The key= parameter handles state binding automatically
        sample = int(
            st.slider(
                "Sample every N crops",
                min_value=1,
                max_value=20,
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

    # Show skipped toggle - placed early so it affects ALL frame loading
    show_skipped_key = f"show_skipped_{ep_id}_{track_id}"
    show_skipped_prev_key = f"{show_skipped_key}::prev"
    prev_show_skipped = st.session_state.get(show_skipped_prev_key, False)
    show_skipped = st.checkbox(
        "Show skipped faces",
        key=show_skipped_key,
        help="Include faces that were auto-skipped due to quality filters (blurry, low confidence)",
    )
    # Clear cache when toggle changes to force re-fetch
    if show_skipped != prev_show_skipped:
        _reset_track_media_state(ep_id, track_id)
        st.session_state[show_skipped_prev_key] = show_skipped

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

    frames_payload = _fetch_track_frames(ep_id, track_id, sample=sample, page=page, page_size=page_size, include_skipped=show_skipped)
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

    # Track-level summary metrics
    track_similarity: float | None = None
    cast_track_score: float | None = None
    track_excluded_frames: int = 0
    track_total_frames: int = 0
    if current_identity:
        track_reps = _fetch_cluster_track_reps_cached(ep_id, current_identity)
        if track_reps and track_reps.get("tracks"):
            for tr in track_reps.get("tracks", []):
                tr_id = coerce_int(tr.get("track_id") or tr.get("track_int"))
                if tr_id == track_id:
                    track_similarity = tr.get("similarity") or tr.get("internal_similarity")
                    cast_track_score = tr.get("cast_track_score")
                    track_excluded_frames = tr.get("excluded_frames", 0)
                    track_total_frames = tr.get("frame_count") or tr.get("faces", 0)
                    break

    frame_sim_values = [f.get("similarity") for f in frames if f.get("similarity") is not None]
    avg_frame_sim = float(np.mean(frame_sim_values)) if frame_sim_values else None
    min_frame_sim = float(np.min(frame_sim_values)) if frame_sim_values else None
    max_frame_sim = float(np.max(frame_sim_values)) if frame_sim_values else None
    outlier_frames = len([s for s in frame_sim_values if s is not None and s < 0.65])
    drift = None
    if min_frame_sim is not None and max_frame_sim is not None:
        drift = max_frame_sim - min_frame_sim

    quality_scores: list[float] = []
    for frame_meta in frames:
        quality = frame_meta.get("quality")
        if isinstance(quality, dict) and quality.get("score") is not None:
            try:
                quality_scores.append(float(quality.get("score")))
            except (TypeError, ValueError):
                continue
    avg_quality = float(np.mean(quality_scores)) if quality_scores else None

    quality_weighted_sim = None
    if frame_sim_values and quality_scores:
        quality_weighted_sim = float(np.mean([s * q for s, q in zip(frame_sim_values, quality_scores)]))

    def _quality_badge(score: float | None) -> str:
        if score is None:
            return ""
        pct = int(score * 100)
        if score >= 0.85:
            color = "#2E7D32"
        elif score >= 0.60:
            color = "#81C784"
        else:
            color = "#EF5350"
        return (
            f'<span style="background-color: {color}; color: white; padding: 2px 6px; '
            f'border-radius: 3px; font-size: 0.8em; font-weight: bold;">Q: {pct}%</span>'
        )

    with st.container(border=True):
        st.markdown("**Track Health**")
        summary_cols = st.columns([1.1, 1.1, 1.2, 1.2])
        with summary_cols[0]:
            if track_similarity is not None:
                # Nov 2024: Enhanced with dropout indicator
                st.markdown(
                    f"Track → Cluster {render_track_with_dropout(track_similarity, track_excluded_frames, track_total_frames)}",
                    unsafe_allow_html=True,
                )
            if cast_track_score is not None:
                st.caption(
                    f"Person Cohesion {render_similarity_badge(cast_track_score, SimilarityType.PERSON_COHESION)}",
                    unsafe_allow_html=True,
                )
        with summary_cols[1]:
            if avg_frame_sim is not None:
                st.markdown(
                    f"Avg Frame {render_similarity_badge(avg_frame_sim, SimilarityType.FRAME)}",
                    unsafe_allow_html=True,
                )
            if min_frame_sim is not None:
                st.caption(
                    f"Min Frame {render_similarity_badge(min_frame_sim, SimilarityType.FRAME)}",
                    unsafe_allow_html=True,
                )
        with summary_cols[2]:
            if avg_quality is not None:
                st.markdown(_quality_badge(avg_quality), unsafe_allow_html=True)
            st.caption(f"Outliers <65%: {outlier_frames}")
        with summary_cols[3]:
            if quality_weighted_sim is not None:
                st.markdown(
                    f"Quality × Sim {render_similarity_badge(quality_weighted_sim, SimilarityType.FRAME)}",
                    unsafe_allow_html=True,
                )
            if drift is not None:
                st.caption(f"Drift (max-min): {drift:.0%}")

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

    # Use the show_skipped value from the checkbox at the top of the track view
    show_skipped_key = f"show_skipped_{ep_id}_{track_id}"
    show_skipped = st.session_state.get(show_skipped_key, False)

    _render_track_media_section(ep_id, track_id, sample=sample, include_skipped=show_skipped)
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
    if integrity:
        if not integrity.get("ok"):
            st.warning(
                "Crops on disk are missing for this track. "
                f"Faces manifest={integrity.get('faces_manifest', 0)} · "
                f"crops={integrity.get('crops_files', 0)}"
            )
        elif integrity.get("all_skipped"):
            faces_skipped = integrity.get("faces_skipped", 0)
            # Check if show_skipped is already enabled
            show_skipped_key = f"show_skipped_{ep_id}_{track_id}"
            show_skipped_enabled = st.session_state.get(show_skipped_key, False)

            st.warning(
                f"**Low-Quality Track:** All {faces_skipped} face{'s' if faces_skipped != 1 else ''} "
                "in this track were marked as too blurry for reliable embedding. "
                "This cluster cannot receive automatic cast suggestions."
            )

            if show_skipped_enabled:
                st.caption("'Show skipped faces' is enabled - the blurry frames are displayed below.")
            else:
                st.caption("Enable 'Show skipped faces' above to view these frames.")

            action_cols_skip = st.columns([1, 1, 1, 1])
            with action_cols_skip[0]:
                if st.button("Unskip all", key=f"unskip_all_{ep_id}_{track_id}", help="Remove skip flag from all faces in this track"):
                    resp = _api_post(f"/episodes/{ep_id}/tracks/{track_id}/unskip_all", {})
                    if resp and resp.get("status") == "unskipped":
                        st.success(f"Unskipped {resp.get('unskipped', 0)} faces")
                        _reset_track_media_state(ep_id, track_id)
                        st.rerun()
                    else:
                        st.error("Failed to unskip faces")
            with action_cols_skip[1]:
                if not show_skipped_enabled:
                    if st.button("Show skipped", key=f"show_skipped_btn_{ep_id}_{track_id}"):
                        st.session_state[show_skipped_key] = True
                        st.session_state[f"{show_skipped_key}::prev"] = True
                        _reset_track_media_state(ep_id, track_id)
                        st.rerun()
            with action_cols_skip[2]:
                # Force Embed / Quality Rescue button
                if st.button("🔧 Force Embed", key=f"force_embed_{ep_id}_{track_id}", help="Re-embed with lower quality threshold (inclusive profile)"):
                    resp = _api_post(
                        f"/episodes/{ep_id}/tracks/{track_id}/force_embed",
                        {"quality_profile": "inclusive", "recompute_centroid": True},
                    )
                    if resp and resp.get("status") in ("rescued", "already_embedded"):
                        unskipped = resp.get("unskipped", 0)
                        st.success(f"Track rescued! Unskipped {unskipped} faces. Refresh Smart Suggestions to see results.")
                        _reset_track_media_state(ep_id, track_id)
                        st.rerun()
                    else:
                        st.error(f"Failed to rescue: {resp.get('error', 'Unknown error')}")
    action_cols = st.columns([1.0, 1.0, 1.0])
    with action_cols[0]:
        targets = [ident.get("identity_id") for ident in identities if ident.get("identity_id") and ident.get("identity_id") != current_identity]
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
    frame_face_ids: Dict[int, str] = {}
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
                    best_face = next(
                        (face for face in faces_for_track if coerce_int(face.get("track_id")) == track_id),
                        None,
                    )
                    if best_face is None:
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
                if frame_idx_int is not None and face_id:
                    frame_face_ids[frame_idx_int] = str(face_id)

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
                    # Show quality score with breakdown (Nov 2024 enhanced)
                    quality = best_face.get("quality") or frame_meta.get("quality")
                    if quality and isinstance(quality, dict):
                        quality_score_value = quality.get("score")
                        if quality_score_value is not None:
                            # Use new quality breakdown badge with components
                            quality_badge = render_quality_breakdown_badge(
                                quality_score_value,
                                detection=quality.get("det_score") or quality.get("detection"),
                                sharpness=quality.get("sharpness") or quality.get("std"),
                                area=quality.get("area") or quality.get("face_area"),
                            )
                            st.markdown(quality_badge, unsafe_allow_html=True)

                    # Show outlier badge ONLY if frame is an outlier (Nov 2024 - reduced noise)
                    similarity = best_face.get("similarity") if isinstance(best_face, dict) else None
                    if similarity is None:
                        similarity = frame_meta.get("similarity")
                    if similarity is not None:
                        # Only show outlier badge if below threshold (50%)
                        outlier_badge = render_outlier_severity_badge(similarity)
                        if outlier_badge:
                            st.markdown(outlier_badge, unsafe_allow_html=True)
                    if skip_reason:
                        st.markdown(f":red[⚠ invalid crop] {skip_reason}")
                    exclusion_entry = face_exclusions.get(str(face_id)) if face_id else None
                    if isinstance(exclusion_entry, dict) and exclusion_entry.get("excluded", True):
                        st.markdown(":orange[🚫 Excluded]")
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
        identity_values = [None] + [ident.get("identity_id") for ident in identities if ident.get("identity_id")]
        identity_labels = ["Create new identity"] + [
            f"{ident.get('identity_id')} · {(ident.get('name') or ident.get('label') or ident.get('identity_id', 'unknown'))}"
            for ident in identities if ident.get("identity_id")
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
        action_cols = st.columns([1, 1, 1])
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
        selected_face_ids = [
            frame_face_ids[frame_idx]
            for frame_idx in selected_frames
            if frame_idx in frame_face_ids
        ]
        if action_cols[2].button("Exclude selected", key=f"track_exclude_selected_{track_id}"):
            if not selected_face_ids:
                st.warning("No face ids available for selected frames.")
            else:
                for face_id in selected_face_ids:
                    _set_face_exclusion(ep_id, face_id, excluded=True, track_id=track_id)
                _invalidate_assignment_caches()
                st.success(f"Excluded {len(selected_face_ids)} face(s).")
                st.rerun()


def _track_identity_edit(ep_id: str, identity_id: str) -> None:
    """Track that an identity was recently edited (for protection during cleanup)."""
    key = f"recently_edited_identities:{ep_id}"
    edited = st.session_state.get(key, {})
    edited[identity_id] = datetime.datetime.now().isoformat()
    # Keep only last 100 edits
    if len(edited) > 100:
        sorted_items = sorted(edited.items(), key=lambda x: x[1], reverse=True)
        edited = dict(sorted_items[:100])
    st.session_state[key] = edited


def _rename_identity(ep_id: str, identity_id: str, label: str) -> None:
    endpoint = f"/identities/{ep_id}/rename"
    payload = {"identity_id": identity_id, "new_label": label}
    if _api_post(endpoint, payload):
        _track_identity_edit(ep_id, identity_id)  # Track edit for cleanup protection
        _invalidate_assignment_caches()  # Clear caches so UI reflects changes
        st.success("Identity renamed.")
        st.rerun()


def _delete_identity(ep_id: str, identity_id: str) -> None:
    endpoint = f"/episodes/{ep_id}/identities/{identity_id}"
    if _api_delete(endpoint):
        _invalidate_assignment_caches()  # Clear caches so UI reflects changes
        st.success("Identity deleted.")
        st.rerun()


def _api_merge(ep_id: str, source_id: str, target_id: str) -> None:
    endpoint = f"/identities/{ep_id}/merge"
    if _api_post(endpoint, {"source_id": source_id, "target_id": target_id}):
        _track_identity_edit(ep_id, target_id)  # Track edit for cleanup protection
        _invalidate_assignment_caches()  # Clear caches so UI reflects changes
        st.success("Identities merged.")
        st.rerun()


def _move_track(ep_id: str, track_id: int, target_identity_id: str | None) -> None:
    endpoint = f"/identities/{ep_id}/move_track"
    payload = {"track_id": track_id, "target_identity_id": target_identity_id}
    resp = _api_post(endpoint, payload)
    if resp:
        if target_identity_id:
            _track_identity_edit(ep_id, target_identity_id)  # Track edit for cleanup protection
        _invalidate_assignment_caches()  # Clear caches so UI reflects changes
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
        # Try both field names as API may return either
        archive_payload["frame_count"] = track_detail.get("faces_count") or track_detail.get("face_count") or 0
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
        # Invalidate caches to ensure UI reflects changes
        _invalidate_assignment_caches()
        # Navigate back within the person/cluster context
        st.toast("Track archived.")
        person_id = st.session_state.get("selected_person")
        if identity_id:
            _set_view("cluster_tracks", person_id=person_id, identity_id=identity_id)
        else:
            _set_view("people")
        st.rerun()


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
        _invalidate_assignment_caches()  # Clear caches so UI reflects changes
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

# Resolve run_id scope (required for run-scoped mutations + artifact reads).
resolved_run_id: str | None = None
qp_candidate = None
try:
    qp_value = st.query_params.get("run_id")
except Exception:
    qp_value = None
if isinstance(qp_value, str):
    qp_candidate = qp_value.strip()
elif isinstance(qp_value, list) and qp_value:
    qp_candidate = str(qp_value[0]).strip()
if qp_candidate:
    try:
        normalized_qp_run_id = run_layout.normalize_run_id(qp_candidate)
    except ValueError:
        normalized_qp_run_id = None
    if normalized_qp_run_id:
        try:
            known_run_ids = run_layout.list_run_ids(ep_id)
        except Exception:
            known_run_ids = []
        try:
            active_run_id = run_layout.read_active_run_id(ep_id)
        except Exception:
            active_run_id = None
        if not known_run_ids or normalized_qp_run_id in known_run_ids or normalized_qp_run_id == active_run_id:
            resolved_run_id = normalized_qp_run_id

if not resolved_run_id:
    try:
        resolved_run_id = run_layout.read_active_run_id(ep_id)
    except Exception:
        resolved_run_id = None

if not resolved_run_id:
    # Fall back to the most recently modified run directory.
    try:
        candidate_run_ids = run_layout.list_run_ids(ep_id)
    except Exception:
        candidate_run_ids = []
    latest_run_id: str | None = None
    latest_mtime = -1.0
    for candidate in candidate_run_ids:
        try:
            mtime = run_layout.run_root(ep_id, candidate).stat().st_mtime
        except (FileNotFoundError, OSError, ValueError):
            continue
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_run_id = candidate
    resolved_run_id = latest_run_id

if not resolved_run_id:
    st.warning("Faces Review requires a run-scoped attempt (run_id). Open Episode Detail to select an attempt.")
    if st.button("Open Episode Detail", key=f"{ep_id}::faces_review_missing_run_id"):
        try:
            st.query_params["ep_id"] = ep_id
        except Exception:
            pass
        helpers.try_switch_page("pages/2_Episode_Detail.py")
    st.stop()

if MOCKED_STREAMLIT and not resolved_run_id:
    resolved_run_id = "test-run"

_CURRENT_RUN_ID = resolved_run_id
try:
    st.query_params["run_id"] = resolved_run_id
except Exception:
    pass

if MOCKED_STREAMLIT:
    artifact_check = faces_review_artifacts.RunArtifactHydration(
        run_id=resolved_run_id or "test-run",
        required=(),
        optional=(),
    )
else:
    artifact_check = _ensure_faces_review_artifacts(ep_id, resolved_run_id)
if artifact_check.hydrated:
    hydrated_key = f"{ep_id}::{resolved_run_id}::faces_review_hydrated"
    if not st.session_state.get(hydrated_key):
        st.session_state[hydrated_key] = True
        st.info(f"Hydrated {len(artifact_check.hydrated)} run artifact(s) from S3 for this attempt.")
if artifact_check.missing_required:
    _render_missing_run_artifacts(ep_id, resolved_run_id, artifact_check)
    st.stop()
if artifact_check.missing_optional:
    st.caption(
        "Some optional run artifacts are missing; cluster quality metrics may be incomplete for this attempt."
    )

_initialize_state(ep_id)

# Check for active Celery jobs and render status (Phase 2 - non-blocking jobs)
_api_base = st.session_state.get("api_base")
if _api_base and session_manager.render_job_status(_api_base):
    # Job is still running - schedule auto-refresh with exponential backoff
    import time as _time
    _poll_count_key = "job_poll_count"
    _poll_count = st.session_state.get(_poll_count_key, 0)
    # Exponential backoff: 2s, 3s, 4s, 5s, 6s... capped at 8s
    _delay = min(2 + _poll_count, 8)
    st.session_state[_poll_count_key] = _poll_count + 1
    _time.sleep(_delay)
    st.rerun()
else:
    # Job complete or no job - reset poll counter
    if "job_poll_count" in st.session_state:
        del st.session_state["job_poll_count"]

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

st.info(
    "Group Clusters can run automatically at the end of **Run Cluster** on the Episode Detail page. "
    "If this attempt has clusters but no grouped people, use the Group Clusters (auto) button below."
)
st.caption("Use existing face bank entries for cross-episode suggestions.")

view_state = st.session_state.get("facebank_view", "people")
ep_meta = helpers.parse_ep_id(ep_id) or {}
season_label: str | None = None
season_value = ep_meta.get("season")
if isinstance(season_value, int):
    season_label = f"S{season_value:02d}"

filter_cast_id = st.session_state.get("filter_cast_id")
bundle = _fetch_faces_review_bundle(ep_id, filter_cast_id=filter_cast_id)
if not bundle:
    st.error("Failed to load Faces Review bundle. Please check that the API is running and the episode has been processed.")
    st.stop()

identities_payload = bundle.get("identities") or {}
cluster_payload = bundle.get("cluster_payload") or {"clusters": []}
people = bundle.get("people") or []
archived_sets = bundle.get("archived_ids") or {}
cast_options = bundle.get("cast_options") or {}
legacy_people_fallback = bool(bundle.get("legacy_people_fallback"))

_render_run_status_panel(ep_id, _CURRENT_RUN_ID, bundle)

if not identities_payload:
    st.error("Failed to load identities data. Please check that the API is running and the episode has been processed.")
    st.stop()

# Show local fallback banner if any local files are being used
_show_local_fallback_banner(cluster_payload)

# Show sync-to-S3 option for missing thumbnails
_render_sync_to_s3_button(ep_id)

# Keep archived sets handy for other views (run-scoped bundle output)
archived_clusters_raw = archived_sets.get("clusters", []) if isinstance(archived_sets, dict) else []
archived_tracks_raw = archived_sets.get("tracks", []) if isinstance(archived_sets, dict) else []
archived_clusters = set(archived_clusters_raw)
archived_tracks = set(archived_tracks_raw)
st.session_state[f"{ep_id}::archived_clusters"] = archived_clusters
st.session_state[f"{ep_id}::archived_tracks"] = archived_tracks

# Consume Improve Faces trigger set by Episode Detail (run-scoped)
if st.session_state.pop(_improve_faces_state_key(ep_id, "trigger"), False):
    _start_improve_faces(ep_id)

# Manual Improve Faces launcher (always available)
action_btn_col1, action_btn_col2, action_btn_col3 = st.columns([1, 1, 2])
with action_btn_col1:
    if st.button("🎯 Improve Faces", key=f"improve_faces_btn_{ep_id}", type="primary"):
        _start_improve_faces(ep_id, force=True)

with action_btn_col2:
    if st.button("🧹 Cleanup Empty Clusters", key=f"cleanup_empty_{ep_id}", help="Remove clusters with no tracks"):
        with st.spinner("Cleaning up empty clusters..."):
            cleanup_resp = _api_post(f"/episodes/{ep_id}/cleanup_empty_clusters")
            if cleanup_resp and cleanup_resp.get("status") == "success":
                removed = cleanup_resp.get("removed_clusters", [])
                if removed:
                    st.success(f"Removed {len(removed)} empty cluster(s): {', '.join(removed[:5])}{'...' if len(removed) > 5 else ''}")
                    _invalidate_assignment_caches()
                    st.rerun()
                else:
                    st.info("No empty clusters found")
            else:
                st.error(f"Cleanup failed: {cleanup_resp.get('error', 'Unknown error') if cleanup_resp else 'API error'}")

# If Improve Faces modal is open, render it and skip the heavy page to keep YES/NO snappy
_render_improve_faces_modal(ep_id)
if st.session_state.get(_improve_faces_state_key(ep_id, "active")):
    st.stop()

cluster_lookup = _clusters_by_identity(cluster_payload)
identities = identities_payload.get("identities", [])
identity_index = {ident["identity_id"]: ident for ident in identities}
# show_slug already defined above - no need to recompute
roster_names = _fetch_roster_names(show_slug)
show_id = show_slug
_LEGACY_PEOPLE_FALLBACK = legacy_people_fallback
if legacy_people_fallback:
    legacy_warn_key = f"{ep_id}::{_CURRENT_RUN_ID}::faces_review_legacy_people"
    if not st.session_state.get(legacy_warn_key):
        st.session_state[legacy_warn_key] = True
        st.warning(
            "Showing legacy/global people for this episode (no run-scoped people found). "
            "These may not match the selected attempt."
        )
people_lookup = {str(person.get("person_id") or ""): person for person in people}

selected_person = st.session_state.get("selected_person")
selected_identity = st.session_state.get("selected_identity")
selected_track = st.session_state.get("selected_track")

# cast_options already comes from the bundle (run-scoped + legacy merge)
assignment_payload = bundle.get("assignments") or {}
face_exclusions = assignment_payload.get("faces") or {}

if view_state == "track" and selected_track is not None:
    _render_track_view(
        ep_id,
        selected_track,
        identities_payload,
        cluster_lookup=cluster_lookup,
        cast_options=cast_options,
        face_exclusions=face_exclusions,
    )
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
        bundle=bundle,
    )


# Render Cluster Comparison Mode in sidebar (Feature 3)
_render_cluster_comparison_mode(ep_id, show_id, cluster_lookup)

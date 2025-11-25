from __future__ import annotations

import datetime
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from concurrent.futures import ThreadPoolExecutor
import math

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
from track_frame_utils import (  # noqa: E402
    best_track_frame_idx,
    coerce_int,
    quality_score,
    scope_track_frames,
    track_faces_debug,
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

    try:
        col1, col2 = st.columns(2)
    except Exception:
        col1 = col2 = st
    with col1:
        _render_similarity_card(
            "#2196F3",
            "Identity Similarity",
            "Frame → Track centroid. Appears as `ID: XX%` badge on each frame.",
            ["≥ 75%: Strong match", "60–74%: Good match", "< 60%: Weak match"],
        )
    with col2:
        _render_similarity_card(
            "#4CAF50",
            "Quality Score",
            "Detection confidence + sharpness + face area. Appears as `Q: XX%` badge.",
            ["≥ 85%: High quality", "60–84%: Medium quality", "< 60%: Low quality"],
        )

    try:
        col3, col4 = st.columns(2)
    except Exception:
        col3 = col4 = st
    with col3:
        _render_similarity_card(
            "#9C27B0",
            "Cast Similarity",
            "Cluster → Cast member match (auto-assignment).",
            ["≥ 0.68 auto-assigns to cast", "0.50–0.67 requires review"],
        )
    with col4:
        _render_similarity_card(
            "#FF9800",
            "Track Similarity",
            "Frame → Track prototype (appearance gate).",
            [
                "≥ 0.82 soft threshold (continue track)",
                "≥ 0.75 hard threshold (force split)",
            ],
        )

    try:
        col5, col6 = st.columns(2)
    except Exception:
        col5 = col6 = st
    with col5:
        _render_similarity_card(
            "#8BC34A",
            "Cluster Similarity",
            "Face → Face grouping via DBSCAN.",
            ["≥ 0.35 grouped in same cluster", "Used for track grouping/merging"],
        )
    with col6:
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


def _cast_carousel_cache() -> Dict[str, Any]:
    return st.session_state.setdefault(_CAST_CAROUSEL_CACHE_KEY, {})


def _cast_people_cache() -> Dict[str, Any]:
    return st.session_state.setdefault(_CAST_PEOPLE_CACHE_KEY, {})


def _track_media_cache() -> Dict[str, Dict[str, Any]]:
    return st.session_state.setdefault(_TRACK_MEDIA_CACHE_KEY, {})


def _track_media_state(ep_id: str, track_id: int, sample: int = 1) -> Dict[str, Any]:
    """Get or create cache state for track media, keyed by ep_id, track_id, AND sample rate."""
    cache = _track_media_cache()
    key = f"{ep_id}::{track_id}::s{sample}"
    if key not in cache:
        cache[key] = {
            "items": [],
            "cursor": None,
            "initialized": False,
            "sample": sample,
            "batch_limit": TRACK_MEDIA_BATCH_LIMIT,
        }
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


def _render_similarity_badge(similarity: float | None, metric: str = "identity") -> str:
    """Render a similarity score as a percentage badge with metric-specific colors."""
    if similarity is None:
        return ""
    value = max(0.0, min(float(similarity), 1.0))
    pct = int(round(value * 100))
    metric_key = (metric or "identity").lower()
    if metric_key == "cluster":
        if value >= 0.35:
            color = "#7CB342"  # Cluster match
        elif value >= 0.20:
            color = "#C5E1A5"
        else:
            color = "#E0E0E0"
    elif metric_key == "track":
        if value >= 0.82:
            color = "#FB8C00"
        elif value >= 0.75:
            color = "#FFB74D"
        else:
            color = "#FFE0B2"
    elif metric_key == "cast":
        if value >= 0.68:
            color = "#8E24AA"
        elif value >= 0.50:
            color = "#CE93D8"
        else:
            color = "#F3E5F5"
    else:  # identity is default
        if value >= 0.75:
            color = "green"
        elif value >= 0.60:
            color = "orange"
        else:
            color = "red"
    return (
        f'<span style="background-color: {color}; color: white; padding: 2px 6px; '
        f'border-radius: 3px; font-size: 0.8em; font-weight: bold;">{pct}%</span>'
    )


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


@st.cache_data(ttl=60)
def _fetch_identities_cached(ep_id: str) -> Dict[str, Any] | None:
    return _safe_api_get(f"/episodes/{ep_id}/identities")


@st.cache_data(ttl=60)
def _fetch_people_cached(show_slug: str | None) -> Dict[str, Any] | None:
    if not show_slug:
        return None
    return _safe_api_get(f"/shows/{show_slug}/people")


@st.cache_data(ttl=60)
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


def _fetch_tracks_meta(ep_id: str, track_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Batch fetch track metadata when possible; fallback to cached per-track fetch."""
    unique_ids = sorted({tid for tid in track_ids if tid is not None})
    if not unique_ids:
        return {}

    # Try batch endpoint first (silently falls back to per-track fetches if unavailable)
    try:
        ids_param = ",".join(str(tid) for tid in unique_ids)
        batch_resp = _safe_api_get(
            f"/episodes/{ep_id}/tracks",
            params={"ids": ids_param, "fields": "id,track_id,faces_count,frames"},
        )
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
        st.session_state["facebank_ep"] = ep_id
        st.session_state["facebank_view"] = "people"
        st.session_state["selected_person"] = None
        st.session_state["selected_identity"] = None
        st.session_state["selected_track"] = None
        st.session_state.pop("facebank_query_applied", None)


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
    identity_id = _single("identity_id") or _single("cluster_id")
    track_id = coerce_int(_single("track_id"))

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
    """Update view state. Streamlit will auto-rerun after callback completes."""
    st.session_state["facebank_view"] = view
    st.session_state["selected_person"] = person_id
    st.session_state["selected_identity"] = identity_id
    st.session_state["selected_track"] = track_id


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
                    "help": "Split incorrectly merged tracks. Low risk - usually beneficial.",
                    "default": True,
                    "risk": "low",
                },
                "reembed": {
                    "label": "Regenerate embeddings (reembed)",
                    "help": "Recalculate face embeddings. Low risk - just regenerates vectors.",
                    "default": True,
                    "risk": "low",
                },
                "recluster": {
                    "label": "Re-cluster faces (recluster)",
                    "help": "⚠️ HIGH RISK: Regenerates identities.json, may undo manual splits.",
                    "default": False,
                    "risk": "high",
                },
                "group_clusters": {
                    "label": "Auto-group clusters (group_clusters)",
                    "help": "Group similar clusters into people. Medium risk - respects seed matching.",
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
        with st.popover("🔄 Refresh Values", help="Recompute similarity scores and refresh suggestions"):
            st.markdown("**Refresh Options:**")

            auto_link_enabled = st.checkbox(
                "🔗 Auto-assign high confidence matches",
                value=True,
                key="auto_link_checkbox",
                help="Automatically assign clusters to cast members when facebank similarity ≥85%",
            )

            if st.button("Run Refresh", key="facebank_refresh_similarity_button", type="primary"):
                with st.spinner("Refreshing similarity values and cast suggestions..."):
                    # Step 1: Trigger similarity index refresh for all identities
                    refresh_resp = _api_post(f"/episodes/{ep_id}/refresh_similarity", {})
                    if not refresh_resp or refresh_resp.get("status") != "success":
                        st.error("Failed to refresh similarity values. Check logs.")
                    else:
                        auto_linked_count = 0

                        # Enhancement #8: Auto-link high confidence matches
                        if auto_link_enabled:
                            auto_link_resp = _api_post(f"/episodes/{ep_id}/auto_link_cast", {})
                            if auto_link_resp and auto_link_resp.get("auto_assigned", 0) > 0:
                                auto_linked_count = auto_link_resp["auto_assigned"]
                                assignments = auto_link_resp.get("assignments", [])
                                for asn in assignments[:5]:  # Show first 5
                                    st.success(
                                        f"✓ Auto-assigned cluster {asn.get('cluster_id')} → "
                                        f"{asn.get('cast_name')} ({int(asn.get('similarity', 0) * 100)}%)"
                                    )
                                if len(assignments) > 5:
                                    st.info(f"... and {len(assignments) - 5} more")

                        # Step 2: Refresh cluster suggestions based on new similarity values
                        suggestions_resp = _safe_api_get(f"/episodes/{ep_id}/cluster_suggestions_from_assigned")

                        # Step 3: Fetch cast suggestions from facebank (Enhancement #1)
                        cast_suggestions_resp = _safe_api_get(f"/episodes/{ep_id}/cast_suggestions")
                        if cast_suggestions_resp and cast_suggestions_resp.get("suggestions"):
                            # Store in session state for display
                            st.session_state[f"cast_suggestions:{ep_id}"] = {
                                sugg["cluster_id"]: sugg.get("cast_suggestions", [])
                                for sugg in cast_suggestions_resp.get("suggestions", [])
                            }

                        # Show summary
                        if auto_linked_count > 0:
                            st.success(f"✅ Refreshed values and auto-assigned {auto_linked_count} cluster(s)!")
                        elif cast_suggestions_resp and cast_suggestions_resp.get("suggestions"):
                            st.success("✅ Refreshed similarity values and cast suggestions!")
                        elif suggestions_resp:
                            st.success("✅ Refreshed similarity values!")
                        else:
                            st.warning("✅ Refreshed similarity values, but no cast suggestions available.")
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
    params: Dict[str, Any] = {"sample": int(sample), "limit": int(limit)}
    if cursor:
        # The backend returns pagination with a 'next_start_after' cursor
        params["start_after"] = cursor
    payload = _safe_api_get(f"/episodes/{ep_id}/tracks/{track_id}/crops", params=params) or {}
    items = payload.get("items", []) if isinstance(payload, dict) else []
    next_cursor = payload.get("next_start_after") if isinstance(payload, dict) else None
    normalized: List[Dict[str, Any]] = []
    for item in items:
        url = item.get("media_url") or item.get("url") or item.get("thumbnail_url")
        resolved = helpers.resolve_thumb(url)
        # Include all crops, even if URL resolution fails temporarily
        # The UI can handle missing images gracefully
        normalized.append(
            {
                "url": resolved or url,  # Use original URL if resolution fails
                "frame_idx": item.get("frame_idx"),
                "track_id": track_id,
                "s3_key": item.get("s3_key"),  # Preserve S3 key for debugging
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
    people_by_cast_id = {p.get("cast_id"): p for p in people if p.get("cast_id")}

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
                    clusters_summary = _safe_api_get(f"/episodes/{ep_id}/people/{person_id}/clusters_summary")
                    if clusters_summary and clusters_summary.get("clusters"):
                        cohesion_scores: List[float] = []
                        for cluster in clusters_summary.get("clusters", []):
                            total_tracks += cluster.get("tracks", 0)
                            total_faces += cluster.get("faces", 0)
                            cohesion = cluster.get("cohesion")
                            if cohesion is not None:
                                cohesion_scores.append(cohesion)
                        if cohesion_scores:
                            avg_cohesion = sum(cohesion_scores) / len(cohesion_scores)

                st.caption(f"**{len(episode_clusters)}** clusters")
                st.caption(f"**{total_tracks}** tracks · **{total_faces}** faces")

                if avg_cohesion is not None:
                    badge = _render_similarity_badge(avg_cohesion, metric="cluster")
                    st.markdown(f"Cohesion {badge}", unsafe_allow_html=True)

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
    tracks_count = counts.get("tracks", 0)
    faces_count = counts.get("faces", 0)
    track_list = cluster_meta.get("tracks", [])

    # Filter out tracks with only 1 frame (likely noise/false positives)
    track_list = [t for t in track_list if t.get("faces", 0) > 1]

    # Recalculate counts after filtering
    tracks_count = len(track_list)
    faces_count = sum(t.get("faces", 0) for t in track_list)

    # Skip clusters with no valid tracks after filtering
    if not track_list or tracks_count == 0:
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

    with st.container(border=True):
        # Header with cluster info
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.markdown(f"### 🔍 Cluster `{cluster_id}`")
            st.caption(f"{tracks_count} track(s) · {faces_count} face(s)")
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

            # Track pagination state for this cluster
            page_key = f"track_page_{cluster_id}"
            if page_key not in st.session_state:
                st.session_state[page_key] = 0

            current_page = st.session_state[page_key]
            total_pages = (num_tracks + max_visible - 1) // max_visible

            # Navigation controls if more than max_visible tracks
            if total_pages > 1:
                col_left, col_center, col_right = st.columns([1, 6, 1])
                with col_left:
                    if st.button("◀", key=f"prev_{cluster_id}", disabled=current_page == 0):
                        st.session_state[page_key] = max(0, current_page - 1)
                        st.rerun()
                with col_center:
                    st.caption(
                        f"Showing tracks {current_page * max_visible + 1}-{min((current_page + 1) * max_visible, num_tracks)} of {num_tracks}"
                    )
                with col_right:
                    if st.button(
                        "▶",
                        key=f"next_{cluster_id}",
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
                            # Extract S3 key from URL if it's a presigned URL (to avoid expiration)
                            # Format: https://bucket.s3.amazonaws.com/key?presign_params
                            # We want just the key part for resolve_thumb to generate fresh presign
                            if "s3.amazonaws.com/" in thumb_url:
                                # Extract key: everything between .com/ and ? (or end if no ?)
                                start_idx = thumb_url.find("s3.amazonaws.com/") + len("s3.amazonaws.com/")
                                end_idx = thumb_url.find("?") if "?" in thumb_url else len(thumb_url)
                                s3_key = thumb_url[start_idx:end_idx]
                                thumb_markup = helpers.thumb_html(
                                    s3_key,
                                    alt=f"Track {track_id}",
                                    hide_if_missing=False,
                                )
                            else:
                                thumb_markup = helpers.thumb_html(
                                    thumb_url,
                                    alt=f"Track {track_id}",
                                    hide_if_missing=False,
                                )
                            st.markdown(thumb_markup, unsafe_allow_html=True)
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
) -> None:
    """Render an auto-detected person card with detailed clusters and bulk assignment."""
    person_id = str(person.get("person_id") or "")
    name = person.get("name") or "(unnamed)"
    aliases = person.get("aliases") or []
    total_clusters = len(person.get("cluster_ids", []) or [])

    # Get suggested cast member if available
    suggested_person_id = None
    suggested_distance = None
    if episode_clusters and suggestions_by_cluster:
        first_cluster = episode_clusters[0].split(":")[-1]
        suggestion = suggestions_by_cluster.get(first_cluster)
        if suggestion:
            suggested_person_id = suggestion.get("suggested_person_id")
            suggested_distance = suggestion.get("distance")

    with st.container(border=True):
        # Name
        st.markdown(f"### 👤 {name}")

        # Show aliases if present
        if aliases:
            alias_text = ", ".join(f"`{a}`" for a in aliases)
            st.caption(f"Aliases: {alias_text}")

        # Metrics
        st.caption(
            f"ID: {person_id} · {total_clusters} cluster(s) overall · " f"{len(episode_clusters)} in this episode"
        )

        # Jump to full cluster view for this person
        if person_id and episode_clusters:
            if st.button(
                "View All Clusters",
                key=f"view_all_clusters_{person_id}",
                use_container_width=True,
            ):
                _set_view("person_clusters", person_id=person_id)
                st.rerun()

        # Bulk assignment for unnamed people
        if not person.get("cast_id") and not person.get("name"):
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
                            people = people_resp.get("people", [])
                            suggested_person = next(
                                (p for p in people if p.get("person_id") == suggested_person_id),
                                None,
                            )
                            if suggested_person:
                                suggested_cast_id = suggested_person.get("cast_id")

                    # Determine default index
                    default_index = 0
                    if suggested_cast_id and suggested_cast_id in cast_options:
                        default_index = list(cast_options.keys()).index(suggested_cast_id)

                    # Use a form to ensure selectbox and button states are synchronized
                    with st.form(key=f"assign_form_{person_id}"):
                        selected_cast_id = st.selectbox(
                            "Select cast member",
                            options=list(cast_options.keys()),
                            format_func=lambda pid: cast_options[pid],
                            index=default_index,
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

                        submit_assign = st.form_submit_button("Assign Cluster")

                        if submit_assign:
                            # Move all clusters to the selected cast member
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
                    result = _bulk_assign_to_new_person(ep_id, show_id, person_id, new_name, episode_clusters)
                    if result:
                        st.success(f"Created '{new_name}' and assigned {len(episode_clusters)} clusters")
                        st.rerun()

        # Fetch clusters summary to show thumbnails
        clusters_summary = _safe_api_get(f"/episodes/{ep_id}/people/{person_id}/clusters_summary")
        if clusters_summary and clusters_summary.get("clusters"):
            st.markdown("**Clusters in this episode:**")

            # Render clusters in a grid (3 per row)
            clusters = clusters_summary.get("clusters", [])
            cols_per_row = 3
            for row_start in range(0, len(clusters), cols_per_row):
                row_clusters = clusters[row_start : row_start + cols_per_row]
                row_cols = st.columns(cols_per_row)

                for idx, cluster in enumerate(row_clusters):
                    with row_cols[idx]:
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
                        cohesion_badge = _render_similarity_badge(cohesion, metric="cluster") if cohesion else ""
                        st.markdown(f"**{cluster_id}** {cohesion_badge}", unsafe_allow_html=True)
                        st.caption(f"{cluster.get('tracks', 0)} tracks · {cluster.get('faces', 0)} faces")

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
                            default_index = 0
                            cast_ids = list(cast_options.keys())
                            if suggested_cast_id and suggested_cast_id in cast_options:
                                default_index = cast_ids.index(suggested_cast_id)
                            with st.form(key=f"assign_cluster_form_{person_id}_{cluster_id}"):
                                selected_cast_id = st.selectbox(
                                    "Assign to cast member",
                                    options=cast_ids,
                                    format_func=lambda cid: cast_options[cid],
                                    index=default_index,
                                    key=f"assign_cast_select_{person_id}_{cluster_id}",
                                )
                                submitted = st.form_submit_button("Assign cluster", use_container_width=True)
                                if submitted:
                                    if _assign_cluster_to_cast(ep_id, show_id, cluster_id, selected_cast_id):
                                        st.success(f"Assigned to {cast_options[selected_cast_id]}")
                                        st.rerun()
                        else:
                            st.caption("No cast members available for assignment.")
        else:
            # Fallback: just show view clusters button
            if st.button("View clusters", key=f"view_clusters_{person_id}"):
                _set_view("person_clusters", person_id=person_id)
                st.rerun()

        # Delete button for auto-clustered people (those without cast_id)
        if not person.get("cast_id"):
            st.markdown("---")
            with st.expander("🗑️ Delete this person"):
                st.warning(
                    f"This will permanently delete **{name}** ({person_id}) and remove all {total_clusters} cluster assignment(s)."
                )
                st.caption("The clusters will remain in identities.json and can be re-assigned later.")

                if st.button(f"Delete {name}", key=f"delete_person_{person_id}", type="secondary"):
                    try:
                        resp = helpers.api_delete(f"/shows/{show_id}/people/{person_id}")
                        st.success(f"Deleted {name} ({person_id})")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to delete person: {exc}")


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
) -> bool:
    """Assign all clusters from source person to a cast member."""
    debug_assign = os.getenv("SCREENALYTICS_DEBUG_ASSIGN_CLUSTER") == "1" or st.session_state.get(
        "debug_assign_cluster"
    )

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
            st.error(
                f"❌ Merge verification FAILED!\n\n"
                f"Expected {expected_final_count} clusters after merge "
                f"(target had {target_cluster_count_before}, adding {expected_cluster_count}), "
                f"but target now has {actual_cluster_count} clusters.\n\n"
                f"**{expected_final_count - actual_cluster_count} clusters were LOST during the merge!**\n\n"
                f"The source person was deleted but clusters were not transferred correctly. "
                f"Check API logs for merge_people errors."
            )
            return False

        _debug("merge verified", {"clusters_transferred": expected_cluster_count})
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

    cast_resp = cast_api_resp if cast_api_resp is not None else _fetch_cast_cached(show_id, season_label)
    raw_cast_entries = cast_resp.get("cast", []) if cast_resp else []
    people_by_cast_id = {p.get("cast_id"): p for p in people if p.get("cast_id")}

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
            _render_cast_gallery(ep_id, cast_gallery_cards)
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
        cast_gallery_cards.append(
            {
                "cast": {
                    "cast_id": cast_id,
                    "name": person.get("name") or "(unnamed)",
                    "aliases": person.get("aliases") or [],
                },
                "person": person,
                "episode_clusters": episode_clusters,
                "featured_thumbnail": person.get("rep_crop"),
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
            episode_auto_people.append(
                {
                    "person": person,
                    "episode_clusters": episode_clusters,
                    "counts": {
                        "clusters": len(episode_clusters),
                        "tracks": total_tracks,
                        "faces": total_faces,
                    },
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
    # Sort auto-people by episode impact: clusters → tracks → faces, then name
    episode_auto_people.sort(
        key=lambda entry: (
            -(entry.get("counts", {}).get("clusters") or 0),
            -(entry.get("counts", {}).get("tracks") or 0),
            -(entry.get("counts", {}).get("faces") or 0),
            entry.get("person", {}).get("name") in (None, ""),
            (entry.get("person", {}).get("name") or "").lower(),
        )
    )

    # --- CAST MEMBERS SECTION ---
    if cast_gallery_cards:
        st.markdown(f"### 🎬 Cast Members ({len(cast_gallery_cards)})")
        st.caption("Cast members with clusters in this episode")
        _render_cast_gallery(ep_id, cast_gallery_cards)

    # --- EPISODE AUTO-PEOPLE SECTION ---
    if episode_auto_people:
        st.markdown("---")
        st.markdown(f"### 👥 Episode Auto-Clustered People ({len(episode_auto_people)})")
        st.caption(f"People auto-detected in episode {ep_id}")

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
            _render_auto_person_card(
                ep_id,
                show_id,
                person,
                episode_clusters,
                cast_options,
                suggestions_by_cluster,
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

        # Sort unassigned clusters by face count (descending - most faces first)
        sorted_clusters = sorted(
            unassigned_clusters,
            key=lambda cid: cluster_lookup.get(cid, {}).get("counts", {}).get("faces", 0),
            reverse=True,
        )

        # Render each unassigned cluster as a suggestion card
        for cluster_id in sorted_clusters:
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
) -> None:
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

    st.caption(f"{len(episode_clusters)} clusters · {total_tracks} tracks · {total_faces} faces")

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
    sort_option = st.selectbox(
        "Sort by:",
        [
            "Track Similarity (Low to High)",
            "Track Similarity (High to Low)",
            "Frame Count (Low to High)",
            "Frame Count (High to Low)",
            "Average Frame Similarity (Low to High)",
            "Average Frame Similarity (High to Low)",
            "Track ID (Low to High)",
            "Track ID (High to Low)",
        ],
        key=f"sort_tracks_{person_id}",
    )

    # Apply sorting based on selection
    if sort_option == "Track Similarity (Low to High)":
        all_tracks.sort(key=lambda t: (t.get("similarity") if t.get("similarity") is not None else 999.0))
    elif sort_option == "Track Similarity (High to Low)":
        all_tracks.sort(
            key=lambda t: (t.get("similarity") if t.get("similarity") is not None else -999.0),
            reverse=True,
        )
    elif sort_option == "Frame Count (Low to High)":
        for track in all_tracks:
            meta = _track_meta(track.get("track_int"))
            frame_count = coerce_int(meta.get("faces_count")) or coerce_int(meta.get("frames_count"))
            if frame_count is None:
                frame_count = len(meta.get("frames", []) or [])
            track["frame_count"] = frame_count or 0
        all_tracks.sort(key=lambda t: t.get("frame_count", 0))
    elif sort_option == "Frame Count (High to Low)":
        for track in all_tracks:
            meta = _track_meta(track.get("track_int"))
            frame_count = coerce_int(meta.get("faces_count")) or coerce_int(meta.get("frames_count"))
            if frame_count is None:
                frame_count = len(meta.get("frames", []) or [])
            track["frame_count"] = frame_count or 0
        all_tracks.sort(key=lambda t: t.get("frame_count", 0), reverse=True)
    elif sort_option == "Average Frame Similarity (Low to High)":
        for track in all_tracks:
            meta = _track_meta(track.get("track_int"))
            frames = meta.get("frames", []) or []
            similarities = [f.get("similarity") for f in frames if f.get("similarity") is not None]
            track["avg_frame_similarity"] = sum(similarities) / len(similarities) if similarities else 999.0
        all_tracks.sort(key=lambda t: t.get("avg_frame_similarity", 999.0))
    elif sort_option == "Average Frame Similarity (High to Low)":
        for track in all_tracks:
            meta = _track_meta(track.get("track_int"))
            frames = meta.get("frames", []) or []
            similarities = [f.get("similarity") for f in frames if f.get("similarity") is not None]
            track["avg_frame_similarity"] = sum(similarities) / len(similarities) if similarities else -999.0
        all_tracks.sort(key=lambda t: t.get("avg_frame_similarity", -999.0), reverse=True)
    elif sort_option == "Track ID (Low to High)":
        all_tracks.sort(
            key=lambda t: (
                int(t.get("track_id", "0").replace("track_", "")) if isinstance(t.get("track_id"), str) else 0
            )
        )
    elif sort_option == "Track ID (High to Low)":
        all_tracks.sort(
            key=lambda t: (
                int(t.get("track_id", "0").replace("track_", "")) if isinstance(t.get("track_id"), str) else 0
            ),
            reverse=True,
        )

    st.markdown(f"**All {len(all_tracks)} Tracks**")

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
            badge_html = _render_similarity_badge(similarity)
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
                        frame_badge = _render_similarity_badge(frame_sim)
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
                if track_id_int is not None:
                    # Move track to different cluster
                    identity_index = {ident["identity_id"]: ident for ident in identities_payload.get("identities", [])}
                    move_targets = [ident_id for ident_id in identity_index if ident_id != cluster_id]
                    if move_targets and st.button("Move track", key=f"move_track_{person_id}_{track_id_int}"):
                        st.session_state[f"show_move_{track_id_int}"] = True
                        st.rerun()
            with col3:
                if track_id_int is not None and st.button(
                    "Delete",
                    key=f"delete_track_{person_id}_{track_id_int}",
                    type="secondary",
                ):
                    _delete_track(ep_id, track_id_int)


def _render_cluster_tracks(
    ep_id: str,
    identity_id: str,
    cluster_lookup: Dict[str, Dict[str, Any]],
    identity_index: Dict[str, Dict[str, Any]],
    show_slug: str | None,
    roster_names: List[str],
    person_id: str | None,
) -> None:
    st.button(
        "← Back to clusters",
        key="facebank_back_person_clusters",
        on_click=lambda: _set_view("person_clusters", person_id=person_id),
    )

    # Fetch track representatives from new endpoint
    track_reps_data = _safe_api_get(f"/episodes/{ep_id}/clusters/{identity_id}/track_reps")
    if not track_reps_data:
        st.error("Failed to load track representatives.")
        return
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
    faces_count = identity_meta.get("size")
    # If size is not set (e.g., newly created identity), compute from track frames
    if faces_count is None or faces_count == 0:
        # Try to count frames from track_ids
        track_ids = identity_meta.get("track_ids", [])
        if track_ids:
            # Fetch track details to get frame counts
            ids_param = ",".join(str(tid) for tid in track_ids)
            tracks_resp = _safe_api_get(
                f"/episodes/{ep_id}/tracks",
                params={"ids": ids_param, "fields": "faces_count"},
            )
            if tracks_resp:
                tracks_data = tracks_resp.get("tracks", [])
                faces_count = sum(t.get("faces_count", 0) for t in tracks_data)
            else:
                faces_count = 0
        else:
            faces_count = 0
    track_reps = track_reps_data.get("tracks", [])

    # Header
    header_parts = [identity_id, f"Tracks: {tracks_count}", f"Faces: {faces_count}"]
    if cohesion is not None:
        header_parts.append(f"Cohesion: {int(cohesion * 100)}%")
    st.subheader(" · ".join(header_parts))

    if display_name or label:
        st.caption(" · ".join([part for part in [display_name, label] if part]))

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

    # Export to Facebank button (only if person_id is assigned)
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
        st.markdown("---")
    else:
        st.info(
            "💡 To export this identity to facebank, first assign it to a cast member using the name controls above."
        )

    # Display all track representatives with similarity scores
    if not track_reps:
        st.info("No track representatives available.")
        return

    st.markdown(f"**All {len(track_reps)} Track(s) with Similarity Scores:**")
    st.caption("Sorted by similarity: lowest (worst match) to highest (best match)")
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

                # Display track ID and similarity badge
                badge_html = _render_similarity_badge(similarity)
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
                        "Delete",
                        key=f"cluster_delete_btn_{identity_id}_{track_id_int}",
                        type="secondary",
                    ):
                        _delete_track(ep_id, track_id_int)


def _render_track_view(ep_id: str, track_id: int, identities_payload: Dict[str, Any]) -> None:
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
    show_slug = _episode_show_slug(ep_id)
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
        if st.button("Delete track", key=f"track_view_delete_{track_id}"):
            _delete_track(ep_id, track_id)

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
                        similarity_badge = _render_similarity_badge(similarity)
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


def _delete_track(ep_id: str, track_id: int) -> None:
    if _api_post(f"/identities/{ep_id}/drop_track", {"track_id": track_id}):
        # Clear selection state for this track
        st.session_state.get("track_frame_selection", {}).pop(track_id, None)
        # Navigate back within the person/cluster context
        st.toast("Track deleted.")
        identity_id = st.session_state.get("selected_identity")
        person_id = st.session_state.get("selected_person")
        if identity_id:
            _set_view("cluster_tracks", person_id=person_id, identity_id=identity_id)
        else:
            _set_view("people")


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
    helpers.set_ep_id(sidebar_ep_id, rerun=False, origin="sidebar")
    ep_id = sidebar_ep_id
if not ep_id:
    ep_id_from_query = helpers.get_ep_id_from_query_params()
    if ep_id_from_query:
        helpers.set_ep_id(ep_id_from_query, rerun=False, update_query_params=False, origin="query")
        ep_id = ep_id_from_query
if not ep_id:
    st.warning("Select an episode from the sidebar to continue.")
    st.stop()
_initialize_state(ep_id)
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

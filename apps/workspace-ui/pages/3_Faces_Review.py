from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import math

import requests
import streamlit as st

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402

cfg = helpers.init_page("Faces & Tracks")
st.title("Faces & Tracks Review")
st.caption(f"Backend: {cfg['backend']} · Bucket: {cfg.get('bucket') or 'n/a'}")

# Similarity Scores Color Key (native Streamlit layout to avoid raw HTML)
with st.expander("📊 Similarity Scores Guide", expanded=False):
    st.markdown("### Similarity Types")

    def _render_similarity_card(
        color: str, title: str, description: str, details: List[str]
    ) -> None:
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

    col1, col2 = st.columns(2)
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

    col3, col4 = st.columns(2)
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

    col5, col6 = st.columns(2)
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

MAX_TRACKS_PER_ROW = 6


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
            color = "#1E88E5"
        elif value >= 0.60:
            color = "#64B5F6"
        else:
            color = "#BBDEFB"
    return (
        f'<span style="background-color: {color}; color: white; padding: 2px 6px; '
        f'border-radius: 3px; font-size: 0.8em; font-weight: bold;">{pct}%</span>'
    )


def _safe_api_get(
    path: str, params: Dict[str, Any] | None = None
) -> Dict[str, Any] | None:
    try:
        return helpers.api_get(path, params=params)
    except requests.RequestException as exc:
        st.error(helpers.describe_error(f"{cfg['api_base']}{path}", exc))
        return None


def _api_post(
    path: str, payload: Dict[str, Any] | None = None
) -> Dict[str, Any] | None:
    try:
        return helpers.api_post(path, payload or {})
    except requests.RequestException as exc:
        st.error(helpers.describe_error(f"{cfg['api_base']}{path}", exc))
        return None


def _api_delete(
    path: str, payload: Dict[str, Any] | None = None
) -> Dict[str, Any] | None:
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
        return st.text_input(
            text_label, value=current_name, key=f"{key_prefix}_input"
        ).strip()
    return choice.strip()


def _save_identity_name(
    ep_id: str, identity_id: str, name: str, show: str | None
) -> None:
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


def _assign_track_name(ep_id: str, track_id: int, name: str, show: str | None) -> None:
    cleaned = name.strip()
    if not cleaned:
        st.warning("Provide a non-empty name before saving.")
        return
    payload: Dict[str, Any] = {"name": cleaned}
    if show:
        payload["show"] = show
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
    name = (
        resp.get("target_name")
        or resp.get("target_identity_id")
        or target_identity_id
        or "target identity"
    )
    st.toast(f"Moved {moved} frame(s) to {name}")
    _refresh_roster_names(show)
    # Only clear selection after successful move
    st.session_state.setdefault("track_frame_selection", {}).pop(track_id, None)
    st.rerun()


def _delete_frames_api(
    ep_id: str, track_id: int, frame_ids: List[int], delete_assets: bool = True
) -> None:
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


def _select_episode() -> str:
    """Episode selector in sidebar with lock/unlock mechanism."""
    with st.sidebar:
        st.markdown("### Episode")

        current = helpers.get_ep_id()

        # Check if selector is locked (default is locked)
        is_locked = not st.session_state.get("episode_selector_unlocked", False)

        if current and is_locked:
            # Show locked state - just display current episode
            ep_meta = helpers.parse_ep_id(current) or {}
            show = ep_meta.get("show", "").upper()
            season = ep_meta.get("season", 0)
            episode = ep_meta.get("episode", 0)
            display_text = (
                f"{show} S{season:02d}E{episode:02d}"
                if show and season and episode
                else current
            )

            st.info(f"🔒 **{display_text}**")

            if st.button(
                "🔓 Change Episode",
                key="unlock_episode_selector",
                use_container_width=True,
            ):
                st.session_state["episode_selector_unlocked"] = True
                st.rerun()

            return current

        # Show unlocked state - full selector
        st.session_state["episode_selector_unlocked"] = True

        tracked_tab, s3_tab = st.tabs(["Tracked", "Browse S3"])

        with tracked_tab:
            payload = _safe_api_get("/episodes")
            options = payload.get("episodes", []) if payload else []
            if options:
                ep_ids = [item["ep_id"] for item in options]
                default_idx = ep_ids.index(current) if current in ep_ids else 0

                selection = st.selectbox(
                    "Select episode",
                    ep_ids,
                    format_func=lambda eid: f"{eid} ({options[ep_ids.index(eid)]['show_slug']})",
                    index=default_idx if ep_ids else 0,
                    key="facebank_tracked_select",
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "✓ Load",
                        key="facebank_load_tracked",
                        use_container_width=True,
                        type="primary",
                    ):
                        st.session_state["episode_selector_unlocked"] = False
                        helpers.set_ep_id(selection)
                with col2:
                    if st.button(
                        "✗ Cancel",
                        key="cancel_episode_change",
                        use_container_width=True,
                    ):
                        st.session_state["episode_selector_unlocked"] = False
                        st.rerun()
            else:
                st.info("No tracked episodes yet.")

        with s3_tab:
            s3_payload = _safe_api_get("/episodes/s3_videos")
            items = s3_payload.get("items", []) if s3_payload else []
            if items:
                labels = [
                    f"{item['ep_id']} · {item.get('last_modified') or 'unknown'}"
                    for item in items
                ]
                idx = st.selectbox(
                    "S3 videos",
                    list(range(len(items))),
                    format_func=lambda i: labels[i],
                    key="facebank_s3_select",
                )
                selected = items[idx]
                if st.button(
                    "Track & Load", key="facebank_track_s3", use_container_width=True
                ):
                    st.session_state["episode_selector_unlocked"] = False
                    _track_episode_from_s3(selected)
            else:
                st.info("No S3 videos exposed by the API.")

        ep_id = helpers.get_ep_id()
        if not ep_id:
            st.warning("Choose an episode to continue.")
            st.stop()

        return ep_id


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
    if st.button(
        "Save name", key=f"{prefix}_save_{identity['identity_id']}", disabled=disabled
    ):
        _save_identity_name(ep_id, identity["identity_id"], resolved, show_slug)


def _initialize_state(ep_id: str) -> None:
    if st.session_state.get("facebank_ep") != ep_id:
        st.session_state["facebank_ep"] = ep_id
        st.session_state["facebank_view"] = "people"
        st.session_state["selected_person"] = None
        st.session_state["selected_identity"] = None
        st.session_state["selected_track"] = None


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
        if st.button("Cluster Cleanup", key="facebank_cleanup_button"):
            payload = helpers.default_cleanup_payload(ep_id)
            with st.spinner("Running cleanup…"):
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
                    details.append(
                        f"tracks {helpers.format_count(tb) or tb} → {helpers.format_count(ta) or ta}"
                    )
                if cbefore is not None and cafter is not None:
                    details.append(
                        f"clusters {helpers.format_count(cbefore) or cbefore} → {helpers.format_count(cafter) or cafter}"
                    )
    with action_cols[2]:
        if st.button(
            "🔄 Refresh Values",
            key="facebank_refresh_similarity_button",
            help="Recompute all similarity scores",
        ):
            with st.spinner("Refreshing similarity values..."):
                # Trigger similarity index refresh for all identities
                refresh_resp = _api_post(f"/episodes/{ep_id}/refresh_similarity", {})
                if refresh_resp and refresh_resp.get("status") == "success":
                    st.success("✅ Refreshed similarity values!")
                    st.rerun()
                else:
                    st.error("Failed to refresh similarity values. Check logs.")
    return detail


def _episode_people(ep_id: str) -> tuple[str | None, List[Dict[str, Any]]]:
    meta = helpers.parse_ep_id(ep_id)
    if not meta:
        return None, []
    show_slug = str(meta.get("show") or "").lower()
    if not show_slug:
        return None, []
    people_resp = _safe_api_get(f"/shows/{show_slug}/people")
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
    limit: int = 2,
    cursor: str | None = None,
) -> tuple[List[Dict[str, Any]], str | None]:
    params: Dict[str, Any] = {"sample": int(sample), "limit": int(limit)}
    if cursor:
        # The backend returns pagination with a 'next_start_after' cursor
        params["start_after"] = cursor
    payload = (
        _safe_api_get(f"/episodes/{ep_id}/tracks/{track_id}/crops", params=params) or {}
    )
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
    payload = _safe_api_get(
        f"/episodes/{ep_id}/tracks/{track_id}/frames", params=params
    )
    if isinstance(payload, list):
        return {"items": payload}
    if isinstance(payload, dict):
        return payload
    return {}


def _render_cast_carousel(
    ep_id: str,
    show_id: str,
) -> None:
    """Render featured cast members carousel at the top - ONLY shows cast with clusters in this episode."""
    # Fetch cast members from Cast API
    cast_api_resp = _safe_api_get(f"/shows/{show_id}/cast")
    if not cast_api_resp:
        return

    cast_members = cast_api_resp.get("cast", [])
    if not cast_members:
        return

    # Get people data to check who has clusters
    people_resp = _safe_api_get(f"/shows/{show_id}/people")
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

    st.markdown("### 🎬 Cast Lineup")
    st.caption("Cast members with clusters in this episode")

    # Create horizontal carousel (max 5 per row)
    cols_per_row = min(len(cast_with_clusters), 5)

    for row_start in range(0, len(cast_with_clusters), cols_per_row):
        row_items = cast_with_clusters[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for idx, (cast, person, episode_clusters) in enumerate(row_items):
            with cols[idx]:
                cast_id = cast.get("cast_id")
                name = cast.get("name", "(unnamed)")

                # Get facebank featured image
                facebank_resp = _safe_api_get(
                    f"/cast/{cast_id}/facebank?show_id={show_id}"
                )
                featured_url = None
                if facebank_resp and facebank_resp.get("featured_seed"):
                    featured_seed = facebank_resp["featured_seed"]
                    featured_url = featured_seed.get("display_url")

                # Display featured image
                if featured_url:
                    thumb_markup = helpers.thumb_html(
                        featured_url, alt=name, hide_if_missing=False
                    )
                    st.markdown(thumb_markup, unsafe_allow_html=True)
                else:
                    st.markdown("_No featured image_")

                # Name
                st.markdown(f"**{name}**")

                # Show cluster count (always > 0 due to filtering)
                cluster_count = len(episode_clusters)
                st.caption(
                    f"✓ {cluster_count} cluster{'s' if cluster_count != 1 else ''}"
                )

                # View detections button
                if st.button(
                    "View", key=f"carousel_view_{cast_id}", use_container_width=True
                ):
                    st.session_state["filter_cast_id"] = cast_id
                    st.session_state["filter_cast_name"] = name
                    st.rerun()

    st.markdown("---")


def _render_cast_gallery(
    ep_id: str,
    cast_cards: List[Dict[str, Any]],
) -> None:
    """Render cast members as a horizontal scrollable gallery."""
    if not cast_cards:
        return

    cols_per_row = min(len(cast_cards), 5) or 1

    for row_start in range(0, len(cast_cards), cols_per_row):
        row_members = cast_cards[row_start : row_start + cols_per_row]
        cols = st.columns(len(row_members))

        for idx, card in enumerate(row_members):
            with cols[idx]:
                cast_info = card.get("cast") or {}
                person = card.get("person") or {}
                cast_id = cast_info.get("cast_id") or person.get("cast_id")
                person_id = person.get("person_id")
                episode_clusters = card.get("episode_clusters", [])

                name = cast_info.get("name") or person.get("name") or "(unnamed)"
                aliases = cast_info.get("aliases") or person.get("aliases") or []

                featured_source = card.get("featured_thumbnail") or person.get(
                    "rep_crop"
                )
                featured_thumb = helpers.resolve_thumb(featured_source)
                if featured_thumb:
                    thumb_markup = helpers.thumb_html(
                        featured_thumb, alt=name, hide_if_missing=False
                    )
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
                    clusters_summary = _safe_api_get(
                        f"/episodes/{ep_id}/people/{person_id}/clusters_summary"
                    )
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
) -> None:
    """Render an unassigned cluster card with suggestion and assignment UI."""
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
        people_resp = _safe_api_get(f"/shows/{show_id}/people")
        if people_resp:
            people = people_resp.get("people", [])
            suggested_person = next(
                (p for p in people if p.get("person_id") == suggested_person_id), None
            )
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
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"### 🔍 Cluster `{cluster_id}`")
            st.caption(f"{tracks_count} track(s) · {faces_count} face(s)")
        with col2:
            # View cluster button
            if st.button("View", key=f"view_unassigned_{cluster_id}"):
                _set_view("cluster_tracks", identity_id=cluster_id)
                st.rerun()
        with col3:
            # Delete cluster button
            if st.button(
                "Delete", key=f"delete_unassigned_{cluster_id}", type="secondary"
            ):
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
                    if st.button(
                        "◀", key=f"prev_{cluster_id}", disabled=current_page == 0
                    ):
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
                        st.session_state[page_key] = min(
                            total_pages - 1, current_page + 1
                        )
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
                                start_idx = thumb_url.find("s3.amazonaws.com/") + len(
                                    "s3.amazonaws.com/"
                                )
                                end_idx = (
                                    thumb_url.find("?")
                                    if "?" in thumb_url
                                    else len(thumb_url)
                                )
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

        # Show suggestion if available
        if suggested_cast_id and suggested_cast_name:
            similarity_pct = (
                int((1 - suggested_distance) * 100)
                if suggested_distance is not None
                else 0
            )
            sugg_col1, sugg_col2 = st.columns([5, 1])
            with sugg_col1:
                st.info(
                    f"✨ Suggested: **{suggested_cast_name}** ({similarity_pct}% similarity)"
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
                            people_resp = _safe_api_get(f"/shows/{show_id}/people")
                            people = (
                                people_resp.get("people", []) if people_resp else []
                            )
                            target_person = next(
                                (
                                    p
                                    for p in people
                                    if p.get("cast_id") == selected_cast_id
                                ),
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
                            resp = _api_post(
                                f"/episodes/{ep_id}/clusters/group", payload
                            )
                            if resp and resp.get("status") == "success":
                                st.success(
                                    f"Assigned cluster to {cast_options[selected_cast_id]}!"
                                )
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
) -> None:
    """Render an auto-detected person card with detailed clusters and bulk assignment."""
    person_id = str(person.get("person_id") or "")
    name = person.get("name") or "(unnamed)"
    aliases = person.get("aliases") or []
    total_clusters = len(person.get("cluster_ids", []) or [])

    # Get suggested cast member if available
    suggested_person_id = None
    suggested_distance = None
    if episode_clusters:
        # Fetch suggestions from API
        suggestions_resp = _safe_api_get(f"/episodes/{ep_id}/cluster_suggestions")
        if suggestions_resp:
            suggestions = suggestions_resp.get("suggestions", [])
            # Find suggestion for this person's first cluster
            first_cluster = (
                episode_clusters[0].split(":")[-1] if episode_clusters else None
            )
            for suggestion in suggestions:
                if suggestion.get("cluster_id") == first_cluster:
                    suggested_person_id = suggestion.get("suggested_person_id")
                    suggested_distance = suggestion.get("distance")
                    break

    with st.container(border=True):
        # Name
        st.markdown(f"### 👤 {name}")

        # Show aliases if present
        if aliases:
            alias_text = ", ".join(f"`{a}`" for a in aliases)
            st.caption(f"Aliases: {alias_text}")

        # Metrics
        st.caption(
            f"ID: {person_id} · {total_clusters} cluster(s) overall · "
            f"{len(episode_clusters)} in this episode"
        )

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
                        people_resp = _safe_api_get(f"/shows/{show_id}/people")
                        if people_resp:
                            people = people_resp.get("people", [])
                            suggested_person = next(
                                (
                                    p
                                    for p in people
                                    if p.get("person_id") == suggested_person_id
                                ),
                                None,
                            )
                            if suggested_person:
                                suggested_cast_id = suggested_person.get("cast_id")

                    # Determine default index
                    default_index = 0
                    if suggested_cast_id and suggested_cast_id in cast_options:
                        default_index = list(cast_options.keys()).index(
                            suggested_cast_id
                        )

                    # Use a form to ensure selectbox and button states are synchronized
                    with st.form(key=f"assign_form_{person_id}"):
                        selected_cast_id = st.selectbox(
                            "Select cast member",
                            options=list(cast_options.keys()),
                            format_func=lambda pid: cast_options[pid],
                            index=default_index,
                            key=f"cast_select_{person_id}",
                        )

                        # Show suggestion info if available
                        if (
                            suggested_cast_id
                            and suggested_cast_id == selected_cast_id
                            and suggested_distance is not None
                        ):
                            similarity_pct = int((1 - suggested_distance) * 100)
                            st.caption(
                                f"✨ Suggested match ({similarity_pct}% similarity)"
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
                    st.info(
                        "No cast members available. Create one first in the Cast page."
                    )
            else:
                new_name = st.text_input(
                    "New person name",
                    key=f"new_name_{person_id}",
                    placeholder="Enter name...",
                )
                if new_name and st.button(
                    "Create & Assign", key=f"create_assign_btn_{person_id}"
                ):
                    # Assign all clusters with this name
                    result = _bulk_assign_to_new_person(
                        ep_id, show_id, person_id, new_name, episode_clusters
                    )
                    if result:
                        st.success(
                            f"Created '{new_name}' and assigned {len(episode_clusters)} clusters"
                        )
                        st.rerun()

        # Fetch clusters summary to show thumbnails
        clusters_summary = _safe_api_get(
            f"/episodes/{ep_id}/people/{person_id}/clusters_summary"
        )
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
                        cohesion_badge = (
                            _render_similarity_badge(cohesion, metric="cluster")
                            if cohesion
                            else ""
                        )
                        st.markdown(
                            f"**{cluster_id}** {cohesion_badge}", unsafe_allow_html=True
                        )
                        st.caption(
                            f"{cluster.get('tracks', 0)} tracks · {cluster.get('faces', 0)} faces"
                        )

                        # View and Delete cluster buttons
                        btn_cols = st.columns([1, 1])
                        with btn_cols[0]:
                            if st.button(
                                "View", key=f"view_cluster_{person_id}_{cluster_id}"
                            ):
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
                                resp = _api_delete(
                                    f"/episodes/{ep_id}/identities/{cluster_id}"
                                )
                                if resp:
                                    st.success(f"Deleted cluster {cluster_id}")
                                    st.rerun()
                                else:
                                    st.error("Failed to delete cluster")
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
                st.caption(
                    "The clusters will remain in identities.json and can be re-assigned later."
                )

                if st.button(
                    f"Delete {name}", key=f"delete_person_{person_id}", type="secondary"
                ):
                    try:
                        resp = helpers.api_delete(
                            f"/shows/{show_id}/people/{person_id}"
                        )
                        st.success(f"Deleted {name} ({person_id})")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to delete person: {exc}")


def _bulk_assign_clusters(
    ep_id: str,
    show_id: str,
    source_person_id: str,
    target_cast_id: str,
    cluster_ids: List[str],
) -> bool:
    """Assign all clusters from source person to a cast member."""
    try:
        # Find or create a person record for this cast_id via API
        people_resp = _safe_api_get(f"/shows/{show_id}/people")
        people = people_resp.get("people", []) if people_resp else []
        target_person = next(
            (p for p in people if p.get("cast_id") == target_cast_id), None
        )

        if not target_person:
            # Fetch cast member details to get the name
            cast_resp = _safe_api_get(f"/shows/{show_id}/cast")
            cast_members = cast_resp.get("cast", []) if cast_resp else []
            cast_member = next(
                (cm for cm in cast_members if cm.get("cast_id") == target_cast_id), None
            )

            if not cast_member:
                st.error(f"Cast member {target_cast_id} not found")
                return False

            # Create a new person record linked to this cast member via API
            create_payload = {
                "name": cast_member.get("name"),
                "cast_id": target_cast_id,
                "aliases": cast_member.get("aliases", []),
            }
            target_person = _api_post(f"/shows/{show_id}/people", create_payload)
            if not target_person:
                st.error("Failed to create person record")
                return False

        # Merge source person into target person via API
        merge_payload = {
            "source_person_id": source_person_id,
            "target_person_id": target_person["person_id"],
        }
        result = _api_post(f"/shows/{show_id}/people/merge", merge_payload)
        return result is not None
    except Exception as exc:
        st.error(f"Failed to assign clusters: {exc}")
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
        people_resp = _safe_api_get(f"/shows/{show_id}/people")
        people = people_resp.get("people", []) if people_resp else []

        # Simple name matching (case-insensitive)
        existing = next(
            (p for p in people if p.get("name", "").lower() == new_name.lower()), None
        )

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

    cast_params: Dict[str, Any] = {"include_featured": "1"}
    if season_label:
        cast_params["season"] = season_label
    cast_api_resp = _safe_api_get(f"/shows/{show_id}/cast", params=cast_params)
    raw_cast_entries = cast_api_resp.get("cast", []) if cast_api_resp else []
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
        st.info(
            "No people found for this show. Run 'Group Clusters (auto)' to create people."
        )
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
    episode_auto_people: List[tuple[Dict[str, Any], List[str]]] = []
    for person in people:
        if person.get("cast_id"):
            continue
        if filter_cast_id and str(person.get("person_id") or "") != str(filter_cast_id):
            continue
        episode_clusters = _episode_cluster_ids(person, ep_id)
        if episode_clusters:
            episode_auto_people.append((person, episode_clusters))

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
    cast_gallery_cards.sort(
        key=lambda card: (card.get("cast", {}).get("name") or "").lower()
    )
    # Sort episode auto-people: named first, then by name
    episode_auto_people.sort(
        key=lambda x: (
            x[0].get("name") is None or x[0].get("name") == "",
            (x[0].get("name") or "").lower(),
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
        st.markdown(
            f"### 👥 Episode Auto-Clustered People ({len(episode_auto_people)})"
        )
        st.caption(f"People auto-detected in episode {ep_id}")

        # Build options: map cast_id to name
        cast_options = {
            cm.get("cast_id"): cm.get("name")
            for cm in deduped_cast_entries
            if cm.get("cast_id") and cm.get("name")
        }

        for person, episode_clusters in episode_auto_people:
            _render_auto_person_card(
                ep_id, show_id, person, episode_clusters, cast_options
            )

    # --- UNASSIGNED CLUSTERS (SUGGESTIONS) SECTION ---
    if unassigned_clusters:
        st.markdown("---")
        header_col1, header_col2, header_col3 = st.columns([3, 1, 1])
        with header_col1:
            st.markdown(
                f"### 🔍 Unassigned Clusters - Review Suggestions ({len(unassigned_clusters)})"
            )
            st.caption(
                "Clusters detected but not yet assigned to cast members. Review and assign manually."
            )
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
                        st.success(
                            f"✅ Saved {save_resp.get('saved_count', 0)} assignment(s)!"
                        )
                    else:
                        st.error("Failed to save progress. Check logs.")
        with header_col3:
            if st.button("🔄 Refresh Suggestions", key=f"refresh_suggestions_{ep_id}"):
                with st.spinner("Refreshing suggestions..."):
                    # Use the new endpoint that compares against assigned clusters
                    suggestions_resp = _safe_api_get(
                        f"/episodes/{ep_id}/cluster_suggestions_from_assigned"
                    )
                    if suggestions_resp:
                        st.success("Suggestions refreshed!")
                    st.rerun()

        # Build options: map cast_id to name
        cast_options = {
            cm.get("cast_id"): cm.get("name")
            for cm in deduped_cast_entries
            if cm.get("cast_id") and cm.get("name")
        }

        # Fetch suggestions from API - now comparing against assigned clusters in this episode
        suggestions_resp = _safe_api_get(
            f"/episodes/{ep_id}/cluster_suggestions_from_assigned"
        )
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

        # Sort unassigned clusters by face count (descending - most faces first)
        sorted_clusters = sorted(
            unassigned_clusters,
            key=lambda cid: cluster_lookup.get(cid, {})
            .get("counts", {})
            .get("faces", 0),
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
            )

    # Show message if filtering but nothing found
    if filter_cast_id and not cast_gallery_cards and not episode_auto_people:
        st.warning(
            f"{filter_cast_name or filter_cast_id} has no clusters in episode {ep_id}."
        )

    # Show message if no people at all
    if (
        not cast_gallery_cards
        and not episode_auto_people
        and not unassigned_clusters
        and not filter_cast_id
    ):
        st.info(
            "No people with clusters in this episode yet. Run 'Group Clusters (auto)' to create people."
        )


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
    clusters_summary = _safe_api_get(
        f"/episodes/{ep_id}/people/{person_id}/clusters_summary"
    )
    if not clusters_summary:
        st.error("Failed to load cluster data.")
        return

    # Collect all tracks from all clusters
    all_tracks = []
    total_tracks = 0
    total_faces = 0
    all_frame_embeddings = (
        []
    )  # Collect all frame embeddings for person-level similarity

    for cluster_data in clusters_summary.get("clusters", []):
        cluster_id = cluster_data["cluster_id"]
        total_tracks += cluster_data.get("tracks", 0)
        total_faces += cluster_data.get("faces", 0)

        # Fetch full track data with frames for this cluster
        track_reps_data = _safe_api_get(
            f"/episodes/{ep_id}/clusters/{cluster_id}/track_reps"
        )
        if track_reps_data:
            for track in track_reps_data.get("tracks", []):
                track["cluster_id"] = cluster_id  # Tag with source cluster
                all_tracks.append(track)

    # Load all face embeddings for this person to compute person-level similarity
    faces_data = _safe_api_get(f"/episodes/{ep_id}/identities")
    if faces_data:
        # Get all cluster IDs for this person
        person_cluster_ids = set(episode_clusters)

        # Load faces.jsonl to get embeddings

        try:
            # We'll compute average embedding across all frames for this person
            # and use that as the person prototype for frame similarity scoring
            pass  # Will compute during frame rendering
        except Exception:
            pass

    st.caption(
        f"{len(episode_clusters)} clusters · {total_tracks} tracks · {total_faces} faces"
    )

    if not all_tracks:
        st.info("No tracks found for this person.")
        return

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
        all_tracks.sort(
            key=lambda t: (
                t.get("similarity") if t.get("similarity") is not None else 999.0
            )
        )
    elif sort_option == "Track Similarity (High to Low)":
        all_tracks.sort(
            key=lambda t: (
                t.get("similarity") if t.get("similarity") is not None else -999.0
            ),
            reverse=True,
        )
    elif sort_option == "Frame Count (Low to High)":
        # Need to fetch frame count for each track
        for track in all_tracks:
            track_id_str = track.get("track_id", "")
            if isinstance(track_id_str, str) and track_id_str.startswith("track_"):
                track_num = track_id_str.replace("track_", "")
                try:
                    track_id_int = int(track_num)
                    track_data = _safe_api_get(
                        f"/episodes/{ep_id}/tracks/{track_id_int}"
                    )
                    track["frame_count"] = (
                        track_data.get("faces_count", 0) if track_data else 0
                    )
                except (TypeError, ValueError):
                    track["frame_count"] = 0
            else:
                track["frame_count"] = 0
        all_tracks.sort(key=lambda t: t.get("frame_count", 0))
    elif sort_option == "Frame Count (High to Low)":
        # Need to fetch frame count for each track
        for track in all_tracks:
            track_id_str = track.get("track_id", "")
            if isinstance(track_id_str, str) and track_id_str.startswith("track_"):
                track_num = track_id_str.replace("track_", "")
                try:
                    track_id_int = int(track_num)
                    track_data = _safe_api_get(
                        f"/episodes/{ep_id}/tracks/{track_id_int}"
                    )
                    track["frame_count"] = (
                        track_data.get("faces_count", 0) if track_data else 0
                    )
                except (TypeError, ValueError):
                    track["frame_count"] = 0
            else:
                track["frame_count"] = 0
        all_tracks.sort(key=lambda t: t.get("frame_count", 0), reverse=True)
    elif sort_option == "Average Frame Similarity (Low to High)":
        # Calculate average frame similarity for each track
        for track in all_tracks:
            track_id_str = track.get("track_id", "")
            if isinstance(track_id_str, str) and track_id_str.startswith("track_"):
                track_num = track_id_str.replace("track_", "")
                try:
                    track_id_int = int(track_num)
                    track_data = _safe_api_get(
                        f"/episodes/{ep_id}/tracks/{track_id_int}"
                    )
                    frames = track_data.get("frames", []) if track_data else []
                    similarities = [
                        f.get("similarity")
                        for f in frames
                        if f.get("similarity") is not None
                    ]
                    track["avg_frame_similarity"] = (
                        sum(similarities) / len(similarities) if similarities else 999.0
                    )
                except (TypeError, ValueError):
                    track["avg_frame_similarity"] = 999.0
            else:
                track["avg_frame_similarity"] = 999.0
        all_tracks.sort(key=lambda t: t.get("avg_frame_similarity", 999.0))
    elif sort_option == "Average Frame Similarity (High to Low)":
        # Calculate average frame similarity for each track
        for track in all_tracks:
            track_id_str = track.get("track_id", "")
            if isinstance(track_id_str, str) and track_id_str.startswith("track_"):
                track_num = track_id_str.replace("track_", "")
                try:
                    track_id_int = int(track_num)
                    track_data = _safe_api_get(
                        f"/episodes/{ep_id}/tracks/{track_id_int}"
                    )
                    frames = track_data.get("frames", []) if track_data else []
                    similarities = [
                        f.get("similarity")
                        for f in frames
                        if f.get("similarity") is not None
                    ]
                    track["avg_frame_similarity"] = (
                        sum(similarities) / len(similarities)
                        if similarities
                        else -999.0
                    )
                except (TypeError, ValueError):
                    track["avg_frame_similarity"] = -999.0
            else:
                track["avg_frame_similarity"] = -999.0
        all_tracks.sort(
            key=lambda t: t.get("avg_frame_similarity", -999.0), reverse=True
        )
    elif sort_option == "Track ID (Low to High)":
        all_tracks.sort(
            key=lambda t: (
                int(t.get("track_id", "0").replace("track_", ""))
                if isinstance(t.get("track_id"), str)
                else 0
            )
        )
    elif sort_option == "Track ID (High to Low)":
        all_tracks.sort(
            key=lambda t: (
                int(t.get("track_id", "0").replace("track_", ""))
                if isinstance(t.get("track_id"), str)
                else 0
            ),
            reverse=True,
        )

    st.markdown(f"**All {len(all_tracks)} Tracks**")

    # Render each track as one row showing up to 6 frames
    for track in all_tracks:
        track_id_str = track.get("track_id", "")
        similarity = track.get("similarity")
        cluster_id = track.get("cluster_id", "unknown")

        # Parse track ID
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

        # Fetch all frames for this track
        track_data = (
            _safe_api_get(f"/episodes/{ep_id}/tracks/{track_id_int}")
            if track_id_int
            else None
        )
        frames = track_data.get("frames", []) if track_data else []

        # Sort frames by similarity (lowest first)
        frames_sorted = sorted(
            frames,
            key=lambda f: (
                f.get("similarity") if f.get("similarity") is not None else 999.0
            ),
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
                        thumb_markup = helpers.thumb_html(
                            resolved, alt=f"Frame {frame_idx}", hide_if_missing=False
                        )
                        st.markdown(thumb_markup, unsafe_allow_html=True)

                        # Show frame similarity badge
                        frame_badge = _render_similarity_badge(frame_sim)
                        st.caption(
                            f"F{frame_idx} {frame_badge}", unsafe_allow_html=True
                        )

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
                    identity_index = {
                        ident["identity_id"]: ident
                        for ident in identities_payload.get("identities", [])
                    }
                    move_targets = [
                        ident_id
                        for ident_id in identity_index
                        if ident_id != cluster_id
                    ]
                    if move_targets and st.button(
                        "Move track", key=f"move_track_{person_id}_{track_id_int}"
                    ):
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
    track_reps_data = _safe_api_get(
        f"/episodes/{ep_id}/clusters/{identity_id}/track_reps"
    )
    if not track_reps_data:
        st.error("Failed to load track representatives.")
        return

    identity_meta = identity_index.get(identity_id, {})
    display_name = identity_meta.get("name")
    label = identity_meta.get("label")

    tracks_count = track_reps_data.get("total_tracks", 0)
    cohesion = track_reps_data.get("cohesion")
    faces_count = identity_meta.get("size") or 0
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
                st.info(
                    "✅ Requirements: Detection score ≥0.75 · Sharpness ≥15 · Similarity ≥0.70"
                )
            with col2:
                if st.button("💾 Export Seeds", key=f"export_seeds_{identity_id}", use_container_width=True, type="primary"):
                    with st.spinner(f"Selecting and exporting seeds for {display_name or person_id_for_export}..."):
                        export_resp = _api_post(
                            f"/episodes/{ep_id}/identities/{identity_id}/export_seeds",
                            {}
                        )
                        if export_resp and export_resp.get("status") == "success":
                            seeds_count = export_resp.get("seeds_exported", 0)
                            seeds_path = export_resp.get("seeds_path", "")
                            st.success(
                                f"✅ Exported {seeds_count} high-quality seeds to facebank!\n\n"
                                f"Path: `{seeds_path}`"
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
    move_targets = [ident_id for ident_id in identity_index if ident_id != identity_id]

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
                thumb_markup = helpers.thumb_html(
                    resolved, alt=f"Track {track_num}", hide_if_missing=False
                )
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

                    if move_targets:
                        target_choice = st.selectbox(
                            "Move to",
                            move_targets,
                            key=f"cluster_move_select_{identity_id}_{track_id_int}",
                            format_func=lambda val: f"{val}",
                        )
                        if st.button(
                            "Move",
                            key=f"cluster_move_btn_{identity_id}_{track_id_int}",
                        ):
                            _move_track(ep_id, track_id_int, target_choice)

                    if st.button(
                        "Delete",
                        key=f"cluster_delete_btn_{identity_id}_{track_id_int}",
                        type="secondary",
                    ):
                        _delete_track(ep_id, track_id_int)


def _render_track_view(
    ep_id: str, track_id: int, identities_payload: Dict[str, Any]
) -> None:
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
    if sample_key not in st.session_state:
        st.session_state[sample_key] = 1
    frame_controls = st.columns([1, 1, 2])
    with frame_controls[0]:
        prev_sample = st.session_state[sample_key]
        sample = int(
            st.slider(
                "Sample every N crops",
                min_value=1,
                max_value=20,
                value=prev_sample,
                key=sample_key,
            )
        )
    page_key = f"track_page_{ep_id}_{track_id}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    if sample != prev_sample:
        st.session_state[page_key] = 1
    current_page = int(st.session_state[page_key])
    with frame_controls[1]:
        page = int(
            st.number_input(
                "Page",
                min_value=1,
                value=current_page,
                step=1,
                key=page_key,
            )
        )
    page = int(st.session_state[page_key])
    page_size = 50
    frames_payload = _fetch_track_frames(
        ep_id, track_id, sample=sample, page=page, page_size=page_size
    )
    frames = frames_payload.get("items", [])
    best_frame_idx = frames_payload.get("best_frame_idx")
    total_sampled = int(frames_payload.get("total") or 0)
    # Preserve zero: only use total_sampled if total_frames is None
    total_frames_raw = frames_payload.get("total_frames")
    total_frames = (
        int(total_frames_raw) if total_frames_raw is not None else total_sampled
    )
    max_page = max(1, math.ceil(total_sampled / page_size)) if total_sampled else 1

    # Reorder frames to show best-quality frame first (if present in current page)
    if best_frame_idx is not None and frames:
        best_frame = None
        best_frame_position = None
        for i, frame in enumerate(frames):
            if frame.get("frame_idx") == best_frame_idx:
                best_frame = frame
                best_frame_position = i
                break

        if (
            best_frame is not None
            and best_frame_position is not None
            and best_frame_position > 0
        ):
            # Move best frame to the front
            frames = (
                [best_frame]
                + frames[:best_frame_position]
                + frames[best_frame_position + 1 :]
            )
    nav_cols = st.columns([1, 1, 3])
    with nav_cols[0]:
        if st.button("Prev page", key=f"track_prev_{track_id}", disabled=page <= 1):
            st.session_state[page_key] = max(1, page - 1)
            st.rerun()
    with nav_cols[1]:
        if st.button(
            "Next page", key=f"track_next_{track_id}", disabled=page >= max_page
        ):
            st.session_state[page_key] = min(max_page, page + 1)
            st.rerun()
    shown = len(frames)
    summary = f"Frames shown: {shown} / {total_sampled or 0} (page {page}/{max_page}) · Sample every {sample}"
    if total_frames:
        summary += f" · Faces tracked: {total_frames}"
    nav_cols[2].caption(summary)
    if current_identity:
        assign_container = st.container(border=True)
        with assign_container:
            st.markdown(f"**Assign Track {track_id} to Cast Name**")
            identity_meta = identity_lookup.get(current_identity, {})
            current_name = identity_meta.get("name") or ""
            if not show_slug:
                st.info("Show slug missing; unable to assign roster names.")
            else:
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
                    _assign_track_name(ep_id, track_id, resolved, show_slug)
    integrity = _safe_api_get(f"/episodes/{ep_id}/tracks/{track_id}/integrity")
    if integrity and not integrity.get("ok"):
        st.warning(
            "Crops on disk are missing for this track. Faces manifest"
            f"={integrity.get('faces_manifest', 0)} · crops={integrity.get('crops_files', 0)}"
        )
    action_cols = st.columns([1.0, 1.0, 1.0])
    with action_cols[0]:
        targets = [
            ident["identity_id"]
            for ident in identities
            if ident["identity_id"] != current_identity
        ]
        if targets:
            move_select_key = (
                f"track_view_move_{ep_id}_{track_id}_{current_identity or 'none'}"
            )
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

    selection_store: Dict[int, set[int]] = st.session_state.setdefault(
        "track_frame_selection", {}
    )
    track_selection = selection_store.setdefault(track_id, set())
    selected_frames: List[int] = []
    if frames:
        st.caption("Select frames to move or delete.")
        cols_per_row = 6
        for row_start in range(0, len(frames), cols_per_row):
            row_frames = frames[row_start : row_start + cols_per_row]
            row_cols = st.columns(len(row_frames))
            for idx, frame_meta in enumerate(row_frames):
                face_id = frame_meta.get("face_id")
                frame_idx = frame_meta.get("frame_idx")
                try:
                    frame_idx_int = int(frame_idx)
                except (TypeError, ValueError):
                    frame_idx_int = None
                thumb_url = frame_meta.get("media_url") or frame_meta.get(
                    "thumbnail_url"
                )
                skip_reason = frame_meta.get("skip")
                with row_cols[idx]:
                    caption = (
                        f"Frame {frame_idx}"
                        if frame_idx is not None
                        else (face_id or "frame")
                    )
                    resolved_thumb = helpers.resolve_thumb(thumb_url)
                    thumb_markup = helpers.thumb_html(
                        resolved_thumb, alt=caption, hide_if_missing=True
                    )
                    if thumb_markup:
                        st.markdown(thumb_markup, unsafe_allow_html=True)
                    else:
                        st.caption("Crop unavailable.")
                    st.caption(caption)
                    # Show best-quality frame badge
                    if frame_idx == best_frame_idx:
                        st.markdown(
                            '<span style="background-color: #4CAF50; color: white; padding: 2px 6px; '
                            'border-radius: 3px; font-size: 0.8em; font-weight: bold;">★ BEST QUALITY</span>',
                            unsafe_allow_html=True,
                        )
                    # Show similarity badge if available
                    similarity = frame_meta.get("similarity")
                    if similarity is not None:
                        similarity_badge = _render_similarity_badge(similarity)
                        st.markdown(similarity_badge, unsafe_allow_html=True)
                        # Show quality score if available
                        quality = frame_meta.get("quality")
                        if quality and isinstance(quality, dict):
                            quality_score = quality.get("score")
                            if quality_score is not None:
                                quality_pct = int(quality_score * 100)
                                if quality_score >= 0.85:
                                    quality_color = "#2E7D32"
                                elif quality_score >= 0.60:
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
            st.info(
                "No frames on this page. Try a smaller page number or lower sampling."
            )
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
        if action_cols[1].button(
            "Delete selected", key=f"track_delete_selected_{track_id}", type="secondary"
        ):
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
    if _api_post(endpoint, payload):
        st.success("Track updated.")
        st.rerun()


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


def _delete_frame(
    ep_id: str, track_id: int, frame_idx: int, delete_assets: bool
) -> None:
    payload = {
        "track_id": track_id,
        "frame_idx": frame_idx,
        "delete_assets": delete_assets,
    }
    if _api_post(f"/identities/{ep_id}/drop_frame", payload):
        # Stay on track view - don't navigate away
        st.success("Frame removed.")
        st.rerun()


ep_id = _select_episode()
_initialize_state(ep_id)
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
    key="faces_review_grouping_strategy",
)
selected_strategy = strategy_labels[selected_label]
facebank_ready = True
if selected_strategy == "facebank":
    confirm = st.text_input(
        "Confirm show slug before regrouping",
        value=show_slug or "",
        key="facebank_show_confirm",
        help="Protect against grouping the wrong show",
    ).strip()
    expected_slug = (show_slug or "").lower()
    facebank_ready = bool(expected_slug and confirm.lower() == expected_slug)
    if not facebank_ready:
        st.info(f"Enter '{show_slug}' to enable facebank regrouping.")

button_label = (
    "Group Clusters (facebank)"
    if selected_strategy == "facebank"
    else "Group Clusters (auto)"
)
caption_text = (
    "Auto-clusters within episode and computes similarity suggestions (no auto-assignment)"
    if selected_strategy == "auto"
    else "Uses existing facebank seeds to align clusters to known cast members"
)
if st.button(
    button_label,
    key="group_clusters_action",
    type="primary",
    disabled=(selected_strategy == "facebank" and not facebank_ready),
):
    payload = {"strategy": selected_strategy}

    if selected_strategy == "auto":
        # Show detailed progress for auto clustering
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        log_expander = st.expander("📋 Detailed Progress Log", expanded=False)

        status_text.text("🚀 Starting auto-clustering...")
        result = None
        error = None

        # Collect log messages
        log_messages = []

        try:
            result = _api_post(f"/episodes/{ep_id}/clusters/group", payload)
        except Exception as exc:
            error = exc

        if error:
            progress_bar.empty()
            status_text.error(f"❌ Clustering failed: {error}")
        elif result:
            # Show progress log if available
            progress_log = result.get("progress_log", [])
            if progress_log:
                for entry in progress_log:
                    progress = entry.get("progress", 0.0)
                    message = entry.get("message", "")
                    step = entry.get("step", "")
                    progress_bar.progress(progress)
                    status_text.text(message)
                    log_messages.append(f"[{int(progress*100)}%] {step}: {message}")

            progress_bar.progress(1.0)

            # Show detailed summary - check both possible locations for log
            # API returns result -> log, but also could be at top level
            log_data = result.get("result", {}).get("log", {}) or result.get("log", {})
            steps = log_data.get("steps", [])
            cleared = next(
                (
                    s.get("cleared_count", 0)
                    for s in steps
                    if s.get("step") == "clear_assignments"
                ),
                0,
            )
            centroids = next(
                (
                    s.get("centroids_count", 0)
                    for s in steps
                    if s.get("step") == "compute_centroids"
                ),
                0,
            )
            merged = next(
                (
                    s.get("merged_count", 0)
                    for s in steps
                    if s.get("step") == "group_within_episode"
                ),
                0,
            )
            suggestions = next(
                (
                    s.get("suggestions_count", 0)
                    for s in steps
                    if s.get("step") == "group_across_episodes"
                ),
                0,
            )

            # Add detailed steps to log
            for step in steps:
                step_name = step.get("step", "")
                step_status = step.get("status", "")
                log_messages.append(f"✓ {step_name}: {step_status}")
                if step_name == "clear_assignments":
                    log_messages.append(
                        f"  → Cleared {step.get('cleared_count', 0)} stale assignments"
                    )
                elif step_name == "compute_centroids":
                    log_messages.append(
                        f"  → Computed {step.get('centroids_count', 0)} centroids"
                    )
                elif step_name == "group_within_episode":
                    log_messages.append(
                        f"  → Merged {step.get('merged_count', 0)} cluster groups"
                    )
                elif step_name == "group_across_episodes":
                    log_messages.append(
                        f"  → Generated {step.get('suggestions_count', 0)} suggestions"
                    )

            # Show log in expander
            with log_expander:
                for msg in log_messages:
                    st.text(msg)

            status_text.success(
                f"✅ Clustering complete!\n"
                f"• Cleared {cleared} stale assignment(s)\n"
                f"• Computed {centroids} centroid(s)\n"
                f"• Merged {merged} cluster group(s)\n"
                f"• Computed {suggestions} similarity suggestion(s)\n\n"
                "Check Episode Auto-Clustered People below to review and assign."
            )
            progress_bar.empty()
            st.rerun()
    else:
        # Original simple spinner for facebank
        with st.spinner("Running cluster grouping..."):
            result = _api_post(f"/episodes/{ep_id}/clusters/group", payload)
            if result:
                matched = result.get("result", {}).get("matched_clusters", 0)
                st.success(
                    f"Facebank regroup complete! {matched} clusters matched to seeds."
                )
                st.rerun()
st.caption(caption_text)

view_state = st.session_state.get("facebank_view", "people")
identities_payload = _safe_api_get(f"/episodes/{ep_id}/identities")
if not identities_payload:
    st.stop()

cluster_payload = _safe_api_get(f"/episodes/{ep_id}/cluster_tracks") or {"clusters": []}
cluster_lookup = _clusters_by_identity(cluster_payload)
identities = identities_payload.get("identities", [])
identity_index = {ident["identity_id"]: ident for ident in identities}
# show_slug already defined above - no need to recompute
roster_names = _fetch_roster_names(show_slug)
show_id, people = _episode_people(ep_id)
ep_meta = helpers.parse_ep_id(ep_id) or {}
season_label: str | None = None
season_value = ep_meta.get("season")
if isinstance(season_value, int):
    season_label = f"S{season_value:02d}"
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
        ep_id, show_id, people, cluster_lookup, identity_index, season_label
    )

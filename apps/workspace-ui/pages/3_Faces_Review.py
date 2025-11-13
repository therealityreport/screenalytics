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

# Inject thumbnail CSS
helpers.inject_thumb_css()

MAX_TRACKS_PER_ROW = 6


def _render_similarity_badge(similarity: float | None) -> str:
    """Render a similarity score as a percentage badge."""
    if similarity is None:
        return ""
    pct = int(similarity * 100)
    # Color coding: green for high similarity, yellow for medium, red for low
    if similarity >= 0.75:
        color = "green"
    elif similarity >= 0.60:
        color = "orange"
    else:
        color = "red"
    return f'<div style="background-color: {color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-weight: bold; display: inline-block;">{pct}%</div>'



def _safe_api_get(path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    try:
        return helpers.api_get(path, params=params)
    except requests.RequestException as exc:
        st.error(helpers.describe_error(f"{cfg['api_base']}{path}", exc))
        return None


def _api_post(path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    try:
        return helpers.api_post(path, payload or {})
    except requests.RequestException as exc:
        st.error(helpers.describe_error(f"{cfg['api_base']}{path}", exc))
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
        return
    moved = resp.get("moved") or len(frame_ids)
    name = resp.get("target_name") or resp.get("target_identity_id") or target_identity_id or "target identity"
    st.toast(f"Moved {moved} frame(s) to {name}")
    _refresh_roster_names(show)
    st.session_state.setdefault("track_frame_selection", {}).pop(track_id, None)
    st.rerun()


def _delete_frames_api(ep_id: str, track_id: int, frame_ids: List[int], delete_assets: bool = True) -> None:
    payload = {"frame_ids": frame_ids, "delete_assets": delete_assets}
    resp = _api_delete(f"/episodes/{ep_id}/tracks/{track_id}/frames", payload)
    if resp is None:
        return
    deleted = resp.get("deleted") or len(frame_ids)
    st.toast(f"Deleted {deleted} frame(s)")
    st.session_state.setdefault("track_frame_selection", {}).pop(track_id, None)
    st.rerun()


def _select_episode() -> str:
    current = helpers.get_ep_id()
    tracked_tab, s3_tab = st.tabs(["Tracked", "Browse S3"])
    with tracked_tab:
        payload = _safe_api_get("/episodes")
        options = payload.get("episodes", []) if payload else []
        if options:
            ep_ids = [item["ep_id"] for item in options]
            default_idx = ep_ids.index(current) if current in ep_ids else 0
            selection = st.selectbox(
                "Tracked episodes",
                ep_ids,
                format_func=lambda eid: f"{eid} ({options[ep_ids.index(eid)]['show_slug']})",
                index=default_idx if ep_ids else 0,
                key="facebank_tracked_select",
            )
            if st.button("Load tracked episode", key="facebank_load_tracked"):
                helpers.set_ep_id(selection)
        else:
            st.info("No tracked episodes yet.")
    with s3_tab:
        s3_payload = _safe_api_get("/episodes/s3_videos")
        items = s3_payload.get("items", []) if s3_payload else []
        if items:
            labels = [f"{item['ep_id']} · {item.get('last_modified') or 'unknown'}" for item in items]
            idx = st.selectbox("S3 videos", list(range(len(items))), format_func=lambda i: labels[i], key="facebank_s3_select")
            selected = items[idx]
            if st.button("Track selected S3 episode", key="facebank_track_s3"):
                _track_episode_from_s3(selected)
        else:
            st.info("No S3 videos exposed by the API.")
    ep_id = helpers.get_ep_id()
    if not ep_id:
        st.warning("Choose an episode from either tab to continue.")
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
    if st.button("Save name", key=f"{prefix}_save_{identity['identity_id']}", disabled=disabled):
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
    action_cols = st.columns([1, 1])
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
                if faces_after is not None:
                    details.append(f"faces {helpers.format_count(faces_after) or faces_after}")
                grouping = (report.get("grouping") or {}) if isinstance(report, dict) else {}
                across = grouping.get("across_episodes") or {}
                assigned = across.get("assigned") or []
                new_people = helpers.coerce_int(across.get("new_people_count"))
                if assigned:
                    details.append(f"matched {len(assigned)} cluster(s) to people")
                if new_people:
                    details.append(f"created {new_people} new people")
                summary_line = " · ".join(details) if details else "Cleanup complete."
                st.success(summary_line or "Cleanup complete.")
                st.caption(f"Report written to data/manifests/{ep_id}/cleanup_report.json")
    return detail


def _episode_people(ep_id: str) -> tuple[str | None, List[Dict[str, Any]]]:
    meta = helpers.parse_ep_id(ep_id)
    if not meta:
        return None, []
    show_slug = str(meta.get("show") or "").upper()
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


def _clusters_by_identity(cluster_payload: Dict[str, Any] | None) -> Dict[str, Dict[str, Any]]:
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
    payload = _safe_api_get(f"/episodes/{ep_id}/tracks/{track_id}/crops", params=params) or {}
    items = payload.get("items", []) if isinstance(payload, dict) else []
    next_cursor = payload.get("next_start_after") if isinstance(payload, dict) else None
    normalized: List[Dict[str, Any]] = []
    for item in items:
        url = item.get("media_url") or item.get("url") or item.get("thumbnail_url")
        resolved = helpers.resolve_thumb(url)
        # Include all crops, even if URL resolution fails temporarily
        # The UI can handle missing images gracefully
        normalized.append({
            "url": resolved or url,  # Use original URL if resolution fails
            "frame_idx": item.get("frame_idx"),
            "track_id": track_id,
            "s3_key": item.get("s3_key"),  # Preserve S3 key for debugging
        })
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


def _render_people_view(
    ep_id: str,
    show_id: str | None,
    people: List[Dict[str, Any]],
    cluster_lookup: Dict[str, Dict[str, Any]],
    identity_index: Dict[str, Dict[str, Any]],
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

    if not people:
        st.info("No people found for this show. Run 'Group Clusters (auto)' to create people.")
        return

    filtered_people = people
    if filter_cast_id:
        filtered_people = [
            person
            for person in people
            if str(person.get("person_id") or "").lower() == str(filter_cast_id).lower()
        ]
        if not filtered_people:
            st.warning(f"{filter_cast_name or filter_cast_id} is not part of this show's roster.")
            return

    episode_people: List[tuple[Dict[str, Any], List[str]]] = []
    for person in filtered_people:
        episode_clusters = _episode_cluster_ids(person, ep_id)
        if not episode_clusters:
            continue
        episode_people.append((person, episode_clusters))

    if not episode_people:
        if filter_cast_id:
            st.info(
                f"{filter_cast_name or filter_cast_id} has no clusters linked to episode {ep_id} yet."
            )
        else:
            st.info("No clusters from this episode are linked to show-level people yet.")
        return

    # Sort people: named first, then unnamed
    episode_people.sort(key=lambda x: (x[0].get("name") is None or x[0].get("name") == "", x[0].get("name") or ""))

    st.markdown(f"**{len(episode_people)} People** in episode {ep_id}")
    for person, episode_clusters in episode_people:
        person_id = str(person.get("person_id") or "")
        name = person.get("name") or "(unnamed)"
        total_clusters = len(person.get("cluster_ids", []) or [])
        with st.container(border=True):
            featured_crop = helpers.resolve_thumb(person.get("rep_crop"))
            if featured_crop:
                st.image(featured_crop, caption="Featured image", width=180)
            st.markdown(f"### 👤 {name}")
            st.caption(
                f"ID: {person_id or 'n/a'} · {total_clusters} cluster(s) overall · "
                f"{len(episode_clusters)} in this episode"
            )

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
                                thumb_markup = helpers.thumb_html(resolved, alt=f"Cluster {cluster_id}", hide_if_missing=False)
                                st.markdown(thumb_markup, unsafe_allow_html=True)

                            # Show cluster ID and cohesion badge
                            cohesion_badge = _render_similarity_badge(cohesion) if cohesion else ""
                            st.markdown(f"**{cluster_id}** {cohesion_badge}", unsafe_allow_html=True)
                            st.caption(f"{cluster.get('tracks', 0)} tracks · {cluster.get('faces', 0)} faces")

                            # View cluster button
                            if st.button("View cluster", key=f"view_cluster_{person_id}_{cluster_id}"):
                                _set_view("cluster_tracks", person_id=person_id, identity_id=cluster_id)
                                st.rerun()
            else:
                # Fallback to text labels if summary unavailable
                labels: List[str] = []
                for identity_id in episode_clusters:
                    cluster_meta = cluster_lookup.get(identity_id, {})
                    identity_meta = identity_index.get(identity_id, {})
                    label = (
                        cluster_meta.get("name")
                        or identity_meta.get("name")
                        or cluster_meta.get("label")
                        or identity_meta.get("label")
                        or identity_id
                    )
                    labels.append(f"{label} (`{identity_id}`)")
                st.markdown("**Clusters in this episode:** " + ", ".join(labels))
                st.button(
                    "View clusters",
                    key=f"view_clusters_{person_id or name}",
                    on_click=lambda pid=person_id: _set_view("person_clusters", person_id=pid),
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
    st.subheader(f"👤 {name} · ID: {person_id}")
    featured_crop = helpers.resolve_thumb(person.get("rep_crop"))
    if featured_crop:
        st.image(featured_crop, caption="Featured image", width=220)
    st.caption(f"{len(episode_clusters)} cluster(s) in this episode · {total_clusters} overall")
    if not episode_clusters:
        st.info("No clusters assigned to this person in this episode yet.")
        return

    # Fetch clusters summary with track representatives from new endpoint
    clusters_summary = _safe_api_get(f"/episodes/{ep_id}/people/{person_id}/clusters_summary")
    if not clusters_summary:
        st.error("Failed to load cluster representatives.")
        return

    identity_index = {ident["identity_id"]: ident for ident in identities_payload.get("identities", [])}

    for cluster_data in clusters_summary.get("clusters", []):
        identity_id = cluster_data["cluster_id"]
        identity_meta = identity_index.get(identity_id, {})
        tracks_count = cluster_data.get("tracks", 0)
        faces_count = cluster_data.get("faces", 0)
        cohesion = cluster_data.get("cohesion")
        track_reps = cluster_data.get("track_reps", [])

        display_name = identity_meta.get("name")
        label = identity_meta.get("label")

        with st.container(border=True):
            title = f"Cluster {identity_id}"
            if display_name:
                title += f" · {display_name}"
            if label and label != display_name:
                title += f" · {label}"
            st.markdown(f"**{title}**")

            # Display metrics
            metrics_parts = [f"Tracks: {tracks_count}", f"Faces: {faces_count}"]
            if cohesion is not None:
                metrics_parts.append(f"Cohesion: {int(cohesion * 100)}%")
            st.caption(" · ".join(metrics_parts))

            _identity_name_controls(
                ep_id=ep_id,
                identity={
                    "identity_id": identity_id,
                    "name": display_name,
                    "label": label,
                },
                show_slug=show_slug,
                roster_names=roster_names,
                prefix=f"person_cluster_{person_id}_{identity_id}",
            )

            # Render all track representatives with similarity badges
            if track_reps:
                st.markdown(f"**All {len(track_reps)} Track(s):**")
                # Render in rows of MAX_TRACKS_PER_ROW
                for row_start in range(0, len(track_reps), MAX_TRACKS_PER_ROW):
                    row_tracks = track_reps[row_start : row_start + MAX_TRACKS_PER_ROW]
                    cols = st.columns(len(row_tracks))
                    for idx, track_rep in enumerate(row_tracks):
                        with cols[idx]:
                            crop_url = track_rep.get("crop_url")
                            track_id = track_rep.get("track_id", "")
                            similarity = track_rep.get("similarity")

                            # Parse numeric track_id for display
                            track_num = track_id.replace("track_", "") if isinstance(track_id, str) else track_id

                            resolved = helpers.resolve_thumb(crop_url)
                            thumb_markup = helpers.thumb_html(
                                resolved, alt=f"Track {track_num}", hide_if_missing=False
                            )
                            st.markdown(thumb_markup, unsafe_allow_html=True)

                            # Display track ID and similarity badge
                            badge_html = _render_similarity_badge(similarity)
                            st.markdown(f"Track {track_num} {badge_html}", unsafe_allow_html=True)
            else:
                st.info("No track representatives available.")

            st.button(
                "View tracks",
                key=f"view_tracks_{person_id}_{identity_id}",
                on_click=lambda pid=person_id, iid=identity_id: _set_view(
                    "cluster_tracks", person_id=pid, identity_id=iid
                ),
            )


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

    # Display all track representatives with similarity badges
    if not track_reps:
        st.info("No track representatives available.")
        return

    st.markdown(f"**All {len(track_reps)} Track(s) with Similarity Scores:**")
    move_targets = [ident_id for ident_id in identity_index if ident_id != identity_id]

    # Render tracks in grid
    for row_start in range(0, len(track_reps), MAX_TRACKS_PER_ROW):
        row_tracks = track_reps[row_start : row_start + MAX_TRACKS_PER_ROW]
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
    frames_payload = _fetch_track_frames(ep_id, track_id, sample=sample, page=page, page_size=page_size)
    frames = frames_payload.get("items", [])
    total_sampled = int(frames_payload.get("total") or 0)
    total_frames = int(frames_payload.get("total_frames") or total_sampled)
    max_page = max(1, math.ceil(total_sampled / page_size)) if total_sampled else 1
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
                face_id = frame_meta.get("face_id")
                frame_idx = frame_meta.get("frame_idx")
                try:
                    frame_idx_int = int(frame_idx)
                except (TypeError, ValueError):
                    frame_idx_int = None
                thumb_url = frame_meta.get("media_url") or frame_meta.get("thumbnail_url")
                skip_reason = frame_meta.get("skip")
                with row_cols[idx]:
                    caption = f"Frame {frame_idx}" if frame_idx is not None else (face_id or "frame")
                    resolved_thumb = helpers.resolve_thumb(thumb_url)
                    thumb_markup = helpers.thumb_html(resolved_thumb, alt=caption, hide_if_missing=True)
                    if thumb_markup:
                        st.markdown(thumb_markup, unsafe_allow_html=True)
                    else:
                        st.caption("Crop unavailable.")
                    st.caption(caption)
                    # Show similarity badge if available
                    similarity = frame_meta.get("similarity")
                    if similarity is not None:
                        similarity_badge = _render_similarity_badge(similarity)
                        st.markdown(similarity_badge, unsafe_allow_html=True)
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
        if action_cols[0].button("Move selected", key=f"track_move_selected_{track_id}", disabled=move_disabled):
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

ep_id = _select_episode()
_initialize_state(ep_id)
episode_detail = _episode_header(ep_id)
if not episode_detail:
    st.stop()
if not helpers.detector_is_face_only(ep_id):
    st.warning(
        "Tracks were generated with a legacy detector. Rerun detect/track with RetinaFace or YOLOv8-face for best results."
    )

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

button_label = "Group Clusters (facebank)" if selected_strategy == "facebank" else "Group Clusters (auto)"
caption_text = (
    "Auto-groups clusters within episode and matches to show-level People"
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
    with st.spinner("Running cluster grouping..."):
        result = _api_post(f"/episodes/{ep_id}/clusters/group", payload)
        if result:
            if selected_strategy == "auto":
                st.success(
                    f"Grouping complete! {result.get('result', {}).get('across_episodes', {}).get('new_people_count', 0)} new people created."
                )
            else:
                matched = result.get("result", {}).get("matched_clusters", 0)
                st.success(f"Facebank regroup complete! {matched} clusters matched to seeds.")
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
show_slug = _episode_show_slug(ep_id)
roster_names = _fetch_roster_names(show_slug)
show_id, people = _episode_people(ep_id)
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
    _render_people_view(ep_id, show_id, people, cluster_lookup, identity_index)

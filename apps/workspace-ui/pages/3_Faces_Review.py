from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

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
    if not show:
        return []
    cache = _roster_cache()
    if show in cache:
        return cache[show]
    payload = _safe_api_get(f"/shows/{show}/cast_names")
    names = payload.get("names", []) if payload else []
    cache[show] = names
    return names


def _refresh_roster_names(show: str | None) -> None:
    if not show:
        return
    _roster_cache().pop(show, None)


def _cluster_offsets() -> Dict[str, int]:
    return st.session_state.setdefault("cluster_track_offsets", {})


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
    st.session_state.get("track_detail_cache", {}).clear()
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
    st.session_state.get("track_detail_cache", {}).pop(track_id, None)
    st.session_state.setdefault("track_frame_selection", {}).pop(track_id, None)
    st.rerun()


def _delete_frames_api(ep_id: str, track_id: int, frame_ids: List[int], delete_assets: bool = True) -> None:
    payload = {"frame_ids": frame_ids, "delete_assets": delete_assets}
    resp = _api_delete(f"/episodes/{ep_id}/tracks/{track_id}/frames", payload)
    if resp is None:
        return
    deleted = resp.get("deleted") or len(frame_ids)
    st.toast(f"Deleted {deleted} frame(s)")
    st.session_state.get("track_detail_cache", {}).pop(track_id, None)
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
        st.session_state["facebank_view"] = "clusters"
        st.session_state["selected_identity"] = None
        st.session_state["selected_track"] = None
    st.session_state.setdefault("cluster_track_offsets", {})
    st.session_state.setdefault("track_detail_cache", {})


def _set_view(view: str, identity_id: str | None = None, track_id: int | None = None) -> None:
    st.session_state["facebank_view"] = view
    st.session_state["selected_identity"] = identity_id
    st.session_state["selected_track"] = track_id
    st.rerun()


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
    st.button(
        "Open Episode Detail",
        key="facebank_open_detail",
        on_click=lambda: helpers.try_switch_page("pages/2_Episode_Detail.py"),
    )
    return detail


def _render_cluster_rows(
    ep_id: str,
    cluster_payload: Dict[str, Any],
    identities_payload: Dict[str, Any],
) -> None:
    clusters = cluster_payload.get("clusters", []) if cluster_payload else []
    if not clusters:
        st.info("No identities yet. Run Faces Harvest and Cluster to populate the facebank.")
        return
    show_slug = _episode_show_slug(ep_id)
    roster_names = _fetch_roster_names(show_slug)
    identity_index = {ident["identity_id"]: ident for ident in identities_payload.get("identities", [])}
    filter_value = st.text_input(
        "Filter clusters by name, label, or id",
        key="cluster_filter",
        placeholder="Start typing…",
    ).strip()
    if filter_value:
        lowered = filter_value.lower()
        clusters = [
            cluster
            for cluster in clusters
            if lowered in str(cluster.get("identity_id", "")).lower()
            or lowered in str(cluster.get("name", "") or "").lower()
            or lowered in str(cluster.get("label", "") or "").lower()
        ]
        if not clusters:
            st.warning("No clusters matched the filter.")
            return
    offsets = _cluster_offsets()
    TRACK_WINDOW = 5
    for cluster in clusters:
        identity_id = cluster["identity_id"]
        identity_meta = identity_index.get(identity_id, {})
        header = f"{identity_id} · Tracks: {cluster['counts']['tracks']} · Faces: {cluster['counts']['faces']}"
        st.subheader(header)
        _identity_name_controls(
            ep_id=ep_id,
            identity={
                "identity_id": identity_id,
                "name": cluster.get("name") or identity_meta.get("name"),
                "label": cluster.get("label") or identity_meta.get("label"),
            },
            show_slug=show_slug,
            roster_names=roster_names,
            prefix=f"cluster_row_{identity_id}",
        )
        tracks = cluster.get("tracks", [])
        if not tracks:
            st.info("No tracks linked to this identity yet.")
            continue
        start_idx = offsets.get(identity_id, 0)
        start_idx = max(0, min(start_idx, max(0, len(tracks) - TRACK_WINDOW)))
        offsets[identity_id] = start_idx
        visible = tracks[start_idx : start_idx + TRACK_WINDOW]
        arrow_cols = st.columns([0.5, 8, 0.5])
        with arrow_cols[0]:
            disabled = start_idx == 0
            if st.button("←", key=f"cluster_left_{identity_id}", disabled=disabled):
                offsets[identity_id] = max(0, start_idx - TRACK_WINDOW)
                st.rerun()
        with arrow_cols[2]:
            disabled = start_idx + TRACK_WINDOW >= len(tracks)
            if st.button("→", key=f"cluster_right_{identity_id}", disabled=disabled):
                offsets[identity_id] = min(len(tracks) - 1, start_idx + TRACK_WINDOW)
                st.rerun()
        # Always use 5 columns for consistent sizing
        grid_cols = st.columns(5)
        for idx, track in enumerate(visible):
            with grid_cols[idx]:
                thumb_url = helpers.resolve_thumb(track.get("rep_thumb_url"))
                st.markdown(helpers.thumb_html(thumb_url, alt=f"Track {track['track_id']}"), unsafe_allow_html=True)
                faces_count = track.get("faces") or 0
                st.caption(f"Track {track['track_id']} · {faces_count} faces")
                if st.button("View track", key=f"view_track_{identity_id}_{track['track_id']}"):
                    st.session_state["selected_identity"] = identity_id
                    _set_view("track", identity_id=identity_id, track_id=track["track_id"])
                move_targets = [
                    ident_id
                    for ident_id in identity_index
                    if ident_id != identity_id
                ]
                if move_targets:
                    target_choice = st.selectbox(
                        "Move track to…",
                        move_targets,
                        key=f"cluster_move_select_{identity_id}_{track['track_id']}",
                        format_func=lambda val: f"{val} · {identity_index.get(val, {}).get('name') or ''}".strip(" ·"),
                    )
                    if st.button("Move track", key=f"cluster_move_btn_{identity_id}_{track['track_id']}"):
                        _move_track(ep_id, track["track_id"], target_choice)
                # Add delete button for each track in cluster view
                if st.button("Delete track", key=f"cluster_delete_btn_{identity_id}_{track['track_id']}", type="secondary"):
                    _delete_track(ep_id, track["track_id"])
        st.divider()




def _render_track_view(ep_id: str, track_id: int, identities_payload: Dict[str, Any]) -> None:
    st.button(
        "← Back to clusters",
        key="facebank_back_clusters",
        on_click=lambda: _set_view("clusters"),
    )
    st.markdown(f"### Track {track_id}")
    identities = identities_payload.get("identities", [])
    current_identity = st.session_state.get("selected_identity")
    show_slug = _episode_show_slug(ep_id)
    roster_names = _fetch_roster_names(show_slug)
    track_detail_cache = st.session_state.setdefault("track_detail_cache", {})
    detail = track_detail_cache.get(track_id)
    if detail is None:
        detail = _safe_api_get(f"/episodes/{ep_id}/tracks/{track_id}")
        if detail is None:
            return
        track_detail_cache[track_id] = detail
    frames = detail.get("frames", [])
    action_cols = st.columns([1.0, 1.0, 1.0])
    with action_cols[0]:
        targets = [ident["identity_id"] for ident in identities if ident["identity_id"] != current_identity]
        if targets:
            target_choice = st.selectbox(
                "Move entire track",
                targets,
                key=f"track_view_move_{track_id}",
            )
            if st.button("Move track", key=f"track_view_move_btn_{track_id}"):
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
        cols_per_row = 5
        for row_start in range(0, len(frames), cols_per_row):
            row_frames = frames[row_start : row_start + cols_per_row]
            row_cols = st.columns(cols_per_row)
            for idx, frame_meta in enumerate(row_frames):
                face_id = frame_meta.get("face_id")
                frame_idx = frame_meta.get("frame_idx")
                try:
                    frame_idx_int = int(frame_idx)
                except (TypeError, ValueError):
                    frame_idx_int = None
                thumb_url = frame_meta.get("thumbnail_url")
                skip_reason = frame_meta.get("skip")
                with row_cols[idx]:
                    caption = f"Frame {frame_idx}" if frame_idx is not None else (face_id or "frame")
                    resolved_thumb = helpers.resolve_thumb(thumb_url)
                    st.markdown(helpers.thumb_html(resolved_thumb, alt=caption), unsafe_allow_html=True)
                    st.caption(caption)
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
        # Clear track detail cache so deleted track doesn't reappear
        st.session_state.get("track_detail_cache", {}).pop(track_id, None)
        # Clear selection state for this track
        st.session_state.get("track_frame_selection", {}).pop(track_id, None)
        # Return to clusters view
        _set_view("clusters")
        st.success("Track deleted.")
        st.rerun()


def _delete_frame(ep_id: str, track_id: int, frame_idx: int, delete_assets: bool) -> None:
    payload = {
        "track_id": track_id,
        "frame_idx": frame_idx,
        "delete_assets": delete_assets,
    }
    if _api_post(f"/identities/{ep_id}/drop_frame", payload):
        # Clear track detail cache for this specific track to reload fresh data
        st.session_state.get("track_detail_cache", {}).pop(track_id, None)
        # Stay on track view - don't navigate away
        st.success("Frame removed.")
        st.rerun()


def _render_people_view(ep_id: str, identities_payload: Dict[str, Any]) -> None:
    """Render People view showing show-level person entities."""
    ep_meta = helpers.parse_ep_id(ep_id)
    if not ep_meta:
        st.error("Unable to parse episode ID.")
        return

    show_id = ep_meta["show"].upper()
    people_resp = _safe_api_get(f"/shows/{show_id}/people")
    if not people_resp:
        st.info("No people found for this show. Run 'Group Clusters (auto)' to create people.")
        return

    people = people_resp.get("people", [])
    if not people:
        st.info("No people found for this show. Run 'Group Clusters (auto)' to create people.")
        return

    # Build map of identity_id to identity metadata
    identities = identities_payload.get("identities", [])
    identity_map = {ident["identity_id"]: ident for ident in identities}

    st.markdown(f"**{len(people)} People** across show {show_id}")

    for person in people:
        person_id = person["person_id"]
        name = person.get("name") or "(unnamed)"
        cluster_ids = person.get("cluster_ids", [])

        # Filter clusters for this episode
        ep_clusters = [cid for cid in cluster_ids if cid.startswith(f"{ep_id}:")]
        ep_cluster_count = len(ep_clusters)

        header = f"👤 {name} · ID: {person_id} · {ep_cluster_count} clusters in this episode · {len(cluster_ids)} total"
        st.subheader(header)

        # Show clusters for this episode
        if ep_clusters:
            st.caption(f"Clusters in {ep_id}:")
            for full_cluster_id in ep_clusters:
                cluster_id = full_cluster_id.split(":", 1)[1] if ":" in full_cluster_id else full_cluster_id
                identity_meta = identity_map.get(cluster_id, {})
                label = identity_meta.get("name") or identity_meta.get("label") or cluster_id
                st.markdown(f"- {label} (`{cluster_id}`)")

        st.divider()


ep_id = _select_episode()
_initialize_state(ep_id)
episode_detail = _episode_header(ep_id)
if not episode_detail:
    st.stop()
if not helpers.detector_is_face_only(ep_id):
    st.warning(
        "Tracks were generated with a legacy detector. Rerun detect/track with RetinaFace or YOLOv8-face for best results."
    )

# Group Clusters button and view switcher
cols = st.columns([1, 1, 2])
with cols[0]:
    if st.button("Group Clusters (auto)", key="group_clusters_auto", type="primary"):
        with st.spinner("Running cluster grouping..."):
            result = _api_post(f"/episodes/{ep_id}/clusters/group", {"strategy": "auto"})
            if result:
                st.success(f"Grouping complete! {result.get('result', {}).get('across_episodes', {}).get('new_people_count', 0)} new people created.")
                st.rerun()
with cols[1]:
    view_mode = st.radio(
        "View mode",
        options=["Clusters", "People"],
        horizontal=True,
        key="view_mode",
        label_visibility="collapsed"
    )
with cols[2]:
    st.caption("Auto-groups clusters within episode and matches to show-level People")

view_state = st.session_state.get("facebank_view", "clusters")
identities_payload = _safe_api_get(f"/episodes/{ep_id}/identities")
if not identities_payload:
    st.stop()

# Determine which view to show
if view_mode == "People":
    _render_people_view(ep_id, identities_payload)
elif view_state == "track" and st.session_state.get("selected_track") is not None:
    _render_track_view(ep_id, st.session_state["selected_track"], identities_payload)
else:
    # Clusters view
    cluster_payload = _safe_api_get(f"/episodes/{ep_id}/cluster_tracks") or {"clusters": []}
    _render_cluster_rows(ep_id, cluster_payload, identities_payload)

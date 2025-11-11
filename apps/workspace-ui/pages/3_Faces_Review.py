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
        st.session_state["facebank_view"] = "people"
        st.session_state["selected_person"] = None
        st.session_state["selected_identity"] = None
        st.session_state["selected_track"] = None
    st.session_state.setdefault("track_detail_cache", {})


def _set_view(
    view: str,
    *,
    person_id: str | None = None,
    identity_id: str | None = None,
    track_id: int | None = None,
) -> None:
    st.session_state["facebank_view"] = view
    st.session_state["selected_person"] = person_id
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
    if not people:
        st.info("No people found for this show. Run 'Group Clusters (auto)' to create people.")
        return

    st.markdown(f"**{len(people)} People** across show {show_id}")
    for person in people:
        person_id = str(person.get("person_id") or "")
        name = person.get("name") or "(unnamed)"
        total_clusters = len(person.get("cluster_ids", []) or [])
        episode_clusters = _episode_cluster_ids(person, ep_id)
        with st.container(border=True):
            st.markdown(f"### 👤 {name}")
            st.caption(
                f"ID: {person_id or 'n/a'} · {total_clusters} cluster(s) overall · "
                f"{len(episode_clusters)} in this episode"
            )
            if episode_clusters:
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
            else:
                st.caption("No clusters linked to this person in this episode yet.")
            st.button(
                "View clusters",
                key=f"view_clusters_{person_id or name}",
                disabled=not episode_clusters,
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
        return
    name = person.get("name") or "(unnamed)"
    total_clusters = len(person.get("cluster_ids", []) or [])
    episode_clusters = _episode_cluster_ids(person, ep_id)
    st.subheader(f"👤 {name} · ID: {person_id}")
    st.caption(f"{len(episode_clusters)} cluster(s) in this episode · {total_clusters} overall")
    if not episode_clusters:
        st.info("No clusters assigned to this person in this episode yet.")
        return
    identity_index = {ident["identity_id"]: ident for ident in identities_payload.get("identities", [])}
    for identity_id in episode_clusters:
        cluster = cluster_lookup.get(identity_id, {})
        identity_meta = identity_index.get(identity_id, {})
        counts = cluster.get("counts", {}) if isinstance(cluster, dict) else {}
        tracks_count = counts.get("tracks") or len(identity_meta.get("track_ids", []) or [])
        faces_count = counts.get("faces") or identity_meta.get("faces") or 0
        display_name = cluster.get("name") or identity_meta.get("name")
        label = cluster.get("label") or identity_meta.get("label")
        with st.container(border=True):
            title = f"Cluster {identity_id}"
            if display_name:
                title += f" · {display_name}"
            if label and label != display_name:
                title += f" · {label}"
            st.markdown(f"**{title}**")
            st.caption(f"Tracks: {tracks_count} · Faces: {faces_count}")
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
    cluster = cluster_lookup.get(identity_id, {})
    identity_meta = identity_index.get(identity_id, {})
    if not cluster and not identity_meta:
        st.warning("Cluster not found. Returning to people list.")
        _set_view("people")
        return
    counts = cluster.get("counts", {}) if isinstance(cluster, dict) else {}
    tracks_count = counts.get("tracks") or len(identity_meta.get("track_ids", []) or [])
    faces_count = counts.get("faces") or identity_meta.get("faces") or 0
    display_name = cluster.get("name") or identity_meta.get("name")
    label = cluster.get("label") or identity_meta.get("label")
    header = f"{identity_id} · Tracks: {tracks_count} · Faces: {faces_count}"
    st.subheader(header)
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

    tracks = cluster.get("tracks", []) if isinstance(cluster, dict) else []
    if not tracks:
        st.info("No tracks linked to this identity yet.")
        return

    move_targets = [ident_id for ident_id in identity_index if ident_id != identity_id]
    cols_per_row = 5
    for row_start in range(0, len(tracks), cols_per_row):
        row_tracks = tracks[row_start : row_start + cols_per_row]
        row_cols = st.columns(cols_per_row)
        for idx, track in enumerate(row_tracks):
            with row_cols[idx]:
                track_id = track.get("track_id")
                thumb_url = helpers.resolve_thumb(track.get("rep_thumb_url"))
                st.markdown(
                    helpers.thumb_html(thumb_url, alt=f"Track {track_id}"),
                    unsafe_allow_html=True,
                )
                faces_count = track.get("faces") or 0
                st.caption(f"Track {track_id} · {faces_count} faces")
                if st.button(
                    "View frames",
                    key=f"view_track_{identity_id}_{track_id}",
                ):
                    _set_view(
                        "track",
                        person_id=person_id,
                        identity_id=identity_id,
                        track_id=int(track_id),
                    )
                if move_targets and track_id is not None:
                    target_choice = st.selectbox(
                        "Move track to…",
                        move_targets,
                        key=f"cluster_move_select_{identity_id}_{track_id}",
                        format_func=lambda val: f"{val} · {identity_index.get(val, {}).get('name') or ''}".strip(" ·"),
                    )
                    if st.button(
                        "Move track",
                        key=f"cluster_move_btn_{identity_id}_{track_id}",
                    ):
                        _move_track(ep_id, int(track_id), target_choice)
                if track_id is not None and st.button(
                    "Delete track",
                    key=f"cluster_delete_btn_{identity_id}_{track_id}",
                    type="secondary",
                ):
                    _delete_track(ep_id, int(track_id))




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
        # Clear track detail cache for this specific track to reload fresh data
        st.session_state.get("track_detail_cache", {}).pop(track_id, None)
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

# Group Clusters action and context hint
cols = st.columns([1, 3])
with cols[0]:
    if st.button("Group Clusters (auto)", key="group_clusters_auto", type="primary"):
        with st.spinner("Running cluster grouping..."):
            result = _api_post(f"/episodes/{ep_id}/clusters/group", {"strategy": "auto"})
            if result:
                st.success(
                    f"Grouping complete! {result.get('result', {}).get('across_episodes', {}).get('new_people_count', 0)} new people created."
                )
                st.rerun()
with cols[1]:
    st.caption("Auto-groups clusters within episode and matches to show-level People")

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

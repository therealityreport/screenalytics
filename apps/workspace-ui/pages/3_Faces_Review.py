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

TRACK_STRIP_LIMIT = 40
TRACK_VIEW_LIMIT = 80


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


def _initialize_state(ep_id: str) -> None:
    if st.session_state.get("facebank_ep") != ep_id:
        st.session_state["facebank_ep"] = ep_id
        st.session_state["facebank_view"] = "grid"
        st.session_state["selected_identity"] = None
        st.session_state["selected_track"] = None
        st.session_state["cluster_row_data"] = {}
        st.session_state["track_view_cache"] = {}
    st.session_state.setdefault("cluster_sample_every", 5)
    st.session_state.setdefault("track_view_cache", {})


def _set_view(view: str, identity_id: str | None = None, track_id: int | None = None) -> None:
    st.session_state["facebank_view"] = view
    st.session_state["selected_identity"] = identity_id
    st.session_state["selected_track"] = track_id
    st.rerun()


def _cluster_cache(identity_id: str) -> Dict[int, Dict[str, Any]]:
    cache = st.session_state.setdefault("cluster_row_data", {})
    return cache.setdefault(identity_id, {})


def _prepare_track_view_state(track_id: int, sample_every: int) -> None:
    cache = st.session_state.setdefault("track_view_cache", {})
    cache[track_id] = {
        "items": [],
        "cursor": None,
        "sample": sample_every,
        "exhausted": False,
    }


def _fetch_track_media(
    ep_id: str,
    track_id: int,
    *,
    sample: int,
    limit: int,
    cursor: str | None = None,
    asset: str = "crops",
) -> tuple[List[Dict[str, Any]], str | None]:
    params = {"sample": max(1, int(sample)), "limit": max(1, int(limit))}
    if cursor:
        params["start_after"] = cursor
    payload = _safe_api_get(f"/episodes/{ep_id}/tracks/{track_id}/{asset}", params=params)
    if not payload:
        return [], None
    if isinstance(payload, dict):
        items = payload.get("items", []) or []
        next_cursor = payload.get("next_start_after")
        return items, next_cursor
    return payload or [], None


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


def _render_identity_grid(ep_id: str, identities_data: Dict[str, Any]) -> None:
    identities = identities_data.get("identities", [])
    if not identities:
        st.info("No identities yet. Run Faces Harvest and Cluster to populate the facebank.")
        return
    cols = st.columns(3)
    for idx, identity in enumerate(identities):
        col = cols[idx % len(cols)]
        with col:
            def extra_actions(identity_id: str = identity["identity_id"], *, current_idx: int = idx):
                if st.button("View cluster", key=f"view_cluster_{identity_id}"):
                    _set_view("cluster", identity_id=identity_id)
                new_label = st.text_input(
                    "Rename",
                    value=identity.get("label") or "",
                    key=f"identity_label_{identity_id}",
                )
                if st.button("Save", key=f"save_label_{identity_id}"):
                    _rename_identity(ep_id, identity_id, new_label)
                merge_options = [item["identity_id"] for item in identities if item["identity_id"] != identity_id]
                if merge_options:
                    merge_target = st.selectbox(
                        "Merge into…",
                        merge_options,
                        key=f"merge_target_{identity_id}",
                    )
                    if st.button("Merge", key=f"merge_btn_{identity_id}"):
                        _api_merge(ep_id, identity_id, merge_target)
                if st.button("Delete", key=f"delete_identity_{identity_id}"):
                    _delete_identity(ep_id, identity_id)
            helpers.identity_card(
                title=f"{identity.get('label') or identity['identity_id']}",
                subtitle=f"Tracks: {len(identity.get('track_ids', []))} · Faces: {identity.get('faces', 0)}",
                image_url=helpers.letterbox_thumb_url(identity.get("rep_thumbnail_url")),
                extra=extra_actions,
            )

    st.caption("Select an identity to see its tracks, or use the inline merge controls above.")


def _render_cluster_view(ep_id: str, identity_id: str) -> None:
    detail = _safe_api_get(f"/episodes/{ep_id}/identities/{identity_id}")
    if not detail:
        return
    st.button("← Back to grid", key="facebank_back_grid", on_click=lambda: _set_view("grid"))
    identity = detail["identity"]
    st.markdown(f"### Identity `{identity_id}`")
    st.caption(f"Tracks: {len(identity.get('track_ids', []))}")
    tracks = detail.get("tracks", [])
    if not tracks:
        st.info("No tracks linked yet.")
        return
    all_identities = (_safe_api_get(f"/episodes/{ep_id}/identities") or {}).get("identities", [])
    current_sample = int(st.session_state.get("cluster_sample_every", 5))
    new_sample = int(
        st.number_input(
            "Sample every N crops (lower for denser previews)",
            min_value=1,
            max_value=20,
            value=current_sample,
            step=1,
        )
    )
    if new_sample != current_sample:
        st.session_state["cluster_sample_every"] = new_sample
        st.session_state.setdefault("cluster_row_data", {})[identity_id] = {}
        st.rerun()
    st.caption(
        f"Showing every {new_sample}ᵗʰ crop (change the slider above to load fewer or more thumbnails)."
    )
    cluster_cache = _cluster_cache(identity_id)
    for track in tracks:
        track_id = track["track_id"]
        cache_entry = cluster_cache.setdefault(
            track_id,
            {"items": [], "cursor": None, "exhausted": False},
        )
        if not cache_entry["items"] and not cache_entry.get("exhausted"):
            fetched, cursor = _fetch_track_media(
                ep_id,
                track_id,
                sample=new_sample,
                limit=TRACK_STRIP_LIMIT,
            )
            cache_entry["items"] = fetched
            cache_entry["cursor"] = cursor
            cache_entry["exhausted"] = cursor is None or not fetched
        st.markdown(f"#### Track {track_id} · {track.get('faces_count', 0)} frames")
        action_cols = st.columns([1.1, 1.3, 1.2])
        with action_cols[0]:
            if st.button("View track", key=f"cluster_view_track_{identity_id}_{track_id}"):
                _prepare_track_view_state(track_id, new_sample)
                _set_view("track", identity_id=identity_id, track_id=track_id)
        with action_cols[1]:
            target_options = [ident["identity_id"] for ident in all_identities if ident["identity_id"] != identity_id]
            if target_options:
                target_choice = st.selectbox(
                    "Move to identity",
                    target_options,
                    key=f"track_move_{identity_id}_{track_id}",
                )
                if st.button("Move", key=f"track_move_btn_{identity_id}_{track_id}"):
                    _move_track(ep_id, track_id, target_choice)
        with action_cols[2]:
            if st.button("Remove from identity", key=f"track_remove_{identity_id}_{track_id}"):
                _move_track(ep_id, track_id, None)
        row_items = cache_entry.get("items", [])
        if row_items:
            helpers.render_track_row(track_id, row_items, thumb_height=200)
        else:
            st.info("No crops available for this track yet. Try lowering the sampling interval.")
        if not cache_entry.get("exhausted") and cache_entry.get("cursor"):
            if st.button("Load more", key=f"cluster_load_more_{identity_id}_{track_id}"):
                more, cursor = _fetch_track_media(
                    ep_id,
                    track_id,
                    sample=new_sample,
                    limit=TRACK_STRIP_LIMIT,
                    cursor=cache_entry.get("cursor"),
                )
                if more:
                    cache_entry["items"].extend(more)
                cache_entry["cursor"] = cursor
                cache_entry["exhausted"] = cursor is None or not more
                st.rerun()
        st.divider()


def _render_track_view(ep_id: str, track_id: int) -> None:
    st.button(
        "← Back to cluster",
        key="facebank_back_cluster",
        on_click=lambda: _set_view("cluster", st.session_state.get("selected_identity")),
    )
    st.markdown(f"### Track {track_id}")
    st.button(
        "Delete track",
        key="facebank_delete_track",
        on_click=lambda: _delete_track(ep_id, track_id),
    )
    cache = st.session_state.setdefault("track_view_cache", {})
    state = cache.setdefault(
        track_id,
        {
            "items": [],
            "cursor": None,
            "sample": st.session_state.get("cluster_sample_every", 5),
            "exhausted": False,
        },
    )
    current_sample = int(state.get("sample", st.session_state.get("cluster_sample_every", 5)))
    new_sample = int(
        st.number_input(
            "Sample every N crops",
            min_value=1,
            max_value=20,
            value=current_sample,
            step=1,
        )
    )
    if new_sample != current_sample:
        state.update({"sample": new_sample, "items": [], "cursor": None, "exhausted": False})
        cache[track_id] = state
        st.rerun()
    if not state["items"] and not state.get("exhausted"):
        fetched, cursor = _fetch_track_media(
            ep_id,
            track_id,
            sample=new_sample,
            limit=TRACK_VIEW_LIMIT,
        )
        state["items"] = fetched
        state["cursor"] = cursor
        state["exhausted"] = cursor is None or not fetched
    items = state.get("items", [])
    if items:
        helpers.render_track_row(track_id, items, thumb_height=240)
    else:
        st.info("No crops available for this track. Try decreasing the sample interval.")
    if not state.get("exhausted") and state.get("cursor"):
        if st.button("Load more", key=f"track_view_load_more_{track_id}"):
            more, cursor = _fetch_track_media(
                ep_id,
                track_id,
                sample=new_sample,
                limit=TRACK_VIEW_LIMIT,
                cursor=state.get("cursor"),
            )
            if more:
                state["items"].extend(more)
            state["cursor"] = cursor
            state["exhausted"] = cursor is None or not more
            st.rerun()


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
        st.success("Track deleted.")
        st.rerun()


def _delete_frame(ep_id: str, track_id: int, frame_idx: int, delete_assets: bool) -> None:
    payload = {
        "track_id": track_id,
        "frame_idx": frame_idx,
        "delete_assets": delete_assets,
    }
    if _api_post(f"/identities/{ep_id}/drop_frame", payload):
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

view_state = st.session_state.get("facebank_view", "grid")
identities_payload = _safe_api_get(f"/episodes/{ep_id}/identities")
if not identities_payload:
    st.stop()

if view_state == "cluster" and st.session_state.get("selected_identity"):
    _render_cluster_view(ep_id, st.session_state["selected_identity"])
elif view_state == "track" and st.session_state.get("selected_track") is not None:
    _render_track_view(ep_id, st.session_state["selected_track"])
else:
    _render_identity_grid(ep_id, identities_payload)

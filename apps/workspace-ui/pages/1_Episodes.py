from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import streamlit as st

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402

cfg = helpers.init_page("Episodes")

# E9 fix: Clean up stale delete/purge state on page init to prevent cross-session issues
if "episodes_delete_target" in st.session_state and st.session_state.get("_episodes_page_loaded"):
    # Only reset if returning to page (not first load)
    st.session_state.pop("episodes_delete_target", None)
if "episodes_purge_open" in st.session_state and st.session_state.get("_episodes_page_loaded"):
    st.session_state.pop("episodes_purge_open", None)
st.session_state["_episodes_page_loaded"] = True

st.title("Episodes Browser")


# E6 fix: Removed duplicate _api_post_json - now using helpers.api_post() directly


def _reset_delete_state() -> None:
    st.session_state.pop("episodes_delete_target", None)


def _reset_purge_state() -> None:
    st.session_state.pop("episodes_purge_open", None)


def _show_single_delete(ep_id: str) -> None:
    st.markdown("#### Delete this episode")
    if st.button("Delete episode", key=f"episodes_delete_btn_{ep_id}"):
        st.session_state["episodes_delete_target"] = ep_id
    target = st.session_state.get("episodes_delete_target")
    if target != ep_id:
        st.caption(
            "Removes the EpisodeStore entry, local video/manifests/frames/embeddings, and cleans up people cluster references."
        )
        return
    with st.container(border=True):
        st.warning(
            f"You are about to delete `{ep_id}`. This removes the EpisodeStore record, "
            f"local data (video, manifests, frames/crops, embeddings, analytics), "
            f"and cleans up orphaned people cluster references. "
            f"S3 artifacts (frames/crops/manifests) will also be deleted."
        )
        cols = st.columns(2)
        with cols[0]:
            if st.button("Confirm delete", type="primary", key=f"confirm_delete_{ep_id}"):
                payload = {"include_s3": True}
                # E6 fix: Use helpers.api_post with try/except instead of duplicate function
                try:
                    resp = helpers.api_post(f"/episodes/{ep_id}/delete", json=payload, timeout=90)
                except requests.RequestException as exc:
                    st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/{ep_id}/delete", exc))
                    resp = None
                if resp is not None:
                    # E7 fix: Validate response structure with defaults
                    deleted = resp.get("deleted") if isinstance(resp.get("deleted"), dict) else {}
                    people_cleanup = deleted.get("people_cleanup") if isinstance(deleted.get("people_cleanup"), dict) else {}
                    msg_parts = [
                        f"Deleted {ep_id}:",
                        f"local dirs={deleted.get('local_dirs', 0)}",
                        f"S3 objects={deleted.get('s3_objects', 0)}",
                    ]
                    # E12 fix: Always show cleanup results (even when count=0) so users know it ran
                    clusters_removed = people_cleanup.get("clusters_removed", 0)
                    msg_parts.append(f"people clusters cleaned={clusters_removed}")
                    st.success(" · ".join(msg_parts))
                    _reset_delete_state()
                    st.rerun()
        with cols[1]:
            if st.button("Cancel", key=f"cancel_delete_{ep_id}"):
                _reset_delete_state()
                st.rerun()


def _show_purge_section() -> None:
    st.markdown("#### Delete ALL episodes & data")
    if st.button("Delete ALL episodes & data", key="episodes_purge_btn"):
        st.session_state["episodes_purge_open"] = True
    if not st.session_state.get("episodes_purge_open"):
        st.caption("Danger zone: wipes every tracked episode.")
        return
    with st.container(border=True):
        st.error(
            "This action deletes every EpisodeStore entry plus optional local/S3 artifacts. "
            "Type DELETE ALL to confirm."
        )
        delete_s3 = st.checkbox(
            "Also delete S3 artifacts (frames/crops/manifests) for every episode",
            value=False,
            key="purge_include_s3",
        )
        confirm_text = st.text_input("Type DELETE ALL to confirm", key="purge_confirm")
        cols = st.columns(2)
        with cols[0]:
            if st.button("Confirm purge", type="primary", key="purge_confirm_btn"):
                payload = {
                    "confirm": confirm_text.strip() if confirm_text else "",
                    "include_s3": delete_s3,
                }
                # E6 fix: Use helpers.api_post with try/except
                try:
                    resp = helpers.api_post("/episodes/delete_all", json=payload, timeout=90)
                except requests.RequestException as exc:
                    st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/delete_all", exc))
                    resp = None
                if resp is not None:
                    # E7 fix: Validate response structure
                    totals = resp.get("deleted") if isinstance(resp.get("deleted"), dict) else {}
                    st.success(
                        "Purged episodes: "
                        f"{resp.get('count', 0)} removed · local dirs cleared={totals.get('local_dirs', 0)} · "
                        f"S3 objects deleted={totals.get('s3_objects', 0)}."
                    )
                    _reset_purge_state()
                    st.rerun()
        with cols[1]:
            if st.button("Cancel purge", key="purge_cancel_btn"):
                _reset_purge_state()
                st.rerun()


try:
    episodes_payload = helpers.api_get("/episodes")
except requests.RequestException as exc:
    st.error(helpers.describe_error(f"{cfg['api_base']}/episodes", exc))
    st.stop()

episodes = episodes_payload.get("episodes", [])
if not episodes:
    st.info("No episodes yet. Use Upload & Run first.")
    st.stop()

search_query = st.text_input("Search", "", help="Filter by ep_id or show.", key="episodes_search").strip().lower()
filtered = [
    ep
    for ep in episodes
    if not search_query or search_query in ep["ep_id"].lower() or search_query in (ep["show_slug"] or "").lower()
]
if not filtered:
    st.warning("No episodes match that filter.")
    st.stop()

helpers.ds(filtered)

option_lookup = {ep["ep_id"]: ep for ep in filtered}
selected_ep_id = st.selectbox(
    "Episode",
    list(option_lookup.keys()),
    format_func=lambda eid: f"{eid} ({option_lookup[eid]['show_slug']})",
    key="episodes_selectbox",
)

if st.button("Open Episode Detail", use_container_width=True, type="primary", key="open_episode_detail_btn"):
    helpers.set_ep_id(selected_ep_id)
    st.switch_page("pages/2_Episode_Detail.py")

# Edit Episode Info section
st.divider()
st.markdown("#### Edit Episode Info")

selected_ep = option_lookup.get(selected_ep_id, {})
edit_key = f"episodes_edit_mode_{selected_ep_id}"

if not st.session_state.get(edit_key):
    # Display current info with edit button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"**Title:** {selected_ep.get('title') or '(no title)'}")
        st.caption(f"**Air Date:** {selected_ep.get('air_date') or '(not set)'}")
    with col2:
        if st.button("Edit", key=f"edit_ep_btn_{selected_ep_id}"):
            st.session_state[edit_key] = True
            st.rerun()
else:
    # Edit form
    with st.form(f"episode_edit_form_{selected_ep_id}"):
        st.caption("Update episode metadata. Show/Season/Episode cannot be changed.")

        new_title = st.text_input(
            "Title",
            value=selected_ep.get("title") or "",
            max_chars=200,
            key=f"edit_title_{selected_ep_id}",
            placeholder="e.g., Amsterdamn",
        )

        # Parse existing air_date for date_input
        existing_date = None
        if selected_ep.get("air_date"):
            try:
                existing_date = datetime.fromisoformat(selected_ep["air_date"]).date()
            except (ValueError, TypeError):
                pass

        new_air_date = st.date_input(
            "Air Date",
            value=existing_date,
            key=f"edit_air_date_{selected_ep_id}",
        )

        btn_cols = st.columns(2)
        with btn_cols[0]:
            submit = st.form_submit_button("Save changes", type="primary")
        with btn_cols[1]:
            cancel = st.form_submit_button("Cancel")

        if cancel:
            st.session_state[edit_key] = False
            st.rerun()

        if submit:
            payload: dict[str, Any] = {}
            # Only include changed fields
            current_title = selected_ep.get("title") or ""
            if new_title != current_title:
                payload["title"] = new_title if new_title else None
            if new_air_date != existing_date:
                payload["air_date"] = new_air_date.isoformat() if new_air_date else None

            if not payload:
                st.info("No changes detected.")
            else:
                try:
                    resp = helpers.api_patch(f"/episodes/{selected_ep_id}", json=payload, timeout=30)
                    st.success(f"Episode {selected_ep_id} updated.")
                    st.session_state[edit_key] = False
                    st.rerun()
                except requests.RequestException as exc:
                    st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/{selected_ep_id}", exc))

st.divider()
_show_single_delete(selected_ep_id)
_show_purge_section()

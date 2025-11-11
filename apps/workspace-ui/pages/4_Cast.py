"""CAST management page for show/season cast members and facebank seeds."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402

cfg = helpers.init_page("Cast")
st.title("Cast Management")
st.caption(f"Backend: {cfg['backend']} · Bucket: {cfg.get('bucket') or 'n/a'}")

# Inject thumbnail CSS
helpers.inject_thumb_css()


def _api_get(path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
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


# Show and season filters
st.subheader("Filters")
cols = st.columns([2, 2, 1])

with cols[0]:
    # Show selector
    show_id = st.text_input("Show ID", value="RHOBH", key="cast_show_id", help="Show ID (e.g., RHOBH)")

with cols[1]:
    # Season selector
    season_options = ["All seasons"] + [f"S{i:02d}" for i in range(1, 20)]
    season_choice = st.selectbox(
        "Season",
        options=season_options,
        index=0,
        key="cast_season_filter",
    )
    season_filter = None if season_choice == "All seasons" else season_choice

with cols[2]:
    if st.button("Refresh", key="cast_refresh", use_container_width=True):
        st.rerun()

# Fetch cast list
if not show_id:
    st.info("Enter a Show ID to view cast")
    st.stop()

params = {}
if season_filter:
    params["season"] = season_filter

cast_resp = _api_get(f"/shows/{show_id}/cast", params=params)
if not cast_resp:
    st.stop()

cast_members = cast_resp.get("cast", [])

if not cast_members:
    st.info(f"No cast members found for {show_id}" + (f" in {season_filter}" if season_filter else ""))

    # Add new cast member button
    if st.button("Add Cast Member", key="cast_add_new"):
        st.session_state["cast_show_add_form"] = True
        st.rerun()

    if st.session_state.get("cast_show_add_form"):
        with st.form("add_cast_form"):
            st.subheader("Add New Cast Member")
            name = st.text_input("Name", key="new_cast_name")
            role = st.selectbox("Role", options=["main", "friend", "guest", "other"], key="new_cast_role")
            status = st.selectbox("Status", options=["active", "past", "inactive"], key="new_cast_status")
            aliases_text = st.text_input("Aliases (comma-separated)", key="new_cast_aliases")
            seasons_text = st.text_input("Seasons (comma-separated, e.g., S05,S06)", key="new_cast_seasons")

            cols = st.columns([1, 1])
            if cols[0].form_submit_button("Create"):
                aliases = [a.strip() for a in aliases_text.split(",") if a.strip()]
                seasons = [s.strip() for s in seasons_text.split(",") if s.strip()]

                payload = {
                    "name": name,
                    "role": role,
                    "status": status,
                    "aliases": aliases,
                    "seasons": seasons,
                }

                result = _api_post(f"/shows/{show_id}/cast", payload)
                if result:
                    st.success(f"Created cast member: {name}")
                    st.session_state.pop("cast_show_add_form", None)
                    st.rerun()

            if cols[1].form_submit_button("Cancel"):
                st.session_state.pop("cast_show_add_form", None)
                st.rerun()

    st.stop()

# Display cast list and detail
st.divider()
st.subheader(f"Cast Members ({len(cast_members)})")

# Add new cast member button
if st.button("Add Cast Member", key="cast_add_new_btn"):
    st.session_state["cast_show_add_form"] = True
    st.rerun()

# Cast list with selection
selected_cast_id = st.session_state.get("selected_cast_id")

# Search/filter
search = st.text_input("Search cast", key="cast_search", placeholder="Name or alias...")
if search:
    search_lower = search.lower()
    cast_members = [
        m for m in cast_members
        if search_lower in m["name"].lower()
        or any(search_lower in alias.lower() for alias in m.get("aliases", []))
    ]

# Cast list
for member in cast_members:
    cast_id = member["cast_id"]
    name = member["name"]
    role = member.get("role", "other")
    status = member.get("status", "active")
    seasons = member.get("seasons", [])
    aliases = member.get("aliases", [])

    # Build display string
    display_parts = [f"**{name}**", role.upper()]
    if aliases:
        display_parts.append(f"({', '.join(aliases[:2])})")
    if seasons:
        display_parts.append(f"Seasons: {', '.join(seasons)}")

    status_emoji = "✅" if status == "active" else ("⏸️" if status == "past" else "❌")
    display = f"{status_emoji} {' · '.join(display_parts)}"

    if st.button(display, key=f"cast_select_{cast_id}", use_container_width=True):
        st.session_state["selected_cast_id"] = cast_id
        st.rerun()

# Cast detail panel
if selected_cast_id:
    st.divider()

    # Fetch cast member detail
    member_resp = _api_get(f"/shows/{show_id}/cast/{selected_cast_id}")
    if not member_resp:
        st.session_state.pop("selected_cast_id", None)
        st.rerun()

    member = member_resp

    st.subheader(f"Cast Detail: {member['name']}")

    # Basic info
    info_cols = st.columns([2, 1, 1])
    with info_cols[0]:
        st.markdown(f"**Name:** {member['name']}")
        if member.get("aliases"):
            st.caption(f"Aliases: {', '.join(member['aliases'])}")
    with info_cols[1]:
        st.markdown(f"**Role:** {member['role']}")
        st.markdown(f"**Status:** {member['status']}")
    with info_cols[2]:
        if member.get("seasons"):
            st.markdown(f"**Seasons:** {', '.join(member['seasons'])}")

    # Facebank section
    st.markdown("---")
    st.markdown("### Facebank")

    # Fetch facebank
    facebank_params = {"show_id": show_id}
    facebank_resp = _api_get(f"/cast/{selected_cast_id}/facebank", params=facebank_params)

    if facebank_resp:
        seeds = facebank_resp.get("seeds", [])
        stats = facebank_resp.get("stats", {})

        # Stats chips
        stat_cols = st.columns(3)
        stat_cols[0].metric("Seed Images", stats.get("total_seeds", 0))
        stat_cols[1].metric("Exemplars", stats.get("total_exemplars", 0))
        stat_cols[2].caption(f"Updated: {stats.get('updated_at', 'never')[:10]}")

        # Seed images grid
        if seeds:
            st.markdown("**Seed Images**")
            # Display seeds in 200×250 thumbnail grid
            cols_per_row = 5
            for row_start in range(0, len(seeds), cols_per_row):
                row_seeds = seeds[row_start : row_start + cols_per_row]
                row_cols = st.columns(cols_per_row)
                for idx, seed in enumerate(row_seeds):
                    with row_cols[idx]:
                        image_uri = seed.get("image_uri")
                        fb_id = seed.get("fb_id")
                        resolved_thumb = helpers.resolve_thumb(image_uri)
                        st.markdown(helpers.thumb_html(resolved_thumb, alt=f"Seed {fb_id}"), unsafe_allow_html=True)
                        st.caption(f"Seed {fb_id[:8]}...")

        # Upload seeds button
        if st.button("Upload Seed Images", key=f"cast_upload_seeds_{selected_cast_id}"):
            st.session_state["cast_show_upload_form"] = True
            st.rerun()

        # Upload form
        if st.session_state.get("cast_show_upload_form"):
            with st.form("upload_seeds_form"):
                st.markdown("**Upload Seed Images**")
                st.caption("Images must contain exactly 1 face. Supported: JPG, PNG")

                uploaded_files = st.file_uploader(
                    "Choose images",
                    type=["jpg", "jpeg", "png"],
                    accept_multiple_files=True,
                    key="cast_upload_files",
                )

                cols = st.columns([1, 1])
                if cols[0].form_submit_button("Upload"):
                    if not uploaded_files:
                        st.error("No files selected")
                    else:
                        # Upload via API
                        base = cfg["api_base"]
                        url = f"{base}/cast/{selected_cast_id}/seeds/upload?show_id={show_id}"

                        files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]

                        try:
                            resp = requests.post(url, files=files, timeout=120)
                            resp.raise_for_status()
                            result = resp.json()

                            uploaded_count = result.get("uploaded", 0)
                            failed_count = result.get("failed", 0)

                            if uploaded_count > 0:
                                st.success(f"Uploaded {uploaded_count} seed(s)")
                            if failed_count > 0:
                                st.warning(f"{failed_count} file(s) failed")
                                for error in result.get("errors", []):
                                    st.error(f"{error['file']}: {error['error']}")

                            st.session_state.pop("cast_show_upload_form", None)
                            st.rerun()

                        except requests.RequestException as exc:
                            st.error(f"Upload failed: {exc}")

                if cols[1].form_submit_button("Cancel"):
                    st.session_state.pop("cast_show_upload_form", None)
                    st.rerun()

    # Actions
    st.markdown("---")
    st.markdown("### Actions")

    action_cols = st.columns([1, 1, 1])

    with action_cols[0]:
        if st.button("Edit Cast Member", key=f"cast_edit_{selected_cast_id}"):
            st.info("Edit functionality coming soon")

    with action_cols[1]:
        if st.button("View Detections", key=f"cast_view_detections_{selected_cast_id}"):
            st.info("Navigation to Episode Review coming soon")

    with action_cols[2]:
        if st.button("Delete Cast Member", key=f"cast_delete_{selected_cast_id}", type="secondary"):
            if st.session_state.get(f"confirm_delete_{selected_cast_id}"):
                result = _api_delete(f"/shows/{show_id}/cast/{selected_cast_id}")
                if result:
                    st.success("Cast member deleted")
                    st.session_state.pop("selected_cast_id", None)
                    st.session_state.pop(f"confirm_delete_{selected_cast_id}", None)
                    st.rerun()
            else:
                st.session_state[f"confirm_delete_{selected_cast_id}"] = True
                st.warning("Click Delete again to confirm")
                st.rerun()

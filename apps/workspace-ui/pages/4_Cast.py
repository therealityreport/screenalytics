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

SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "avif", "heic", "heif"]
SUPPORTED_IMAGE_DESC = "Supported: JPG, PNG, WebP, AVIF, HEIC/HEIF"
SIMULATED_DETECTOR_WARNING = (
    "Seed uploads are using the simulated detector. Install RetinaFace (insightface + buffalo_l) for aligned crops."
)
_CAST_EDIT_FIELDS = ["name", "role", "status", "aliases", "seasons"]


def _reset_cast_edit_state() -> None:
    st.session_state.pop("cast_edit_id", None)
    for field in _CAST_EDIT_FIELDS:
        st.session_state.pop(f"cast_edit_{field}", None)


def _prime_cast_edit_state(member: Dict[str, Any]) -> None:
    st.session_state["cast_edit_id"] = member["cast_id"]
    st.session_state["cast_edit_name"] = member.get("name", "")
    st.session_state["cast_edit_role"] = member.get("role", "other")
    st.session_state["cast_edit_status"] = member.get("status", "active")
    st.session_state["cast_edit_aliases"] = ", ".join(member.get("aliases", []))
    st.session_state["cast_edit_seasons"] = ", ".join(member.get("seasons", []))


def _parse_csv_field(raw: str) -> List[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def _render_cast_edit_form(member: Dict[str, Any], show_id: str) -> None:
    cast_id = member["cast_id"]
    if "cast_edit_name" not in st.session_state:
        _prime_cast_edit_state(member)

    st.markdown("**Edit Cast Member**")
    st.caption("Update metadata and click Save to persist changes.")
    with st.form(f"cast_edit_form_{cast_id}"):
        name = st.text_input("Name*", key="cast_edit_name")
        form_cols = st.columns(2)
        with form_cols[0]:
            role = st.selectbox("Role", options=["main", "friend", "guest", "other"], key="cast_edit_role")
        with form_cols[1]:
            status = st.selectbox("Status", options=["active", "past", "inactive"], key="cast_edit_status")
        aliases_raw = st.text_input("Aliases (comma-separated)", key="cast_edit_aliases")
        seasons_raw = st.text_input("Seasons (comma-separated)", key="cast_edit_seasons")

        button_cols = st.columns([1, 1])
        submit = button_cols[0].form_submit_button("Save changes", type="primary")
        cancel = button_cols[1].form_submit_button("Cancel")

        if cancel:
            _reset_cast_edit_state()
            st.rerun()

        if submit:
            if not name.strip():
                st.error("Name is required.")
            else:
                payload = {
                    "name": name.strip(),
                    "role": role,
                    "status": status,
                    "aliases": _parse_csv_field(aliases_raw),
                    "seasons": _parse_csv_field(seasons_raw),
                }
                result = _api_patch(f"/shows/{show_id}/cast/{cast_id}", payload)
                if result:
                    st.success("Cast member updated.")
                    _reset_cast_edit_state()
                    st.session_state["selected_cast_id"] = cast_id
                    st.rerun()


def _mark_featured_seed(show_id: str, cast_id: str, seed_id: str) -> None:
    path = f"/cast/{cast_id}/seeds/{seed_id}/feature?show_id={show_id}"
    resp = _api_post(path)
    if resp:
        st.success("Featured image updated.")
        st.rerun()


def _delete_seed(show_id: str, cast_id: str, seed_id: str) -> None:
    """Delete a single seed from the facebank."""
    path = f"/cast/{cast_id}/seeds?show_id={show_id}"
    payload = {"seed_ids": [seed_id]}
    resp = _api_delete(path, payload)
    if resp:
        deleted = resp.get("deleted", 0)
        if deleted > 0:
            st.success(f"Deleted seed {seed_id[:8]}...")
        else:
            st.warning("Seed not found or already deleted")
        st.rerun()


def _resolve_api_url(url: str | None) -> str | None:
    """Convert relative API URLs to full URLs, or return HTTP(S) URLs as-is."""
    if not url:
        return None
    # If it's already a full URL or data URL, return as-is
    if url.startswith(("http://", "https://", "data:")):
        return url
    # If it's a relative API path, prepend the API base
    if url.startswith("/"):
        api_base = cfg.get("api_base") or st.session_state.get("api_base") or "http://localhost:8000"
        return f"{api_base}{url}"
    return url


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


def _api_patch(path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    base = st.session_state.get("api_base")
    if not base:
        st.error("API base URL missing; re-run init_page().")
        return None
    try:
        resp = requests.patch(f"{base}{path}", json=payload or {}, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(helpers.describe_error(f"{base}{path}", exc))
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


def _warn_if_simulated(payload: Dict[str, Any] | None) -> None:
    if not payload:
        return
    detector = (payload.get("detector") or "").lower()
    if detector and detector != "retinaface":
        message = payload.get("detector_message") or SIMULATED_DETECTOR_WARNING
        st.warning(message)


# Show creation form
with st.expander("➕ Add New Show", expanded=False):
    st.caption("Create a show slug before adding cast members. Use lowercase letters/numbers/dashes (e.g., rhobh).")
    with st.form("cast_new_show_form"):
        new_show_id = st.text_input("Show slug*", key="cast_new_show_slug", placeholder="e.g., rhobh")
        new_show_title = st.text_input("Display name", key="cast_new_show_title", placeholder="e.g., RHOBH")
        new_show_full_name = st.text_input("Full name", key="cast_new_show_full_name", placeholder="e.g., The Real Housewives of Beverly Hills")
        new_show_imdb = st.text_input("IMDb Series ID", key="cast_new_show_imdb", placeholder="e.g., tt1720601")
        create_show = st.form_submit_button("Create show", type="primary")

        if create_show:
            slug = (new_show_id or "").strip()
            title = (new_show_title or "").strip()
            full_name = (new_show_full_name or "").strip()
            imdb_series_id = (new_show_imdb or "").strip()
            if not slug:
                st.error("Show slug is required")
            else:
                payload = {
                    "show_id": slug,
                    "title": title or None,
                    "full_name": full_name or None,
                    "imdb_series_id": imdb_series_id or None,
                }
                result = _api_post("/shows", payload)
                if result:
                    show_slug = result.get("show_id", slug)
                    helpers.remember_custom_show(show_slug)
                    st.success(f"Registered show {show_slug}")
                    st.session_state["cast_show_select"] = show_slug
                    st.rerun()

# Show and season filters
st.subheader("Filters")
show_options = helpers.known_shows()
if not show_options:
    st.info("No shows available yet. Add a show via the Upload page first.")
    st.stop()
cols = st.columns([3, 2, 1])

with cols[0]:
    show_id = st.selectbox(
        "Show",
        options=show_options,
        key="cast_show_select",
        help="Select a show to manage cast and seed faces",
    )
    prev_show = st.session_state.get("cast_active_show")
    if prev_show and prev_show != show_id:
        _reset_cast_edit_state()
        st.session_state.pop("selected_cast_id", None)
    st.session_state["cast_active_show"] = show_id

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

params = {}
if season_filter:
    params["season"] = season_filter

cast_resp = _api_get(f"/shows/{show_id}/cast", params=params)
if not cast_resp:
    st.stop()

cast_members = cast_resp.get("cast", [])

# Check if we're in add form mode
show_add_form = st.session_state.get("cast_show_add_form") and not st.session_state.get("selected_cast_id")

if not cast_members and not show_add_form:
    st.info(f"No cast members found for {show_id}" + (f" in {season_filter}" if season_filter else ""))

    # Add new cast member button
    if st.button("Add Cast Member", key="cast_add_new"):
        st.session_state["cast_show_add_form"] = True
        st.session_state.pop("selected_cast_id", None)
        st.rerun()

    st.stop()

# Display cast list and detail
st.divider()
st.subheader(f"Cast Members ({len(cast_members)})" if cast_members else "Cast Members")

# Add new cast member button
if st.button("Add Cast Member", key="cast_add_new_btn"):
    st.session_state["cast_show_add_form"] = True
    st.session_state.pop("selected_cast_id", None)
    st.rerun()

# Add cast member form (appears when button is clicked)
if show_add_form:
    with st.container(border=True):
        st.subheader("➕ Add New Cast Member")

        # Step 1: Cast member details
        if not st.session_state.get("new_cast_created"):
            with st.form("add_cast_details_form"):
                st.markdown("**Cast Member Information**")
                name = st.text_input("Name*", key="new_cast_name", placeholder="e.g., Kyle Richards")

                col1, col2 = st.columns(2)
                with col1:
                    role = st.selectbox("Role", options=["main", "friend", "guest", "other"], key="new_cast_role")
                with col2:
                    status = st.selectbox("Status", options=["active", "past", "inactive"], key="new_cast_status")

                aliases_text = st.text_input(
                    "Aliases (comma-separated)",
                    key="new_cast_aliases",
                    placeholder="e.g., Kyle, Kyle R"
                )
                seasons_text = st.text_input(
                    "Seasons (comma-separated)",
                    key="new_cast_seasons",
                    placeholder="e.g., S01,S02,S03"
                )

                cols = st.columns([1, 1])
                if cols[0].form_submit_button("Create Cast Member", type="primary"):
                    if not name or not name.strip():
                        st.error("Name is required")
                    else:
                        aliases = [a.strip() for a in aliases_text.split(",") if a.strip()]
                        seasons = [s.strip() for s in seasons_text.split(",") if s.strip()]

                        payload = {
                            "name": name.strip(),
                            "role": role,
                            "status": status,
                            "aliases": aliases,
                            "seasons": seasons,
                        }

                        result = _api_post(f"/shows/{show_id}/cast", payload)
                        if result:
                            st.session_state["new_cast_created"] = result["cast_id"]
                            st.session_state["new_cast_name"] = name.strip()
                            st.rerun()

                if cols[1].form_submit_button("Cancel"):
                    st.session_state.pop("cast_show_add_form", None)
                    st.rerun()

        # Step 2: Upload seed images (optional)
        else:
            new_cast_id = st.session_state["new_cast_created"]
            new_cast_name = st.session_state.get("new_cast_name", "new member")

            st.success(f"✓ Created cast member: **{new_cast_name}**")
            st.markdown("**Optional: Upload Seed Images**")
            st.caption("Upload photos/portraits of this person to improve face recognition. Images must contain exactly 1 face.")

            with st.form("upload_seeds_after_create_form"):
                uploaded_files = st.file_uploader(
                    "Choose seed images",
                    type=SUPPORTED_IMAGE_TYPES,
                    accept_multiple_files=True,
                    key="new_cast_upload_files",
                    help=f"Select one or more photos ({SUPPORTED_IMAGE_DESC})"
                )

                cols = st.columns([1, 1, 1, 1])
                if cols[0].form_submit_button("Upload Seeds", type="primary"):
                    if uploaded_files:
                        # Upload via API
                        base = cfg["api_base"]
                        url = f"{base}/cast/{new_cast_id}/seeds/upload?show_id={show_id}"
                        files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]

                        try:
                            resp = requests.post(url, files=files, timeout=120)
                            resp.raise_for_status()
                            result = resp.json()
                            _warn_if_simulated(result)

                            uploaded_count = result.get("uploaded", 0)
                            failed_count = result.get("failed", 0)

                            if uploaded_count > 0:
                                st.success(f"Uploaded {uploaded_count} seed(s)")
                            if failed_count > 0:
                                st.warning(f"{failed_count} file(s) failed")
                                for error in result.get("errors", []):
                                    st.error(f"{error['file']}: {error['error']}")

                            # Clean up and view the new cast member
                            st.session_state.pop("cast_show_add_form", None)
                            st.session_state.pop("new_cast_created", None)
                            st.session_state.pop("new_cast_name", None)
                            st.session_state["selected_cast_id"] = new_cast_id
                            st.rerun()

                        except requests.RequestException as exc:
                            st.error(f"Upload failed: {exc}")
                    else:
                        st.warning("No files selected")

                if cols[1].form_submit_button("Add Another"):
                    # Reset form to add another cast member
                    st.session_state.pop("new_cast_created", None)
                    st.session_state.pop("new_cast_name", None)
                    # Keep cast_show_add_form True to stay in add mode
                    st.rerun()

                if cols[2].form_submit_button("View Member"):
                    # Skip upload and view the new member
                    st.session_state.pop("cast_show_add_form", None)
                    st.session_state.pop("new_cast_created", None)
                    st.session_state.pop("new_cast_name", None)
                    st.session_state["selected_cast_id"] = new_cast_id
                    st.rerun()

                if cols[3].form_submit_button("Done"):
                    # Cancel and return to list
                    st.session_state.pop("cast_show_add_form", None)
                    st.session_state.pop("new_cast_created", None)
                    st.session_state.pop("new_cast_name", None)
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

# Cast gallery view
cols_per_row = 3
for row_start in range(0, len(cast_members), cols_per_row):
    row_members = cast_members[row_start : row_start + cols_per_row]
    row_cols = st.columns(cols_per_row)

    for idx, member in enumerate(row_members):
        with row_cols[idx]:
            cast_id = member["cast_id"]
            name = member["name"]
            role = member.get("role", "other")
            status = member.get("status", "active")
            seasons = member.get("seasons", [])

            # Fetch facebank for featured image
            facebank_resp = _api_get(f"/cast/{cast_id}/facebank?show_id={show_id}")
            featured_seed_id = facebank_resp.get("featured_seed_id") if facebank_resp else None
            featured_seed = None
            if facebank_resp and featured_seed_id:
                seeds = facebank_resp.get("seeds", [])
                featured_seed = next((seed for seed in seeds if seed.get("fb_id") == featured_seed_id), None)

            # Get featured image URL
            image_url = None
            if featured_seed:
                image_url = _resolve_api_url(helpers.seed_display_source(featured_seed))
                resolved_thumb = helpers.resolve_thumb(image_url)
            else:
                resolved_thumb = None

            # Render 4:5 portrait frame
            thumb_markup = helpers.thumb_html(resolved_thumb, alt=name, hide_if_missing=False)
            st.markdown(thumb_markup, unsafe_allow_html=True)

            # Name and seasons
            status_emoji = "✅" if status == "active" else ("⏸️" if status == "past" else "❌")
            st.markdown(f"**{status_emoji} {name}**")
            if seasons:
                st.caption(f"Seasons: {', '.join(seasons)}")
            else:
                st.caption(role.upper())

            # Select button
            if st.button("View Details", key=f"cast_select_{cast_id}", use_container_width=True):
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

    if st.session_state.get("cast_edit_id") and st.session_state["cast_edit_id"] != selected_cast_id:
        _reset_cast_edit_state()

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

    edit_mode = st.session_state.get("cast_edit_id") == selected_cast_id
    if edit_mode:
        _render_cast_edit_form(member, show_id)

    # Facebank section
    st.markdown("---")
    st.markdown("### Facebank")

    # Fetch facebank
    facebank_params = {"show_id": show_id}
    facebank_resp = _api_get(f"/cast/{selected_cast_id}/facebank", params=facebank_params)

    if facebank_resp:
        seeds = facebank_resp.get("seeds", [])
        stats = facebank_resp.get("stats", {})
        featured_seed_id = facebank_resp.get("featured_seed_id")
        similarity_stats = facebank_resp.get("similarity") or {}
        summary_stats = similarity_stats.get("summary") if isinstance(similarity_stats, dict) else None
        avg_similarity = summary_stats.get("mean") if isinstance(summary_stats, dict) else None
        per_seed_similarity = similarity_stats.get("per_seed") if isinstance(similarity_stats, dict) else {}

        # Stats chips
        stat_cols = st.columns(4)
        stat_cols[0].metric("Seed Images", stats.get("total_seeds", 0))
        stat_cols[1].metric("Exemplars", stats.get("total_exemplars", 0))
        if avg_similarity is not None:
            stat_cols[2].metric("Avg Similarity", f"{int(round(avg_similarity * 100))}%")
        else:
            stat_cols[2].metric("Avg Similarity", "n/a")
        updated_label = stats.get("updated_at", "never") or "never"
        stat_cols[3].caption(f"Updated: {updated_label[:10]}")

        featured_seed = next((seed for seed in seeds if seed.get("fb_id") == featured_seed_id), None)
        if featured_seed:
            st.markdown("**⭐ Featured Image**")
            image_uri = _resolve_api_url(helpers.seed_display_source(featured_seed))
            featured_thumb = helpers.resolve_thumb(image_uri)
            if featured_thumb:
                st.image(featured_thumb, width=220)
            st.caption("Select another seed below to change the featured image.")

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
                        image_uri = _resolve_api_url(helpers.seed_display_source(seed))
                        fb_id = seed.get("fb_id")
                        resolved_thumb = helpers.resolve_thumb(image_uri)
                        st.markdown(helpers.thumb_html(resolved_thumb, alt=f"Seed {fb_id}"), unsafe_allow_html=True)
                        if seed.get("featured"):
                            st.caption("⭐ Featured")
                        else:
                            st.caption(f"Seed {fb_id[:8]}...")
                            if st.button("☆ Feature", key=f"feature_seed_{fb_id}"):
                                _mark_featured_seed(show_id, selected_cast_id, fb_id)
                        sim_info = per_seed_similarity.get(fb_id) if isinstance(per_seed_similarity, dict) else None
                        if isinstance(sim_info, dict) and sim_info.get("mean") is not None:
                            sim_pct = int(round(sim_info["mean"] * 100))
                            st.caption(f"{sim_pct}% avg similarity")

                        # Delete button for all seeds (including featured)
                        confirm_key = f"confirm_delete_seed_{fb_id}"
                        if st.button("🗑️ Delete", key=f"delete_seed_{fb_id}", type="secondary"):
                            if st.session_state.get(confirm_key):
                                _delete_seed(show_id, selected_cast_id, fb_id)
                                st.session_state.pop(confirm_key, None)
                            else:
                                st.session_state[confirm_key] = True
                                st.rerun()

                        if st.session_state.get(confirm_key):
                            st.caption("⚠️ Click again to confirm")

        # Upload seeds button
        if st.button("Upload Seed Images", key=f"cast_upload_seeds_{selected_cast_id}"):
            st.session_state["cast_show_upload_form"] = True
            st.rerun()

        # Upload form
        if st.session_state.get("cast_show_upload_form"):
            with st.form("upload_seeds_form"):
                st.markdown("**Upload Seed Images**")
                st.caption(f"Images must contain exactly 1 face. {SUPPORTED_IMAGE_DESC}")

                uploaded_files = st.file_uploader(
                    "Choose images",
                    type=SUPPORTED_IMAGE_TYPES,
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
                            _warn_if_simulated(result)

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
        if edit_mode:
            st.button("Editing…", key=f"cast_edit_disabled_{selected_cast_id}", disabled=True)
        else:
            if st.button("Edit Cast Member", key=f"cast_edit_{selected_cast_id}"):
                _prime_cast_edit_state(member)
                st.rerun()

    with action_cols[1]:
        if st.button("View Detections", key=f"cast_view_detections_{selected_cast_id}"):
            # Store cast filter in session state for Faces Review page
            st.session_state["filter_cast_id"] = selected_cast_id
            st.session_state["filter_cast_name"] = member["name"]
            st.switch_page("pages/3_Faces_Review.py")

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

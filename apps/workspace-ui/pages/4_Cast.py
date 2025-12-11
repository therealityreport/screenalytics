"""CAST management page for show/season cast members and facebank seeds."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402
from similarity_badges import SimilarityType, render_similarity_badge  # noqa: E402

cfg = helpers.init_page("Cast")

# Cast View header
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, rgba(156, 39, 176, 0.25), rgba(33, 150, 243, 0.25));
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    ">
        <span style="font-size: 24px; font-weight: 600; color: #000;">🎭 Cast View</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("Cast Management")
st.caption(f"Backend: {cfg['backend']} · Bucket: {cfg.get('bucket') or 'n/a'}")

# Inject thumbnail CSS
helpers.inject_thumb_css()

SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "avif", "heic", "heif"]
SUPPORTED_IMAGE_DESC = "Supported: JPG, PNG, WebP, AVIF, HEIC/HEIF"
SIMULATED_DETECTOR_WARNING = (
    "Seed uploads are using the simulated detector. Install RetinaFace (insightface + buffalo_l) for aligned crops."
)
_CAST_EDIT_FIELDS = ["name", "full_name", "role", "status", "aliases", "seasons", "imdb_id", "social_instagram", "social_twitter"]


def _reset_cast_edit_state() -> None:
    st.session_state.pop("cast_edit_id", None)
    for field in _CAST_EDIT_FIELDS:
        st.session_state.pop(f"cast_edit_{field}", None)


def _prime_cast_edit_state(member: Dict[str, Any]) -> None:
    st.session_state["cast_edit_id"] = member["cast_id"]
    st.session_state["cast_edit_name"] = member.get("name", "")
    st.session_state["cast_edit_full_name"] = member.get("full_name", "") or ""
    st.session_state["cast_edit_role"] = member.get("role", "other")
    st.session_state["cast_edit_status"] = member.get("status", "active")
    st.session_state["cast_edit_aliases"] = ", ".join(member.get("aliases", []))
    st.session_state["cast_edit_seasons"] = ", ".join(member.get("seasons", []))
    st.session_state["cast_edit_imdb_id"] = member.get("imdb_id", "") or ""
    social = member.get("social", {}) or {}
    st.session_state["cast_edit_social_instagram"] = social.get("instagram", "") or ""
    st.session_state["cast_edit_social_twitter"] = social.get("twitter", "") or ""


def _parse_csv_field(raw: str) -> List[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def _render_cast_edit_form(member: Dict[str, Any], show_id: str) -> None:
    cast_id = member["cast_id"]
    if "cast_edit_name" not in st.session_state:
        _prime_cast_edit_state(member)

    st.markdown("### Edit Cast Member")
    st.caption("Update cast member information and click Save to persist changes.")
    with st.form(f"cast_edit_form_{cast_id}"):
        # Names section
        name_cols = st.columns(2)
        with name_cols[0]:
            name = st.text_input("Display Name*", key="cast_edit_name", help="Primary display name")
        with name_cols[1]:
            full_name = st.text_input("Full Name", key="cast_edit_full_name", help="Legal/full name")

        # Role and status
        role_cols = st.columns(2)
        with role_cols[0]:
            role = st.selectbox(
                "Role",
                options=["main", "friend", "guest", "other"],
                key="cast_edit_role",
            )
        with role_cols[1]:
            status = st.selectbox("Status", options=["active", "past", "inactive"], key="cast_edit_status")

        # Aliases and seasons
        aliases_raw = st.text_input(
            "Aliases (comma-separated)",
            key="cast_edit_aliases",
            help="Alternative names/nicknames for matching (e.g., Kyle, Kyle R)",
        )
        seasons_raw = st.text_input(
            "Seasons (comma-separated)",
            key="cast_edit_seasons",
            help="Season codes this person appears in (e.g., S01, S02, S03)",
        )

        # External IDs
        st.markdown("**External Links**")
        imdb_id = st.text_input(
            "IMDb ID",
            key="cast_edit_imdb_id",
            help="IMDb person identifier (e.g., nm0000001)",
            placeholder="nm0000001",
        )

        # Social media
        social_cols = st.columns(2)
        with social_cols[0]:
            instagram = st.text_input(
                "Instagram Handle",
                key="cast_edit_social_instagram",
                placeholder="@username",
            )
        with social_cols[1]:
            twitter = st.text_input(
                "Twitter/X Handle",
                key="cast_edit_social_twitter",
                placeholder="@username",
            )

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
                # Build social dict only with non-empty values
                social = {}
                if instagram and instagram.strip():
                    social["instagram"] = instagram.strip().lstrip("@")
                if twitter and twitter.strip():
                    social["twitter"] = twitter.strip().lstrip("@")

                payload = {
                    "name": name.strip(),
                    "full_name": full_name.strip() if full_name else None,
                    "role": role,
                    "status": status,
                    "aliases": _parse_csv_field(aliases_raw),
                    "seasons": _parse_csv_field(seasons_raw),
                    "imdb_id": imdb_id.strip() if imdb_id else None,
                    "social": social if social else None,
                }
                result = _api_patch(f"/shows/{show_id}/cast/{cast_id}", payload)
                if result:
                    st.success("Cast member updated.")
                    _reset_cast_edit_state()
                    time.sleep(0.5)
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


def _navigate_to_cast_detail(cast_id: str, show_id: str, ep_id: str | None = None) -> None:
    """Select a cast member within this page and persist context in query params."""
    params = dict(st.query_params)
    params["cast_id"] = cast_id
    if show_id:
        params["show_id"] = show_id
    if ep_id:
        params["ep_id"] = ep_id
    st.query_params = params
    st.session_state["selected_cast_id"] = cast_id


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
    base = cfg.get("api_base") or st.session_state.get("api_base")
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
    base = cfg.get("api_base") or st.session_state.get("api_base")
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


def _format_seed_upload_error(resp: requests.Response, exc: requests.HTTPError) -> str:
    detail: str | Dict[str, Any] | List[Any] | None = None
    try:
        payload = resp.json()
        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("message") or payload.get("error") or None
            if detail is None and payload:
                detail = json.dumps(payload)
        else:
            detail = payload
    except (json.JSONDecodeError, ValueError):
        detail = resp.text or None
    if isinstance(detail, (dict, list)):
        detail = json.dumps(detail)
    message = detail or helpers.describe_error(resp.url or "upload", exc)
    return f"Upload failed: {message}"


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
        new_show_full_name = st.text_input(
            "Full name",
            key="cast_new_show_full_name",
            placeholder="e.g., The Real Housewives of Beverly Hills",
        )
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

# If ep_id is provided in the query string, default the Cast page to that episode's show
ep_id_param = helpers.get_ep_id_from_query_params()
ep_show_slug = None
if ep_id_param:
    episode_detail = _api_get(f"/episodes/{ep_id_param}")
    if episode_detail and isinstance(episode_detail, dict):
        ep_show_slug = (episode_detail.get("show_slug") or episode_detail.get("show") or "").strip().lower()

# Normalize show options for lookup and set session default from ep_id when available
option_lookup = {opt.lower(): opt for opt in show_options}
if ep_show_slug and ep_show_slug in option_lookup:
    st.session_state["cast_show_select"] = option_lookup[ep_show_slug]
show_id_param = (st.query_params.get("show_id") or "").lower() if "show_id" in st.query_params else ""
if show_id_param and show_id_param in option_lookup:
    st.session_state["cast_show_select"] = option_lookup[show_id_param]
cast_id_param = st.query_params.get("cast_id") if "cast_id" in st.query_params else ""
if cast_id_param:
    st.session_state["selected_cast_id"] = cast_id_param

# Ensure session state value is valid
selected_show_value = st.session_state.get("cast_show_select")
if selected_show_value not in show_options:
    selected_show_value = show_options[0]
    st.session_state["cast_show_select"] = selected_show_value

cols = st.columns([3, 2, 1])

with cols[0]:
    show_id = st.selectbox(
        "Show",
        options=show_options,
        index=show_options.index(selected_show_value),
        key="cast_show_select",
        help="Select a show to manage cast and seed faces",
    )
    prev_show = st.session_state.get("cast_active_show")
    if prev_show and prev_show != show_id:
        _reset_cast_edit_state()
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
show_add_form = st.session_state.get("cast_show_add_form")

if not cast_members and not show_add_form:
    st.info(f"No cast members found for {show_id}" + (f" in {season_filter}" if season_filter else ""))

    # Add new cast member button
    if st.button("Add Cast Member", key="cast_add_new"):
        st.session_state["cast_show_add_form"] = True
        st.rerun()

    st.stop()

# Display cast list and detail
st.divider()
st.subheader(f"Cast Members ({len(cast_members)})" if cast_members else "Cast Members")

# Add new cast member button
if st.button("Add Cast Member", key="cast_add_new_btn"):
    st.session_state["cast_show_add_form"] = True
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
                    role = st.selectbox(
                        "Role",
                        options=["main", "friend", "guest", "other"],
                        key="new_cast_role",
                    )
                with col2:
                    status = st.selectbox(
                        "Status",
                        options=["active", "past", "inactive"],
                        key="new_cast_status",
                    )

                aliases_text = st.text_input(
                    "Aliases (comma-separated)",
                    key="new_cast_aliases",
                    placeholder="e.g., Kyle, Kyle R",
                )
                seasons_text = st.text_input(
                    "Seasons (comma-separated)",
                    key="new_cast_seasons",
                    placeholder="e.g., S01,S02,S03",
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
                            st.session_state["new_cast_name_stored"] = name.strip()
                            st.rerun()

                if cols[1].form_submit_button("Cancel"):
                    st.session_state.pop("cast_show_add_form", None)
                    st.rerun()

        # Step 2: Upload seed images (optional)
        else:
            new_cast_id = st.session_state.get("new_cast_created")
            if not new_cast_id:
                # State was unexpectedly cleared - return to step 1
                st.warning("Cast creation state was lost. Please create the cast member again.")
                st.session_state.pop("cast_show_add_form", None)
                st.rerun()
            new_cast_name = st.session_state.get("new_cast_name_stored", "new member")

            st.success(f"✓ Created cast member: **{new_cast_name}**")
            st.markdown("**Optional: Upload Seed Images**")
            st.caption(
                "Upload photos/portraits of this person to improve face recognition. Images must contain exactly 1 face."
            )

            with st.form("upload_seeds_after_create_form"):
                uploaded_files = st.file_uploader(
                    "Choose seed images",
                    type=SUPPORTED_IMAGE_TYPES,
                    accept_multiple_files=True,
                    key="new_cast_upload_files",
                    help=f"Select one or more photos ({SUPPORTED_IMAGE_DESC})",
                )

                cols = st.columns([1, 1, 1, 1])
                if cols[0].form_submit_button("Upload Seeds", type="primary"):
                    if uploaded_files:
                        # Upload via API
                        base = cfg["api_base"]
                        url = f"{base}/cast/{new_cast_id}/seeds/upload?show_id={show_id}"
                        files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]

                        try:
                            with st.spinner(f"Uploading {len(uploaded_files)} seed(s)..."):
                                resp = requests.post(url, files=files, timeout=120)
                            handled_error = False
                            try:
                                resp.raise_for_status()
                            except requests.HTTPError as exc:
                                if resp.status_code == 422:
                                    st.error(_format_seed_upload_error(resp, exc))
                                elif resp.status_code >= 500:
                                    st.error(f"Server error ({resp.status_code}): Please try again later.")
                                elif resp.status_code >= 400:
                                    st.error(f"Upload failed ({resp.status_code}): {exc}")
                                else:
                                    st.error(f"Unexpected error: {exc}")
                                handled_error = True

                            if not handled_error:
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
                                st.session_state.pop("new_cast_name_stored", None)
                                _navigate_to_cast_detail(new_cast_id, show_id, helpers.get_ep_id())

                        except requests.RequestException as exc:
                            st.error(helpers.describe_error(url, exc))
                    else:
                        st.warning("No files selected")

                if cols[1].form_submit_button("Add Another"):
                    # Reset form to add another cast member
                    st.session_state.pop("new_cast_created", None)
                    st.session_state.pop("new_cast_name_stored", None)
                    # Keep cast_show_add_form True to stay in add mode
                    st.rerun()

                if cols[2].form_submit_button("View Member"):
                    # Skip upload and view the new member
                    st.session_state.pop("cast_show_add_form", None)
                    st.session_state.pop("new_cast_created", None)
                    st.session_state.pop("new_cast_name_stored", None)
                    _navigate_to_cast_detail(new_cast_id, show_id, helpers.get_ep_id())

                if cols[3].form_submit_button("Done"):
                    # Cancel and return to list
                    st.session_state.pop("cast_show_add_form", None)
                    st.session_state.pop("new_cast_created", None)
                    st.session_state.pop("new_cast_name_stored", None)
                    st.rerun()

# Search/filter
search = st.text_input("Search cast", key="cast_search", placeholder="Name or alias...")
if search:
    search_lower = search.lower()
    cast_members = [
        m
        for m in cast_members
        if search_lower in m["name"].lower() or any(search_lower in alias.lower() for alias in m.get("aliases", []))
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
                featured_seed = next(
                    (seed for seed in seeds if seed.get("fb_id") == featured_seed_id),
                    None,
                )

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
            status_emoji = "🟢" if status == "active" else ("🔵" if status == "past" else "⚫")
            st.markdown(f"**{status_emoji} {name}**")
            if seasons:
                st.caption(f"Seasons: {', '.join(seasons)}")
            else:
                st.caption(role.upper())

            # Select button
            if st.button("View Details", key=f"cast_select_{cast_id}", use_container_width=True):
                _navigate_to_cast_detail(cast_id, show_id, helpers.get_ep_id())
                st.rerun()

selected_cast_id = st.session_state.get("selected_cast_id")
if not selected_cast_id and cast_id_param:
    selected_cast_id = cast_id_param
    st.session_state["selected_cast_id"] = selected_cast_id

if selected_cast_id:
    st.divider()

    member_resp = _api_get(f"/shows/{show_id}/cast/{selected_cast_id}")
    if not member_resp:
        st.session_state.pop("selected_cast_id", None)
        # Clear query params to prevent infinite loop on invalid cast_id
        params = dict(st.query_params)
        params.pop("cast_id", None)
        st.query_params = params
        st.rerun()
    member = member_resp

    st.subheader(f"Cast Detail: {member['name']}")

    info_cols = st.columns([2, 1, 1])
    with info_cols[0]:
        st.markdown(f"**Name:** {member['name']}")
        if member.get("aliases"):
            st.caption(f"Aliases: {', '.join(member['aliases'])}")
    with info_cols[1]:
        st.markdown(f"**Role:** {member.get('role', 'other')}")
        st.markdown(f"**Status:** {member.get('status', 'active')}")
    with info_cols[2]:
        if member.get("seasons"):
            st.markdown(f"**Seasons:** {', '.join(member['seasons'])}")

    st.markdown("---")
    st.markdown("### Facebank")

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

        stat_cols = st.columns(4)
        stat_cols[0].metric("Seed Images", stats.get("total_seeds", 0))
        stat_cols[1].metric("Exemplars", stats.get("total_exemplars", 0))
        if avg_similarity is not None:
            avg_badge = render_similarity_badge(avg_similarity, SimilarityType.CAST)
            stat_cols[2].markdown(f"**Avg Similarity** {avg_badge}", unsafe_allow_html=True)
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
                st.image(featured_thumb, width=helpers.THUMB_W)
            st.caption("Select another seed below to change the featured image.")

        if seeds:
            st.markdown("**Seed Images**")
            cols_per_row = 5
            for row_start in range(0, len(seeds), cols_per_row):
                row_seeds = seeds[row_start : row_start + cols_per_row]
                row_cols = st.columns(cols_per_row)
                for idx, seed in enumerate(row_seeds):
                    with row_cols[idx]:
                        image_uri = _resolve_api_url(helpers.seed_display_source(seed))
                        fb_id = seed.get("fb_id")
                        resolved_thumb = helpers.resolve_thumb(image_uri)
                        st.markdown(
                            helpers.thumb_html(resolved_thumb, alt=f"Seed {fb_id}"),
                            unsafe_allow_html=True,
                        )
                        if seed.get("featured"):
                            st.caption("⭐ Featured")
                        else:
                            st.caption(f"Seed {fb_id[:8]}...")
                            if st.button("☆ Feature", key=f"feature_seed_{fb_id}"):
                                _mark_featured_seed(show_id, selected_cast_id, fb_id)
                        sim_info = per_seed_similarity.get(fb_id) if isinstance(per_seed_similarity, dict) else None
                        if isinstance(sim_info, dict) and isinstance(sim_info.get("mean"), (int, float)):
                            sim_badge = render_similarity_badge(sim_info["mean"], SimilarityType.CAST)
                            st.markdown(sim_badge, unsafe_allow_html=True)

                        confirm_key = f"confirm_delete_seed_{fb_id}"
                        is_confirming = st.session_state.get(confirm_key, False)

                        if is_confirming:
                            st.caption("⚠️ Click again to confirm")

                        if st.button("🗑️ Delete", key=f"delete_seed_{fb_id}", type="secondary"):
                            if is_confirming:
                                st.session_state.pop(confirm_key, None)
                                _delete_seed(show_id, selected_cast_id, fb_id)
                            else:
                                st.session_state[confirm_key] = True
                                st.rerun()

        if st.button("Upload Seed Images", key=f"cast_upload_seeds_{selected_cast_id}"):
            st.session_state["cast_show_upload_form"] = True
            st.rerun()

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
                        base = cfg["api_base"]
                        url = f"{base}/cast/{selected_cast_id}/seeds/upload?show_id={show_id}"

                        files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]

                        try:
                            with st.spinner(f"Uploading {len(uploaded_files)} seed(s)..."):
                                resp = requests.post(url, files=files, timeout=120)
                            handled_error = False
                            try:
                                resp.raise_for_status()
                            except requests.HTTPError as exc:
                                if resp.status_code == 422:
                                    st.error(_format_seed_upload_error(resp, exc))
                                elif resp.status_code >= 500:
                                    st.error(f"Server error ({resp.status_code}): Please try again later.")
                                elif resp.status_code >= 400:
                                    st.error(f"Upload failed ({resp.status_code}): {exc}")
                                else:
                                    st.error(f"Unexpected error: {exc}")
                                handled_error = True

                            if not handled_error:
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
                            st.error(helpers.describe_error(url, exc))

                if cols[1].form_submit_button("Cancel"):
                    st.session_state.pop("cast_show_upload_form", None)
                    st.rerun()

    st.markdown("---")

    # Show edit form if in edit mode for this cast member
    if st.session_state.get("cast_edit_id") == selected_cast_id:
        _render_cast_edit_form(member, show_id)
    else:
        st.markdown("### Actions")

        action_cols = st.columns([1, 1, 1])

        with action_cols[0]:
            if st.button("Edit Cast Member", key=f"cast_edit_{selected_cast_id}"):
                _prime_cast_edit_state(member)
                st.rerun()

        with action_cols[1]:
            if st.button("View Detections", key=f"cast_view_detections_{selected_cast_id}"):
                st.session_state["filter_cast_id"] = selected_cast_id
                st.session_state["filter_cast_name"] = member["name"]
                st.switch_page("pages/3_Faces_Review.py")

        with action_cols[2]:
            confirm_delete_key = f"confirm_delete_{selected_cast_id}"
            is_delete_confirming = st.session_state.get(confirm_delete_key, False)

            if is_delete_confirming:
                st.warning("⚠️ Click Delete again to confirm")

            if st.button(
                "Delete Cast Member",
                key=f"cast_delete_{selected_cast_id}",
                type="secondary",
            ):
                if is_delete_confirming:
                    result = _api_delete(f"/shows/{show_id}/cast/{selected_cast_id}")
                    if result:
                        st.success("Cast member deleted")
                        st.session_state.pop("selected_cast_id", None)
                        st.session_state.pop(confirm_delete_key, None)
                        # Clear query params for clean URL
                        params = dict(st.query_params)
                        params.pop("cast_id", None)
                        st.query_params = params
                        st.rerun()
                else:
                    st.session_state[confirm_delete_key] = True
                    st.rerun()

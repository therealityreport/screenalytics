"""TRR Metadata viewer - Display canonical metadata from Postgres."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODULE_DIR = Path(__file__).resolve().parent.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

import ui_helpers as helpers  # noqa: E402

# Initialize page configuration
cfg = helpers.init_page("TRR Metadata")

st.title("📊 TRR Metadata (Postgres)")

st.markdown(
    """
This page displays **canonical metadata** from the TRR Postgres database (`core.*` tables).

Enter a show slug (e.g., `RHOSLC`, `RHOBH`) and click **Fetch from TRR DB** to retrieve:
- Show information
- Seasons
- Episodes
- Cast members

All data is **read-only** and sourced from the TRR BACKEND metadata database.
"""
)

st.divider()

# Input section
col1, col2 = st.columns([3, 1])

with col1:
    show_slug = st.text_input(
        "Show slug",
        value=st.session_state.get("trr_show_slug", "RHOSLC"),
        placeholder="e.g., RHOSLC, RHOBH",
        help="Enter the show slug to fetch metadata from TRR database",
    ).strip()

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    fetch_button = st.button("🔍 Fetch from TRR DB", type="primary", use_container_width=True)

# Fetch metadata when button is clicked
if fetch_button and show_slug:
    st.session_state["trr_show_slug"] = show_slug

    with st.spinner(f"Fetching metadata for {show_slug}..."):
        try:
            meta = helpers.fetch_trr_metadata(show_slug)
            st.session_state["trr_metadata"] = meta
            st.session_state["trr_metadata_error"] = None
            st.success(f"✅ Successfully fetched metadata for **{show_slug}**")
        except Exception as exc:
            st.session_state["trr_metadata"] = None
            st.session_state["trr_metadata_error"] = str(exc)
            st.error(f"❌ Failed to fetch metadata: {exc}")

# Display metadata if available
meta = st.session_state.get("trr_metadata")
error = st.session_state.get("trr_metadata_error")

if error and not meta:
    st.warning(
        f"Could not fetch metadata for `{show_slug}`. "
        "Ensure TRR_DB_URL is configured and the show exists in the database."
    )

if meta:
    st.divider()

    # Show section
    with st.expander("📺 Show", expanded=True):
        show = meta["show"]
        if show:
            col_a, col_b = st.columns(2)

            with col_a:
                st.metric("Show Slug", show.get("show_slug", "N/A"))
                st.metric("Title", show.get("title", "N/A"))
                st.metric("Franchise", show.get("franchise") or "N/A")

            with col_b:
                st.metric("Network", show.get("network") or "N/A")
                st.metric("Active", "✅ Yes" if show.get("is_active") else "❌ No")
                if show.get("imdb_series_id"):
                    st.caption(f"IMDb: {show.get('imdb_series_id')}")
                if show.get("tmdb_series_id"):
                    st.caption(f"TMDb: {show.get('tmdb_series_id')}")

            st.json(show, expanded=False)
        else:
            st.info("No show data available.")

    # Seasons & Episodes section
    with st.expander("📅 Seasons & Episodes", expanded=False):
        seasons = meta.get("seasons", [])
        episodes = meta.get("episodes", [])

        if seasons:
            st.subheader(f"Seasons ({len(seasons)})")
            st.dataframe(
                seasons,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No seasons found for this show.")

        st.divider()

        if episodes:
            st.subheader(f"Episodes ({len(episodes)})")
            st.dataframe(
                episodes,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No episodes found for this show.")

    # Cast section
    with st.expander("👥 Cast", expanded=False):
        cast = meta.get("cast", [])

        if cast:
            st.subheader(f"Cast Members ({len(cast)})")
            st.dataframe(
                cast,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No cast members found for this show yet.")

# Help section
with st.expander("ℹ️ About TRR Metadata", expanded=False):
    st.markdown(
        """
    ### What is TRR Metadata?

    TRR (The Reality Report) maintains a **canonical metadata database** in Postgres that stores:
    - Show information (title, network, IMDb/TMDb IDs)
    - Season and episode details
    - Cast member information

    SCREANALYTICS connects to this database in **read-only mode** to:
    - Display canonical metadata for shows
    - Validate episode data against official records
    - Provide cast member information for face recognition workflows

    ### Database Schema

    The metadata is stored in the `core` schema with these tables:
    - `core.shows` - Show information
    - `core.seasons` - Season records per show
    - `core.episodes` - Episode records per season
    - `core.cast` - Cast members per show

    All schema migrations and data updates are managed by **TRR BACKEND**.

    ### Configuration

    Set the `TRR_DB_URL` environment variable to connect:
    ```bash
    export TRR_DB_URL=postgresql://user:pass@host:5432/trr_metadata
    ```
    """
    )

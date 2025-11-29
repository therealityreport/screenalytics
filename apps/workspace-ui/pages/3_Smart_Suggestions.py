"""Smart Suggestions Page - Sub-page of Faces Review.

This page shows cast assignment suggestions for unassigned clusters,
allowing batch review and one-click accept/dismiss actions.

Each suggestion is displayed as its own row with up to 6 frame thumbnails.
"""

from __future__ import annotations

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

cfg = helpers.init_page("Smart Suggestions")
st.title("Smart Suggestions")

# Episode selector
sidebar_ep_id = None
try:
    sidebar_ep_id = helpers.render_sidebar_episode_selector()
except Exception:
    sidebar_ep_id = None

ep_id = helpers.get_ep_id()
if not ep_id and sidebar_ep_id:
    helpers.set_ep_id(sidebar_ep_id, rerun=False)
    ep_id = sidebar_ep_id
if not ep_id:
    ep_id_from_query = helpers.get_ep_id_from_query_params()
    if ep_id_from_query:
        helpers.set_ep_id(ep_id_from_query, rerun=False)
        ep_id = ep_id_from_query
if not ep_id:
    st.warning("Select an episode from the sidebar to continue.")
    st.stop()

# API helpers
_api_base = st.session_state.get("api_base", "http://localhost:8000")


def _safe_api_get(path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    """Fetch from API, return None on error."""
    try:
        resp = requests.get(f"{_api_base}{path}", params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _api_post(path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    """POST to API, return response or None on error."""
    try:
        resp = requests.post(f"{_api_base}{path}", json=payload or {}, timeout=30)
        if resp.status_code in (200, 201):
            return resp.json()
    except Exception:
        pass
    return None


def _api_patch(path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    """PATCH to API, return response or None on error."""
    try:
        resp = requests.patch(f"{_api_base}{path}", json=payload or {}, timeout=30)
        if resp.status_code in (200, 201):
            return resp.json()
    except Exception:
        pass
    return None


def _recompute_cast_suggestions(ep_id: str, show_slug: str | None) -> Dict[str, List[Dict[str, Any]]]:
    """Persist assignments and recompute cast suggestions from latest embeddings."""
    st.session_state.pop(f"cast_suggestions:{ep_id}", None)
    st.session_state.pop(f"dismissed_suggestions:{ep_id}", None)
    if show_slug:
        st.session_state.pop(f"people_cache:{show_slug}", None)
    st.session_state.pop(f"cluster_tracks:{ep_id}", None)
    st.session_state.pop(f"identities:{ep_id}", None)
    _api_post(f"/episodes/{ep_id}/save_assignments", {})

    cache_buster = int(time.time() * 1000)
    suggestions_resp = _safe_api_get(f"/episodes/{ep_id}/cast_suggestions", params={"_t": cache_buster})
    suggestions_map: Dict[str, List[Dict[str, Any]]] = {}
    if suggestions_resp:
        for entry in suggestions_resp.get("suggestions", []):
            cid = entry.get("cluster_id")
            if not cid:
                continue
            suggestions_map[cid] = entry.get("cast_suggestions", []) or []

    if suggestions_map:
        st.session_state[f"cast_suggestions:{ep_id}"] = suggestions_map
    return suggestions_map


def _fetch_people_cached(show_id: str) -> Dict[str, Any] | None:
    """Fetch people for a show (cached)."""
    cache_key = f"people_cache:{show_id}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = _safe_api_get(f"/shows/{show_id}/people")
    return st.session_state.get(cache_key)


# CSS for fixed 147x184px thumbnails (4:5 aspect ratio)
THUMB_CSS = """
<style>
/* Fixed size thumbnail frames */
.thumb-row {
    display: flex;
    gap: 8px;
    align-items: flex-start;
}
.thumb-frame {
    width: 147px;
    height: 184px;
    border-radius: 6px;
    overflow: hidden;
    background: #2a2a2a;
    flex-shrink: 0;
}
.thumb-frame img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
</style>
"""
st.markdown(THUMB_CSS, unsafe_allow_html=True)

# Get episode metadata
ep_meta = helpers.parse_ep_id(ep_id) or {}
show_slug = ep_meta.get("show_slug", ep_id.rsplit("-", 2)[0] if "-" in ep_id else ep_id)
season_label = None
season_value = ep_meta.get("season")
if isinstance(season_value, int):
    season_label = f"S{season_value:02d}"

# Fetch required data
cluster_payload = _safe_api_get(f"/episodes/{ep_id}/cluster_tracks") or {"clusters": []}
cast_api_resp = _safe_api_get(f"/shows/{show_slug}/cast" + (f"?season={season_label}" if season_label else ""))
people_resp = _fetch_people_cached(show_slug)
unlinked_resp = _safe_api_get(f"/episodes/{ep_id}/unlinked_entities")

# Build cluster lookup
cluster_lookup: Dict[str, Dict[str, Any]] = {}
for cluster in cluster_payload.get("clusters", []):
    identity_id = cluster.get("identity_id")
    if identity_id:
        cluster_lookup[identity_id] = cluster

# Build cast options
cast_options: Dict[str, str] = {}
cast_members_list: List[Dict[str, Any]] = []
if cast_api_resp:
    cast_members_list = cast_api_resp.get("cast", [])
    cast_options = {
        cm.get("cast_id"): cm.get("name")
        for cm in cast_members_list
        if cm.get("cast_id") and cm.get("name")
    }

# Identify auto-clustered people (people without cast_id)
people = people_resp.get("people", []) if people_resp else []
people_by_id = {p.get("person_id"): p for p in people if p.get("person_id")}
unlinked_entities = unlinked_resp.get("entities", []) if unlinked_resp else []
auto_clustered_people: List[Dict[str, Any]] = []
unlinked_cluster_ids: set[str] = set()
for entity in unlinked_entities:
    cluster_ids = [cid for cid in entity.get("cluster_ids", []) if cid]
    if not cluster_ids:
        continue
    if entity.get("entity_type") == "person":
        person_id = entity.get("entity_id")
        person = people_by_id.get(person_id) or entity.get("person") or {"person_id": person_id}
        auto_clustered_people.append({
            "person": person,
            "person_id": person_id,
            "name": person.get("name", f"Person {person_id}"),
            "episode_clusters": cluster_ids,
        })
    else:
        unlinked_cluster_ids.update(cluster_ids)

# Navigation back to Faces Review
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("← Faces Review"):
        helpers.try_switch_page("pages/3_Faces_Review.py")

st.markdown("---")

# Get or fetch cast suggestions
suggestions_key = f"cast_suggestions:{ep_id}"
cast_suggestions = st.session_state.get(suggestions_key, {})

# Dismissed suggestions tracking
dismissed_key = f"dismissed_suggestions:{ep_id}"
dismissed = st.session_state.setdefault(dismissed_key, set())

# Auto-refresh suggestions once when none are cached
auto_attempt_key = f"cast_suggestions_attempted:{ep_id}"
if not cast_suggestions and not st.session_state.get(auto_attempt_key, False):
    with st.spinner("Analyzing cast vs unassigned clusters..."):
        cast_suggestions = _recompute_cast_suggestions(ep_id, show_slug)
        st.session_state[auto_attempt_key] = True
        dismissed = st.session_state.setdefault(dismissed_key, set())

# Collect clusters with suggestions, sorted by confidence
suggestion_entries = []
for cluster_id, cluster_data in cluster_lookup.items():
    if unlinked_cluster_ids and cluster_id not in unlinked_cluster_ids:
        continue
    # Skip if dismissed
    if cluster_id in dismissed:
        continue

    # Check if already assigned (has person_id)
    if cluster_data.get("person_id"):
        continue

    # Get cast suggestions for this cluster
    cluster_suggestions = cast_suggestions.get(cluster_id, [])
    if cluster_suggestions:
        best_suggestion = cluster_suggestions[0]

        # Collect track thumbnails (max 5)
        tracks = cluster_data.get("tracks", [])
        thumb_urls = []
        for track in tracks[:5]:
            url = track.get("rep_thumb_url") or track.get("rep_media_url")
            if url:
                thumb_urls.append(url)

        suggestion_entries.append({
            "cluster_id": cluster_id,
            "cluster_data": cluster_data,
            "suggestion": best_suggestion,
            "all_suggestions": cluster_suggestions,
            "faces": cluster_data.get("counts", {}).get("faces", 0),
            "tracks": cluster_data.get("counts", {}).get("tracks", 0),
            "cohesion": cluster_data.get("cohesion"),
            "thumb_urls": thumb_urls,
        })

# Collect person-level suggestions by aggregating cluster suggestions
person_suggestion_entries = []
for person_entry in auto_clustered_people:
    person_id = person_entry["person_id"]
    if f"person:{person_id}" in dismissed:
        continue
    episode_clusters = person_entry["episode_clusters"]
    suggestion_bucket: Dict[str, Dict[str, Any]] = {}
    total_faces = 0
    total_tracks = 0
    thumb_urls: List[str] = []

    for cluster_id in episode_clusters:
        cluster_data = cluster_lookup.get(cluster_id, {})
        counts = cluster_data.get("counts", {}) if isinstance(cluster_data, dict) else {}
        total_faces += counts.get("faces", 0)
        total_tracks += counts.get("tracks", 0)

        # Aggregate suggestions by cast_id, keep the strongest similarity
        cluster_suggestions = cast_suggestions.get(cluster_id, [])
        for sugg in cluster_suggestions:
            cast_id = sugg.get("cast_id")
            if not cast_id:
                continue
            current = suggestion_bucket.get(cast_id)
            if not current or sugg.get("similarity", 0) > current.get("similarity", 0):
                suggestion_bucket[cast_id] = dict(sugg)

        # Collect thumbnails (limit to 5 total)
        tracks = cluster_data.get("tracks", [])
        for track in tracks:
            url = track.get("rep_thumb_url") or track.get("rep_media_url")
            if url and len(thumb_urls) < 5:
                thumb_urls.append(url)

    if not suggestion_bucket:
        continue

    sorted_suggs = sorted(suggestion_bucket.values(), key=lambda x: x.get("similarity", 0), reverse=True)
    person_suggestion_entries.append({
        "person": person_entry["person"],
        "person_id": person_id,
        "name": person_entry["name"],
        "episode_clusters": episode_clusters,
        "best_suggestion": sorted_suggs[0],
        "all_suggestions": sorted_suggs,
        "faces": total_faces,
        "tracks": total_tracks,
        "thumb_urls": thumb_urls,
    })

# Sort options
SORT_OPTIONS = {
    "Similarity (High → Low)": ("similarity", True),
    "Similarity (Low → High)": ("similarity", False),
    "Frames (Most → Least)": ("faces", True),
    "Frames (Least → Most)": ("faces", False),
    "Tracks (Most → Least)": ("tracks", True),
    "Tracks (Least → Most)": ("tracks", False),
    "Suggested Name (A → Z)": ("name", False),
    "Suggested Name (Z → A)": ("name", True),
}


def _sort_entries(entries: List[Dict[str, Any]], sort_key: str, reverse: bool) -> List[Dict[str, Any]]:
    """Sort suggestion entries by the specified key."""
    if sort_key == "similarity":
        return sorted(entries, key=lambda x: x["suggestion"].get("similarity", 0), reverse=reverse)
    elif sort_key == "faces":
        return sorted(entries, key=lambda x: x.get("faces", 0), reverse=reverse)
    elif sort_key == "tracks":
        return sorted(entries, key=lambda x: x.get("tracks", 0), reverse=reverse)
    elif sort_key == "name":
        return sorted(
            entries,
            key=lambda x: (x["suggestion"].get("name") or "").lower(),
            reverse=reverse,
        )
    return entries


# Stats bar (fallback to legacy unassigned count if endpoint empty)
total_unassigned = len(unlinked_entities) or sum(1 for c in cluster_lookup.values() if not c.get("person_id"))
with_suggestions = len(suggestion_entries) + len(person_suggestion_entries)
dismissed_count = len(dismissed)
auto_clustered_count = len(auto_clustered_people)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Needs Cast Assignment", total_unassigned)
with col2:
    st.metric("Auto-Clustered People", auto_clustered_count)
with col3:
    st.metric("With Suggestions", with_suggestions)
with col4:
    st.metric("Dismissed", dismissed_count)
with col5:
    # Refresh button
    if st.button("Refresh Suggestions", key="refresh_all_suggestions", use_container_width=True):
        with st.spinner("Recomputing suggestions from latest cast assignments..."):
            new_suggestions = _recompute_cast_suggestions(ep_id, show_slug)
            if new_suggestions:
                st.toast(f"Found {len(new_suggestions)} cluster(s) with suggestions")
                st.session_state[auto_attempt_key] = True
                st.rerun()
            else:
                st.info("No suggestions available yet - all clusters may already be assigned")

st.markdown("---")

# Sort control
sort_col1, sort_col2 = st.columns([1, 3])
with sort_col1:
    selected_sort = st.selectbox(
        "Sort by",
        options=list(SORT_OPTIONS.keys()),
        index=0,
        key="smart_suggestions_sort",
    )

# Apply sort
sort_key, sort_reverse = SORT_OPTIONS[selected_sort]
suggestion_entries = _sort_entries(suggestion_entries, sort_key, sort_reverse)

if not suggestion_entries and not person_suggestion_entries:
    st.info("No pending suggestions. Click 'Refresh Suggestions' to find matches for unassigned clusters.")
    # Clear dismissed button if there are dismissed entries
    if dismissed:
        if st.button("Clear All Dismissed", key="clear_all_dismissed", use_container_width=True):
            st.session_state[dismissed_key] = set()
            st.rerun()
    st.stop()


def render_suggestion_row(entry: Dict[str, Any], idx: int) -> None:
    """Render a single suggestion as a full-width row with thumbnails."""
    cluster_id = entry["cluster_id"]
    cluster_data = entry["cluster_data"]
    suggestion = entry["suggestion"]
    cast_id = suggestion.get("cast_id")
    cast_name = suggestion.get("name") or cast_options.get(cast_id, cast_id)
    similarity = suggestion.get("similarity", 0)
    confidence = suggestion.get("confidence", "low")
    source = suggestion.get("source", "facebank")
    faces_used = suggestion.get("faces_used")
    faces = entry["faces"]
    tracks = entry["tracks"]
    cohesion = entry["cohesion"]
    thumb_urls = entry["thumb_urls"]
    track_list = cluster_data.get("tracks", [])

    # Build source label with extra info for frame-based suggestions
    source_label = source
    if source == "frame" and faces_used:
        source_label = f"frame ({faces_used} face{'s' if faces_used > 1 else ''})"

    # Determine which similarity to show: cluster cohesion for multi-track, internal similarity for single-track
    similarity_value = None
    similarity_label = None
    similarity_type = None
    if tracks > 1 and cohesion is not None:
        # Multi-track cluster: show cluster cohesion
        similarity_value = cohesion
        similarity_label = "Cluster Similarity"
        similarity_type = SimilarityType.CLUSTER
    elif track_list:
        # Single/few-track cluster: show track internal similarity (frame consistency within track)
        # Try internal_similarity first (how similar rep frame is to track centroid)
        track = track_list[0]
        internal_sim = track.get("internal_similarity")
        if internal_sim is not None:
            similarity_value = internal_sim
            similarity_label = "Track Consistency"
            similarity_type = SimilarityType.TRACK
        else:
            # Fallback to track-to-cluster similarity if available
            track_sim = track.get("similarity")
            if track_sim is not None:
                similarity_value = track_sim
                similarity_label = "Track Similarity"
                similarity_type = SimilarityType.TRACK

    with st.container(border=True):
        # Row layout: thumbnails on left (fixed 147x184px each, max 5), info on right
        thumb_col, info_col, action_col = st.columns([5, 3, 1])

        with thumb_col:
            # Render thumbnails as HTML with fixed 147x184px size
            if thumb_urls:
                thumbs_html = '<div class="thumb-row">'
                for url in thumb_urls[:5]:
                    thumbs_html += f'<div class="thumb-frame"><img src="{url}" /></div>'
                thumbs_html += '</div>'
                st.markdown(thumbs_html, unsafe_allow_html=True)
            else:
                st.caption("No thumbnails available")

        with info_col:
            # Cluster info
            st.markdown(f"**Cluster `{cluster_id}`**")
            st.caption(f"{faces} faces · {tracks} tracks")

            if similarity_value is not None and similarity_label and similarity_type:
                sim_badge = render_similarity_badge(similarity_value, similarity_type, show_label=True, custom_label=similarity_label)
                st.markdown(sim_badge, unsafe_allow_html=True)

            st.markdown("")  # Spacer

            # Suggestion with confidence aligned to cast thresholds (68/50)
            cast_badge = render_similarity_badge(similarity, SimilarityType.CAST, show_label=True)
            conf_level = "high" if similarity >= 0.68 else "medium" if similarity >= 0.50 else "low"
            conf_colors = {"high": "#4CAF50", "medium": "#FF9800", "low": "#F44336"}
            conf_color = conf_colors.get(conf_level, "#9E9E9E")

            # Ambiguity margin (top1 vs top2)
            margin_pct = None
            all_suggestions = entry.get("all_suggestions") or []
            if len(all_suggestions) > 1:
                runner_up = all_suggestions[1].get("similarity", 0) or 0
                margin_pct = int(max(similarity - runner_up, 0) * 100)

            margin_html = f" · Δ {margin_pct}%" if margin_pct is not None else ""
            st.markdown(
                f'<span style="background-color: {conf_color}; color: white; '
                f'padding: 3px 8px; border-radius: 4px; font-size: 0.9em; font-weight: bold;">'
                f'{conf_level.upper()}</span> {cast_badge} → **{cast_name}** '
                f'<span style="font-size: 0.75em; color: #888;">({source_label}){margin_html}</span>',
                unsafe_allow_html=True,
            )

            # Alternative suggestions
            alt_suggestions = entry["all_suggestions"][1:3]
            if alt_suggestions:
                alt_text = " · ".join([
                    f"{alt.get('name', 'Unknown')} ({int(alt.get('similarity', 0) * 100)}%)"
                    for alt in alt_suggestions
                ])
                st.caption(f"Also: {alt_text}")

        with action_col:
            # Action buttons - stacked vertically
            if st.button("✓ Accept", key=f"sp_accept_{cluster_id}", use_container_width=True):
                # Find or create person for this cast member
                people_resp = _fetch_people_cached(show_slug)
                people = people_resp.get("people", []) if people_resp else []
                target_person = next(
                    (p for p in people if p.get("cast_id") == cast_id),
                    None,
                )
                target_person_id = target_person.get("person_id") if target_person else None

                payload = {
                    "strategy": "manual",
                    "cluster_ids": [cluster_id],
                    "target_person_id": target_person_id,
                    "cast_id": cast_id,
                }
                resp = _api_post(f"/episodes/{ep_id}/clusters/group", payload)
                if resp and resp.get("status") == "success":
                    st.toast(f"Assigned to {cast_name}")
                    # Remove from suggestions
                    if cluster_id in cast_suggestions:
                        del cast_suggestions[cluster_id]
                    st.rerun()
                else:
                    st.error("Failed to assign")

            if st.button("👁 View", key=f"sp_view_{cluster_id}", use_container_width=True):
                # Navigate to Faces Review with this cluster selected
                st.session_state["facebank_ep"] = ep_id  # Prevent state reset
                st.session_state["facebank_view"] = "cluster_tracks"
                st.session_state["selected_identity"] = cluster_id
                st.session_state["selected_person"] = None  # Clear person selection
                st.session_state["selected_track"] = None  # Clear track selection
                # Set query params to match (prevents _hydrate_view_from_query from overriding)
                st.query_params["view"] = "cluster"
                st.query_params["cluster"] = cluster_id
                if "person" in st.query_params:
                    del st.query_params["person"]
                if "track" in st.query_params:
                    del st.query_params["track"]
                helpers.try_switch_page("pages/3_Faces_Review.py")

            if st.button("✗ Skip", key=f"sp_dismiss_{cluster_id}", use_container_width=True):
                dismissed.add(cluster_id)
                st.rerun()


# --- UNIFIED NEEDS-ASSIGNMENT SECTION ---
if suggestion_entries or person_suggestion_entries:
    st.markdown("### 🔍 Needs Cast Assignment")
    st.caption("Clusters and auto-people that are not linked to cast. Review and accept suggestions.")
    for idx, entry in enumerate(suggestion_entries):
        render_suggestion_row(entry, idx)

    for person_entry in person_suggestion_entries:
        person_id = person_entry["person_id"]
        person_name = person_entry["name"]
        episode_clusters = person_entry["episode_clusters"]
        best_suggestion = person_entry["best_suggestion"]
        all_suggestions = person_entry["all_suggestions"]
        faces = person_entry["faces"]
        tracks = person_entry["tracks"]
        person_thumb_urls = person_entry["thumb_urls"]

        with st.container(border=True):
            thumb_col, info_col, action_col = st.columns([5, 3, 1])

            with thumb_col:
                if person_thumb_urls:
                    thumbs_html = '<div class="thumb-row">'
                    for url in person_thumb_urls[:5]:
                        thumbs_html += f'<div class="thumb-frame"><img src="{url}" /></div>'
                    thumbs_html += '</div>'
                    st.markdown(thumbs_html, unsafe_allow_html=True)
                else:
                    st.caption("No thumbnails available")

            with info_col:
                st.markdown(f"**{person_name}** `{person_id}`")
                st.caption(f"{len(episode_clusters)} cluster(s) · {tracks} tracks · {faces} faces")

                if best_suggestion:
                    cast_id = best_suggestion.get("cast_id")
                    cast_name = best_suggestion.get("name") or cast_options.get(cast_id, cast_id)
                    similarity = best_suggestion.get("similarity", 0)
                    source = best_suggestion.get("source", "facebank")

                    cast_badge = render_similarity_badge(similarity, SimilarityType.CAST, show_label=True)
                    conf_level = "high" if similarity >= 0.68 else "medium" if similarity >= 0.50 else "low"
                    conf_colors = {"high": "#4CAF50", "medium": "#FF9800", "low": "#F44336"}
                    conf_color = conf_colors.get(conf_level, "#9E9E9E")
                    margin_pct = None
                    if len(all_suggestions) > 1:
                        runner_up = all_suggestions[1].get("similarity", 0) or 0
                        margin_pct = int(max(similarity - runner_up, 0) * 100)

                    st.markdown(
                        f'<span style="background-color: {conf_color}; color: white; '
                        f'padding: 3px 8px; border-radius: 4px; font-size: 0.9em; font-weight: bold;">'
                        f'{conf_level.upper()}</span> {cast_badge} → **{cast_name}** '
                        f'<span style="font-size: 0.75em; color: #888;">({source})'
                        f"{' · Δ ' + str(margin_pct) + '%' if margin_pct is not None else ''}</span>",
                        unsafe_allow_html=True,
                    )

                    alt_suggestions = all_suggestions[1:3]
                    if alt_suggestions:
                        alt_text = " · ".join([
                            f"{alt.get('name', 'Unknown')} ({int(alt.get('similarity', 0) * 100)}%)"
                            for alt in alt_suggestions
                        ])
                        st.caption(f"Also: {alt_text}")
                else:
                    st.caption("💡 Click 'Refresh Suggestions' to find cast matches")

            with action_col:
                if best_suggestion:
                    cast_id = best_suggestion.get("cast_id")
                    cast_name = best_suggestion.get("name") or cast_options.get(cast_id, cast_id)

                    if st.button("✓ Link", key=f"sp_link_person_{person_id}", use_container_width=True):
                        payload = {"cast_id": cast_id}
                        resp = _api_patch(f"/shows/{show_slug}/people/{person_id}", payload)
                        if resp:
                            st.toast(f"Linked {person_name} to {cast_name}")
                            people_cache_key = f"people_cache:{show_slug}"
                            if people_cache_key in st.session_state:
                                del st.session_state[people_cache_key]
                            st.rerun()
                        else:
                            st.error("Failed to link")

                if st.button("👁 View", key=f"sp_view_person_{person_id}", use_container_width=True):
                    st.session_state["facebank_ep"] = ep_id
                    st.session_state["facebank_view"] = "person_clusters"
                    st.session_state["selected_person"] = person_id
                    st.session_state["selected_identity"] = None
                    st.session_state["selected_track"] = None
                    st.query_params["view"] = "person"
                    st.query_params["person"] = person_id
                    if "cluster" in st.query_params:
                        del st.query_params["cluster"]
                    if "track" in st.query_params:
                        del st.query_params["track"]
                    helpers.try_switch_page("pages/3_Faces_Review.py")

                if st.button("✗ Skip", key=f"sp_dismiss_person_{person_id}", use_container_width=True):
                    dismissed.add(f"person:{person_id}")
                    st.rerun()

# Clear all dismissed button at bottom
if dismissed:
    st.markdown("---")
    if st.button("Clear All Dismissed", key="clear_dismissed_bottom", use_container_width=True):
        st.session_state[dismissed_key] = set()
        st.rerun()

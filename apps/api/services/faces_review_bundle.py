"""Run-scoped Faces Review bundle builder (no Streamlit dependencies)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from py_screenalytics import run_layout

from apps.api.services.assignments import load_assignment_state
from apps.api.services.assignment_resolver import resolve_cluster_assignment, resolve_track_assignment
from apps.api.services.archive import ArchiveService
from apps.api.services.cast import CastService
from apps.api.services.grouping import GroupingService
from apps.api.services.identities import cluster_track_summary, load_identities
from apps.api.services.people import PeopleService
from apps.api.services.run_state import run_state_service
from apps.api.services.run_validator import validate_run_integrity

LOGGER = logging.getLogger(__name__)

_EP_ID_REGEX = re.compile(r"^(?P<show>.+)-s(?P<season>\d{2})e(?P<episode>\d{2})$", re.IGNORECASE)


def _parse_ep_id(ep_id: str) -> Optional[Dict[str, Any]]:
    match = _EP_ID_REGEX.match(ep_id)
    if not match:
        return None
    show = match.group("show")
    try:
        season = int(match.group("season"))
        episode = int(match.group("episode"))
    except ValueError:
        return None
    return {"show": show, "season": season, "episode": episode}


def _split_cluster_ids(
    cluster_ids: Iterable[str],
    ep_id: str,
    run_id: str | None,
) -> Tuple[List[str], List[str]]:
    run_cluster_ids: List[str] = []
    legacy_cluster_ids: List[str] = []
    run_prefix = f"{ep_id}:{run_id}:" if run_id else None
    episode_prefix = f"{ep_id}:"
    for raw in cluster_ids:
        if not isinstance(raw, str):
            continue
        if run_prefix and raw.startswith(run_prefix):
            run_cluster_ids.append(raw[len(run_prefix):])
            continue
        if raw.startswith(episode_prefix):
            legacy_cluster_ids.append(raw[len(episode_prefix):])
    return run_cluster_ids, legacy_cluster_ids


def _select_people_for_run(
    people: List[Dict[str, Any]],
    ep_id: str,
    run_id: str | None,
) -> Tuple[List[Dict[str, Any]], bool]:
    if not run_id:
        return people, False
    run_people: List[Dict[str, Any]] = []
    legacy_people: List[Dict[str, Any]] = []
    for person in people:
        cluster_ids = person.get("cluster_ids", []) if isinstance(person, dict) else []
        run_clusters, legacy_clusters = _split_cluster_ids(cluster_ids, ep_id, run_id)
        if run_clusters:
            run_people.append(person)
        elif legacy_clusters:
            legacy_people.append(person)
    if run_people:
        return run_people, False
    if legacy_people:
        return legacy_people, True
    return [], False


def _normalize_cast_entries(
    cast_entries: Iterable[Dict[str, Any]],
    people: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    for entry in cast_entries:
        if not isinstance(entry, dict):
            continue
        cast_id = entry.get("cast_id")
        if not cast_id:
            continue
        deduped[cast_id] = dict(entry)
    for person in people:
        cast_id = person.get("cast_id") if isinstance(person, dict) else None
        if not cast_id or cast_id in deduped:
            continue
        deduped[cast_id] = {
            "cast_id": cast_id,
            "name": person.get("name") or "(unnamed)",
            "aliases": person.get("aliases") or [],
            "legacy": True,
        }
    return list(deduped.values())


def _normalize_track_entry(track: Dict[str, Any]) -> Dict[str, Any]:
    entry = dict(track)
    thumb_url = (
        entry.get("crop_url")
        or entry.get("rep_thumb_url")
        or entry.get("rep_media_url")
    )
    if thumb_url:
        entry.setdefault("crop_url", thumb_url)
        entry["thumb_url"] = thumb_url
    return entry


def _normalize_cluster_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(entry)
    tracks = normalized.get("tracks")
    if not isinstance(tracks, list):
        tracks = []
    normalized_tracks = [_normalize_track_entry(t) for t in tracks if isinstance(t, dict)]
    normalized["tracks"] = normalized_tracks
    return normalized


def _filter_clusters(
    clusters: Iterable[Dict[str, Any]],
    archived_clusters: set[str],
    archived_tracks: set[int],
    *,
    include_archived: bool,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for entry in clusters:
        if not isinstance(entry, dict):
            continue
        cluster_id = entry.get("identity_id") or entry.get("cluster_id")
        if not include_archived and cluster_id and str(cluster_id) in archived_clusters:
            continue
        normalized = _normalize_cluster_entry(entry)
        tracks = normalized.get("tracks") if isinstance(normalized.get("tracks"), list) else []
        tracks_filtered = False
        if not include_archived and tracks:
            cleaned_tracks = []
            for track in tracks:
                tid = track.get("track_id") or track.get("track") or track.get("track_int")
                try:
                    tid_int = int(tid)
                except (TypeError, ValueError):
                    tid_int = None
                if tid_int is not None and tid_int in archived_tracks:
                    continue
                cleaned_tracks.append(track)
            tracks_filtered = len(cleaned_tracks) != len(tracks)
            tracks = cleaned_tracks
            normalized["tracks"] = tracks

        counts = normalized.get("counts")
        counts = dict(counts) if isinstance(counts, dict) else {}
        if tracks or tracks_filtered:
            counts["tracks"] = len(tracks)
            faces_values = [
                track.get("faces")
                for track in tracks
                if track.get("faces") is not None
            ]
            if faces_values:
                faces_total = 0
                for value in faces_values:
                    try:
                        faces_total += int(value)
                    except (TypeError, ValueError):
                        continue
                counts["faces"] = faces_total
            elif tracks_filtered:
                counts["faces"] = 0
        normalized["counts"] = counts
        filtered.append(normalized)
    return filtered


def _build_cluster_lookup(clusters: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for entry in clusters:
        cluster_id = entry.get("identity_id") or entry.get("cluster_id")
        if not cluster_id:
            continue
        lookup[str(cluster_id)] = entry
    return lookup


def _best_thumb_for_clusters(
    cluster_ids: Iterable[str],
    cluster_lookup: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    for cluster_id in cluster_ids:
        cluster = cluster_lookup.get(str(cluster_id))
        if not cluster:
            continue
        for track in cluster.get("tracks", []) or []:
            thumb_url = track.get("crop_url") or track.get("rep_thumb_url") or track.get("rep_media_url")
            if thumb_url:
                return thumb_url
    return None


def build_faces_review_bundle(
    ep_id: str,
    run_id: str | None = None,
    *,
    filter_cast_id: str | None = None,
    include_archived: bool = False,
    data_root: Path | str | None = None,
) -> Dict[str, Any]:
    """Build a run-scoped Faces Review bundle for deterministic UI rendering."""
    run_id_norm = run_layout.normalize_run_id(run_id) if run_id else None
    parsed = _parse_ep_id(ep_id) or {}
    show_id = parsed.get("show")

    cast_service = CastService(data_root)
    people_service = PeopleService(data_root)
    grouping_service = GroupingService(data_root=data_root, run_id=run_id_norm)
    archive_service = ArchiveService(data_root)

    cast_entries = cast_service.list_cast(show_id) if show_id else []
    people = people_service.list_people(show_id) if show_id else []
    people, legacy_people_fallback = _select_people_for_run(people, ep_id, run_id_norm)

    cast_entries = _normalize_cast_entries(cast_entries, people)
    cast_options: Dict[str, str] = {}
    for entry in cast_entries:
        cast_id = entry.get("cast_id")
        name = entry.get("name")
        if cast_id and name:
            cast_options[cast_id] = name
    for person in people:
        cast_id = person.get("cast_id")
        name = person.get("name")
        if cast_id and name and cast_id not in cast_options:
            cast_options[cast_id] = name

    identities_payload = load_identities(ep_id, run_id=run_id_norm)
    cluster_payload = cluster_track_summary(ep_id, run_id=run_id_norm) or {"clusters": []}

    archived_ids = {"clusters": [], "tracks": []}
    if show_id:
        archived = archive_service.list_archived(show_id, episode_id=ep_id, limit=500)
        items = archived.get("items", []) if archived else []
        archived_clusters = {
            str(item.get("cluster_id") or item.get("identity_id") or item.get("original_id"))
            for item in items
            if item.get("type") == "cluster"
        }
        archived_tracks = set()
        for item in items:
            if item.get("type") != "track":
                continue
            tid = item.get("track_id") or item.get("original_id")
            try:
                archived_tracks.add(int(tid))
            except (TypeError, ValueError):
                continue
        archived_ids = {
            "clusters": sorted([cid for cid in archived_clusters if cid]),
            "tracks": sorted(archived_tracks),
        }
    else:
        archived_clusters = set()
        archived_tracks = set()

    archived_clusters = set(archived_ids["clusters"])
    archived_tracks = set(archived_ids["tracks"])
    filtered_clusters = _filter_clusters(
        cluster_payload.get("clusters", []),
        archived_clusters,
        archived_tracks,
        include_archived=include_archived,
    )
    cluster_payload = dict(cluster_payload)
    cluster_payload["clusters"] = filtered_clusters
    cluster_lookup = _build_cluster_lookup(filtered_clusters)

    assignment_state = load_assignment_state(
        ep_id,
        run_id_norm,
        data_root=data_root,
        include_inferred=True,
    )
    cluster_assignments = assignment_state.get("cluster_assignments", {})
    track_overrides = assignment_state.get("track_overrides", {})

    for cluster in filtered_clusters:
        cluster_id = cluster.get("identity_id") or cluster.get("cluster_id")
        cluster_assignment = resolve_cluster_assignment(cluster_id, cluster_assignments)
        cluster["assigned_cast_id"] = cluster_assignment.get("cast_id")
        cluster["assignment"] = cluster_assignment
        for track in cluster.get("tracks", []) or []:
            track_id = track.get("track_id") or track.get("track") or track.get("track_int")
            track_assignment = resolve_track_assignment(
                track_id,
                str(cluster_id) if cluster_id else None,
                cluster_assignments,
                track_overrides,
            )
            track["effective_cast_id"] = track_assignment.get("cast_id")
            track["assignment"] = track_assignment

    try:
        unlinked = grouping_service.list_unlinked_entities(ep_id)
        unlinked_entities = unlinked.get("entities", []) if unlinked else []
    except FileNotFoundError as exc:
        LOGGER.info("[faces-review-bundle] %s", exc)
        unlinked_entities = []

    if not include_archived:
        filtered_entities = []
        for entity in unlinked_entities:
            cluster_ids = [cid for cid in (entity.get("cluster_ids") or []) if cid]
            cluster_ids = [cid for cid in cluster_ids if cid not in archived_clusters]
            if not cluster_ids:
                continue
            entry = dict(entity)
            entry["cluster_ids"] = cluster_ids
            filtered_entities.append(entry)
        unlinked_entities = filtered_entities

    if filter_cast_id:
        people = [
            person for person in people
            if str(person.get("cast_id") or "") == str(filter_cast_id)
        ]
        cast_entries = [
            entry for entry in cast_entries
            if str(entry.get("cast_id") or "") == str(filter_cast_id)
        ]
        filtered_entities = []
        for entity in unlinked_entities:
            if entity.get("entity_type") == "person":
                person = entity.get("person") or {}
                if str(person.get("cast_id") or "") != str(filter_cast_id):
                    continue
            filtered_entities.append(entity)
        unlinked_entities = filtered_entities

    for person in people:
        run_clusters, legacy_clusters = _split_cluster_ids(
            person.get("cluster_ids", []) or [],
            ep_id,
            run_id_norm,
        )
        if run_id_norm:
            episode_clusters = run_clusters if run_clusters else (legacy_clusters if legacy_people_fallback else [])
        else:
            episode_clusters = legacy_clusters
        person["episode_clusters"] = episode_clusters

    for entity in unlinked_entities:
        cluster_ids = entity.get("cluster_ids", []) or []
        tracks = 0
        faces = 0
        for cid in cluster_ids:
            cluster = cluster_lookup.get(str(cid))
            if not cluster:
                continue
            counts = cluster.get("counts") or {}
            try:
                tracks += int(counts.get("tracks") or 0)
            except (TypeError, ValueError):
                pass
            try:
                faces += int(counts.get("faces") or 0)
            except (TypeError, ValueError):
                pass
        entity["tracks"] = tracks
        entity["faces"] = faces

    cast_gallery_cards: List[Dict[str, Any]] = []
    people_by_cast_id = {
        person.get("cast_id"): person
        for person in people
        if person.get("cast_id")
    }
    for entry in cast_entries:
        cast_id = entry.get("cast_id")
        person = people_by_cast_id.get(cast_id)
        if not person:
            continue
        episode_clusters = person.get("episode_clusters") or []
        if not episode_clusters:
            continue
        featured_thumb = entry.get("featured_thumbnail_url") or person.get("rep_crop")
        if not featured_thumb:
            featured_thumb = _best_thumb_for_clusters(episode_clusters, cluster_lookup)
        cast_gallery_cards.append(
            {
                "cast": entry,
                "person": person,
                "episode_clusters": episode_clusters,
                "featured_thumbnail": featured_thumb,
            }
        )

    cast_gallery_cards.sort(key=lambda card: (card.get("cast", {}).get("name") or "").lower())

    run_state_payload: Dict[str, Any] | None = None
    status_snapshot: Dict[str, Any] | None = None
    validation_payload: Dict[str, Any] | None = None
    if run_id_norm:
        try:
            run_state_bundle = run_state_service.get_state(ep_id=ep_id, run_id=run_id_norm)
            run_state_payload = run_state_bundle.get("run_state") if isinstance(run_state_bundle, dict) else None
            status_snapshot = run_state_bundle.get("status_snapshot") if isinstance(run_state_bundle, dict) else None
        except Exception as exc:
            LOGGER.debug("[faces-review-bundle] run_state unavailable for %s/%s: %s", ep_id, run_id_norm, exc)
        try:
            validation_payload = validate_run_integrity(ep_id, run_id_norm, data_root=data_root)
        except Exception as exc:
            LOGGER.debug("[faces-review-bundle] validator unavailable for %s/%s: %s", ep_id, run_id_norm, exc)

    return {
        "ep_id": ep_id,
        "run_id": run_id_norm or "legacy",
        "show_id": show_id,
        "legacy_people_fallback": legacy_people_fallback,
        "cast_entries": cast_entries,
        "cast_gallery_cards": cast_gallery_cards,
        "cast_options": cast_options,
        "people": people,
        "unlinked_entities": unlinked_entities,
        "archived_ids": archived_ids,
        "assignments": {
            "clusters": assignment_state.get("cluster_assignments_raw", cluster_assignments),
            "tracks": assignment_state.get("track_overrides", {}),
            "faces": assignment_state.get("face_exclusions", {}),
            "summary": assignment_state.get("summary", {}),
        },
        "run_state": run_state_payload,
        "status_snapshot": status_snapshot,
        "validation": validation_payload,
        "cluster_payload": cluster_payload,
        "identities": identities_payload,
    }

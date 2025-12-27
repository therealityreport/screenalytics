"""Run-scoped assignment storage for Faces Review."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from py_screenalytics import run_layout

from apps.api.services.identities import load_identities, write_identities
from apps.api.services.people import PeopleService

LOGGER = logging.getLogger(__name__)

_EP_ID_REGEX = re.compile(r"^(?P<show>.+)-s(?P<season>\d{2})e(?P<episode>\d{2})$", re.IGNORECASE)


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_show_id(ep_id: str) -> str | None:
    match = _EP_ID_REGEX.match(ep_id)
    if not match:
        return None
    return match.group("show")


def _normalize_assignment_map(raw: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for key, value in raw.items():
        if not key:
            continue
        if isinstance(value, dict):
            normalized[str(key)] = dict(value)
    return normalized


def _identity_ids(payload: Dict[str, Any]) -> List[str]:
    identities = payload.get("identities")
    if not isinstance(identities, list):
        return []
    ids: List[str] = []
    for entry in identities:
        if not isinstance(entry, dict):
            continue
        identity_id = entry.get("identity_id") or entry.get("id")
        if identity_id:
            ids.append(str(identity_id))
    return ids


def _track_to_cluster(payload: Dict[str, Any]) -> Dict[int, str]:
    identities = payload.get("identities")
    if not isinstance(identities, list):
        return {}
    mapping: Dict[int, str] = {}
    for entry in identities:
        if not isinstance(entry, dict):
            continue
        identity_id = entry.get("identity_id") or entry.get("id")
        if not identity_id:
            continue
        for raw in entry.get("track_ids", []) or []:
            try:
                track_id = int(str(raw).replace("track_", ""))
            except (TypeError, ValueError):
                continue
            mapping[track_id] = str(identity_id)
    return mapping


def _infer_cluster_assignments(
    payload: Dict[str, Any],
    show_id: str | None,
    *,
    data_root: Path | str | None = None,
) -> Dict[str, Dict[str, Any]]:
    if not show_id:
        return {}
    people_service = PeopleService(data_root)
    people = people_service.list_people(show_id)
    person_to_cast = {
        person.get("person_id"): person.get("cast_id")
        for person in people
        if person.get("person_id") and person.get("cast_id")
    }
    identities = payload.get("identities")
    if not isinstance(identities, list):
        return {}
    inferred: Dict[str, Dict[str, Any]] = {}
    for entry in identities:
        if not isinstance(entry, dict):
            continue
        identity_id = entry.get("identity_id") or entry.get("id")
        person_id = entry.get("person_id")
        if not identity_id or not person_id:
            continue
        cast_id = person_to_cast.get(person_id)
        if not cast_id:
            continue
        inferred[str(identity_id)] = {
            "cast_id": cast_id,
            "assigned_by": "auto",
            "source": "auto",
        }
    return inferred


def _apply_assignment_update(
    existing: Dict[str, Any] | None,
    *,
    cast_id: str | None,
    source: str,
    updated_by: str | None,
    unassigned: bool | None = None,
) -> Tuple[Dict[str, Any], bool]:
    entry = dict(existing or {})
    assigned_by = entry.get("assigned_by")
    if source == "manual":
        assigned_by = "user"
    elif source == "auto":
        assigned_by = "auto"
    new_entry = dict(entry)
    new_entry["cast_id"] = cast_id
    new_entry["assigned_by"] = assigned_by
    new_entry["source"] = source
    if unassigned is not None:
        new_entry["unassigned"] = unassigned
    if updated_by is not None:
        new_entry["updated_by"] = updated_by
    if new_entry.get("updated_at") is None:
        new_entry["updated_at"] = entry.get("updated_at")
    if new_entry.get("timestamp") is None:
        new_entry["timestamp"] = entry.get("timestamp")

    def _matches() -> bool:
        return (
            entry.get("cast_id") == new_entry.get("cast_id")
            and entry.get("source") == new_entry.get("source")
            and entry.get("assigned_by") == new_entry.get("assigned_by")
            and entry.get("updated_by") == new_entry.get("updated_by")
            and entry.get("unassigned") == new_entry.get("unassigned")
        )

    if _matches():
        return entry, False

    now = _now_iso()
    new_entry["updated_at"] = now
    new_entry["timestamp"] = now
    return new_entry, True


def load_assignment_state(
    ep_id: str,
    run_id: str | None,
    *,
    data_root: Path | str | None = None,
    include_inferred: bool = True,
) -> Dict[str, Any]:
    run_id_norm = run_layout.normalize_run_id(run_id) if run_id else None
    payload = load_identities(ep_id, run_id=run_id_norm)
    manual_assignments = _normalize_assignment_map(payload.get("manual_assignments"))
    track_overrides = _normalize_assignment_map(payload.get("track_overrides"))
    face_exclusions = _normalize_assignment_map(payload.get("face_exclusions"))

    cluster_assignments = dict(manual_assignments)
    if include_inferred:
        show_id = _parse_show_id(ep_id)
        inferred = _infer_cluster_assignments(payload, show_id, data_root=data_root)
        for cluster_id, entry in inferred.items():
            cluster_assignments.setdefault(cluster_id, entry)

    summary = summarize_assignments(payload, manual_assignments, track_overrides, face_exclusions)
    return {
        "ep_id": ep_id,
        "run_id": run_id_norm or "legacy",
        "cluster_assignments": cluster_assignments,
        "cluster_assignments_raw": manual_assignments,
        "track_overrides": track_overrides,
        "face_exclusions": face_exclusions,
        "summary": summary,
    }


def summarize_assignments(
    payload: Dict[str, Any],
    cluster_assignments: Dict[str, Dict[str, Any]],
    track_overrides: Dict[str, Dict[str, Any]],
    face_exclusions: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    identity_ids = _identity_ids(payload)
    assigned_cluster_ids = {
        cid
        for cid, entry in cluster_assignments.items()
        if isinstance(entry, dict) and entry.get("cast_id")
    }
    assigned_clusters = len([cid for cid in identity_ids if cid in assigned_cluster_ids])
    overrides = sum(
        1 for entry in track_overrides.values()
        if isinstance(entry, dict) and entry.get("cast_id")
    )
    exclusions = sum(
        1 for entry in face_exclusions.values()
        if isinstance(entry, dict) and entry.get("excluded", True)
    )
    return {
        "clusters_total": len(identity_ids),
        "clusters_assigned": assigned_clusters,
        "clusters_unassigned": max(len(identity_ids) - assigned_clusters, 0),
        "track_overrides": overrides,
        "face_exclusions": exclusions,
    }


def set_cluster_assignment(
    ep_id: str,
    run_id: str,
    *,
    cluster_id: str,
    cast_id: str | None,
    source: str = "manual",
    updated_by: str | None = None,
) -> Dict[str, Any]:
    run_id_norm = run_layout.normalize_run_id(run_id)
    payload = load_identities(ep_id, run_id=run_id_norm)
    identity_ids = _identity_ids(payload)
    if cluster_id not in identity_ids:
        raise ValueError("cluster_not_found")
    manual_assignments = _normalize_assignment_map(payload.get("manual_assignments"))

    changed = False
    if not cast_id:
        entry, entry_changed = _apply_assignment_update(
            manual_assignments.get(cluster_id),
            cast_id=None,
            source=source,
            updated_by=updated_by,
            unassigned=True,
        )
        if entry_changed or cluster_id not in manual_assignments:
            manual_assignments[cluster_id] = entry
            changed = True
    else:
        entry, entry_changed = _apply_assignment_update(
            manual_assignments.get(cluster_id),
            cast_id=cast_id,
            source=source,
            updated_by=updated_by,
            unassigned=False,
        )
        if entry_changed or cluster_id not in manual_assignments:
            manual_assignments[cluster_id] = entry
            changed = True

    if changed:
        payload["manual_assignments"] = manual_assignments
        path = write_identities(ep_id, payload, run_id=run_id_norm)
        LOGGER.info("[assignments] Updated cluster assignment for %s/%s", ep_id, cluster_id)
        LOGGER.debug("[assignments] identities updated: %s", path)

    summary = summarize_assignments(payload, manual_assignments, _normalize_assignment_map(payload.get("track_overrides")), _normalize_assignment_map(payload.get("face_exclusions")))
    return {
        "changed": changed,
        "assignments": {
            "clusters": manual_assignments,
        },
        "summary": summary,
    }


def set_track_override(
    ep_id: str,
    run_id: str,
    *,
    track_id: int,
    cast_id: str | None,
    source: str = "manual",
    updated_by: str | None = None,
) -> Dict[str, Any]:
    run_id_norm = run_layout.normalize_run_id(run_id)
    payload = load_identities(ep_id, run_id=run_id_norm)
    track_map = _track_to_cluster(payload)
    if track_id not in track_map:
        raise ValueError("track_not_found")
    track_overrides = _normalize_assignment_map(payload.get("track_overrides"))
    key = str(track_id)

    changed = False
    if not cast_id:
        if key in track_overrides:
            track_overrides.pop(key, None)
            changed = True
    else:
        entry, entry_changed = _apply_assignment_update(
            track_overrides.get(key),
            cast_id=cast_id,
            source=source,
            updated_by=updated_by,
        )
        if entry_changed or key not in track_overrides:
            track_overrides[key] = entry
            changed = True

    if changed:
        payload["track_overrides"] = track_overrides
        path = write_identities(ep_id, payload, run_id=run_id_norm)
        LOGGER.info("[assignments] Updated track override for %s/%s", ep_id, track_id)
        LOGGER.debug("[assignments] identities updated: %s", path)

    summary = summarize_assignments(payload, _normalize_assignment_map(payload.get("manual_assignments")), track_overrides, _normalize_assignment_map(payload.get("face_exclusions")))
    return {
        "changed": changed,
        "assignments": {
            "tracks": track_overrides,
        },
        "summary": summary,
    }


def set_face_exclusion(
    ep_id: str,
    run_id: str,
    *,
    face_id: str,
    excluded: bool = True,
    reason: str | None = None,
    source: str = "manual",
    updated_by: str | None = None,
    track_id: int | None = None,
) -> Dict[str, Any]:
    run_id_norm = run_layout.normalize_run_id(run_id)
    payload = load_identities(ep_id, run_id=run_id_norm)
    face_exclusions = _normalize_assignment_map(payload.get("face_exclusions"))
    key = str(face_id)

    changed = False
    if not excluded:
        if key in face_exclusions:
            face_exclusions.pop(key, None)
            changed = True
    else:
        entry = dict(face_exclusions.get(key) or {})
        if (
            entry.get("excluded")
            and entry.get("reason") == reason
            and entry.get("source") == source
            and entry.get("updated_by") == updated_by
            and entry.get("track_id") == track_id
        ):
            return {
                "changed": False,
                "assignments": {"faces": face_exclusions},
                "summary": summarize_assignments(
                    payload,
                    _normalize_assignment_map(payload.get("manual_assignments")),
                    _normalize_assignment_map(payload.get("track_overrides")),
                    face_exclusions,
                ),
            }
        entry["excluded"] = True
        entry["reason"] = reason
        entry["source"] = source
        if updated_by is not None:
            entry["updated_by"] = updated_by
        if track_id is not None:
            entry["track_id"] = track_id
        entry["updated_at"] = _now_iso()
        face_exclusions[key] = entry
        changed = True

    if changed:
        payload["face_exclusions"] = face_exclusions
        path = write_identities(ep_id, payload, run_id=run_id_norm)
        LOGGER.info("[assignments] Updated face exclusion for %s/%s", ep_id, face_id)
        LOGGER.debug("[assignments] identities updated: %s", path)

    summary = summarize_assignments(payload, _normalize_assignment_map(payload.get("manual_assignments")), _normalize_assignment_map(payload.get("track_overrides")), face_exclusions)
    return {
        "changed": changed,
        "assignments": {
            "faces": face_exclusions,
        },
        "summary": summary,
    }

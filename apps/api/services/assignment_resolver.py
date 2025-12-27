"""Pure assignment resolver helpers for Faces Review + downstream consumers."""

from __future__ import annotations

from typing import Any, Dict


def _normalize_assignment_entry(entry: Dict[str, Any] | None, *, assignment_type: str) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        return {
            "cast_id": None,
            "source": None,
            "updated_at": None,
            "updated_by": None,
            "assignment_type": assignment_type,
        }
    assigned_by = entry.get("assigned_by")
    source = entry.get("source")
    if not source:
        if assigned_by == "user":
            source = "manual"
        elif assigned_by == "auto":
            source = "auto"
    updated_at = entry.get("updated_at") or entry.get("timestamp")
    return {
        "cast_id": entry.get("cast_id"),
        "source": source,
        "updated_at": updated_at,
        "updated_by": entry.get("updated_by"),
        "assignment_type": assignment_type,
    }


def resolve_cluster_assignment(
    cluster_id: str | None,
    cluster_assignments: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if not cluster_id:
        return _normalize_assignment_entry(None, assignment_type="cluster_assignment")
    entry = cluster_assignments.get(str(cluster_id))
    return _normalize_assignment_entry(entry, assignment_type="cluster_assignment")


def resolve_track_assignment(
    track_id: int | str | None,
    cluster_id: str | None,
    cluster_assignments: Dict[str, Dict[str, Any]],
    track_overrides: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if track_id is not None:
        override = track_overrides.get(str(track_id))
        if isinstance(override, dict) and override.get("cast_id"):
            resolved = _normalize_assignment_entry(override, assignment_type="track_override")
            resolved["override"] = True
            return resolved
    resolved = resolve_cluster_assignment(cluster_id, cluster_assignments)
    resolved["override"] = False
    return resolved


def resolve_face_assignment(
    face_id: str | None,
    track_id: int | str | None,
    cluster_id: str | None,
    cluster_assignments: Dict[str, Dict[str, Any]],
    track_overrides: Dict[str, Dict[str, Any]],
    face_exclusions: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    exclusion = face_exclusions.get(str(face_id)) if face_id else None
    if isinstance(exclusion, dict) and exclusion.get("excluded", True):
        return {
            "cast_id": None,
            "source": exclusion.get("source"),
            "updated_at": exclusion.get("updated_at"),
            "updated_by": exclusion.get("updated_by"),
            "assignment_type": "face_exclusion",
            "excluded": True,
            "exclusion_reason": exclusion.get("reason"),
        }
    resolved = resolve_track_assignment(
        track_id,
        cluster_id,
        cluster_assignments,
        track_overrides,
    )
    resolved["excluded"] = False
    return resolved

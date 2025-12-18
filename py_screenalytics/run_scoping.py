from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def marker_status_success(marker_payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(marker_payload, Mapping):
        return False
    status = str(marker_payload.get("status") or "").strip().lower()
    if status != "success":
        return False
    if marker_payload.get("error"):
        return False
    return True


def marker_run_id_matches(marker_payload: Mapping[str, Any] | None, run_id: str | None) -> bool:
    if not run_id:
        return True
    if not isinstance(marker_payload, Mapping):
        return False
    marker_run_id = marker_payload.get("run_id")
    return isinstance(marker_run_id, str) and marker_run_id.strip() == run_id


def should_synthesize_detect_track_success(
    *,
    run_id: str | None,
    marker_payload: Mapping[str, Any] | None,
    detections_exists: bool,
    tracks_exists: bool,
) -> bool:
    """True only when detect/track manifests + run-scoped success marker agree."""
    if not detections_exists or not tracks_exists:
        return False
    if not marker_status_success(marker_payload):
        return False
    if not marker_run_id_matches(marker_payload, run_id):
        return False
    return True


def json_payload_matches_run_id(payload: Any, run_id: str | None) -> bool:
    if not run_id:
        return True
    if not isinstance(payload, dict):
        return False
    file_run_id = payload.get("run_id")
    return isinstance(file_run_id, str) and file_run_id.strip() == run_id


def json_file_matches_run_id(path: Path, run_id: str | None) -> bool:
    """True when JSON exists and run_id matches (or run_id is None for legacy)."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return json_payload_matches_run_id(payload, run_id)


"""Run-scoped processing state persistence + artifact pointers."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

from py_screenalytics import run_layout
from apps.api.services.storage import StorageService

from apps.api.services.run_persistence import run_persistence_service

LOGGER = logging.getLogger(__name__)

RUN_STATE_STAGES = (
    "detect_track",
    "faces_embed",
    "cluster",
    "screentime",
    "export",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _hash_params(params: Dict[str, Any] | None) -> str | None:
    if params is None:
        return None
    try:
        payload = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_params_hash(params: Dict[str, Any] | None) -> str | None:
    """Expose stable params hashing for idempotent job triggers."""
    return _hash_params(params)


def _split_stage_state(raw: Any) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    if not isinstance(raw, dict):
        return {}, None
    if "run_state" in raw or "status_snapshot" in raw:
        run_state = raw.get("run_state")
        run_state = run_state if isinstance(run_state, dict) else {}
        status_snapshot = raw.get("status_snapshot")
        status_snapshot = status_snapshot if isinstance(status_snapshot, dict) else None
        return run_state, status_snapshot
    return {}, raw


def _ensure_stage_defaults(stages: Dict[str, Any]) -> Dict[str, Any]:
    for stage in RUN_STATE_STAGES:
        entry = stages.get(stage)
        if not isinstance(entry, dict):
            entry = {}
        entry.setdefault("state", "pending")
        entry.setdefault("progress", 0.0)
        entry.setdefault("updated_at", None)
        stages[stage] = entry
    return stages


_STORAGE = StorageService()


def _artifact_entry(*, s3_key: str | None) -> Dict[str, Any]:
    entry = {"s3_key": s3_key, "exists": False}
    if s3_key:
        entry["exists"] = _STORAGE.object_exists(s3_key)
    return entry


def _build_artifact_pointers(ep_id: str, run_id: str) -> Dict[str, Any]:
    run_id_norm = run_layout.normalize_run_id(run_id)
    faces_key = run_layout.run_artifact_s3_key(ep_id, run_id_norm, "faces.jsonl")
    tracks_key = run_layout.run_artifact_s3_key(ep_id, run_id_norm, "tracks.jsonl")
    track_reps_key = run_layout.run_artifact_s3_key(ep_id, run_id_norm, "track_reps.jsonl")
    identities_key = run_layout.run_artifact_s3_key(ep_id, run_id_norm, "identities.json")
    embeddings_key = run_layout.run_artifact_s3_key(ep_id, run_id_norm, "faces.npy")
    faces_manifest_exists = _STORAGE.object_exists(faces_key)
    track_reps_exists = _STORAGE.object_exists(track_reps_key)
    embeddings_exists = _STORAGE.object_exists(embeddings_key)
    tracks_exists = _STORAGE.object_exists(tracks_key)
    faces_source: str | None = None
    if faces_manifest_exists:
        faces_source = "manifest"
    elif embeddings_exists or track_reps_exists:
        faces_source = "embeddings"
    elif tracks_exists:
        faces_source = "tracks"

    faces_entry = _artifact_entry(
        s3_key=faces_key,
    )
    faces_entry["manifest_key"] = faces_key
    faces_entry["manifest_exists"] = faces_manifest_exists
    faces_entry["source"] = faces_source

    return {
        "run_prefix": run_layout.run_s3_prefix(ep_id, run_id_norm),
        "tracks": _artifact_entry(
            s3_key=tracks_key,
        ),
        "faces": faces_entry,
        "suggestions": _artifact_entry(
            s3_key=run_layout.run_artifact_s3_key(ep_id, run_id_norm, "suggestions.json"),
        ),
        "identities": _artifact_entry(
            s3_key=identities_key,
        ),
        "track_reps": _artifact_entry(
            s3_key=track_reps_key,
        ),
        "crops": {
            "s3_prefix": f"{run_layout.run_s3_prefix(ep_id, run_id_norm)}crops/",
            "s3_layout": run_layout.get_run_s3_layout(ep_id, run_id_norm).s3_layout,
        },
        "embeddings": _artifact_entry(
            s3_key=embeddings_key,
        ),
        "exports": {
            "screentime_csv": _artifact_entry(
                s3_key=run_layout.run_artifact_s3_key(ep_id, run_id_norm, "analytics/screentime.csv"),
            ),
            "screentime_json": _artifact_entry(
                s3_key=run_layout.run_artifact_s3_key(ep_id, run_id_norm, "analytics/screentime.json"),
            ),
            "run_debug_pdf": _artifact_entry(
                s3_key=run_layout.run_export_s3_key(ep_id, run_id_norm, "run_debug.pdf"),
            ),
            "debug_bundle_zip": _artifact_entry(
                s3_key=run_layout.run_export_s3_key(ep_id, run_id_norm, "debug_bundle.zip"),
            ),
        },
    }


class RunStateService:
    """Persistence helper for run-scoped processing state."""

    def get_state(self, *, ep_id: str, run_id: str) -> Dict[str, Any]:
        run_id_norm = run_layout.normalize_run_id(run_id)
        run_row = run_persistence_service.get_run(ep_id=ep_id, run_id=run_id_norm) or {}
        run_state, status_snapshot = _split_stage_state(run_row.get("stage_state_json"))
        stages = run_state.get("stages") if isinstance(run_state.get("stages"), dict) else {}
        stages = _ensure_stage_defaults(stages)
        run_state = dict(run_state)
        run_state.setdefault("ep_id", ep_id)
        run_state.setdefault("run_id", run_id_norm)
        run_state["stages"] = stages
        run_state["updated_at"] = run_state.get("updated_at") or _now_iso()
        run_state["artifacts"] = _build_artifact_pointers(ep_id, run_id_norm)
        return {
            "run_state": run_state,
            "status_snapshot": status_snapshot,
        }

    def update_status_snapshot(
        self,
        *,
        ep_id: str,
        run_id: str,
        status_snapshot: Dict[str, Any],
    ) -> None:
        run_id_norm = run_layout.normalize_run_id(run_id)
        run_persistence_service.ensure_run(ep_id=ep_id, run_id=run_id_norm)
        run_row = run_persistence_service.get_run(ep_id=ep_id, run_id=run_id_norm) or {}
        run_state, _ = _split_stage_state(run_row.get("stage_state_json"))
        payload = {
            "run_state": run_state or {},
            "status_snapshot": status_snapshot,
        }
        run_persistence_service.update_run_stage_state(run_id=run_id_norm, stage_state_json=payload)

    def update_stage(
        self,
        *,
        ep_id: str,
        run_id: str,
        stage: str,
        state: str,
        progress: float | None = None,
        last_error: str | None = None,
        params: Dict[str, Any] | None = None,
        params_hash: str | None = None,
        job_id: str | None = None,
        source: str | None = None,
    ) -> Dict[str, Any]:
        run_id_norm = run_layout.normalize_run_id(run_id)
        run_persistence_service.ensure_run(ep_id=ep_id, run_id=run_id_norm)
        run_row = run_persistence_service.get_run(ep_id=ep_id, run_id=run_id_norm) or {}
        run_state, status_snapshot = _split_stage_state(run_row.get("stage_state_json"))

        stages = run_state.get("stages") if isinstance(run_state.get("stages"), dict) else {}
        entry = stages.get(stage)
        entry = dict(entry) if isinstance(entry, dict) else {}

        entry["state"] = state
        if progress is not None:
            entry["progress"] = float(progress)
        if params is not None:
            entry["params"] = params
        params_hash = params_hash or _hash_params(params)
        if params_hash:
            entry["params_hash"] = params_hash
        if job_id is not None:
            entry["job_id"] = job_id
        if source is not None:
            entry["source"] = source
        if last_error is not None:
            entry["last_error"] = last_error
        elif state in {"queued", "running", "done", "pending"}:
            entry["last_error"] = None

        entry["updated_at"] = _now_iso()
        stages[stage] = entry

        run_state = dict(run_state)
        run_state.setdefault("ep_id", ep_id)
        run_state.setdefault("run_id", run_id_norm)
        run_state["stages"] = stages
        run_state["updated_at"] = _now_iso()
        run_state["artifacts"] = _build_artifact_pointers(ep_id, run_id_norm)

        payload = {
            "run_state": run_state,
            "status_snapshot": status_snapshot,
        }
        run_persistence_service.update_run_stage_state(run_id=run_id_norm, stage_state_json=payload)
        return run_state

    def current_stage_entry(
        self,
        *,
        ep_id: str,
        run_id: str,
        stage: str,
    ) -> Dict[str, Any] | None:
        state = self.get_state(ep_id=ep_id, run_id=run_id)
        run_state = state.get("run_state") if isinstance(state, dict) else None
        if not isinstance(run_state, dict):
            return None
        stages = run_state.get("stages") if isinstance(run_state.get("stages"), dict) else {}
        entry = stages.get(stage)
        return dict(entry) if isinstance(entry, dict) else None


run_state_service = RunStateService()

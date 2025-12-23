from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from py_screenalytics import run_layout
from py_screenalytics.artifacts import get_path
from py_screenalytics.episode_status import Stage, StageStatus, normalize_stage_key
from py_screenalytics.run_gates import GateReason, get_stage_prereqs

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
_DIGEST_CACHE_FILENAME = "digests.json"


@dataclass(frozen=True)
class StageErrorInfo:
    code: str
    message: str
    exception_summary: str | None = None


@dataclass(frozen=True)
class StageBlockedInfo:
    reasons: list[GateReason]
    suggested_actions: list[str]


def stage_manifest_path(ep_id: str, run_id: str, stage: Stage | str) -> Path:
    stage_key = _coerce_stage_key(stage)
    run_root = run_layout.run_root(ep_id, run_layout.normalize_run_id(run_id))
    return run_root / "manifests" / f"{stage_key}.json"


def write_stage_manifest(
    episode_id: str,
    run_id: str,
    stage: Stage | str,
    status: StageStatus | str,
    *,
    started_at: datetime | str | None,
    finished_at: datetime | str | None,
    duration_s: float | None,
    inputs: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, str] | Iterable[Mapping[str, Any]] | None = None,
    model_versions: Mapping[str, str] | None = None,
    thresholds: Mapping[str, Any] | None = None,
    counts: Mapping[str, int] | None = None,
    error: StageErrorInfo | None = None,
    blocked: StageBlockedInfo | None = None,
    include_media: bool = True,
) -> Path:
    run_id_norm = run_layout.normalize_run_id(run_id)
    stage_key = _coerce_stage_key(stage)
    status_value = _coerce_status(status)
    payload = _build_manifest_payload(
        episode_id=episode_id,
        run_id=run_id_norm,
        stage_key=stage_key,
        status=status_value,
        started_at=_serialize_dt(started_at),
        finished_at=_serialize_dt(finished_at),
        duration_s=duration_s,
        inputs=inputs,
        artifacts=artifacts,
        model_versions=model_versions,
        thresholds=thresholds,
        counts=counts,
        error=error,
        blocked=blocked,
        include_media=include_media,
    )
    path = stage_manifest_path(episode_id, run_id_norm, stage_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def _build_manifest_payload(
    *,
    episode_id: str,
    run_id: str,
    stage_key: str,
    status: str,
    started_at: str | None,
    finished_at: str | None,
    duration_s: float | None,
    inputs: Mapping[str, Any] | None,
    artifacts: Mapping[str, str] | Iterable[Mapping[str, Any]] | None,
    model_versions: Mapping[str, str] | None,
    thresholds: Mapping[str, Any] | None,
    counts: Mapping[str, int] | None,
    error: StageErrorInfo | None,
    blocked: StageBlockedInfo | None,
    include_media: bool,
) -> dict[str, Any]:
    run_root = run_layout.run_root(episode_id, run_id)
    cache = _load_digest_cache(run_root)
    input_payload = dict(inputs) if isinstance(inputs, Mapping) else {}
    if "artifacts" not in input_payload:
        input_payload["artifacts"] = _input_artifacts(
            episode_id,
            run_id,
            stage_key,
            cache,
        )
    if include_media:
        media_inputs = _media_inputs(episode_id, cache)
        if media_inputs:
            input_payload.setdefault("media", media_inputs)
    artifact_entries = _artifact_entries(artifacts, cache)
    model_versions_payload = {str(k): str(v) for k, v in (model_versions or {}).items()}
    thresholds_payload = {str(k): v for k, v in (thresholds or {}).items()}
    counts_payload = {str(k): int(v) for k, v in (counts or {}).items() if isinstance(v, int)}

    if duration_s is None:
        duration_s = _duration_s(started_at, finished_at)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "run_id": run_id,
        "stage": stage_key,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration_s,
        "inputs": input_payload,
        "model_versions": model_versions_payload,
        "thresholds": thresholds_payload,
        "counts": counts_payload,
        "artifacts": artifact_entries,
    }

    if error is not None:
        payload["error"] = {
            "error_code": error.code,
            "error_message": error.message,
            "exception_summary": error.exception_summary,
        }
    if blocked is not None:
        payload["blocked"] = {
            "blocked_reasons": [reason.as_dict() for reason in blocked.reasons],
            "suggested_actions": list(blocked.suggested_actions),
        }

    _persist_digest_cache(run_root, cache)
    return payload


def _artifact_entries(
    artifacts: Mapping[str, str] | Iterable[Mapping[str, Any]] | None,
    cache: dict[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if artifacts is None:
        return entries
    if isinstance(artifacts, Mapping):
        for name, path in artifacts.items():
            if not isinstance(name, str) or not isinstance(path, str):
                continue
            entries.append(_artifact_entry(name, Path(path), cache))
        return entries
    for entry in artifacts:
        if not isinstance(entry, Mapping):
            continue
        label = entry.get("logical_name") or entry.get("label")
        path = entry.get("path")
        if isinstance(label, str) and isinstance(path, str):
            entries.append(_artifact_entry(label, Path(path), cache))
    return entries


def _artifact_entry(label: str, path: Path, cache: dict[str, Any]) -> dict[str, Any]:
    return {
        "logical_name": label,
        "path": str(path),
        "sha256": _sha256(path, cache),
    }


def _input_artifacts(
    episode_id: str,
    run_id: str,
    stage_key: str,
    cache: dict[str, Any],
) -> list[dict[str, Any]]:
    prereqs = get_stage_prereqs(stage_key)
    artifacts: list[dict[str, Any]] = []
    run_root = run_layout.run_root(episode_id, run_id)
    for req in prereqs.artifacts:
        path = run_root / req.rel_path
        artifacts.append(
            {
                "logical_name": req.logical_name,
                "path": str(path),
                "sha256": _sha256(path, cache),
            }
        )
    return artifacts


def _media_inputs(episode_id: str, cache: dict[str, Any]) -> list[dict[str, Any]]:
    media: list[dict[str, Any]] = []
    try:
        video_path = get_path(episode_id, "video")
    except Exception:
        return media
    if video_path.exists():
        media.append(_artifact_entry("video", video_path, cache))
    return media


def _sha256(path: Path, cache: dict[str, Any]) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    cached = _cached_digest(path, cache)
    if cached:
        return cached
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    value = digest.hexdigest()
    _store_digest(path, value, cache)
    return value


def _load_digest_cache(run_root: Path) -> dict[str, Any]:
    path = run_root / _DIGEST_CACHE_FILENAME
    if not path.exists():
        return {"version": 1, "items": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "items": {}}
    if not isinstance(payload, dict):
        return {"version": 1, "items": {}}
    if not isinstance(payload.get("items"), dict):
        payload["items"] = {}
    return payload


def _persist_digest_cache(run_root: Path, cache: dict[str, Any]) -> None:
    path = run_root / _DIGEST_CACHE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    tmp.replace(path)


def _cached_digest(path: Path, cache: dict[str, Any]) -> str | None:
    items = cache.get("items")
    if not isinstance(items, dict):
        return None
    key = str(path)
    entry = items.get(key)
    if not isinstance(entry, dict):
        return None
    try:
        size = int(entry.get("size"))
        mtime = float(entry.get("mtime"))
    except (TypeError, ValueError):
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    if stat.st_size != size or float(stat.st_mtime) != mtime:
        return None
    digest = entry.get("sha256")
    return digest if isinstance(digest, str) else None


def _store_digest(path: Path, digest: str, cache: dict[str, Any]) -> None:
    items = cache.setdefault("items", {})
    if not isinstance(items, dict):
        return
    try:
        stat = path.stat()
    except OSError:
        return
    items[str(path)] = {
        "sha256": digest,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


def _duration_s(started_at: str | None, finished_at: str | None) -> float | None:
    if not started_at or not finished_at:
        return None
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    delta = (end - start).total_seconds()
    return delta if delta >= 0 else None


def _serialize_dt(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _coerce_stage_key(stage: Stage | str) -> str:
    if isinstance(stage, Stage):
        return stage.value
    normalized = normalize_stage_key(stage)
    if not normalized:
        raise ValueError(f"Unknown stage key: {stage!r}")
    return normalized


def _coerce_status(status: StageStatus | str) -> str:
    if isinstance(status, StageStatus):
        value = status.name
    else:
        value = str(status)
    return value.strip().upper()

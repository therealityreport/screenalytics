from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Iterable

from py_screenalytics import run_layout

LOGGER = logging.getLogger(__name__)

FACES_REVIEW_REQUIRED_ARTIFACTS = (
    "identities.json",
    "tracks.jsonl",
)

FACES_REVIEW_OPTIONAL_ARTIFACTS = (
    "faces.jsonl",
    "cluster.json",
    "cluster_centroids.json",
    "track_reps.jsonl",
    "faces_embed.json",
)


@dataclass
class RunArtifactHydration:
    run_id: str
    required: tuple[str, ...]
    optional: tuple[str, ...]
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    missing_remote: list[str] = field(default_factory=list)
    hydrated: list[str] = field(default_factory=list)
    storage_enabled: bool = False

    @property
    def ok(self) -> bool:
        return not self.missing_required


def resolve_run_artifact_paths(
    ep_id: str,
    run_id: str,
    rel_paths: Iterable[str],
) -> dict[str, Path]:
    run_id_norm = run_layout.normalize_run_id(run_id)
    run_root = run_layout.run_root(ep_id, run_id_norm)
    return {rel_path: run_root / rel_path for rel_path in rel_paths}


def ensure_run_artifacts_local(
    ep_id: str,
    run_id: str,
    required: Iterable[str],
    *,
    optional: Iterable[str] = (),
    storage=None,
) -> RunArtifactHydration:
    run_id_norm = run_layout.normalize_run_id(run_id)
    required_tuple = tuple(required)
    optional_tuple = tuple(optional)
    result = RunArtifactHydration(
        run_id=run_id_norm,
        required=required_tuple,
        optional=optional_tuple,
    )
    run_root = run_layout.run_root(ep_id, run_id_norm)

    required_paths = resolve_run_artifact_paths(ep_id, run_id_norm, required_tuple)
    optional_paths = resolve_run_artifact_paths(ep_id, run_id_norm, optional_tuple)

    for rel_path, local_path in required_paths.items():
        if not local_path.exists():
            result.missing_required.append(rel_path)

    for rel_path, local_path in optional_paths.items():
        if not local_path.exists():
            result.missing_optional.append(rel_path)

    if not result.missing_required:
        return result

    if storage is None:
        try:
            from apps.api.services.storage import StorageService

            storage = StorageService()
        except Exception as exc:
            LOGGER.warning("[faces-review] Storage init failed: %s", exc)
            return result

    try:
        result.storage_enabled = bool(storage.s3_enabled())
    except Exception:
        result.storage_enabled = False

    if not result.storage_enabled:
        result.missing_remote = list(result.missing_required)
        return result

    for rel_path in list(result.missing_required):
        local_path = run_root / rel_path
        s3_key = None
        try:
            keys = run_layout.run_artifact_s3_keys_for_read(ep_id, run_id_norm, rel_path)
        except Exception as exc:
            LOGGER.debug("[faces-review] Failed to resolve S3 keys for %s: %s", rel_path, exc)
            keys = []
        for candidate in keys:
            try:
                if storage.object_exists(candidate):
                    s3_key = candidate
                    break
            except Exception as exc:
                LOGGER.debug("[faces-review] S3 exists check failed for %s: %s", candidate, exc)
        if not s3_key:
            result.missing_remote.append(rel_path)
            continue
        payload = storage.download_bytes(s3_key)
        if payload is None:
            result.missing_remote.append(rel_path)
            continue
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(payload)
            result.hydrated.append(rel_path)
        except OSError as exc:
            LOGGER.warning("[faces-review] Failed to write %s: %s", local_path, exc)
            result.missing_remote.append(rel_path)

    result.missing_required = [
        rel_path for rel_path in required_tuple if not (run_root / rel_path).exists()
    ]
    return result

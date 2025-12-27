"""Run-scoped integrity validator for Faces Review artifacts."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from py_screenalytics import run_layout
from py_screenalytics.artifacts import get_path

from apps.api.services.run_state import run_state_service

LOGGER = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    def _generator() -> Iterable[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row
    return _generator()


def _parse_track_id(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.replace("track_", "")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _crop_exists(
    frames_root: Path,
    run_id: str,
    rel_path: str,
) -> bool:
    cleaned = rel_path.lstrip("/\\")
    if cleaned.startswith("crops/"):
        cleaned = cleaned[len("crops/") :]
    candidates = [
        frames_root / "runs" / run_id / "crops" / cleaned,
        frames_root / "crops" / cleaned,
    ]
    return any(candidate.exists() for candidate in candidates)


def validate_run_integrity(
    ep_id: str,
    run_id: str,
    *,
    data_root: Path | str | None = None,
) -> Dict[str, Any]:
    """Return a structured validation report for run artifacts."""
    run_id_norm = run_layout.normalize_run_id(run_id)
    run_root = run_layout.run_root(ep_id, run_id_norm)
    now = _now_iso()

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "tracks": 0,
        "faces": 0,
        "identities": 0,
        "track_reps": 0,
    }

    if not run_root.exists():
        errors.append(
            {
                "code": "missing_run_root",
                "message": "Run artifacts folder not found on disk.",
                "details": {"run_root": str(run_root)},
            }
        )
        return {
            "ep_id": ep_id,
            "run_id": run_id_norm,
            "generated_at": now,
            "errors": errors,
            "warnings": warnings,
            "stats": stats,
            "summary": {
                "blocking": True,
                "error_count": len(errors),
                "warning_count": len(warnings),
                "blocking_issues": [err.get("code") for err in errors],
            },
        }

    tracks_path = run_root / "tracks.jsonl"
    faces_path = run_root / "faces.jsonl"
    identities_path = run_root / "identities.json"
    reps_path = run_root / "track_reps.jsonl"

    artifact_pointers: Dict[str, Any] = {}
    faces_artifacts: Dict[str, Any] = {}
    try:
        run_state_bundle = run_state_service.get_state(ep_id=ep_id, run_id=run_id_norm)
        run_state_payload = run_state_bundle.get("run_state") if isinstance(run_state_bundle, dict) else None
        if isinstance(run_state_payload, dict):
            artifact_pointers = run_state_payload.get("artifacts") if isinstance(run_state_payload.get("artifacts"), dict) else {}
            faces_artifacts = artifact_pointers.get("faces") if isinstance(artifact_pointers.get("faces"), dict) else {}
    except Exception as exc:
        LOGGER.debug("Run state lookup failed for validator (%s/%s): %s", ep_id, run_id_norm, exc)
        artifact_pointers = {}
        faces_artifacts = {}

    faces_source = faces_artifacts.get("source")
    faces_manifest_path = faces_artifacts.get("manifest_path") or str(faces_path)
    faces_manifest_key = faces_artifacts.get("manifest_key") or faces_artifacts.get("s3_key")
    faces_manifest_exists = faces_artifacts.get("manifest_exists")
    if faces_manifest_exists is None:
        faces_manifest_exists = faces_path.exists()
    if faces_source is None:
        if faces_manifest_exists:
            faces_source = "manifest"
        elif reps_path.exists() or (Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser() / "embeds" / ep_id / "runs" / run_id_norm / "faces.npy").exists():
            faces_source = "embeddings"
        elif tracks_path.exists():
            faces_source = "tracks"

    track_ids: Set[int] = set()
    track_rows: Dict[int, Dict[str, Any]] = {}
    face_ids: Set[str] = set()
    identity_ids: Set[str] = set()
    identity_track_ids: Set[int] = set()
    rep_track_ids: Set[int] = set()
    rep_lookup: Dict[int, Dict[str, Any]] = {}
    faces_estimate = 0
    faces_estimate_present = False
    missing_tracks_for_faces = 0
    missing_tracks_for_identities = 0
    missing_tracks_for_reps = 0
    missing_crops_for_reps = 0

    if not tracks_path.exists():
        errors.append(
            {
                "code": "missing_tracks",
                "message": "tracks.jsonl is missing for this run.",
                "details": {"path": str(tracks_path)},
            }
        )
    else:
        for row in _iter_jsonl(tracks_path):
            track_id = _parse_track_id(row.get("track_id") or row.get("track"))
            if track_id is None:
                continue
            track_ids.add(track_id)
            track_rows[track_id] = row
            faces_count = row.get("faces_count")
            if faces_count is not None:
                try:
                    faces_estimate += int(faces_count)
                    faces_estimate_present = True
                except (TypeError, ValueError):
                    pass
        stats["tracks"] = len(track_ids)

    if not faces_manifest_exists:
        if faces_source in (None, "manifest"):
            errors.append(
                {
                    "code": "missing_faces",
                    "message": "faces.jsonl is missing for this run.",
                    "details": {
                        "path": faces_manifest_path,
                        "faces_source": faces_source,
                        "manifest_key": faces_manifest_key,
                    },
                }
            )
        else:
            warnings.append(
                {
                    "code": "faces_manifest_missing",
                    "message": "faces.jsonl is missing; using alternate faces source.",
                    "details": {
                        "path": faces_manifest_path,
                        "faces_source": faces_source,
                        "manifest_key": faces_manifest_key,
                    },
                }
            )
    else:
        for row in _iter_jsonl(faces_path):
            track_id = _parse_track_id(row.get("track_id") or row.get("track"))
            if track_id is not None and track_id not in track_ids and track_ids:
                missing_tracks_for_faces += 1
            face_id = row.get("face_id")
            if face_id:
                face_ids.add(str(face_id))
        stats["faces"] = len(face_ids) if face_ids else stats.get("faces", 0)

    if not identities_path.exists():
        errors.append(
            {
                "code": "missing_identities",
                "message": "identities.json is missing for this run.",
                "details": {"path": str(identities_path)},
            }
        )
        identities_payload: Dict[str, Any] = {}
    else:
        payload = _read_json(identities_path)
        identities_payload = payload if isinstance(payload, dict) else {}
        identities = identities_payload.get("identities")
        if isinstance(identities, list):
            for entry in identities:
                if not isinstance(entry, dict):
                    continue
                identity_id = entry.get("identity_id") or entry.get("id")
                if identity_id:
                    identity_ids.add(str(identity_id))
                for raw in entry.get("track_ids", []) or []:
                    try:
                        track_id = int(str(raw).replace("track_", ""))
                    except (TypeError, ValueError):
                        continue
                    identity_track_ids.add(track_id)
                    if track_id not in track_ids and track_ids:
                        missing_tracks_for_identities += 1
        stats["identities"] = len(identity_ids)

    if reps_path.exists():
        for row in _iter_jsonl(reps_path):
            track_id = _parse_track_id(row.get("track_id") or row.get("track"))
            if track_id is not None:
                rep_track_ids.add(track_id)
                rep_lookup[track_id] = row
                if track_id not in track_ids and track_ids:
                    missing_tracks_for_reps += 1

            crop_rel = row.get("best_crop_rel_path") or row.get("crop_rel_path")
            if isinstance(crop_rel, str):
                frames_root = get_path(ep_id, "frames_root")
                if not _crop_exists(frames_root, run_id_norm, crop_rel):
                    missing_crops_for_reps += 1
        stats["track_reps"] = len(rep_track_ids)

    if missing_tracks_for_faces:
        errors.append(
            {
                "code": "faces_missing_tracks",
                "message": "Some faces reference missing track IDs.",
                "details": {"count": missing_tracks_for_faces},
            }
        )
    if missing_tracks_for_identities:
        errors.append(
            {
                "code": "identities_missing_tracks",
                "message": "Some identities reference missing track IDs.",
                "details": {"count": missing_tracks_for_identities},
            }
        )
    if missing_tracks_for_reps:
        errors.append(
            {
                "code": "track_reps_missing_tracks",
                "message": "Some track representatives reference missing track IDs.",
                "details": {"count": missing_tracks_for_reps},
            }
        )

    unclustered_by_reason: Dict[str, List[int]] = {}
    if track_ids and identity_track_ids:
        missing_cluster_refs = track_ids - identity_track_ids
        if missing_cluster_refs:
            reps_present = reps_path.exists()
            for track_id in sorted(missing_cluster_refs):
                reason = "not_in_identities"
                track_row = track_rows.get(track_id, {})
                frame_count = track_row.get("frame_count") or track_row.get("frames")
                try:
                    if frame_count is not None and int(frame_count) <= 1:
                        reason = "filtered_single_frame"
                except (TypeError, ValueError):
                    pass
                if reason == "not_in_identities":
                    if track_row.get("excluded") or track_row.get("excluded_reason") or track_row.get("filtered_reason") or track_row.get("skip_reason"):
                        reason = "excluded_by_params"
                if reason == "not_in_identities":
                    rep = rep_lookup.get(track_id)
                    if rep and rep.get("no_embeddings"):
                        reason = "missing_embeddings"
                    elif reps_present and track_id not in rep_lookup:
                        reason = "missing_embeddings"

                unclustered_by_reason.setdefault(reason, []).append(track_id)

            samples: List[int] = []
            for reason_tracks in unclustered_by_reason.values():
                for tid in reason_tracks:
                    samples.append(tid)
                    if len(samples) >= 10:
                        break
                if len(samples) >= 10:
                    break
            warnings.append(
                {
                    "code": "tracks_without_clusters",
                    "message": "Some tracks are not assigned to any cluster identity.",
                    "details": {
                        "count": len(missing_cluster_refs),
                        "by_reason": {k: len(v) for k, v in unclustered_by_reason.items()},
                        "sample_track_ids": samples,
                    },
                }
            )

    manual_assignments = identities_payload.get("manual_assignments") if isinstance(identities_payload, dict) else None
    track_overrides = identities_payload.get("track_overrides") if isinstance(identities_payload, dict) else None
    face_exclusions = identities_payload.get("face_exclusions") if isinstance(identities_payload, dict) else None

    if isinstance(manual_assignments, dict):
        for cluster_id in manual_assignments.keys():
            if identity_ids and str(cluster_id) not in identity_ids:
                errors.append(
                    {
                        "code": "assignment_missing_cluster",
                        "message": "Cluster assignment references a missing cluster ID.",
                        "details": {"cluster_id": str(cluster_id)},
                    }
                )
    if isinstance(track_overrides, dict):
        for track_id_raw in track_overrides.keys():
            try:
                track_id = int(str(track_id_raw))
            except (TypeError, ValueError):
                continue
            if track_ids and track_id not in track_ids:
                errors.append(
                    {
                        "code": "assignment_missing_track",
                        "message": "Track override references a missing track ID.",
                        "details": {"track_id": track_id},
                    }
                )
    if isinstance(face_exclusions, dict):
        for face_id in face_exclusions.keys():
            if face_ids and str(face_id) not in face_ids:
                errors.append(
                    {
                        "code": "exclusion_missing_face",
                        "message": "Face exclusion references a missing face ID.",
                        "details": {"face_id": str(face_id)},
                    }
                )

    if missing_crops_for_reps:
        warnings.append(
            {
                "code": "missing_crops",
                "message": "Some track representatives are missing crop files.",
                "details": {"count": missing_crops_for_reps},
            }
        )

    embeddings_path = None
    if data_root is None:
        data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
    embeds_root = Path(data_root) / "embeds" / ep_id / "runs" / run_id_norm
    embeddings_path = embeds_root / "faces.npy"
    if stats.get("faces", 0) and not embeddings_path.exists():
        warnings.append(
            {
                "code": "missing_embeddings",
                "message": "Embeddings file missing for faces.",
                "details": {"path": str(embeddings_path)},
            }
        )
    elif embeddings_path.exists() and stats.get("faces", 0):
        try:
            import numpy as np  # type: ignore

            embeddings = np.load(str(embeddings_path), mmap_mode="r")
            embed_count = int(embeddings.shape[0]) if hasattr(embeddings, "shape") else None
            if embed_count is not None and face_ids and embed_count != len(face_ids):
                warnings.append(
                    {
                        "code": "embedding_count_mismatch",
                        "message": "Embeddings count does not match faces count.",
                        "details": {"embeddings": embed_count, "faces": len(face_ids)},
                    }
                )
        except Exception as exc:  # pragma: no cover - best effort
            LOGGER.debug("Embedding check skipped: %s", exc)

    if faces_estimate_present and not stats.get("faces"):
        stats["faces_estimate"] = faces_estimate

    return {
        "ep_id": ep_id,
        "run_id": run_id_norm,
        "generated_at": now,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
        "artifact_pointers": {
            "faces": {
                "source": faces_source,
                "manifest_path": faces_manifest_path,
                "manifest_key": faces_manifest_key,
                "manifest_exists": faces_manifest_exists,
                "exists": faces_artifacts.get("exists") if faces_artifacts else None,
            }
        },
        "summary": {
            "blocking": len(errors) > 0,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "blocking_issues": [err.get("code") for err in errors],
            "unclustered_tracks": sum(len(v) for v in unclustered_by_reason.values()) if unclustered_by_reason else 0,
            "unclustered_tracks_by_reason": {k: len(v) for k, v in unclustered_by_reason.items()},
            "unclustered_track_samples": [tid for tids in unclustered_by_reason.values() for tid in tids[:3]][:10],
        },
    }

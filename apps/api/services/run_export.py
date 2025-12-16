"""Run debug bundle export service.

Builds a single zip that captures everything needed to reconstruct and debug a
specific (ep_id, run_id) flow end-to-end.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from py_screenalytics import run_layout
from py_screenalytics.artifacts import get_path

LOGGER = logging.getLogger(__name__)


_DEFAULT_EXCLUDE_FILENAMES: set[str] = {
    "run_summary.json",
    "jobs.json",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _safe_add_file(zip_handle: zipfile.ZipFile, src: Path, *, arcname: str) -> None:
    try:
        if src.exists() and src.is_file():
            zip_handle.write(src, arcname=arcname)
    except OSError as exc:
        LOGGER.warning("[export] Failed to add file %s: %s", src, exc)


def _safe_add_dir(
    zip_handle: zipfile.ZipFile,
    root: Path,
    *,
    arc_prefix: str,
    include_predicate=None,
) -> None:
    if not root.exists() or not root.is_dir():
        return
    try:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if include_predicate and not include_predicate(path):
                continue
            arcname = f"{arc_prefix.rstrip('/')}/{rel}"
            _safe_add_file(zip_handle, path, arcname=arcname)
    except OSError as exc:
        LOGGER.warning("[export] Failed to add directory %s: %s", root, exc)


def _identity_assignments_snapshot(identities_payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = identities_payload or {}
    manual_assignments = payload.get("manual_assignments") if isinstance(payload, dict) else None
    if not isinstance(manual_assignments, dict):
        manual_assignments = {}
    identities = payload.get("identities") if isinstance(payload, dict) else None
    if not isinstance(identities, list):
        identities = []

    rows: list[dict[str, Any]] = []
    for identity in identities:
        if not isinstance(identity, dict):
            continue
        identity_id = identity.get("identity_id") or identity.get("id")
        if not identity_id:
            continue
        identity_id_str = str(identity_id)
        person_id = identity.get("person_id")
        meta = manual_assignments.get(identity_id_str) if isinstance(manual_assignments, dict) else None
        if not isinstance(meta, dict):
            meta = {}
        assigned_by = meta.get("assigned_by")
        method = "manual" if assigned_by == "user" else ("auto" if assigned_by == "auto" else None)
        rows.append(
            {
                "identity_id": identity_id_str,
                "person_id": person_id,
                "name": identity.get("name"),
                "label": identity.get("label"),
                "track_ids": identity.get("track_ids") or [],
                "method": method,
                "assigned_by": assigned_by,
                "cast_id": meta.get("cast_id"),
                "timestamp": meta.get("timestamp"),
            }
        )

    return {
        "schema_version": 1,
        "counts": {
            "identities_total": len(identities),
            "assigned_identities": sum(1 for row in rows if row.get("person_id")),
        },
        "assignments": rows,
    }


def _jobs_snapshot(run_root: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    if not run_root.exists():
        return jobs

    for path in sorted(run_root.glob("*.json")):
        if path.name in _DEFAULT_EXCLUDE_FILENAMES:
            continue
        if path.name in {"identities.json", "cluster_centroids.json", "group_progress.json", "group_log.json"}:
            continue
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        # Heuristic: job markers typically include one of these.
        if not any(key in data for key in ("phase", "stage", "job_type", "status", "started_at", "finished_at")):
            continue
        job = dict(data)
        job.setdefault("source_path", path.name)
        jobs.append(job)

    # Include any progress_* files as job-like traces.
    for path in sorted(run_root.glob("progress*.json")):
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        jobs.append(
            {
                "job_type": "progress",
                "source_path": path.name,
                "payload": data,
            }
        )

    return jobs


def build_run_debug_bundle_zip(
    *,
    ep_id: str,
    run_id: str,
    include_artifacts: bool = True,
    include_images: bool = False,
    include_logs: bool = True,
) -> tuple[str, str]:
    """Build a run-scoped debug bundle zip.

    Returns:
        (zip_path, download_filename)
    """
    run_id_norm = run_layout.normalize_run_id(run_id)
    run_root = run_layout.run_root(ep_id, run_id_norm)
    if not run_root.exists():
        raise FileNotFoundError(f"Run not found on disk: {run_root}")

    runs_root = run_layout.runs_root(ep_id)
    manifests_root = get_path(ep_id, "detections").parent

    identities_payload = _read_json(run_root / "identities.json")
    identities_payload = identities_payload if isinstance(identities_payload, dict) else None

    db_error: str | None = None
    run_row: dict[str, Any] | None = None
    job_runs: list[dict[str, Any]] = []
    identity_locks: list[dict[str, Any]] = []
    suggestion_batches: list[dict[str, Any]] = []
    suggestions_rows: list[dict[str, Any]] = []
    suggestion_applies: list[dict[str, Any]] = []
    try:
        from apps.api.services.run_persistence import run_persistence_service

        run_row = run_persistence_service.get_run(ep_id=ep_id, run_id=run_id_norm)
        job_runs = run_persistence_service.list_job_runs(ep_id=ep_id, run_id=run_id_norm)
        identity_locks = run_persistence_service.list_identity_locks(ep_id=ep_id, run_id=run_id_norm)
        suggestion_batches = run_persistence_service.list_suggestion_batches(ep_id=ep_id, run_id=run_id_norm, limit=250)
        for batch in suggestion_batches:
            batch_id = batch.get("batch_id") if isinstance(batch, dict) else None
            if batch_id:
                suggestions_rows.extend(
                    run_persistence_service.list_suggestions(
                        ep_id=ep_id,
                        run_id=run_id_norm,
                        batch_id=str(batch_id),
                        include_dismissed=True,
                    )
                )
        suggestion_applies = run_persistence_service.list_suggestion_applies(ep_id=ep_id, run_id=run_id_norm)
    except Exception as exc:
        db_error = str(exc)

    run_summary: dict[str, Any] = {
        "schema_version": 2,
        "ep_id": ep_id,
        "run_id": run_id_norm,
        "generated_at": _now_iso(),
        "paths": {
            "run_root": str(run_root),
        },
        "db": {
            "run_record": run_row,
            "error": db_error,
        },
        "toggles": {
            "include_artifacts": bool(include_artifacts),
            "include_images": bool(include_images),
            "include_logs": bool(include_logs),
        },
        "artifacts_present": {
            "detections_jsonl": (run_root / "detections.jsonl").exists(),
            "tracks_jsonl": (run_root / "tracks.jsonl").exists(),
            "track_metrics_json": (run_root / "track_metrics.json").exists(),
            "faces_jsonl": (run_root / "faces.jsonl").exists(),
            "faces_npy": (Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser() / "embeds" / ep_id / "runs" / run_id_norm / "faces.npy").exists(),
            "identities_json": (run_root / "identities.json").exists(),
            "cluster_centroids_json": (run_root / "cluster_centroids.json").exists(),
            "track_reps_jsonl": (run_root / "track_reps.jsonl").exists(),
            "face_review_state_json": (run_root / "face_review_state.json").exists(),
            "dismissed_suggestions_json": (run_root / "dismissed_suggestions.json").exists(),
            "group_log_json": (run_root / "group_log.json").exists(),
        },
        "cluster_config": (identities_payload or {}).get("config") if identities_payload else None,
        "cluster_stats": (identities_payload or {}).get("stats") if identities_payload else None,
    }

    jobs_payload = {
        "schema_version": 2,
        "ep_id": ep_id,
        "run_id": run_id_norm,
        "source": "job_runs" if not db_error else "job_runs_failed_fallback_to_disk",
        "jobs": job_runs,
        "supplemental_marker_jobs": _jobs_snapshot(run_root),
        "db_error": db_error,
    }

    assignments_payload = _identity_assignments_snapshot(identities_payload)
    assignments_payload["ep_id"] = ep_id
    assignments_payload["run_id"] = run_id_norm
    assignments_payload["generated_at"] = run_summary["generated_at"]

    locks_payload = {
        "schema_version": 1,
        "ep_id": ep_id,
        "run_id": run_id_norm,
        "generated_at": run_summary["generated_at"],
        "locks": identity_locks,
        "source": "identity_locks",
        "db_error": db_error,
    }

    batches_payload = {
        "schema_version": 1,
        "ep_id": ep_id,
        "run_id": run_id_norm,
        "generated_at": run_summary["generated_at"],
        "batches": suggestion_batches,
        "source": "suggestion_batches",
        "db_error": db_error,
    }

    suggestions_payload = {
        "schema_version": 1,
        "ep_id": ep_id,
        "run_id": run_id_norm,
        "generated_at": run_summary["generated_at"],
        "suggestions": suggestions_rows,
        "source": "suggestions",
        "db_error": db_error,
    }

    applied_payload = {
        "schema_version": 1,
        "ep_id": ep_id,
        "run_id": run_id_norm,
        "generated_at": run_summary["generated_at"],
        "applies": suggestion_applies,
        "source": "suggestion_applies",
        "db_error": db_error,
    }

    tmp = tempfile.NamedTemporaryFile(prefix="screenalytics_run_debug_", suffix=".zip", delete=False)
    zip_path = tmp.name
    tmp.close()

    try:
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_handle:
            zip_handle.writestr("run_summary.json", json.dumps(run_summary, indent=2, ensure_ascii=False))
            zip_handle.writestr("jobs.json", json.dumps(jobs_payload, indent=2, ensure_ascii=False))
            zip_handle.writestr("identity_assignments.json", json.dumps(assignments_payload, indent=2, ensure_ascii=False))
            zip_handle.writestr("identity_locks.json", json.dumps(locks_payload, indent=2, ensure_ascii=False))
            zip_handle.writestr(
                "smart_suggestion_batches.json",
                json.dumps(batches_payload, indent=2, ensure_ascii=False),
            )
            zip_handle.writestr("smart_suggestions.json", json.dumps(suggestions_payload, indent=2, ensure_ascii=False))
            zip_handle.writestr(
                "smart_suggestions_applied.json",
                json.dumps(applied_payload, indent=2, ensure_ascii=False),
            )

            # Run-scoped artifacts
            if include_artifacts:
                for filename in (
                    "detections.jsonl",
                    "tracks.jsonl",
                    "track_metrics.json",
                    "faces.jsonl",
                    "identities.json",
                    "cluster_centroids.json",
                    "track_reps.jsonl",
                    "faces_ops.jsonl",
                ):
                    _safe_add_file(zip_handle, run_root / filename, arcname=filename)
                _safe_add_dir(zip_handle, run_root / "body_tracking", arc_prefix="body_tracking")
                _safe_add_dir(zip_handle, run_root / "analytics", arc_prefix="analytics")

                faces_npy = (
                    Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
                    / "embeds"
                    / ep_id
                    / "runs"
                    / run_id_norm
                    / "faces.npy"
                )
                _safe_add_file(zip_handle, faces_npy, arcname="faces.npy")

            # Images (thumbs/crops/frames) can be huge - gated behind include_images.
            if include_images:
                frames_root = get_path(ep_id, "frames_root")
                run_frames = frames_root / "runs" / run_id_norm
                _safe_add_dir(zip_handle, run_frames / "thumbs", arc_prefix="images/thumbs")
                _safe_add_dir(zip_handle, run_frames / "crops", arc_prefix="images/crops")
                _safe_add_dir(zip_handle, run_frames / "frames", arc_prefix="images/frames")

            # Logs (episode-wide) - stored under manifests/{ep_id}/logs
            if include_logs:
                logs_dir = manifests_root / "logs"
                _safe_add_dir(zip_handle, logs_dir, arc_prefix="logs")
                _safe_add_file(zip_handle, runs_root / run_layout.ACTIVE_RUN_FILENAME, arcname="active_run.json")

                # Include legacy phase markers for context (single file per phase; may not match this run).
                for phase in (
                    "detect_track",
                    "faces_embed",
                    "cluster",
                    "episode_cleanup",
                    "body_tracking",
                    "body_tracking_fusion",
                ):
                    _safe_add_file(zip_handle, runs_root / f"{phase}.json", arcname=f"legacy_markers/{phase}.json")

                for filename in ("cleanup_report.json",):
                    _safe_add_file(zip_handle, manifests_root / filename, arcname=f"legacy_artifacts/{filename}")

    except Exception:
        try:
            os.unlink(zip_path)
        except OSError:
            pass
        raise

    download_name = f"screenalytics_{ep_id}_{run_id_norm}_run_debug_bundle.zip"
    return zip_path, download_name

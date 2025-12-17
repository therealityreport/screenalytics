"""Run debug bundle export service.

Builds a single zip that captures everything needed to reconstruct and debug a
specific (ep_id, run_id) flow end-to-end.

Also provides PDF report generation for Screen Time Run Debug Reports.
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

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

    allowed_suffixes = {".json", ".jsonl", ".log", ".txt"}

    def _allowlisted_bundle_path(path: Path) -> bool:
        return path.suffix.lower() in allowed_suffixes

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

            # Always include the PDF debug report for quick inspection.
            try:
                pdf_bytes, _pdf_name = build_screentime_run_debug_pdf(ep_id=ep_id, run_id=run_id_norm)
            except Exception as exc:
                LOGGER.warning("[export] Failed to build debug report PDF: %s", exc)
            else:
                zip_handle.writestr("debug_report.pdf", pdf_bytes)

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
                _safe_add_dir(
                    zip_handle,
                    run_root / "body_tracking",
                    arc_prefix="body_tracking",
                    include_predicate=_allowlisted_bundle_path,
                )
                _safe_add_dir(
                    zip_handle,
                    run_root / "analytics",
                    arc_prefix="analytics",
                    include_predicate=_allowlisted_bundle_path,
                )

            # Logs (episode-wide) - stored under manifests/{ep_id}/logs
            if include_logs:
                logs_dir = manifests_root / "logs"
                _safe_add_dir(
                    zip_handle,
                    logs_dir,
                    arc_prefix="logs",
                    include_predicate=_allowlisted_bundle_path,
                )
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


# ---------------------------------------------------------------------------
# PDF Report Generation
# ---------------------------------------------------------------------------


def _count_jsonl_lines(path: Path) -> int:
    """Count lines in a JSONL file."""
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _count_jsonl_lines_optional(path: Path) -> int | None:
    """Count JSONL lines, returning None when the file is missing/unreadable."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return None


def _na_artifact(path: Path, label: str) -> str:
    """Return an N/A message for missing/unreadable artifacts."""
    if not path.exists():
        return f"N/A (missing {label})"
    return f"N/A (unreadable {label})"


def _format_optional_count(count: int | None, *, path: Path, label: str) -> str:
    """Format an optional count, returning an N/A message when missing/unreadable."""
    if count is None:
        return _na_artifact(path, label)
    return str(count)


def _file_size_str(path: Path) -> str:
    """Return human-readable file size."""
    if not path.exists():
        return "N/A"
    try:
        size = path.stat().st_size
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    except OSError:
        return "N/A"


def _artifact_row(path: Path, name: str | None = None) -> tuple[str, str, str]:
    """Return (filename, status, size) tuple for an artifact."""
    filename = name or path.name
    if path.exists():
        return (filename, "Present", _file_size_str(path))
    return (filename, "Missing", "-")


def _bundle_status(path: Path, *, in_allowlist: bool) -> str:
    """Return 'In Bundle' status based on file existence and allowlist membership.

    - "Yes" if file exists AND is in bundle allowlist
    - "No" if file exists but NOT in bundle allowlist
    - "N/A" if file is missing (nothing to bundle)
    """
    if not path.exists():
        return "N/A"
    return "Yes" if in_allowlist else "No"


def _load_yaml_config(config_name: str) -> dict[str, Any]:
    """Load a YAML config file from config/pipeline/."""
    config_path = Path("config/pipeline") / config_name
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _get_git_sha() -> str:
    """Get the current git SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "N/A"


def _format_yaml_subset(config: dict[str, Any], keys: list[str]) -> str:
    """Format a subset of a YAML config as a compact string."""
    subset = {k: config.get(k) for k in keys if k in config}
    if not subset:
        return "N/A"
    lines = []
    for k, v in subset.items():
        if isinstance(v, dict):
            # Flatten nested dict
            for k2, v2 in v.items():
                lines.append(f"{k}.{k2}: {v2}")
        else:
            lines.append(f"{k}: {v}")
    return " | ".join(lines)


def _yaml_table_rows(config: dict[str, Any], keys: list[str] | None = None) -> list[tuple[str, str]]:
    """Extract config values as table rows."""
    rows = []
    items = config.items() if keys is None else [(k, config.get(k)) for k in keys if k in config]
    for key, value in items:
        if value is None:
            continue
        if isinstance(value, dict):
            # Flatten nested dict to top level only
            for k2, v2 in value.items():
                if not isinstance(v2, (dict, list)):
                    rows.append((f"{key}.{k2}", str(v2)))
        elif isinstance(value, list):
            rows.append((key, str(value)))
        else:
            rows.append((key, str(value)))
    return rows


def _diagnostic_note(condition: bool, message: str) -> str | None:
    """Return a diagnostic note if condition is met."""
    return message if condition else None


def _format_percent(
    numerator: float | int | None,
    denominator: float | int | None,
    *,
    na: str = "N/A",
    signed: bool = False,
) -> str:
    """Format a percentage safely, returning `na` when denominator is 0/invalid."""
    try:
        numer = float(numerator or 0.0)
        denom = float(denominator or 0.0)
    except (TypeError, ValueError):
        return na
    if denom <= 0:
        return na
    pct = numer / denom * 100.0
    return f"{pct:+.1f}%" if signed else f"{pct:.1f}%"


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_ffprobe_fraction(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            return None
        return as_float if as_float > 0 else None
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if "/" in cleaned:
        num_str, den_str = cleaned.split("/", 1)
        try:
            num = float(num_str)
            den = float(den_str)
        except (TypeError, ValueError):
            return None
        if den == 0:
            return None
        fps = num / den
        return fps if fps > 0 else None
    try:
        fps = float(cleaned)
    except (TypeError, ValueError):
        return None
    return fps if fps > 0 else None


def _ffprobe_video_metadata(video_path: Path) -> dict[str, Any]:
    """Best-effort video metadata via ffprobe (duration_s, avg_fps, nb_frames)."""
    if not video_path.exists():
        return {"ok": False, "error": "missing_video", "duration_s": None, "avg_fps": None, "nb_frames": None}
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=duration,avg_frame_rate,nb_frames",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except FileNotFoundError:
        return {"ok": False, "error": "ffprobe_not_found", "duration_s": None, "avg_fps": None, "nb_frames": None}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "ffprobe_timeout", "duration_s": None, "avg_fps": None, "nb_frames": None}
    if result.returncode != 0:
        return {
            "ok": False,
            "error": "ffprobe_failed",
            "duration_s": None,
            "avg_fps": None,
            "nb_frames": None,
        }
    try:
        payload = json.loads(result.stdout or "")
    except json.JSONDecodeError:
        return {"ok": False, "error": "ffprobe_bad_json", "duration_s": None, "avg_fps": None, "nb_frames": None}
    streams = payload.get("streams") if isinstance(payload, dict) else None
    stream0 = streams[0] if isinstance(streams, list) and streams else None
    if not isinstance(stream0, dict):
        return {"ok": False, "error": "ffprobe_no_stream", "duration_s": None, "avg_fps": None, "nb_frames": None}

    duration_s = _parse_ffprobe_fraction(stream0.get("duration"))
    avg_fps = _parse_ffprobe_fraction(stream0.get("avg_frame_rate"))
    nb_frames_raw = stream0.get("nb_frames")
    nb_frames: int | None = None
    if isinstance(nb_frames_raw, int):
        nb_frames = nb_frames_raw
    elif isinstance(nb_frames_raw, str) and nb_frames_raw.isdigit():
        nb_frames = int(nb_frames_raw)
    return {"ok": True, "error": None, "duration_s": duration_s, "avg_fps": avg_fps, "nb_frames": nb_frames}


def _opencv_video_metadata(video_path: Path) -> dict[str, Any]:
    """Best-effort video metadata via OpenCV (fps, frame_count, width, height)."""
    if not video_path.exists():
        return {
            "ok": False,
            "error": "missing_video",
            "fps": None,
            "frame_count": None,
            "width": None,
            "height": None,
        }
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None  # type: ignore
    if cv2 is None:
        return {
            "ok": False,
            "error": "opencv_unavailable",
            "fps": None,
            "frame_count": None,
            "width": None,
            "height": None,
        }

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return {
                "ok": False,
                "error": "opencv_open_failed",
                "fps": None,
                "frame_count": None,
                "width": None,
                "height": None,
            }
        fps_probe = cap.get(cv2.CAP_PROP_FPS)
        frames_probe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width_probe = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height_probe = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = float(fps_probe) if fps_probe else None
        frame_count = int(frames_probe) if frames_probe else None
        width = int(width_probe) if width_probe else None
        height = int(height_probe) if height_probe else None
        return {
            "ok": True,
            "error": None,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
        }
    finally:
        cap.release()


def _face_detection_frame_stats(detections_path: Path) -> dict[str, Any]:
    """Compute observed frame stats from detections.jsonl (unique frames + median stride)."""
    if not detections_path.exists():
        return {
            "ok": False,
            "error": "missing_detections_jsonl",
            "frames_observed": None,
            "stride_median": None,
        }

    frame_indices: set[int] = set()
    try:
        with detections_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                frame_idx = row.get("frame_idx")
                if isinstance(frame_idx, int):
                    frame_indices.add(frame_idx)
    except OSError:
        return {
            "ok": False,
            "error": "read_error_detections_jsonl",
            "frames_observed": None,
            "stride_median": None,
        }

    frames_sorted = sorted(frame_indices)
    if len(frames_sorted) < 2:
        return {
            "ok": True,
            "error": None,
            "frames_observed": len(frames_sorted),
            "stride_median": None,
        }
    deltas = [b - a for a, b in zip(frames_sorted, frames_sorted[1:]) if b - a > 0]
    if not deltas:
        stride_median = None
    else:
        import statistics

        stride_median = int(statistics.median(deltas))
    return {
        "ok": True,
        "error": None,
        "frames_observed": len(frames_sorted),
        "stride_median": stride_median,
    }


def _format_mtime(path: Path) -> str:
    try:
        ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0)
    except OSError:
        return "N/A"
    return ts.isoformat().replace("+00:00", "Z")


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "1", "yes", "y", "on"}:
            return True
        if cleaned in {"false", "0", "no", "n", "off"}:
            return False
    return None


def _find_body_tracking_enabled(payload: Any) -> tuple[bool, str] | None:
    """Best-effort extraction of body_tracking.enabled from job/request payloads."""
    if not isinstance(payload, dict):
        return None

    def _check(root: dict[str, Any], prefix: str) -> tuple[bool, str] | None:
        body_tracking_block = root.get("body_tracking")
        if isinstance(body_tracking_block, dict):
            enabled = _coerce_bool(body_tracking_block.get("enabled"))
            if enabled is not None:
                return enabled, f"{prefix}.body_tracking.enabled"
        enabled = _coerce_bool(root.get("body_tracking_enabled"))
        if enabled is not None:
            return enabled, f"{prefix}.body_tracking_enabled"
        enabled = _coerce_bool(root.get("enable_body_tracking"))
        if enabled is not None:
            return enabled, f"{prefix}.enable_body_tracking"
        return None

    # Common nesting patterns: request_json.options / request_json.requested
    options = payload.get("options")
    if isinstance(options, dict):
        hit = _check(options, "options")
        if hit is not None:
            return hit
    requested = payload.get("requested")
    if isinstance(requested, dict):
        hit = _check(requested, "requested")
        if hit is not None:
            return hit

    # Direct top-level fallback.
    return _check(payload, "request_json")


def _track_fusion_overlap_diagnostics(
    *,
    face_tracks_path: Path,
    body_tracks_path: Path,
    iou_threshold: float,
    min_overlap_ratio: float,
    face_in_upper_body: bool,
    upper_body_fraction: float,
) -> dict[str, Any]:
    """Compute overlap gating counts for fusion diagnostics.

    This is best-effort and operates on available run artifacts:
    - face: tracks.jsonl (bboxes_sampled)
    - body: body_tracks.jsonl (detections list)
    """
    if not face_tracks_path.exists() or not body_tracks_path.exists():
        return {
            "ok": False,
            "error": "missing_inputs",
            "comparisons_total": 0,
            "comparisons_considered": 0,
            "pairs_considered": 0,
            "comparisons_passing": 0,
            "pairs_passing": 0,
            "pairs_passing_min_frames": 0,
            "frames_with_candidates": 0,
        }

    # Index body detections by frame.
    body_by_frame: dict[int, list[tuple[int, list[float]]]] = {}
    try:
        with body_tracks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                body_track_id = row.get("track_id")
                if not isinstance(body_track_id, int):
                    continue
                detections = row.get("detections")
                if not isinstance(detections, list):
                    continue
                for det in detections:
                    if not isinstance(det, dict):
                        continue
                    frame_idx = det.get("frame_idx")
                    bbox = det.get("bbox")
                    if not isinstance(frame_idx, int) or not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                    body_by_frame.setdefault(frame_idx, []).append((body_track_id, bbox))
    except OSError:
        return {
            "ok": False,
            "error": "read_error_body_tracks",
            "comparisons_total": 0,
            "comparisons_considered": 0,
            "pairs_considered": 0,
            "comparisons_passing": 0,
            "pairs_passing": 0,
            "pairs_passing_min_frames": 0,
            "frames_with_candidates": 0,
        }

    def _face_in_upper(face_box: list[float], body_box: list[float]) -> bool:
        if not face_in_upper_body:
            return True
        try:
            fy = (float(face_box[1]) + float(face_box[3])) / 2.0
            by1 = float(body_box[1])
            by2 = float(body_box[3])
        except (TypeError, ValueError, IndexError):
            return False
        upper_y = by1 + max(by2 - by1, 0.0) * float(upper_body_fraction)
        return fy <= upper_y

    def _iou_and_overlap_ratio(face_box: list[float], body_box: list[float]) -> tuple[float, float]:
        try:
            fx1, fy1, fx2, fy2 = (float(v) for v in face_box)
            bx1, by1, bx2, by2 = (float(v) for v in body_box)
        except (TypeError, ValueError):
            return 0.0, 0.0
        ix1, iy1 = max(fx1, bx1), max(fy1, by1)
        ix2, iy2 = min(fx2, bx2), min(fy2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0, 0.0
        face_area = max(0.0, fx2 - fx1) * max(0.0, fy2 - fy1)
        body_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = face_area + body_area - inter
        iou = inter / union if union > 0 else 0.0
        overlap_ratio = inter / face_area if face_area > 0 else 0.0
        return iou, overlap_ratio

    comparisons_total = 0
    comparisons_considered = 0
    comparisons_passing = 0
    pairs_considered: set[tuple[int, int]] = set()
    pairs_passing: set[tuple[int, int]] = set()
    passing_frames_by_pair: dict[tuple[int, int], int] = {}
    frames_with_candidates: set[int] = set()

    try:
        with face_tracks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                face_track_id = row.get("track_id")
                if not isinstance(face_track_id, int):
                    continue
                samples = row.get("bboxes_sampled")
                if not isinstance(samples, list):
                    continue
                for sample in samples:
                    if not isinstance(sample, dict):
                        continue
                    frame_idx = sample.get("frame_idx")
                    face_box = sample.get("bbox_xyxy")
                    if not isinstance(frame_idx, int) or not isinstance(face_box, list) or len(face_box) != 4:
                        continue
                    bodies = body_by_frame.get(frame_idx)
                    if not bodies:
                        continue
                    frames_with_candidates.add(frame_idx)
                    for body_track_id, body_box in bodies:
                        comparisons_total += 1
                        if not _face_in_upper(face_box, body_box):
                            continue
                        comparisons_considered += 1
                        pair = (face_track_id, body_track_id)
                        pairs_considered.add(pair)
                        iou, overlap_ratio = _iou_and_overlap_ratio(face_box, body_box)
                        if iou >= float(iou_threshold) and overlap_ratio >= float(min_overlap_ratio):
                            comparisons_passing += 1
                            pairs_passing.add(pair)
                            passing_frames_by_pair[pair] = passing_frames_by_pair.get(pair, 0) + 1
    except OSError:
        return {
            "ok": False,
            "error": "read_error_face_tracks",
            "comparisons_total": 0,
            "comparisons_considered": 0,
            "pairs_considered": 0,
            "comparisons_passing": 0,
            "pairs_passing": 0,
            "pairs_passing_min_frames": 0,
            "frames_with_candidates": 0,
        }

    pairs_passing_min_frames = sum(1 for count in passing_frames_by_pair.values() if count >= 3)

    return {
        "ok": True,
        "error": None,
        "comparisons_total": comparisons_total,
        "comparisons_considered": comparisons_considered,
        "pairs_considered": len(pairs_considered),
        "comparisons_passing": comparisons_passing,
        "pairs_passing": len(pairs_passing),
        "pairs_passing_min_frames": pairs_passing_min_frames,
        "frames_with_candidates": len(frames_with_candidates),
    }


def _face_tracks_duration_fallback(tracks_path: Path) -> dict[str, Any]:
    """Compute a face-only duration fallback from tracks.jsonl (approx)."""
    if not tracks_path.exists():
        return {
            "ok": False,
            "error": "missing_tracks_jsonl",
            "tracks_count": 0,
            "total_duration_s": None,
        }
    tracks_count = 0
    total_duration_s = 0.0
    try:
        with tracks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                first_ts = row.get("first_ts")
                last_ts = row.get("last_ts")
                if first_ts is None or last_ts is None:
                    continue
                try:
                    duration = float(last_ts) - float(first_ts)
                except (TypeError, ValueError):
                    continue
                tracks_count += 1
                total_duration_s += max(duration, 0.0)
    except OSError:
        return {
            "ok": False,
            "error": "read_error_tracks_jsonl",
            "tracks_count": 0,
            "total_duration_s": None,
        }
    return {
        "ok": True,
        "error": None,
        "tracks_count": tracks_count,
        "total_duration_s": total_duration_s,
    }


def build_screentime_run_debug_pdf(
    *,
    ep_id: str,
    run_id: str,
) -> tuple[bytes, str]:
    """Build a Screen Time Run Debug Report PDF.

    Returns:
        (pdf_bytes, download_filename)
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError("reportlab is required to build PDF debug reports (pip install reportlab)") from exc

    run_id_norm = run_layout.normalize_run_id(run_id)
    run_root = run_layout.run_root(ep_id, run_id_norm)
    if not run_root.exists():
        raise FileNotFoundError(f"Run not found on disk: {run_root}")

    s3_layout = run_layout.get_run_s3_layout(ep_id, run_id_norm)

    manifests_root = get_path(ep_id, "detections").parent
    # Run-scoped body_tracking is authoritative for this report; legacy artifacts are diagnostic only.
    body_tracking_dir = run_root / "body_tracking"
    legacy_body_tracking_dir = manifests_root / "body_tracking"

    # Load artifact paths (local-first; optionally materialized from S3 in delete-local mode).
    detections_path = run_root / "detections.jsonl"
    tracks_path = run_root / "tracks.jsonl"
    faces_path = run_root / "faces.jsonl"
    identities_path = run_root / "identities.json"
    track_metrics_path = run_root / "track_metrics.json"
    body_detections_path = body_tracking_dir / "body_detections.jsonl"
    body_tracks_path = body_tracking_dir / "body_tracks.jsonl"
    track_fusion_path = body_tracking_dir / "track_fusion.json"
    screentime_comparison_path = body_tracking_dir / "screentime_comparison.json"
    face_alignment_path = run_root / "face_alignment" / "aligned_faces.jsonl"
    cluster_centroids_path = run_root / "cluster_centroids.json"
    faces_npy = (
        Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
        / "embeds"
        / ep_id
        / "runs"
        / run_id_norm
        / "faces.npy"
    )

    # If local run artifacts were deleted after S3 sync, hydrate allowlisted artifacts from S3
    # so the PDF reflects the run-scoped state instead of reporting false "missing" signals.
    # NOTE: download mtimes reflect hydration time, not artifact generation time.
    hydrated_from_s3 = False
    hydrated_paths: dict[str, Path] = {}
    hydrated_s3_keys: dict[str, str] = {}
    hydrated_locations: dict[str, str] = {}
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        if any(not (run_root / filename).exists() for filename in run_layout.RUN_ARTIFACT_ALLOWLIST):
            from apps.api.services.storage import StorageService

            storage = StorageService()
            if storage.s3_enabled():
                temp_dir = tempfile.TemporaryDirectory(prefix="screenalytics_run_hydrate_")
                hydrate_root = Path(temp_dir.name)

                for filename in sorted(run_layout.RUN_ARTIFACT_ALLOWLIST):
                    local_path = run_root / filename
                    if local_path.exists():
                        continue
                    for s3_key in run_layout.run_artifact_s3_keys_for_read(ep_id, run_id_norm, filename):
                        try:
                            if not storage.object_exists(s3_key):
                                continue
                        except Exception as exc:
                            LOGGER.warning("[export] Failed to check S3 object existence for %s: %s", s3_key, exc)
                            continue
                        payload = storage.download_bytes(s3_key)
                        if payload is None:
                            continue
                        hydrated_s3_keys[filename] = s3_key
                        hydrated_locations[filename] = (
                            "legacy" if s3_key.startswith(s3_layout.legacy_prefix) else "canonical"
                        )
                        break
                    else:
                        continue
                    dest_path = hydrate_root / filename
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    dest_path.write_bytes(payload)
                    hydrated_paths[filename] = dest_path
                    hydrated_from_s3 = True
    except Exception as exc:
        LOGGER.warning("[export] Failed to hydrate run artifacts from S3: %s", exc)

    def _resolved_path(path: Path, rel_name: str) -> Path:
        resolved = hydrated_paths.get(rel_name)
        return resolved if resolved is not None else path

    detections_path = _resolved_path(detections_path, "detections.jsonl")
    tracks_path = _resolved_path(tracks_path, "tracks.jsonl")
    faces_path = _resolved_path(faces_path, "faces.jsonl")
    identities_path = _resolved_path(identities_path, "identities.json")
    track_metrics_path = _resolved_path(track_metrics_path, "track_metrics.json")
    cluster_centroids_path = _resolved_path(cluster_centroids_path, "cluster_centroids.json")
    face_alignment_path = _resolved_path(face_alignment_path, "face_alignment/aligned_faces.jsonl")
    body_detections_path = _resolved_path(body_detections_path, "body_tracking/body_detections.jsonl")
    body_tracks_path = _resolved_path(body_tracks_path, "body_tracking/body_tracks.jsonl")
    track_fusion_path = _resolved_path(track_fusion_path, "body_tracking/track_fusion.json")
    screentime_comparison_path = _resolved_path(
        screentime_comparison_path,
        "body_tracking/screentime_comparison.json",
    )
    # Keep downstream derived paths pointing at the resolved body_tracking directory.
    body_tracking_dir = body_tracks_path.parent
    detect_track_marker_path = _resolved_path(run_root / "detect_track.json", "detect_track.json")
    body_tracking_marker_path = _resolved_path(run_root / "body_tracking.json", "body_tracking.json")

    # Load JSON artifact data
    identities_payload = _read_json(identities_path)
    identities_data = identities_payload if isinstance(identities_payload, dict) else {}
    track_metrics_payload = _read_json(track_metrics_path)
    track_metrics_data = track_metrics_payload if isinstance(track_metrics_payload, dict) else {}
    track_fusion_data = _read_json(track_fusion_path) if track_fusion_path.exists() else None
    screentime_data = _read_json(screentime_comparison_path) if screentime_comparison_path.exists() else None

    # Load YAML configs
    detection_config = _load_yaml_config("detection.yaml")
    tracking_config = _load_yaml_config("tracking.yaml")
    embedding_config = _load_yaml_config("embedding.yaml")
    clustering_config = _load_yaml_config("clustering.yaml")
    body_detection_config = _load_yaml_config("body_detection.yaml")
    track_fusion_config = _load_yaml_config("track_fusion.yaml")
    screentime_config = _load_yaml_config("screen_time_v2.yaml")

    # Load DB data
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

    # Build PDF
    buffer = io.BytesIO()
    page_compression = 1
    if os.environ.get("SCREENALYTICS_PDF_NO_COMPRESSION", "").strip().lower() in {"1", "true", "yes"}:
        page_compression = 0
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        pageCompression=page_compression,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=16,
        spaceAfter=8,
        textColor=colors.HexColor("#1a365d"),
    )
    subsection_style = ParagraphStyle(
        "SubsectionHeading",
        parent=styles["Heading3"],
        fontSize=11,
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = styles["Normal"]
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=styles["Normal"],
        leftIndent=20,
        bulletIndent=10,
    )
    warning_style = ParagraphStyle(
        "Warning",
        parent=styles["Normal"],
        textColor=colors.HexColor("#c53030"),
        fontSize=9,
        leftIndent=10,
    )
    note_style = ParagraphStyle(
        "Note",
        parent=styles["Normal"],
        textColor=colors.HexColor("#2b6cb0"),
        fontSize=9,
        fontName="Helvetica-Oblique",
        leftIndent=10,
    )
    config_style = ParagraphStyle(
        "ConfigText",
        parent=styles["Normal"],
        fontSize=8,
        fontName="Courier",
        textColor=colors.HexColor("#4a5568"),
        leftIndent=10,
    )
    # Table cell style for wrapping text
    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontSize=9,
        leading=11,  # Line spacing for wrapped text
    )
    cell_style_small = ParagraphStyle(
        "TableCellSmall",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
    )

    def _wrap_cell(text: str, style: ParagraphStyle = cell_style) -> Paragraph:
        """Wrap text in a Paragraph for table cell text wrapping."""
        # Escape XML special characters for ReportLab
        safe_text = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return Paragraph(safe_text, style)

    def _wrap_row(row: list, style: ParagraphStyle = cell_style) -> list:
        """Wrap all cells in a row for text wrapping."""
        return [_wrap_cell(cell, style) for cell in row]

    story: list[Any] = []

    # =========================================================================
    # COVER / EXECUTIVE SUMMARY
    # =========================================================================
    story.append(Paragraph("Screen Time Run Debug Report", title_style))
    story.append(Spacer(1, 12))

    summary_data = [
        ["Episode ID", ep_id],
        ["Run ID", run_id_norm],
        ["Generated At", _now_iso()],
        ["Git SHA", _get_git_sha()],
        ["Run Root", str(run_root)],
        ["S3 Layout (write)", s3_layout.s3_layout],
        ["S3 Run Prefix (write)", s3_layout.write_prefix],
    ]
    summary_table = Table(summary_data, colWidths=[1.5 * inch, 5 * inch])
    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e2e8f0")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (0, 0), (0, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(summary_table)
    story.append(Spacer(1, 12))
    if hydrated_from_s3:
        story.append(
            Paragraph(
                "<b>Note:</b> hydrated_from_s3: <b>true</b>. Some run-scoped artifacts were loaded from S3 because local copies were missing "
                "(commonly due to STORAGE_DELETE_LOCAL_AFTER_SYNC=1). File mtimes reflect hydration time.",
                note_style,
            )
        )
        hydrated_items = sorted(
            (key, hydrated_locations.get(filename, "unknown"))
            for filename, key in hydrated_s3_keys.items()
        )
        story.append(Paragraph(f"Hydrated keys ({len(hydrated_items)}):", subsection_style))
        for key, location in hydrated_items:
            story.append(Paragraph(f"&bull; {key} (artifact_location={location})", bullet_style))
        story.append(Spacer(1, 8))

    # =========================================================================
    # RUN HEALTH (Quick Sanity Check)
    # =========================================================================
    body_config_enabled = bool(body_detection_config.get("body_tracking", {}).get("enabled", False))

    # Run-scoped body artifacts (authoritative for this run_id).
    body_detect_count_run = _count_jsonl_lines_optional(body_detections_path)
    body_track_count_run = _count_jsonl_lines_optional(body_tracks_path)
    body_artifacts_exist_run = body_detections_path.exists() and body_tracks_path.exists()

    # Legacy body artifacts (diagnostic only; promoted from some run_id).
    legacy_body_detections_path = legacy_body_tracking_dir / "body_detections.jsonl"
    legacy_body_tracks_path = legacy_body_tracking_dir / "body_tracks.jsonl"
    legacy_track_fusion_path = legacy_body_tracking_dir / "track_fusion.json"
    legacy_screentime_comparison_path = legacy_body_tracking_dir / "screentime_comparison.json"
    legacy_body_detect_count = _count_jsonl_lines_optional(legacy_body_detections_path)
    legacy_body_track_count = _count_jsonl_lines_optional(legacy_body_tracks_path)
    legacy_body_artifacts_exist = (
        (legacy_body_detect_count or 0) > 0
        or (legacy_body_track_count or 0) > 0
        or legacy_track_fusion_path.exists()
        or legacy_screentime_comparison_path.exists()
    )

    # Determine whether body tracking ran for this run_id.
    # IMPORTANT: Do not treat legacy presence as evidence of run-scoped execution.
    # Require the run-scoped artifacts to exist (even if empty).
    run_body_tracking_marker = _read_json(body_tracking_marker_path)
    body_tracking_ran_effective = body_artifacts_exist_run
    body_tracking_effective_source = "run artifacts" if body_artifacts_exist_run else "missing artifacts"

    # Legacy marker run_id (helps detect stale/out-of-scope body artifacts).
    legacy_body_marker = _read_json(manifests_root / "runs" / "body_tracking.json")
    legacy_body_marker_run_id = legacy_body_marker.get("run_id") if isinstance(legacy_body_marker, dict) else None
    legacy_body_same_run = legacy_body_marker_run_id == run_id_norm if legacy_body_marker_run_id else False
    # Legacy artifacts are always episode-level (not run-scoped). Treat as out-of-scope for this run_id,
    # even when they were promoted from the same run, to avoid conflating legacy with run-scoped execution.
    legacy_body_out_of_scope = legacy_body_artifacts_exist
    legacy_body_only_present = legacy_body_artifacts_exist and not body_artifacts_exist_run

    override_source: str = "N/A"
    preset_name: str | None = None
    cli_command: str | None = None
    if body_config_enabled != bool(body_tracking_ran_effective) or legacy_body_only_present:
        override_source_hit: tuple[bool, str] | None = None
        for job_run in job_runs:
            if not isinstance(job_run, dict):
                continue
            req = job_run.get("request_json")
            if cli_command is None and isinstance(req, dict) and isinstance(req.get("command"), list):
                cli_command = " ".join(str(part) for part in req.get("command") if part is not None)
            override_source_hit = _find_body_tracking_enabled(job_run.get("request_json"))
            if override_source_hit is not None:
                break
        if override_source_hit is not None:
            override_source = f"job_runs.request_json ({override_source_hit[1]})"
        else:
            run_cfg = run_row.get("config_json") if isinstance(run_row, dict) else None
            override_source_hit = _find_body_tracking_enabled({"options": run_cfg} if isinstance(run_cfg, dict) else None)
            if override_source_hit is not None:
                override_source = f"runs.config_json ({override_source_hit[1]})"
            else:
                override_source = "Override source not found; artifacts may be stale for this run_id."
        if preset_name is None:
            for job_run in job_runs:
                if not isinstance(job_run, dict):
                    continue
                req = job_run.get("request_json")
                if not isinstance(req, dict):
                    continue
                for key in ("options", "requested"):
                    block = req.get(key)
                    if isinstance(block, dict):
                        preset = block.get("preset") or block.get("profile")
                        if isinstance(preset, str) and preset.strip():
                            preset_name = preset.strip()
                            break
                if preset_name is not None:
                    break
        if preset_name is None and isinstance(run_row, dict):
            run_cfg = run_row.get("config_json")
            if isinstance(run_cfg, dict):
                preset = run_cfg.get("preset") or run_cfg.get("profile")
                if isinstance(preset, str) and preset.strip():
                    preset_name = preset.strip()

    track_fusion_payload = track_fusion_data if isinstance(track_fusion_data, dict) else None
    track_fusion_available: bool | None
    if not track_fusion_path.exists():
        track_fusion_available = False
    elif track_fusion_payload is None:
        track_fusion_available = None  # exists but unreadable
    else:
        track_fusion_available = True

    screentime_payload = screentime_data if isinstance(screentime_data, dict) else None
    screentime_available: bool | None
    if not screentime_comparison_path.exists():
        screentime_available = False
    elif screentime_payload is None:
        screentime_available = None  # exists but unreadable
    else:
        screentime_available = True

    tracked_ids_fused_total: int | None = None
    if track_fusion_payload is not None:
        tracked_ids_raw = track_fusion_payload.get("num_fused_identities")
        if isinstance(tracked_ids_raw, int):
            tracked_ids_fused_total = tracked_ids_raw
        else:
            identities_block = track_fusion_payload.get("identities")
            if isinstance(identities_block, dict):
                tracked_ids_fused_total = len(identities_block)

    screentime_summary: dict[str, Any] | None = None
    if screentime_payload is not None:
        summary_block = screentime_payload.get("summary")
        # Only set screentime_summary if it's a proper dict; otherwise keep None for N/A display
        screentime_summary = summary_block if isinstance(summary_block, dict) else None

    # Face-only fallback for cases where screentime_comparison.json is missing.
    face_tracks_count_run = _count_jsonl_lines_optional(tracks_path)
    face_tracks_fallback = _face_tracks_duration_fallback(tracks_path)
    face_tracks_present = (face_tracks_count_run or 0) > 0

    actual_fused_pairs: int | None = None
    if track_fusion_payload is not None:
        fusion_identities = track_fusion_payload.get("identities")
        if isinstance(fusion_identities, dict):
            actual_fused_pairs = 0
            for identity_data in fusion_identities.values():
                if not isinstance(identity_data, dict):
                    continue
                face_tids = identity_data.get("face_track_ids", [])
                body_tids = identity_data.get("body_track_ids", [])
                if face_tids and body_tids:
                    actual_fused_pairs += 1

    db_connected = db_error is None

    # Health status helper
    def _health_status(ok: bool | None) -> str:
        if ok is True:
            return "✓ Yes"
        if ok is False:
            return "✗ No"
        return "—"

    def _status_color(ok: bool | None):
        if ok is True:
            return colors.green
        if ok is False:
            return colors.red
        return colors.HexColor("#718096")

    body_detect_display = str(body_detect_count_run) if body_detect_count_run is not None else "N/A"
    body_track_display = str(body_track_count_run) if body_track_count_run is not None else "N/A"
    face_tracks_detail = f"{face_tracks_count_run} tracks" if face_tracks_count_run is not None else "N/A"
    if face_tracks_fallback.get("ok") and isinstance(face_tracks_fallback.get("total_duration_s"), (int, float)):
        face_tracks_detail = f"{face_tracks_detail} | ≈{float(face_tracks_fallback['total_duration_s']):.1f}s total span"

    fused_pairs_ok: bool | None
    fused_pairs_detail: str
    if actual_fused_pairs is None:
        fused_pairs_ok = None
        if track_fusion_available is False:
            fused_pairs_detail = "N/A (missing body_tracking/track_fusion.json)"
        elif track_fusion_available is None:
            fused_pairs_detail = "N/A (unreadable body_tracking/track_fusion.json)"
        else:
            fused_pairs_detail = "N/A"
    else:
        fused_pairs_ok = actual_fused_pairs > 0
        fused_pairs_detail = f"{actual_fused_pairs} pair(s)"

    health_data = [
        _wrap_row(["Health Check", "Status", "Details"]),
        _wrap_row(["DB Connected", _health_status(db_connected),
         "OK" if db_connected else f"Error: {db_error[:50]}..." if db_error and len(db_error) > 50 else (db_error or "Unknown")]),
        _wrap_row(["Face Tracks Present", _health_status(face_tracks_present), face_tracks_detail]),
        _wrap_row([
            "Body Tracking Ran (run-scoped)",
            _health_status(bool(body_tracking_ran_effective)),
            f"YAML enabled={'true' if body_config_enabled else 'false'} | dets={body_detect_display} tracks={body_track_display}",
        ]),
        _wrap_row([
            "Track Fusion Output Present",
            _health_status(track_fusion_available),
            "Present" if track_fusion_available is True else (
                "Missing body_tracking/track_fusion.json" if track_fusion_available is False else "Unreadable track_fusion.json"
            ),
        ]),
        _wrap_row(["Face-Body Pairs Fused", _health_status(fused_pairs_ok), fused_pairs_detail]),
        _wrap_row([
            "Screen Time Comparison Present",
            _health_status(screentime_available),
            "Present" if screentime_available is True else (
                "Missing body_tracking/screentime_comparison.json" if screentime_available is False else "Unreadable screentime_comparison.json"
            ),
        ]),
        _wrap_row([
            "Legacy Body Artifacts Present",
            _health_status(legacy_body_artifacts_exist),
            f"legacy_run_id={legacy_body_marker_run_id or 'unknown'} | out_of_scope={'yes' if legacy_body_out_of_scope else 'no'}",
        ]),
    ]
    health_table = Table(health_data, colWidths=[1.8 * inch, 0.8 * inch, 3.9 * inch])
    status_values: list[bool | None] = [
        db_connected,
        face_tracks_present,
        bool(body_tracking_ran_effective),
        track_fusion_available,
        fused_pairs_ok,
        screentime_available,
        legacy_body_artifacts_exist,
    ]
    status_styles = []
    for idx, status in enumerate(status_values, start=1):
        status_styles.append(("TEXTCOLOR", (1, idx), (1, idx), _status_color(status)))
    health_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a5568")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            *status_styles,
        ])
    )
    story.append(Paragraph("Run Health", subsection_style))
    story.append(health_table)

    if body_config_enabled != bool(body_tracking_ran_effective) or legacy_body_only_present:
        story.append(Paragraph(
            "⚠️ <b>Body Tracking Diagnostic:</b> body_tracking.enabled (YAML) does not match the effective run-scoped state "
            "or only legacy artifacts are present.",
            warning_style
        ))
        diag_rows = [
            ["Field", "Value"],
            ["body_tracking.enabled (YAML)", str(body_config_enabled)],
            ["body_tracking.ran_effective (run-scoped)", str(bool(body_tracking_ran_effective))],
            ["override_source", override_source],
            ["preset/profile (if any)", str(preset_name or "N/A")],
            ["cli_command (if captured)", (cli_command[:120] + "...") if cli_command and len(cli_command) > 120 else (cli_command or "N/A")],
            ["legacy_body_tracking.run_id (marker)", str(legacy_body_marker_run_id or "N/A")],
            ["legacy_body_tracking.out_of_scope", "yes" if legacy_body_out_of_scope else "no"],
            ["run/body_tracking/body_detections.jsonl mtime", _format_mtime(body_detections_path)],
            ["run/body_tracking/body_tracks.jsonl mtime", _format_mtime(body_tracks_path)],
            ["legacy/body_tracking/body_detections.jsonl mtime", _format_mtime(legacy_body_detections_path)],
            ["legacy/body_tracking/body_tracks.jsonl mtime", _format_mtime(legacy_body_tracks_path)],
            ["legacy/body_tracking/track_fusion.json mtime", _format_mtime(legacy_track_fusion_path)],
            ["legacy/body_tracking/screentime_comparison.json mtime", _format_mtime(legacy_screentime_comparison_path)],
        ]
        diag_table = Table(diag_rows, colWidths=[3.1 * inch, 2.4 * inch])
        diag_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )
        story.append(diag_table)

    story.append(Spacer(1, 12))

    # Executive summary stats
    identities_list = identities_data.get("identities", [])
    if not isinstance(identities_list, list):
        identities_list = []
    identities_count_run = len(identities_list) if isinstance(identities_payload, dict) else None
    identities_count_display = _format_optional_count(
        identities_count_run,
        path=identities_path,
        label="identities.json",
    )
    cluster_stats = identities_data.get("stats", {})
    metrics = track_metrics_data.get("metrics", {})

    num_face_tracks = face_tracks_count_run or 0
    num_body_tracks = body_track_count_run or 0
    face_tracks_exec = _format_optional_count(face_tracks_count_run, path=tracks_path, label="tracks.jsonl")
    body_tracks_exec = _format_optional_count(
        body_track_count_run,
        path=body_tracks_path,
        label="body_tracking/body_tracks.jsonl",
    )
    if track_fusion_available is True and tracked_ids_fused_total is not None:
        tracked_ids_exec = str(tracked_ids_fused_total)
    elif track_fusion_available is True:
        tracked_ids_exec = "N/A"
    else:
        tracked_ids_exec = _na_artifact(track_fusion_path, "body_tracking/track_fusion.json")
    if screentime_summary is None:
        screentime_gain_exec = _na_artifact(screentime_comparison_path, "body_tracking/screentime_comparison.json")
    else:
        screentime_gain_exec = f"{_safe_float(screentime_summary.get('total_duration_gain', 0)):.2f}s"

    exec_stats = [
        ["Metric", "Value"],
        ["Total Face Tracks", face_tracks_exec],
        ["Total Body Tracks", body_tracks_exec],
        ["Face Clusters (identities.json)", identities_count_display],
        ["Tracked IDs (from fusion output)", tracked_ids_exec],
        ["Screen Time Gain (from comparison)", screentime_gain_exec],
    ]
    exec_table = Table(exec_stats, colWidths=[3 * inch, 2 * inch])
    exec_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f7fafc")),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(Paragraph("Executive Summary", subsection_style))
    story.append(exec_table)

    # Clarification note for "Tracked IDs"
    story.append(Paragraph(
        "<b>Note:</b> 'Tracked IDs' comes from <b>body_tracking/track_fusion.json</b> (union of face + body after fusion), "
        "NOT the number of actually fused face-body pairs. If that artifact is missing for this run_id, the value is shown as "
        "<b>N/A</b> (not <b>0</b>).",
        note_style
    ))

    # =========================================================================
    # SECTION 0: RUN INPUTS & LINEAGE
    # =========================================================================
    story.append(Paragraph("0. Run Inputs &amp; Lineage", section_style))

    # Get storage backend configuration for display
    try:
        from apps.api.services.validation import validate_storage_backend_config
        storage_config = validate_storage_backend_config()
        storage_backend = storage_config.backend
        storage_bucket = storage_config.bucket or "N/A"
        storage_endpoint = storage_config.s3_endpoint or "N/A"
        storage_is_fallback = storage_config.is_fallback
        storage_fallback_reason = storage_config.fallback_reason or ""
        storage_has_creds = storage_config.has_credentials
    except Exception:
        storage_backend = os.environ.get("STORAGE_BACKEND", "local").lower()
        storage_bucket = os.environ.get("SCREENALYTICS_S3_BUCKET") or os.environ.get("AWS_S3_BUCKET") or "N/A"
        storage_endpoint = os.environ.get("SCREENALYTICS_OBJECT_STORE_ENDPOINT") or "N/A"
        storage_is_fallback = False
        storage_fallback_reason = ""
        storage_has_creds = False

    # Format storage backend display string
    if storage_backend in ("s3", "minio", "hybrid"):
        storage_display = f"{storage_backend} (bucket={storage_bucket})"
        if storage_endpoint != "N/A":
            storage_display = f"{storage_backend} (bucket={storage_bucket}, endpoint={storage_endpoint})"
    else:
        storage_display = storage_backend

    if storage_is_fallback:
        storage_display = f"{storage_display} [FALLBACK: {storage_fallback_reason}]"

    lineage_data = [
        ["Input", "Value"],
        ["Episode ID", ep_id],
        ["Run ID", run_id_norm],
        ["Git SHA", _get_git_sha()],
        ["Generated At", _now_iso()],
        ["Artifact Store", storage_display],
        ["DB Connected", "Yes" if db_connected else f"No ({db_error[:40]}...)" if db_error and len(db_error) > 40 else f"No ({db_error or 'unknown'})"],
    ]

    # Video metadata sources (split, labeled, and validated for mismatches).
    video_path = get_path(ep_id, "video")
    ffprobe_meta = _ffprobe_video_metadata(video_path)
    opencv_meta = _opencv_video_metadata(video_path)
    detect_track_marker = _read_json(detect_track_marker_path) if detect_track_marker_path.exists() else None

    ffprobe_duration_s = ffprobe_meta.get("duration_s") if ffprobe_meta.get("ok") else None
    ffprobe_fps = ffprobe_meta.get("avg_fps") if ffprobe_meta.get("ok") else None
    ffprobe_frames = ffprobe_meta.get("nb_frames") if ffprobe_meta.get("ok") else None

    opencv_fps = opencv_meta.get("fps") if opencv_meta.get("ok") else None
    opencv_frames = opencv_meta.get("frame_count") if opencv_meta.get("ok") else None
    width_value = opencv_meta.get("width") if opencv_meta.get("ok") else None
    height_value = opencv_meta.get("height") if opencv_meta.get("ok") else None

    marker_duration_s = None
    marker_fps = None
    marker_frames_total = None
    marker_stride_requested = None
    marker_gate_enabled: bool | None = None
    marker_gate_auto_rerun: dict[str, Any] | None = None
    if isinstance(detect_track_marker, dict):
        marker_duration_s = _parse_ffprobe_fraction(detect_track_marker.get("video_duration_sec"))
        marker_fps = _parse_ffprobe_fraction(detect_track_marker.get("fps"))
        marker_frames_raw = detect_track_marker.get("frames_total")
        if isinstance(marker_frames_raw, int):
            marker_frames_total = marker_frames_raw
        elif isinstance(marker_frames_raw, str) and marker_frames_raw.isdigit():
            marker_frames_total = int(marker_frames_raw)
        stride_raw = detect_track_marker.get("stride")
        if isinstance(stride_raw, int):
            marker_stride_requested = stride_raw
        elif isinstance(stride_raw, str) and stride_raw.isdigit():
            marker_stride_requested = int(stride_raw)
        tracking_gate = detect_track_marker.get("tracking_gate")
        if isinstance(tracking_gate, dict):
            enabled_raw = tracking_gate.get("enabled")
            if isinstance(enabled_raw, bool):
                marker_gate_enabled = enabled_raw
            auto_rerun_raw = tracking_gate.get("auto_rerun")
            if isinstance(auto_rerun_raw, dict):
                marker_gate_auto_rerun = auto_rerun_raw

    def _fmt_duration_s(value: float | None) -> str:
        if value is None:
            return "unknown"
        return f"{value:.3f}s"

    def _fmt_fps(value: float | None) -> str:
        if value is None:
            return "unknown"
        return f"{value:.3f} fps"

    def _fmt_int(value: int | None) -> str:
        if value is None:
            return "unknown"
        return str(value)

    resolved_duration_s: float | None = ffprobe_duration_s or marker_duration_s
    resolved_fps: float | None = ffprobe_fps or marker_fps or opencv_fps
    # Include opencv_frames as final fallback per Cursor bot feedback
    resolved_frames: int | None = ffprobe_frames or marker_frames_total or opencv_frames

    def _resolved_source(ffprobe_val: Any, marker_val: Any, opencv_val: Any = None) -> str:
        """Determine which source provided the resolved value."""
        if ffprobe_val is not None:
            return "ffprobe"
        if marker_val is not None:
            return "detect_track.marker"
        if opencv_val is not None:
            return "opencv"
        return "unknown"

    if resolved_duration_s is not None:
        dur_source = _resolved_source(ffprobe_duration_s, marker_duration_s)
        lineage_data.append(
            [
                "Video Duration (resolved)",
                f"{resolved_duration_s:.2f}s ({dur_source})",
            ]
        )
    if resolved_fps is not None:
        fps_source = _resolved_source(ffprobe_fps, marker_fps, opencv_fps)
        lineage_data.append(
            [
                "Video Frame Rate (resolved)",
                f"{resolved_fps:.3f} fps ({fps_source})",
            ]
        )
    if resolved_frames is not None:
        frames_source = _resolved_source(ffprobe_frames, marker_frames_total, opencv_frames)
        lineage_data.append(
            [
                "Total Video Frames (resolved)",
                f"{resolved_frames} ({frames_source})",
            ]
        )
    if width_value is not None or height_value is not None:
        lineage_data.append(["Resolution (OpenCV)", f"{width_value or '?'}x{height_value or '?'}"])

    lineage_data.extend([
        ["ffprobe.duration_s", _fmt_duration_s(ffprobe_duration_s)],
        ["ffprobe.avg_fps", _fmt_fps(ffprobe_fps)],
        ["ffprobe.nb_frames", _fmt_int(ffprobe_frames)],
        ["opencv.fps", _fmt_fps(opencv_fps)],
        ["opencv.frame_count", _fmt_int(opencv_frames)],
    ])
    if marker_duration_s is not None or marker_fps is not None or marker_frames_total is not None:
        lineage_data.extend([
            ["detect_track.marker.video_duration_s", _fmt_duration_s(marker_duration_s)],
            ["detect_track.marker.fps", _fmt_fps(marker_fps)],
            ["detect_track.marker.frames_total", _fmt_int(marker_frames_total)],
        ])
    if marker_stride_requested is not None:
        lineage_data.append(["Face Detection Stride (requested)", str(marker_stride_requested)])
    if marker_gate_enabled is not None:
        lineage_data.append(["Appearance Gate Enabled", "true" if marker_gate_enabled else "false"])
    if marker_gate_auto_rerun is not None:
        triggered = marker_gate_auto_rerun.get("triggered")
        selected = marker_gate_auto_rerun.get("selected")
        reason = marker_gate_auto_rerun.get("reason")
        if triggered is True:
            lineage_data.append(
                [
                    "Appearance Gate Auto-Rerun",
                    f"true (selected={selected or 'unknown'}, reason={reason or 'unknown'})",
                ]
            )
        elif triggered is False:
            lineage_data.append(["Appearance Gate Auto-Rerun", f"false (reason={reason or 'unknown'})"])

    def _rel_diff(a: float, b: float) -> float:
        denom = max(abs(a), abs(b))
        if denom <= 0:
            return 0.0
        return abs(a - b) / denom

    mismatches: list[str] = []

    def _check_mismatch(label: str, a_name: str, a_value: float | int | None, b_name: str, b_value: float | int | None) -> None:
        if a_value is None or b_value is None:
            return
        diff = _rel_diff(float(a_value), float(b_value))
        if diff > 0.10:
            mismatches.append(f"{label}: {a_name} vs {b_name} ({diff * 100:.1f}% diff)")

    opencv_duration_s: float | None = None
    if isinstance(opencv_frames, int) and isinstance(opencv_fps, float) and opencv_fps > 0:
        opencv_duration_s = float(opencv_frames) / opencv_fps

    _check_mismatch("Duration", "ffprobe.duration_s", ffprobe_duration_s, "detect_track.marker.video_duration_s", marker_duration_s)
    _check_mismatch("Duration", "ffprobe.duration_s", ffprobe_duration_s, "opencv.frame_count/opencv.fps", opencv_duration_s)
    _check_mismatch("Duration", "detect_track.marker.video_duration_s", marker_duration_s, "opencv.frame_count/opencv.fps", opencv_duration_s)
    _check_mismatch("FPS", "ffprobe.avg_fps", ffprobe_fps, "detect_track.marker.fps", marker_fps)
    _check_mismatch("FPS", "ffprobe.avg_fps", ffprobe_fps, "opencv.fps", opencv_fps)
    _check_mismatch("FPS", "detect_track.marker.fps", marker_fps, "opencv.fps", opencv_fps)
    _check_mismatch("Frames", "ffprobe.nb_frames", ffprobe_frames, "detect_track.marker.frames_total", marker_frames_total)
    _check_mismatch("Frames", "ffprobe.nb_frames", ffprobe_frames, "opencv.frame_count", opencv_frames)
    _check_mismatch("Frames", "detect_track.marker.frames_total", marker_frames_total, "opencv.frame_count", opencv_frames)

    if mismatches:
        story.append(
            Paragraph(
                "<b>⚠️ metadata mismatch</b>: video metadata sources disagree by &gt;10% ("
                + "; ".join(mismatches)
                + "). Values in this section may be unreliable.",
                note_style,
            )
        )

    # Face detection observed frame stats (most reliable for stride/frame count).
    det_stats = _face_detection_frame_stats(detections_path)
    if det_stats.get("ok"):
        lineage_data.append(["Face Detection Frames Observed", str(det_stats.get("frames_observed"))])
        stride_median = det_stats.get("stride_median")
        lineage_data.append(
            ["Face Detection Stride (observed)", str(stride_median) if stride_median is not None else "unknown"]
        )
    else:
        lineage_data.append(["Face Detection Frames Observed", "unknown"])
        lineage_data.append(["Face Detection Stride (observed)", "unknown"])

    detection_stride_cfg = None
    for key in ("stride", "detect_every_n_frames"):
        if key in detection_config:
            detection_stride_cfg = detection_config.get(key)
            break
    if detection_stride_cfg is not None:
        lineage_data.append(["Face Detection Stride (config)", str(detection_stride_cfg)])
    else:
        lineage_data.append(["Face Detection Stride (config)", "N/A (not specified in detection.yaml)"])
    body_stride_cfg: int | None = None
    try:
        body_stride_cfg_raw = body_detection_config.get("person_detection", {}).get("detect_every_n_frames")
        body_stride_cfg = int(body_stride_cfg_raw) if body_stride_cfg_raw is not None else None
    except (TypeError, ValueError):
        body_stride_cfg = None
    if body_stride_cfg is not None:
        lineage_data.append(["Body Pipeline Stride (detect_every_n_frames)", str(body_stride_cfg)])
        if bool(body_tracking_ran_effective):
            if isinstance(resolved_frames, int) and body_stride_cfg > 0:
                body_frames_expected = (resolved_frames + body_stride_cfg - 1) // body_stride_cfg
                lineage_data.append(["Body Frames Processed (expected)", str(body_frames_expected)])
            else:
                lineage_data.append(["Body Frames Processed (expected)", "N/A"])
        else:
            lineage_data.append(["Body Frames Processed (expected)", "N/A (body tracking not run)"])

    # Model versions
    lineage_data.extend([
        ["Face Detector", detection_config.get("model_id", "retinaface_r50")],
        ["Face Tracker", "ByteTrack"],
        ["Embedding Model", embedding_config.get("embedding", {}).get("backend", "tensorrt") + " / ArcFace R100"],
        ["Body Detector", body_detection_config.get("person_detection", {}).get("model", "yolov8n")],
        ["Body Re-ID", body_detection_config.get("person_reid", {}).get("model", "osnet_x1_0")],
    ])

    lineage_table = Table(lineage_data, colWidths=[2 * inch, 4.5 * inch])
    lineage_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ])
    )
    story.append(lineage_table)

    # =========================================================================
    # SECTION 1: FACE DETECT
    # =========================================================================
    story.append(Paragraph("1. Face Detect", section_style))
    detect_count = _count_jsonl_lines_optional(detections_path)
    detect_count_display = _format_optional_count(
        detect_count,
        path=detections_path,
        label="detections.jsonl",
    )
    story.append(Paragraph(f"Total face detections: <b>{detect_count_display}</b>", body_style))

    # Configuration used with explanations
    story.append(Paragraph("Configuration (detection.yaml):", subsection_style))
    detect_config_rows = [
        _wrap_row(["Setting", "Value", "Description", "Tuning"], cell_style_small),
        _wrap_row([
            "confidence_th",
            str(detection_config.get("confidence_th", "N/A")),
            "Min confidence to accept a detection",
            "Lower: More faces (+ false positives) | Higher: Fewer, more confident",
        ], cell_style_small),
        _wrap_row([
            "min_size",
            str(detection_config.get("min_size", "N/A")),
            "Min face size in pixels",
            "Lower: Detect smaller/distant faces | Higher: Ignore small faces",
        ], cell_style_small),
        _wrap_row([
            "iou_th",
            str(detection_config.get("iou_th", "N/A")),
            "NMS IoU threshold for duplicate removal",
            "Lower: More aggressive duplicate removal | Higher: Keep overlapping",
        ], cell_style_small),
        _wrap_row([
            "wide_shot_mode",
            str(detection_config.get("wide_shot_mode", "N/A")),
            "Enhanced detection for distant faces",
            "Enable for wide shots with small faces",
        ], cell_style_small),
        _wrap_row([
            "wide_shot_confidence_th",
            str(detection_config.get("wide_shot_confidence_th", "N/A")),
            "Confidence threshold in wide shot mode",
            "Lower: More small faces | Higher: Stricter detection",
        ], cell_style_small),
        _wrap_row([
            "enable_person_fallback",
            str(detection_config.get("enable_person_fallback", "N/A")),
            "Use body detection when face fails",
            "Enable if missing faces when people turn away",
        ], cell_style_small),
    ]
    detect_config_table = Table(detect_config_rows, colWidths=[1.2 * inch, 0.5 * inch, 1.6 * inch, 2.2 * inch])
    detect_config_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("FONTNAME", (0, 1), (1, -1), "Courier"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(detect_config_table)

    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(
        Paragraph(
            f"&bull; detections.jsonl ({_file_size_str(detections_path)}) - {detect_count_display} records",
            bullet_style,
        )
    )

    # =========================================================================
    # SECTION 2: FACE TRACK
    # =========================================================================
    story.append(Paragraph("2. Face Track", section_style))
    track_count = _count_jsonl_lines_optional(tracks_path)
    track_count_display = _format_optional_count(
        track_count,
        path=tracks_path,
        label="tracks.jsonl",
    )
    story.append(Paragraph(f"Total face tracks: <b>{track_count_display}</b>", body_style))

    track_stats = [
        ["Metric", "Value"],
        ["Tracks Born", str(metrics.get("tracks_born", "N/A"))],
        ["Tracks Lost", str(metrics.get("tracks_lost", "N/A"))],
        ["ID Switches", str(metrics.get("id_switches", "N/A"))],
        ["Forced Splits", str(metrics.get("forced_splits", "N/A"))],
        ["Scene Cuts", str(track_metrics_data.get("scene_cuts", {}).get("count", "N/A"))],
    ]
    track_table = Table(track_stats, colWidths=[2 * inch, 2 * inch])
    track_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(track_table)

    # Diagnostic notes for alarming metrics
    forced_splits = metrics.get("forced_splits", 0)
    id_switches = metrics.get("id_switches", 0)
    if isinstance(forced_splits, (int, float)) and forced_splits > 50:
        story.append(Paragraph(
            f"⚠️ High forced splits ({forced_splits}): Appearance gate is aggressively splitting tracks. "
            "Consider disabling gate_enabled in tracking.yaml or adjusting appearance thresholds.",
            warning_style
        ))
    if isinstance(id_switches, (int, float)) and id_switches > 20:
        story.append(Paragraph(
            f"⚠️ High ID switches ({id_switches}): Tracker losing and re-acquiring faces frequently. "
            "Consider increasing track_buffer or lowering match_thresh in tracking.yaml.",
            warning_style
        ))

    # Configuration used with explanations
    story.append(Paragraph("Configuration (tracking.yaml):", subsection_style))
    gate_enabled_effective: str = str(tracking_config.get("gate_enabled", "N/A"))
    tracking_gate_meta = metrics.get("tracking_gate") if isinstance(metrics, dict) else None
    if isinstance(tracking_gate_meta, dict):
        enabled_raw = tracking_gate_meta.get("enabled")
        if isinstance(enabled_raw, bool):
            gate_enabled_effective = ("true" if enabled_raw else "false") + " (effective)"
    track_config_rows = [
        _wrap_row(["Setting", "Value", "Description", "Tuning"], cell_style_small),
        _wrap_row([
            "track_thresh",
            str(tracking_config.get("track_thresh", "N/A")),
            "Min confidence to continue tracking",
            "Lower: Track low-confidence faces | Higher: Drop uncertain faces",
        ], cell_style_small),
        _wrap_row([
            "match_thresh",
            str(tracking_config.get("match_thresh", "N/A")),
            "IoU threshold for bbox matching",
            "Lower: Match fast-moving faces | Higher: Reduce ID switches",
        ], cell_style_small),
        _wrap_row([
            "track_buffer",
            str(tracking_config.get("track_buffer", "N/A")),
            "Frames to keep track alive when lost",
            "Lower: Faster cleanup | Higher: Bridge brief occlusions",
        ], cell_style_small),
        _wrap_row([
            "new_track_thresh",
            str(tracking_config.get("new_track_thresh", "N/A")),
            "Min confidence to start a new track",
            "Lower: More new tracks | Higher: Fewer, more confident tracks",
        ], cell_style_small),
        _wrap_row([
            "gate_enabled",
            gate_enabled_effective,
            "Appearance-based track splitting",
            "Enable to split when face changes; disable if too many splits",
        ], cell_style_small),
    ]
    track_config_table = Table(track_config_rows, colWidths=[1.2 * inch, 0.5 * inch, 1.6 * inch, 2.2 * inch])
    track_config_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("FONTNAME", (0, 1), (1, -1), "Courier"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(track_config_table)

    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(
        Paragraph(
            f"&bull; tracks.jsonl ({_file_size_str(tracks_path)}) - {track_count_display} records",
            bullet_style,
        )
    )
    story.append(Paragraph(f"&bull; track_metrics.json ({_file_size_str(track_metrics_path)})", bullet_style))

    # =========================================================================
    # SECTION 3: FACE HARVEST / EMBED
    # =========================================================================
    story.append(Paragraph("3. Face Harvest / Embed", section_style))
    faces_count = _count_jsonl_lines_optional(faces_path)
    aligned_count = _count_jsonl_lines_optional(face_alignment_path)
    faces_count_display = _format_optional_count(
        faces_count,
        path=faces_path,
        label="faces.jsonl",
    )
    aligned_count_display = _format_optional_count(
        aligned_count,
        path=face_alignment_path,
        label="face_alignment/aligned_faces.jsonl",
    )

    story.append(Paragraph(f"Harvested faces: <b>{faces_count_display}</b>", body_style))
    story.append(Paragraph(f"Aligned faces: <b>{aligned_count_display}</b>", body_style))

    # Diagnostic for alignment drop
    if isinstance(faces_count, int) and isinstance(aligned_count, int) and faces_count > 0 and aligned_count > 0:
        alignment_rate = aligned_count / faces_count * 100
        if alignment_rate < 70:
            story.append(
                Paragraph(
                    f"⚠️ Low alignment rate ({alignment_rate:.1f}%): Many faces rejected by quality gating. "
                    "Consider lowering face_alignment.min_alignment_quality in embedding.yaml.",
                    warning_style,
                )
            )

    # Configuration used with explanations
    story.append(Paragraph("Configuration (embedding.yaml):", subsection_style))
    emb_cfg = embedding_config.get("embedding", {})
    face_align_cfg = embedding_config.get("face_alignment", {})
    embed_config_rows = [
        _wrap_row(["Setting", "Value", "Description", "Tuning"], cell_style_small),
        _wrap_row(["backend", str(emb_cfg.get("backend", "N/A")), "Embedding computation backend", "tensorrt: Fast GPU | pytorch: Compatible fallback"], cell_style_small),
        _wrap_row(["face_alignment.enabled", str(face_align_cfg.get("enabled", "N/A")), "Apply face alignment before embedding", "Enable for better embeddings; disable for speed"], cell_style_small),
        _wrap_row([
            "min_alignment_quality",
            str(face_align_cfg.get("min_alignment_quality", "N/A")),
            "Min quality score to embed a face",
            "Lower: Embed more faces (+ noise) | Higher: Only high-quality faces",
        ], cell_style_small),
        _wrap_row([
            "embedding_dim",
            str(embedding_config.get("output", {}).get("embedding_dim", "512")),
            "Output vector dimensions",
            "512 standard for ArcFace",
        ], cell_style_small),
    ]
    embed_config_table = Table(embed_config_rows, colWidths=[1.2 * inch, 0.5 * inch, 1.6 * inch, 2.2 * inch])
    embed_config_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("FONTNAME", (0, 1), (1, -1), "Courier"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(embed_config_table)

    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(
        Paragraph(
            f"&bull; faces.jsonl ({_file_size_str(faces_path)}) - {faces_count_display} records",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            f"&bull; face_alignment/aligned_faces.jsonl ({_file_size_str(face_alignment_path)}) - {aligned_count_display} records",
            bullet_style,
        )
    )
    story.append(Paragraph(f"&bull; faces.npy ({_file_size_str(faces_npy)})", bullet_style))

    # =========================================================================
    # SECTION 4: BODY DETECT
    # =========================================================================
    story.append(Paragraph("4. Body Detect", section_style))
    body_detect_count = body_detect_count_run
    body_detect_count_display = _format_optional_count(
        body_detect_count,
        path=body_detections_path,
        label="body_tracking/body_detections.jsonl",
    )
    story.append(Paragraph(f"Total body detections: <b>{body_detect_count_display}</b>", body_style))

    # Configuration used with explanations
    story.append(Paragraph("Configuration (body_detection.yaml):", subsection_style))
    person_det_cfg = body_detection_config.get("person_detection", {})

    override_source_value = "N/A"
    override_source_desc = "No override detected"
    if override_source.startswith("job_runs.request_json"):
        override_source_value = "job_runs.request_json"
        override_source_desc = override_source
    elif override_source.startswith("runs.config_json"):
        override_source_value = "runs.config_json"
        override_source_desc = override_source
    elif override_source.startswith("Override source not found"):
        override_source_value = "unknown"
        override_source_desc = override_source

    body_detect_config_rows = [
        _wrap_row(["Setting", "Value", "Description", "Tuning"], cell_style_small),
        _wrap_row([
            "body_tracking.enabled (YAML)",
            str(body_config_enabled),
            "Config from body_detection.yaml",
            "Enable to run body tracking in detect_track",
        ], cell_style_small),
        _wrap_row([
            "body_tracking.ran_effective",
            str(bool(body_tracking_ran_effective)),
            "Resolved from run-scoped marker/artifacts",
            "If false but legacy artifacts exist, may be stale",
        ], cell_style_small),
        _wrap_row([
            "body_tracking.override_source",
            override_source_value,
            override_source_desc,
            "If unknown, verify artifact scope + mtimes",
        ], cell_style_small),
        _wrap_row([
            "model",
            str(person_det_cfg.get("model", "N/A")),
            "YOLO model variant",
            "yolov8n: Fast | yolov8s/m: Accurate, slower",
        ], cell_style_small),
        _wrap_row([
            "confidence_threshold",
            str(person_det_cfg.get("confidence_threshold", "N/A")),
            "Min confidence for person detection",
            "Lower: More bodies (+ FP) | Higher: Fewer, confident",
        ], cell_style_small),
        _wrap_row([
            "min_height_px",
            str(person_det_cfg.get("min_height_px", "N/A")),
            "Min body height in pixels",
            "Lower: Detect distant | Higher: Ignore small",
        ], cell_style_small),
        _wrap_row([
            "detect_every_n_frames",
            str(person_det_cfg.get("detect_every_n_frames", "N/A")),
            "Frame stride for detection",
            "Lower: More detections | Higher: Faster",
        ], cell_style_small),
    ]
    body_detect_config_table = Table(
        body_detect_config_rows, colWidths=[1.2 * inch, 0.5 * inch, 1.6 * inch, 2.2 * inch]
    )
    body_detect_config_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("FONTNAME", (0, 1), (1, -1), "Courier"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(body_detect_config_table)

    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(Paragraph(
        f"&bull; body_tracking/body_detections.jsonl ({_file_size_str(body_detections_path)}) - {body_detect_count_display} records",
        bullet_style,
    ))
    if legacy_body_artifacts_exist or body_artifacts_exist_run:
        story.append(Paragraph("Body Artifact Timestamps (mtime, UTC):", subsection_style))
        mtime_table = Table(
            [
                ["Artifact", "Run-scoped", "Legacy"],
                ["body_detections.jsonl", _format_mtime(body_detections_path), _format_mtime(legacy_body_detections_path)],
                ["body_tracks.jsonl", _format_mtime(body_tracks_path), _format_mtime(legacy_body_tracks_path)],
                ["track_fusion.json", _format_mtime(track_fusion_path), _format_mtime(legacy_track_fusion_path)],
                ["screentime_comparison.json", _format_mtime(screentime_comparison_path), _format_mtime(legacy_screentime_comparison_path)],
            ],
            colWidths=[1.8 * inch, 2.0 * inch, 2.0 * inch],
        )
        mtime_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        story.append(mtime_table)

    # =========================================================================
    # SECTION 5: BODY TRACK
    # =========================================================================
    story.append(Paragraph("5. Body Track", section_style))
    body_track_count = body_track_count_run
    body_track_count_display = _format_optional_count(
        body_track_count,
        path=body_tracks_path,
        label="body_tracking/body_tracks.jsonl",
    )
    story.append(Paragraph(f"Total body tracks: <b>{body_track_count_display}</b>", body_style))

    # Configuration used with explanations
    story.append(Paragraph("Configuration (body_detection.yaml → person_tracking):", subsection_style))
    person_track_cfg = body_detection_config.get("person_tracking", {})
    body_track_config_rows = [
        _wrap_row(["Setting", "Value", "Description", "Tuning"], cell_style_small),
        _wrap_row(["tracker", str(person_track_cfg.get("tracker", "N/A")), "Tracking algorithm", "bytetrack: Fast | botsort/strongsort: More features"], cell_style_small),
        _wrap_row([
            "track_thresh",
            str(person_track_cfg.get("track_thresh", "N/A")),
            "Min confidence to continue body track",
            "Lower: Track uncertain | Higher: Drop low-confidence",
        ], cell_style_small),
        _wrap_row([
            "new_track_thresh",
            str(person_track_cfg.get("new_track_thresh", "N/A")),
            "Min confidence to start new body track",
            "Lower: More new tracks | Higher: Fewer, confident",
        ], cell_style_small),
        _wrap_row([
            "match_thresh",
            str(person_track_cfg.get("match_thresh", "N/A")),
            "IoU threshold for body bbox matching",
            "Lower: Match moving | Higher: Stricter matching",
        ], cell_style_small),
        _wrap_row([
            "track_buffer",
            str(person_track_cfg.get("track_buffer", "N/A")),
            "Frames to keep lost body track alive",
            "Lower: Faster cleanup | Higher: Bridge occlusions",
        ], cell_style_small),
        _wrap_row([
            "id_offset",
            str(person_track_cfg.get("id_offset", "N/A")),
            "Starting ID for body tracks",
            "Prevents ID collision with face tracks",
        ], cell_style_small),
    ]
    body_track_config_table = Table(body_track_config_rows, colWidths=[1.2 * inch, 0.5 * inch, 1.6 * inch, 2.2 * inch])
    body_track_config_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("FONTNAME", (0, 1), (1, -1), "Courier"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(body_track_config_table)

    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(Paragraph(
        f"&bull; body_tracking/body_tracks.jsonl ({_file_size_str(body_tracks_path)}) - {body_track_count_display} records",
        bullet_style,
    ))

    # =========================================================================
    # SECTION 6: TRACK FUSION
    # =========================================================================
    story.append(Paragraph("6. Track Fusion", section_style))

    # Note: actual_fused_pairs is computed earlier in the Run Health section

    fusion_status: str
    if face_tracks_count_run is None:
        fusion_status = (
            "SKIPPED (missing tracks.jsonl)"
            if not tracks_path.exists()
            else "SKIPPED (unreadable tracks.jsonl)"
        )
    elif body_track_count_run is None:
        fusion_status = (
            "SKIPPED (missing body_tracking/body_tracks.jsonl)"
            if not body_tracks_path.exists()
            else "SKIPPED (unreadable body_tracking/body_tracks.jsonl)"
        )
    elif track_fusion_available is True:
        fusion_status = "OK"
    elif track_fusion_available is False:
        fusion_status = "NOT RUN (missing body_tracking/track_fusion.json)"
    else:
        fusion_status = "ERROR (unreadable body_tracking/track_fusion.json)"

    face_tracks_input = _format_optional_count(face_tracks_count_run, path=tracks_path, label="tracks.jsonl")
    body_tracks_input = _format_optional_count(
        body_track_count_run,
        path=body_tracks_path,
        label="body_tracking/body_tracks.jsonl",
    )
    if track_fusion_available is True and tracked_ids_fused_total is not None:
        tracked_ids_value = str(tracked_ids_fused_total)
    elif track_fusion_available is True:
        tracked_ids_value = "N/A"
    else:
        tracked_ids_value = _na_artifact(track_fusion_path, "body_tracking/track_fusion.json")
    if actual_fused_pairs is not None:
        fused_pairs_value = str(actual_fused_pairs)
    elif track_fusion_available is True:
        fused_pairs_value = "N/A"
    else:
        fused_pairs_value = _na_artifact(track_fusion_path, "body_tracking/track_fusion.json")

    fusion_stats = [
        _wrap_row(["Metric", "Value", "Notes"]),
        _wrap_row(["Fusion Status", fusion_status, "Run-scoped inputs + outputs"]),
        _wrap_row(["Face Tracks (input)", face_tracks_input, "From run tracks.jsonl"]),
        _wrap_row(["Body Tracks (input)", body_tracks_input, "From run body_tracking/body_tracks.jsonl"]),
        _wrap_row(["Tracked IDs (from fusion output)", tracked_ids_value, "From body_tracking/track_fusion.json"]),
        _wrap_row(["Actual Fused Pairs", fused_pairs_value, "Mappings with both face AND body track IDs"]),
    ]
    fusion_table = Table(fusion_stats, colWidths=[2.0 * inch, 1.5 * inch, 2.0 * inch])
    fusion_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(fusion_table)

    # Fusion Diagnostics (intermediate counts so "0 fused pairs" is explainable)
    iou_cfg = track_fusion_config.get("iou_association", {})
    reid_cfg = track_fusion_config.get("reid_handoff", {})

    try:
        iou_threshold = float(iou_cfg.get("iou_threshold", 0.50))
    except (TypeError, ValueError):
        iou_threshold = 0.50
    try:
        min_overlap_ratio = float(iou_cfg.get("min_overlap_ratio", 0.70))
    except (TypeError, ValueError):
        min_overlap_ratio = 0.70
    face_in_upper_body = bool(iou_cfg.get("face_in_upper_body", True))
    try:
        upper_body_fraction = float(iou_cfg.get("upper_body_fraction", 0.5))
    except (TypeError, ValueError):
        upper_body_fraction = 0.5

    def _safe_int(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.isdigit():
                return int(cleaned)
        return None

    def _format_dist(dist: Any) -> str:
        if not isinstance(dist, dict):
            return "N/A"
        parts: list[str] = []
        for key in ("min", "median", "p95", "max"):
            val = dist.get(key)
            if isinstance(val, (int, float)):
                parts.append(f"{key}={float(val):.4f}")
        return " ".join(parts) if parts else "N/A"

    fusion_diag_payload = None
    if track_fusion_payload is not None:
        diag_block = track_fusion_payload.get("diagnostics")
        fusion_diag_payload = diag_block if isinstance(diag_block, dict) else None

    if fusion_diag_payload:
        candidate_overlaps = _safe_int(fusion_diag_payload.get("candidate_overlaps"))
        overlap_ratio_pass = _safe_int(fusion_diag_payload.get("overlap_ratio_pass"))
        iou_pass_count = _safe_int(fusion_diag_payload.get("iou_pass"))
        iou_pairs_count = _safe_int(fusion_diag_payload.get("iou_pairs"))
        reid_pairs_count = _safe_int(fusion_diag_payload.get("reid_pairs"))
        hybrid_pairs_count = _safe_int(fusion_diag_payload.get("hybrid_pairs"))
        final_pairs_count = _safe_int(fusion_diag_payload.get("final_pairs"))
        reid_comparisons = _safe_int(fusion_diag_payload.get("reid_comparisons"))
        reid_pass = _safe_int(fusion_diag_payload.get("reid_pass"))

        candidates_value = f"{candidate_overlaps} comparisons" if candidate_overlaps is not None else "N/A"
        candidates_notes = "From track_fusion.json diagnostics"
        overlap_value = f"{overlap_ratio_pass} comparisons" if overlap_ratio_pass is not None else "N/A"
        overlap_notes = f"threshold=min_overlap_ratio≥{min_overlap_ratio:.2f}"
        iou_value = f"{iou_pass_count} comparisons" if iou_pass_count is not None else "N/A"
        iou_notes = f"threshold=iou≥{iou_threshold:.3f} (and overlap_ratio pass)"
        iou_dist_value = _format_dist(fusion_diag_payload.get("iou_distribution"))
        overlap_dist_value = _format_dist(fusion_diag_payload.get("overlap_ratio_distribution"))

        reid_value = f"{reid_comparisons} comparisons" if reid_comparisons is not None else "N/A"
        reid_notes = "From track_fusion.json diagnostics"
        match_value = f"{reid_pass} matches" if reid_pass is not None else "N/A"
        match_notes = f"threshold={reid_cfg.get('similarity_threshold', 'N/A')}"

        final_pairs_detail = (
            f"{final_pairs_count} pairs (iou={iou_pairs_count or 0}, reid={reid_pairs_count or 0}, hybrid={hybrid_pairs_count or 0})"
            if final_pairs_count is not None
            else "N/A"
        )
        final_pairs_notes = (
            "final_pairs == iou_pairs + reid_pairs + hybrid_pairs"
            if (
                final_pairs_count is not None
                and iou_pairs_count is not None
                and reid_pairs_count is not None
                and hybrid_pairs_count is not None
                and final_pairs_count == (iou_pairs_count + reid_pairs_count + hybrid_pairs_count)
            )
            else "From track_fusion.json diagnostics"
        )
    else:
        overlap_diag = _track_fusion_overlap_diagnostics(
            face_tracks_path=tracks_path,
            body_tracks_path=body_tracks_path,
            iou_threshold=iou_threshold,
            min_overlap_ratio=min_overlap_ratio,
            face_in_upper_body=face_in_upper_body,
            upper_body_fraction=upper_body_fraction,
        )

        if overlap_diag.get("ok"):
            candidates_value = f"{overlap_diag['comparisons_considered']} comparisons"
            candidates_notes = (
                f"{overlap_diag['pairs_considered']} pairs across {overlap_diag['frames_with_candidates']} frames"
            )
            overlap_value = "N/A"
            overlap_notes = "N/A (not recorded in legacy diagnostics)"
            iou_value = f"{overlap_diag['comparisons_passing']} comparisons"
            iou_notes = (
                f"{overlap_diag['pairs_passing']} pairs; {overlap_diag['pairs_passing_min_frames']} pairs with ≥3 frames"
            )
            iou_dist_value = "N/A"
            overlap_dist_value = "N/A"
        else:
            candidates_value = "N/A"
            candidates_notes = str(overlap_diag.get("error") or "unknown")
            overlap_value = "N/A"
            overlap_notes = str(overlap_diag.get("error") or "unknown")
            iou_value = "N/A"
            iou_notes = str(overlap_diag.get("error") or "unknown")
            iou_dist_value = "N/A"
            overlap_dist_value = "N/A"

        reid_enabled = bool(reid_cfg.get("enabled", False))
        reid_similarity = reid_cfg.get("similarity_threshold")
        body_embeddings_path = body_tracking_dir / "body_embeddings.npy"
        if not reid_enabled:
            reid_value = "N/A"
            reid_notes = "reid_handoff.enabled=false"
            match_value = "N/A"
            match_notes = "Re-ID disabled"
        elif not body_embeddings_path.exists():
            reid_value = "N/A"
            reid_notes = "missing body_tracking/body_embeddings.npy"
            match_value = "N/A"
            match_notes = f"threshold={reid_similarity if reid_similarity is not None else 'N/A'}"
        else:
            reid_value = "N/A"
            reid_notes = "Re-ID comparison counts are not recorded in current artifacts"
            match_value = "N/A"
            match_notes = f"threshold={reid_similarity if reid_similarity is not None else 'N/A'}"

        final_pairs_detail = fused_pairs_value
        final_pairs_notes = "From body_tracking/track_fusion.json"

    story.append(Paragraph("Fusion Diagnostics", subsection_style))
    fusion_diag = [
        _wrap_row(["Step", "Count", "Notes"]),
        _wrap_row(["Candidate overlaps considered", candidates_value, candidates_notes]),
        _wrap_row(["Overlaps passing overlap_ratio", overlap_value, overlap_notes]),
        _wrap_row(["Overlaps passing IoU threshold", iou_value, iou_notes]),
        _wrap_row(["IoU distribution (min/median/p95/max)", iou_dist_value, "Sampled over candidate overlaps"]),
        _wrap_row(["Overlap ratio distribution (min/median/p95/max)", overlap_dist_value, "Sampled over candidate overlaps"]),
        _wrap_row(["Re-ID comparisons performed", reid_value, reid_notes]),
        _wrap_row(["Matches passing similarity threshold", match_value, match_notes]),
        _wrap_row(["Final fused pairs", final_pairs_detail, final_pairs_notes]),
    ]
    fusion_diag_table = Table(fusion_diag, colWidths=[2.0 * inch, 1.0 * inch, 2.5 * inch])
    fusion_diag_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(fusion_diag_table)

    # Configuration used with explanations
    story.append(Paragraph("Configuration (track_fusion.yaml):", subsection_style))
    fusion_config_rows = [
        _wrap_row(["Setting", "Value", "Description", "Tuning"], cell_style_small),
        _wrap_row([
            "track_fusion.enabled",
            str(track_fusion_config.get("track_fusion", {}).get("enabled", "N/A")),
            "Master switch for face-body fusion",
            "Enable to link face and body tracks for screen time",
        ], cell_style_small),
        _wrap_row([
            "iou_threshold",
            str(iou_cfg.get("iou_threshold", "N/A")),
            "Min IoU for face-in-body association",
            "Lower: More associations | Higher: Stricter spatial overlap",
        ], cell_style_small),
        _wrap_row([
            "min_overlap_ratio",
            str(iou_cfg.get("min_overlap_ratio", "N/A")),
            "Min face area inside body bbox",
            "Lower: Associate partial overlaps | Higher: Require face fully inside body",
        ], cell_style_small),
        _wrap_row([
            "reid_handoff.enabled",
            str(reid_cfg.get("enabled", "N/A")),
            "Use Re-ID when face disappears",
            "Enable to continue tracking via body when face turns away",
        ], cell_style_small),
        _wrap_row([
            "similarity_threshold",
            str(reid_cfg.get("similarity_threshold", "N/A")),
            "Min Re-ID similarity for handoff",
            "Lower: More handoffs (+ errors) | Higher: Conservative",
        ], cell_style_small),
    ]
    fusion_config_table = Table(fusion_config_rows, colWidths=[1.2 * inch, 0.5 * inch, 1.6 * inch, 2.2 * inch])
    fusion_config_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("FONTNAME", (0, 1), (1, -1), "Courier"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(fusion_config_table)

    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(Paragraph(f"&bull; body_tracking/track_fusion.json ({_file_size_str(track_fusion_path)})", bullet_style))

    # =========================================================================
    # SECTION 7: CLUSTER
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("7. Cluster", section_style))

    # Calculate singleton stats (treat derived metrics as N/A when identities.json is missing/unreadable).
    singleton_count: int | None
    singleton_frac: float | None
    if identities_count_run is None:
        singleton_count = None
        singleton_frac = None
    else:
        singleton_count = sum(1 for identity in identities_list if len(identity.get("track_ids", [])) == 1)
        singleton_frac = singleton_count / len(identities_list) if identities_list else 0.0

    faces_in_clusters = cluster_stats.get("faces") if identities_count_run is not None else None
    mixed_tracks = cluster_stats.get("mixed_tracks") if identities_count_run is not None else None
    outlier_tracks = cluster_stats.get("outlier_tracks") if identities_count_run is not None else None
    low_cohesion = cluster_stats.get("low_cohesion_identities") if identities_count_run is not None else None
    singleton_count_display = _format_optional_count(singleton_count, path=identities_path, label="identities.json")
    if singleton_frac is None:
        singleton_frac_display = _na_artifact(identities_path, "identities.json")
    else:
        singleton_frac_display = f"{singleton_frac:.1%} of total"
    faces_in_clusters_display = _format_optional_count(
        faces_in_clusters if isinstance(faces_in_clusters, int) else None,
        path=identities_path,
        label="identities.json",
    )
    mixed_tracks_display = _format_optional_count(
        mixed_tracks if isinstance(mixed_tracks, int) else None,
        path=identities_path,
        label="identities.json",
    )
    outlier_tracks_display = _format_optional_count(
        outlier_tracks if isinstance(outlier_tracks, int) else None,
        path=identities_path,
        label="identities.json",
    )
    low_cohesion_display = _format_optional_count(
        low_cohesion if isinstance(low_cohesion, int) else None,
        path=identities_path,
        label="identities.json",
    )

    cluster_table_data = [
        _wrap_row(["Metric", "Value", "Notes"]),
        _wrap_row(["Clusters (identities)", identities_count_display, "Unique identity groups"]),
        _wrap_row(["Total Faces in Clusters", faces_in_clusters_display, "Sum of face samples"]),
        _wrap_row(["Singleton Clusters", singleton_count_display, singleton_frac_display]),
        _wrap_row(["Mixed Tracks", mixed_tracks_display, "Tracks with multiple people (error)"]),
        _wrap_row(["Outlier Tracks", outlier_tracks_display, "Rejected from clusters"]),
        _wrap_row(["Low Cohesion", low_cohesion_display, "Clusters with poor internal similarity"]),
    ]
    cluster_table = Table(cluster_table_data, colWidths=[2 * inch, 1 * inch, 2.5 * inch])
    cluster_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(cluster_table)

    # Diagnostic notes
    singleton_high = isinstance(singleton_frac, (int, float)) and singleton_frac > 0.5
    mixed_high = isinstance(mixed_tracks, (int, float)) and mixed_tracks > 5
    if singleton_high and mixed_high:
        story.append(
            Paragraph(
                f"⚠️ High singleton fraction ({singleton_frac:.1%}) <i>and</i> high mixed tracks ({mixed_tracks}). "
                "<b>Do this first:</b> reduce tracking fragmentation (e.g., high forced_splits / gate over-splitting) "
                "so tracks are longer and more consistent. "
                "<b>Then:</b> increase min_identity_sim to reduce mixed-person clusters before adjusting cluster_thresh.",
                warning_style,
            )
        )
    elif singleton_high:
        story.append(Paragraph(
            f"⚠️ High singleton fraction ({singleton_frac:.1%}): Over half of clusters have only 1 track. "
            "Consider lowering cluster_thresh in clustering.yaml (currently "
            f"{clustering_config.get('cluster_thresh', 'N/A')}) to merge more aggressively.",
            warning_style
        ))
    elif mixed_high:
        story.append(Paragraph(
            f"⚠️ High mixed tracks ({mixed_tracks}): Some clusters contain tracks from different people. "
            "Consider increasing min_identity_sim (preferred) or raising cluster_thresh to separate people better.",
            warning_style
        ))

    # Configuration used with explanations
    story.append(Paragraph("Configuration (clustering.yaml):", subsection_style))
    singleton_merge_cfg = clustering_config.get("singleton_merge", {})
    cluster_config_rows = [
        _wrap_row(["Setting", "Value", "Description", "Tuning"], cell_style_small),
        _wrap_row(["Algorithm", "Agglomerative", "Hierarchical clustering method", "Groups similar embeddings bottom-up"], cell_style_small),
        _wrap_row(["Similarity Metric", "Cosine similarity", "How similarity is measured", "Distance: 1 - cosine_similarity"], cell_style_small),
        _wrap_row([
            "cluster_thresh",
            str(clustering_config.get("cluster_thresh", "N/A")),
            "Cosine similarity threshold to merge",
            "Lower: Merge aggressively | Higher: Stricter",
        ], cell_style_small),
        _wrap_row([
            "min_cluster_size",
            str(clustering_config.get("min_cluster_size", "N/A")),
            "Min tracks per cluster",
            "Higher: Filter brief appearances",
        ], cell_style_small),
        _wrap_row([
            "min_identity_sim",
            str(clustering_config.get("min_identity_sim", "N/A")),
            "Min similarity to cluster centroid",
            "Lower: Accept outliers | Higher: Eject dissimilar",
        ], cell_style_small),
        _wrap_row([
            "singleton_merge",
            str(singleton_merge_cfg.get("enabled", "N/A")),
            "Second-pass merge for singletons",
            "Enable if many singletons; disable if over-merging",
        ], cell_style_small),
        _wrap_row([
            "merge.similarity_thresh",
            str(singleton_merge_cfg.get("similarity_thresh", "N/A")),
            "Looser threshold for singleton merge",
            "Lower: Merge more | Higher: Conservative",
        ], cell_style_small),
    ]
    cluster_config_table = Table(cluster_config_rows, colWidths=[1.2 * inch, 0.5 * inch, 1.6 * inch, 2.2 * inch])
    cluster_config_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("FONTNAME", (0, 1), (1, -1), "Courier"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(cluster_config_table)
    story.append(Paragraph(
        "<b>Note:</b> cluster_thresh interpreted as <b>similarity</b> threshold.",
        note_style
    ))

    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(
        Paragraph(
            f"&bull; identities.json ({_file_size_str(identities_path)}) - {identities_count_display} identities",
            bullet_style,
        )
    )
    story.append(Paragraph(f"&bull; cluster_centroids.json ({_file_size_str(cluster_centroids_path)})", bullet_style))

    # =========================================================================
    # SECTION 8: FACES REVIEW (DB State)
    # =========================================================================
    story.append(Paragraph("8. Faces Review (DB State)", section_style))
    assigned_count = sum(1 for identity in identities_list if identity.get("person_id")) if identities_count_run is not None else None
    assigned_count_display = _format_optional_count(
        assigned_count,
        path=identities_path,
        label="identities.json",
    )
    unassigned_count = (
        (identities_count_run - assigned_count)
        if identities_count_run is not None and isinstance(assigned_count, int)
        else None
    )
    unassigned_count_display = _format_optional_count(
        unassigned_count,
        path=identities_path,
        label="identities.json",
    )

    # Show "unavailable" for DB-sourced data when DB is not connected
    if db_error:
        locked_count_str = "unavailable (DB error)"
    else:
        locked_count = len([lock for lock in identity_locks if lock.get("locked")])
        locked_count_str = str(locked_count)

    review_stats = [
        ["Metric", "Value"],
        ["Total Identities", identities_count_display],
        ["Assigned to Cast", assigned_count_display],
        ["Locked Identities", locked_count_str],
        ["Unassigned", unassigned_count_display],
    ]
    review_table = Table(review_stats, colWidths=[2 * inch, 2 * inch])
    review_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(review_table)

    if db_error:
        story.append(Paragraph(
            f"⚠️ <b>DB Error:</b> {db_error}",
            warning_style
        ))
        story.append(Paragraph(
            "<b>Impact:</b> DB-sourced counts (locks, batches, suggestions) are unavailable in this report due to DB connection error.",
            note_style
        ))
        manual_assignments = identities_data.get("manual_assignments")
        manual_assignments_count = len(manual_assignments) if isinstance(manual_assignments, dict) else 0
        if manual_assignments_count > 0:
            story.append(Paragraph(
                f"Manual assignments loaded from identities.json fallback: <b>{manual_assignments_count}</b>",
                note_style
            ))

    story.append(Paragraph("Data Sources:", subsection_style))
    story.append(Paragraph("&bull; identity_locks table (DB)", bullet_style))
    story.append(Paragraph("&bull; identities.json (manual_assignments)", bullet_style))

    # =========================================================================
    # SECTION 9: SMART SUGGESTIONS
    # =========================================================================
    story.append(Paragraph("9. Smart Suggestions", section_style))

    # Show "unavailable" for DB-sourced data when DB is not connected
    if db_error:
        suggestion_stats = [
            ["Metric", "Value"],
            ["Suggestion Batches", "unavailable (DB error)"],
            ["Total Suggestions", "unavailable"],
            ["Dismissed", "unavailable"],
            ["Applied", "unavailable"],
            ["Pending", "unavailable"],
        ]
    else:
        dismissed_count = sum(1 for s in suggestions_rows if s.get("dismissed"))
        applied_count = len(suggestion_applies)
        pending_count = len(suggestions_rows) - dismissed_count - applied_count
        suggestion_stats = [
            ["Metric", "Value"],
            ["Suggestion Batches", str(len(suggestion_batches))],
            ["Total Suggestions", str(len(suggestions_rows))],
            ["Dismissed", str(dismissed_count)],
            ["Applied", str(applied_count)],
            ["Pending", str(pending_count)],
        ]

    suggestion_table = Table(suggestion_stats, colWidths=[2 * inch, 2 * inch])
    suggestion_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(suggestion_table)

    story.append(Paragraph("Data Sources:", subsection_style))
    story.append(Paragraph("&bull; suggestion_batches table (DB)", bullet_style))
    story.append(Paragraph("&bull; suggestions table (DB)", bullet_style))
    story.append(Paragraph("&bull; suggestion_applies table (DB)", bullet_style))

    # =========================================================================
    # SECTION 10: SCREEN TIME ANALYZE
    # =========================================================================
    story.append(Paragraph("10. Screen Time Analyze", section_style))

    duration_gain: float | None
    if screentime_summary is None:
        duration_gain = None
        story.append(Paragraph(
            f"⚠️ Screen Time Analyze is unavailable: {_na_artifact(screentime_comparison_path, 'body_tracking/screentime_comparison.json')}.",
            warning_style,
        ))
        if face_tracks_fallback.get("ok") and isinstance(face_tracks_fallback.get("total_duration_s"), (int, float)):
            approx_duration_s = float(face_tracks_fallback["total_duration_s"])
            fallback_table = Table(
                [
                    ["Metric", "Value", "Notes"],
                    ["Face-only screen time (approx)", f"{approx_duration_s:.2f}s", "Sum of per-track spans from tracks.jsonl"],
                ],
                colWidths=[2.2 * inch, 1.2 * inch, 2.1 * inch],
            )
            fallback_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ])
            )
            story.append(Spacer(1, 6))
            story.append(fallback_table)
        else:
            story.append(Paragraph(
                "Face-only fallback is unavailable because run tracks.jsonl is missing or unreadable.",
                note_style,
            ))
        story.append(Spacer(1, 8))
        unavailable = _na_artifact(screentime_comparison_path, "body_tracking/screentime_comparison.json")
        story.append(Paragraph(f"Total tracked IDs analyzed: <b>{unavailable}</b>", body_style))
        story.append(Paragraph(f"Tracked IDs with gain: <b>{unavailable}</b>", body_style))
    else:
        face_total_s = _safe_float(
            screentime_summary.get("face_total_s", screentime_summary.get("total_face_only_duration", 0))
        )
        body_total_s = _safe_float(
            screentime_summary.get("body_total_s", screentime_summary.get("total_body_duration", 0))
        )
        fused_total_s = _safe_float(
            screentime_summary.get("fused_total_s", screentime_summary.get("total_fused_duration", 0))
        )
        combined_total_s = _safe_float(
            screentime_summary.get("combined_total_s", screentime_summary.get("total_combined_duration", 0))
        )
        gain_total_s = _safe_float(
            screentime_summary.get("gain_total_s", screentime_summary.get("total_duration_gain", 0))
        )
        duration_gain = gain_total_s
        gain_vs_combined_pct = _format_percent(gain_total_s, combined_total_s, na="N/A")
        gain_vs_face_only_pct = _format_percent(gain_total_s, face_total_s, na="N/A")

        screentime_breakdown = [
            ["Metric", "Duration", "% of Combined"],
            ["Face baseline total", f"{face_total_s:.2f}s", _format_percent(face_total_s, combined_total_s, na="N/A")],
            ["Body total (absolute)", f"{body_total_s:.2f}s", _format_percent(body_total_s, combined_total_s, na="N/A")],
            ["Face∩Body overlap total", f"{fused_total_s:.2f}s", _format_percent(fused_total_s, combined_total_s, na="N/A")],
            ["Combined total (Face ∪ Body)", f"{combined_total_s:.2f}s", "100%"],
            ["Gain vs Face baseline (Combined − Face)", f"+{gain_total_s:.2f}s", gain_vs_combined_pct],
        ]
        screentime_table = Table(screentime_breakdown, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch])
        screentime_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#c6f6d5")),
            ])
        )
        story.append(screentime_table)

        # Explicitly report the two gain % references so "0.0%" isn't printed when face_only == 0.
        story.append(Spacer(1, 6))
        gain_pct_table = Table(
            [
                ["Gain % Reference", "Value"],
                ["Gain vs Combined Total", gain_vs_combined_pct],
                ["Gain vs Face-only Total", gain_vs_face_only_pct],
            ],
            colWidths=[2.2 * inch, 2.8 * inch],
        )
        gain_pct_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(gain_pct_table)

        # Adjust wording based on whether fusion actually occurred
        if actual_fused_pairs is not None and actual_fused_pairs > 0:
            story.append(Paragraph(
                f"<b>Note:</b> 'Gain vs Face baseline' is the incremental time added by body visibility "
                f"(Combined − Face baseline), from {actual_fused_pairs} fused face-body pair(s).",
                note_style
            ))
        elif body_tracking_ran_effective and (body_track_count_run or 0) > 0:
            story.append(Paragraph(
                "<b>Note:</b> Body tracking ran but no face-body pairs were successfully fused. "
                "This may indicate IoU/Re-ID thresholds need tuning, or faces and bodies didn't overlap temporally.",
                warning_style
            ))
        else:
            story.append(Paragraph(
                "<b>Note:</b> Body tracking was not enabled or produced no tracks. "
                "'Gain from Body Tracking' will be zero without body-based fusion.",
                note_style
            ))

        # Additional stats
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"Total tracked IDs analyzed: <b>{screentime_summary.get('total_identities', 0)}</b>", body_style))
        story.append(Paragraph(f"Tracked IDs with gain: <b>{screentime_summary.get('identities_with_gain', 0)}</b>", body_style))

    # Configuration used with explanations
    story.append(Paragraph("Configuration (screen_time_v2.yaml):", subsection_style))
    preset = screentime_config.get("preset", "bravo_default")
    presets = screentime_config.get("screen_time_presets", {})
    active_preset = presets.get(preset, {})
    screentime_config_rows = [
        _wrap_row(["Setting", "Value", "Description", "Tuning"], cell_style_small),
        _wrap_row(["Active Preset", preset, "Named configuration profile", "bravo_default: Loose | stricter: Moderate | strict: Conservative"], cell_style_small),
        _wrap_row([
            "quality_min",
            str(active_preset.get("quality_min", "N/A")),
            "Min detection quality to count",
            "Lower: Count blurry faces | Higher: Only clear",
        ], cell_style_small),
        _wrap_row([
            "gap_tolerance_s",
            str(active_preset.get("gap_tolerance_s", "N/A")),
            "Max gap to merge into one segment",
            "Lower: More segments | Higher: Merge through cuts",
        ], cell_style_small),
        _wrap_row([
            "screen_time_mode",
            str(active_preset.get("screen_time_mode", "N/A")),
            "How to compute screen time",
            "tracks: Full spans | faces: Only detected frames",
        ], cell_style_small),
        _wrap_row([
            "edge_padding_s",
            str(active_preset.get("edge_padding_s", "N/A")),
            "Padding added to segment edges",
            "Lower: Conservative | Higher: Generous counting",
        ], cell_style_small),
        _wrap_row([
            "track_coverage_min",
            str(active_preset.get("track_coverage_min", "N/A")),
            "Min detection coverage per track",
            "Lower: Count sparse | Higher: Require consistent",
        ], cell_style_small),
    ]
    screentime_config_table = Table(screentime_config_rows, colWidths=[1.2 * inch, 0.5 * inch, 1.6 * inch, 2.2 * inch])
    screentime_config_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("FONTNAME", (0, 1), (1, -1), "Courier"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(screentime_config_table)

    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(Paragraph(f"&bull; body_tracking/screentime_comparison.json ({_file_size_str(screentime_comparison_path)})", bullet_style))

    # =========================================================================
    # SECTION 11: WHAT LIKELY NEEDS TUNING
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("11. What Likely Needs Tuning", section_style))
    story.append(Paragraph(
        "Based on the metrics above, here are suggested areas to investigate:",
        body_style
    ))
    story.append(Spacer(1, 8))

    tuning_suggestions: list[tuple[str, str, str]] = []

    # Detection tuning
    if detect_count is None:
        tuning_suggestions.append((
            "Face Detection",
            "detections.jsonl missing or unreadable",
            "Rerun detect/track or verify run-scoped artifacts are present in S3 (runs/.../{run_id}/detections.jsonl)",
        ))
    elif detect_count == 0:
        tuning_suggestions.append((
            "Face Detection",
            "No detections found",
            "Lower confidence_th in detection.yaml, enable wide_shot_mode, or check video quality"
        ))
    elif detect_count < 100:
        tuning_suggestions.append((
            "Face Detection",
            f"Low detection count ({detect_count})",
            "Consider lowering confidence_th or min_size in detection.yaml"
        ))

    # Track tuning
    if isinstance(forced_splits, (int, float)) and forced_splits > 50:
        tuning_suggestions.append((
            "Face Tracking",
            f"High forced splits ({forced_splits})",
            "Disable gate_enabled in tracking.yaml or adjust appearance thresholds to reduce false splits"
        ))
    if isinstance(id_switches, (int, float)) and id_switches > 20:
        tuning_suggestions.append((
            "Face Tracking",
            f"High ID switches ({id_switches})",
            "Increase track_buffer in tracking.yaml or lower match_thresh"
        ))

    # Embedding tuning
    if isinstance(faces_count, int) and isinstance(aligned_count, int) and faces_count > 0 and aligned_count > 0:
        alignment_rate = aligned_count / faces_count * 100
        if alignment_rate < 70:
            tuning_suggestions.append((
                "Face Embedding",
                f"Low alignment rate ({alignment_rate:.1f}%)",
                "Lower min_alignment_quality in embedding.yaml (currently "
                f"{embedding_config.get('face_alignment', {}).get('min_alignment_quality', 'N/A')})"
            ))

    # Cluster tuning
    singleton_high = isinstance(singleton_frac, (int, float)) and singleton_frac > 0.5
    mixed_high = isinstance(mixed_tracks, (int, float)) and mixed_tracks > 5
    if singleton_high and mixed_high:
        tuning_suggestions.append((
            "Clustering",
            f"High singleton fraction ({singleton_frac:.1%}) and mixed tracks ({mixed_tracks})",
            "First reduce tracking fragmentation (e.g., forced_splits / gate over-splitting), then increase "
            "min_identity_sim. Adjust cluster_thresh only after tracking is stable.",
        ))
    elif singleton_high:
        tuning_suggestions.append((
            "Clustering",
            f"High singleton fraction ({singleton_frac:.1%})",
            f"Lower cluster_thresh (currently {clustering_config.get('cluster_thresh', 'N/A')}) "
            "or enable singleton_merge",
        ))
    elif mixed_high:
        tuning_suggestions.append((
            "Clustering",
            f"Mixed tracks ({mixed_tracks})",
            "Increase min_identity_sim (preferred) or raise cluster_thresh to separate people better",
        ))

    # Body tracking tuning
    if body_detect_count == 0 and body_detection_config.get("body_tracking", {}).get("enabled"):
        tuning_suggestions.append((
            "Body Detection",
            "No body detections despite being enabled",
            "Check video quality or lower person_detection.confidence_threshold"
        ))
    if actual_fused_pairs == 0 and num_face_tracks > 0 and num_body_tracks > 0:
        tuning_suggestions.append((
            "Track Fusion",
            "No face-body pairs fused",
            "Lower iou_association.iou_threshold or reid_handoff.similarity_threshold in track_fusion.yaml"
        ))

    # Screen time tuning
    if duration_gain == 0 and num_body_tracks > 0:
        tuning_suggestions.append((
            "Screen Time",
            "No duration gain from body tracking",
            "Verify track fusion is working; check screentime_comparison.json for details"
        ))

    if tuning_suggestions:
        tuning_table_data = [_wrap_row(["Stage", "Issue", "Suggested Action"])]
        for suggestion in tuning_suggestions:
            tuning_table_data.append(_wrap_row(list(suggestion)))
        tuning_table = Table(tuning_table_data, colWidths=[1.0 * inch, 1.8 * inch, 2.7 * inch])
        tuning_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fc8181")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#fff5f5"), colors.white]),
            ])
        )
        story.append(tuning_table)
    else:
        story.append(Paragraph(
            "✓ No obvious tuning issues detected. All metrics are within expected ranges.",
            note_style
        ))

    # =========================================================================
    # APPENDIX: ARTIFACT MANIFEST
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Appendix: Artifact Manifest", section_style))
    story.append(Paragraph("Complete listing of all referenced artifacts with status, size, and record counts:", body_style))
    story.append(Spacer(1, 8))

    # "In Bundle" indicates whether the artifact is included when exporting as ZIP with include_artifacts=True
    # ZIP bundle includes: .json, .jsonl files from run root + body_tracking/ + analytics/
    # Excludes: .npy files, face_alignment/ subdirectory
    # If a file is missing, "In Bundle" shows "N/A" (nothing to include)
    artifact_data = [
        ["Artifact", "Status", "Size", "Records", "Stage", "In Bundle"],
        (*_artifact_row(detections_path), detect_count_display, "Face Detect", _bundle_status(detections_path, in_allowlist=True)),
        (*_artifact_row(tracks_path), track_count_display, "Face Track", _bundle_status(tracks_path, in_allowlist=True)),
        (*_artifact_row(track_metrics_path), "-", "Face Track", _bundle_status(track_metrics_path, in_allowlist=True)),
        (*_artifact_row(faces_path), faces_count_display, "Face Harvest", _bundle_status(faces_path, in_allowlist=True)),
        (*_artifact_row(face_alignment_path, "face_alignment/aligned_faces.jsonl"), aligned_count_display, "Face Embed", _bundle_status(face_alignment_path, in_allowlist=False)),
        (*_artifact_row(faces_npy), "-", "Face Embed", _bundle_status(faces_npy, in_allowlist=False)),
        (*_artifact_row(identities_path), identities_count_display, "Cluster", _bundle_status(identities_path, in_allowlist=True)),
        (*_artifact_row(cluster_centroids_path), "-", "Cluster", _bundle_status(cluster_centroids_path, in_allowlist=True)),
        (*_artifact_row(body_detections_path, "body_tracking/body_detections.jsonl"), body_detect_count_display, "Body Detect", _bundle_status(body_detections_path, in_allowlist=True)),
        (*_artifact_row(body_tracks_path, "body_tracking/body_tracks.jsonl"), body_track_count_display, "Body Track", _bundle_status(body_tracks_path, in_allowlist=True)),
        (*_artifact_row(track_fusion_path, "body_tracking/track_fusion.json"), "-", "Track Fusion", _bundle_status(track_fusion_path, in_allowlist=True)),
        (*_artifact_row(screentime_comparison_path, "body_tracking/screentime_comparison.json"), "-", "Screen Time", _bundle_status(screentime_comparison_path, in_allowlist=True)),
    ]
    artifact_table = Table(artifact_data, colWidths=[2.5 * inch, 0.6 * inch, 0.6 * inch, 0.5 * inch, 0.9 * inch, 0.6 * inch])
    artifact_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ])
    )
    story.append(artifact_table)

    # Legacy artifact manifest (diagnostic only; may be out-of-scope for this run_id).
    story.append(Spacer(1, 12))
    story.append(Paragraph("Legacy Artifact Manifest (out-of-scope)", subsection_style))
    story.append(Paragraph(
        f"Legacy artifacts live under episode manifests (not run-scoped). marker_run_id={legacy_body_marker_run_id or 'unknown'}; "
        f"out_of_scope={'yes' if legacy_body_out_of_scope else 'no'}.",
        note_style,
    ))
    if legacy_body_artifacts_exist:
        legacy_body_detect_display = _format_optional_count(
            legacy_body_detect_count,
            path=legacy_body_detections_path,
            label="legacy/body_tracking/body_detections.jsonl",
        )
        legacy_body_track_display = _format_optional_count(
            legacy_body_track_count,
            path=legacy_body_tracks_path,
            label="legacy/body_tracking/body_tracks.jsonl",
        )
        legacy_artifact_data = [
            ["Artifact", "Status", "Size", "Records", "Stage", "In Bundle"],
            (*_artifact_row(legacy_body_detections_path, "legacy/body_tracking/body_detections.jsonl"), legacy_body_detect_display, "Legacy Body Detect", _bundle_status(legacy_body_detections_path, in_allowlist=False)),
            (*_artifact_row(legacy_body_tracks_path, "legacy/body_tracking/body_tracks.jsonl"), legacy_body_track_display, "Legacy Body Track", _bundle_status(legacy_body_tracks_path, in_allowlist=False)),
            (*_artifact_row(legacy_track_fusion_path, "legacy/body_tracking/track_fusion.json"), "-", "Legacy Track Fusion", _bundle_status(legacy_track_fusion_path, in_allowlist=False)),
            (*_artifact_row(legacy_screentime_comparison_path, "legacy/body_tracking/screentime_comparison.json"), "-", "Legacy Screen Time", _bundle_status(legacy_screentime_comparison_path, in_allowlist=False)),
        ]
        legacy_table = Table(legacy_artifact_data, colWidths=[2.5 * inch, 0.6 * inch, 0.6 * inch, 0.5 * inch, 0.9 * inch, 0.6 * inch])
        legacy_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a5568")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
            ])
        )
        story.append(legacy_table)
    else:
        story.append(Paragraph("No legacy body_tracking artifacts found.", note_style))

    # Build PDF
    try:
        doc.build(story)
        pdf_bytes = buffer.getvalue()
    finally:
        buffer.close()
        if temp_dir is not None:
            temp_dir.cleanup()

    download_name = f"screenalytics_{ep_id}_{run_id_norm}_debug_report.pdf"
    return pdf_bytes, download_name


# ---------------------------------------------------------------------------
# S3 Export Upload
# ---------------------------------------------------------------------------


@dataclass
class ExportUploadResult:
    """Result of exporting and optionally uploading to S3."""

    success: bool
    s3_key: str | None = None
    s3_bucket: str | None = None
    s3_url: str | None = None
    error: str | None = None
    bytes_uploaded: int = 0
    upload_time_ms: float = 0.0


def _get_export_s3_key(ep_id: str, run_id: str, filename: str) -> str:
    """Generate deterministic S3 key for an export file.

    Canonical format: runs/{show}/s{ss}/e{ee}/{run_id}/exports/{filename}

    Falls back to legacy (write) prefix only when ep_id parsing fails:
        runs/{ep_id}/{run_id}/exports/{filename}
    """
    return run_layout.run_export_s3_key(ep_id, run_id, filename)


def validate_s3_for_export() -> tuple[bool, str | None, dict[str, Any]]:
    """Validate that S3 is properly configured for export uploads.

    Returns:
        (is_valid, error_message, config_details)
    """
    try:
        from apps.api.services.validation import validate_storage_backend_config
        config = validate_storage_backend_config()
    except ImportError:
        # Fallback to manual check
        backend = os.environ.get("STORAGE_BACKEND", "local").lower()
        if backend == "local":
            return True, None, {"backend": backend, "s3_enabled": False}
        return False, "Validation module not available", {"backend": backend}

    details = {
        "backend": config.backend,
        "bucket": config.bucket,
        "region": config.region,
        "s3_endpoint": config.s3_endpoint,
        "has_credentials": config.has_credentials,
        "is_fallback": config.is_fallback,
        "s3_enabled": config.backend in ("s3", "minio", "hybrid"),
    }

    # Local backend is always valid (just won't upload)
    if config.backend == "local":
        return True, None, details

    # For S3/MinIO/hybrid, check required configuration
    if config.is_fallback:
        return False, config.fallback_reason, details

    if not config.bucket:
        return False, "S3 bucket not configured (set SCREENALYTICS_S3_BUCKET or AWS_S3_BUCKET)", details

    # Note: credentials may come from instance profiles, so we only warn, not fail
    if not config.has_credentials and config.backend == "minio":
        return False, "MinIO requires explicit credentials", details

    return True, None, details


def upload_export_to_s3(
    *,
    ep_id: str,
    run_id: str,
    file_bytes: bytes,
    filename: str,
    content_type: str = "application/octet-stream",
    fail_on_error: bool = True,
) -> ExportUploadResult:
    """Upload an export file to S3.

    Args:
        ep_id: Episode ID
        run_id: Run ID
        file_bytes: File content as bytes
        filename: Filename for the export (e.g., "debug_report.pdf")
        content_type: MIME type for the upload
        fail_on_error: If True, raise exception on failure; if False, return error in result

    Returns:
        ExportUploadResult with upload status and S3 location

    Raises:
        RuntimeError: If fail_on_error=True and upload fails
    """
    import time

    # Validate S3 configuration
    is_valid, error_msg, config = validate_s3_for_export()

    # If not S3-enabled, return success without upload
    if not config.get("s3_enabled"):
        LOGGER.info("[export-s3] S3 not enabled (backend=%s), skipping upload", config.get("backend"))
        return ExportUploadResult(success=True, error="S3 not enabled, file not uploaded")

    if not is_valid:
        msg = f"S3 configuration invalid: {error_msg}"
        LOGGER.error("[export-s3] %s", msg)
        if fail_on_error:
            raise RuntimeError(msg)
        return ExportUploadResult(success=False, error=msg)

    # Resolve S3 key layout (canonical preferred; legacy only if ep_id parsing fails).
    layout = run_layout.get_run_s3_layout(ep_id, run_id)
    cleaned_filename = (filename or "").strip().lstrip("/\\")
    s3_key = f"{layout.write_prefix}exports/{cleaned_filename}"
    bucket = config.get("bucket")
    legacy_key = f"{layout.legacy_prefix}exports/{cleaned_filename}"
    if layout.s3_layout == "canonical" and legacy_key != s3_key:
        # Guard: avoid double-writing into both canonical + legacy prefixes.
        LOGGER.info(
            "[export-s3] s3_layout=canonical key=%s legacy_suppressed=%s",
            s3_key,
            legacy_key,
        )
    elif layout.s3_layout == "legacy":
        LOGGER.warning("[export-s3] s3_layout=legacy key=%s", s3_key)

    # Try to upload
    start_time = time.time()
    try:
        from apps.api.services.storage import StorageService
        storage = StorageService()

        LOGGER.info(
            "[export-s3] Uploading %s to s3://%s/%s (%d bytes)",
            cleaned_filename,
            bucket,
            s3_key,
            len(file_bytes),
        )

        if not storage.upload_bytes(file_bytes, s3_key, content_type=content_type):
            msg = f"Upload failed for s3://{bucket}/{s3_key}"
            LOGGER.error("[export-s3] %s", msg)
            if fail_on_error:
                raise RuntimeError(msg)
            return ExportUploadResult(success=False, error=msg, s3_key=s3_key, s3_bucket=bucket)

        elapsed_ms = (time.time() - start_time) * 1000
        LOGGER.info("[export-s3] Successfully uploaded %s to s3://%s/%s in %.1fms", filename, bucket, s3_key, elapsed_ms)

        # Generate presigned URL for convenience
        presigned_url = None
        try:
            presigned_url = storage.presign_get(s3_key, expiration=3600)
        except Exception:
            pass  # URL generation is best-effort

        return ExportUploadResult(
            success=True,
            s3_key=s3_key,
            s3_bucket=bucket,
            s3_url=presigned_url,
            bytes_uploaded=len(file_bytes),
            upload_time_ms=elapsed_ms,
        )

    except Exception as exc:
        elapsed_ms = (time.time() - start_time) * 1000
        msg = f"S3 upload failed for {filename}: {exc}"
        LOGGER.exception("[export-s3] %s", msg)
        if fail_on_error:
            raise RuntimeError(msg) from exc
        return ExportUploadResult(
            success=False,
            error=msg,
            s3_key=s3_key,
            s3_bucket=bucket,
            upload_time_ms=elapsed_ms,
        )


def build_and_upload_debug_pdf(
    *,
    ep_id: str,
    run_id: str,
    upload_to_s3: bool = True,
    fail_on_s3_error: bool = False,
    write_index: bool = True,
) -> tuple[bytes, str, ExportUploadResult | None]:
    """Build debug PDF and optionally upload to S3.

    Args:
        ep_id: Episode ID
        run_id: Run ID
        upload_to_s3: Whether to upload to S3 (if configured)
        fail_on_s3_error: If True, raise on S3 upload failure
        write_index: If True, write export_index.json marker file

    Returns:
        (pdf_bytes, download_filename, upload_result or None)
    """
    pdf_bytes, download_name = build_screentime_run_debug_pdf(ep_id=ep_id, run_id=run_id)

    upload_result = None
    if upload_to_s3:
        upload_result = upload_export_to_s3(
            ep_id=ep_id,
            run_id=run_id,
            file_bytes=pdf_bytes,
            filename="debug_report.pdf",
            content_type="application/pdf",
            fail_on_error=fail_on_s3_error,
        )

    # Write export index marker for later lookup
    if write_index:
        try:
            from apps.api.services.run_artifact_store import write_export_index, ArtifactSyncResult

            # Convert ExportUploadResult to ArtifactSyncResult-like object for write_export_index
            artifact_result = None
            if upload_result:
                s3_layout = run_layout.get_run_s3_layout(ep_id, run_id)
                artifact_result = ArtifactSyncResult(
                    success=upload_result.success,
                    uploaded_count=1 if upload_result.success else 0,
                    bytes_uploaded=upload_result.bytes_uploaded,
                    sync_time_ms=upload_result.upload_time_ms,
                    s3_prefix=f"{s3_layout.write_prefix}exports/",
                    s3_bucket=upload_result.s3_bucket,
                    errors=[upload_result.error] if upload_result.error else [],
                    uploaded_files=[upload_result.s3_key] if upload_result.s3_key else [],
                )

            write_export_index(
                ep_id=ep_id,
                run_id=run_id,
                export_type="pdf",
                export_key=upload_result.s3_key if upload_result else None,
                export_bytes=len(pdf_bytes),
                upload_result=artifact_result,
            )
        except Exception as exc:
            LOGGER.warning("[export] Failed to write export index: %s", exc)

    return pdf_bytes, download_name, upload_result


def build_and_upload_debug_bundle(
    *,
    ep_id: str,
    run_id: str,
    include_artifacts: bool = True,
    include_logs: bool = True,
    upload_to_s3: bool = True,
    fail_on_s3_error: bool = False,
    write_index: bool = True,
) -> tuple[str, str, ExportUploadResult | None]:
    """Build debug bundle ZIP and optionally upload to S3.

    Args:
        ep_id: Episode ID
        run_id: Run ID
        include_artifacts: Include run artifacts in ZIP
        include_logs: Include logs in ZIP
        upload_to_s3: Whether to upload to S3 (if configured)
        fail_on_s3_error: If True, raise on S3 upload failure
        write_index: If True, write export_index.json marker file

    Returns:
        (zip_path, download_filename, upload_result or None)
    """
    zip_path, download_name = build_run_debug_bundle_zip(
        ep_id=ep_id,
        run_id=run_id,
        include_artifacts=include_artifacts,
        include_logs=include_logs,
    )

    upload_result = None
    zip_bytes_len = 0
    if upload_to_s3:
        try:
            with open(zip_path, "rb") as f:
                zip_bytes = f.read()
            zip_bytes_len = len(zip_bytes)
            upload_result = upload_export_to_s3(
                ep_id=ep_id,
                run_id=run_id,
                file_bytes=zip_bytes,
                filename="debug_bundle.zip",
                content_type="application/zip",
                fail_on_error=fail_on_s3_error,
            )
        except Exception as exc:
            LOGGER.warning("[export-s3] Failed to read ZIP for S3 upload: %s", exc)
            if fail_on_s3_error:
                raise

    # Write export index marker for later lookup
    if write_index:
        try:
            from apps.api.services.run_artifact_store import write_export_index, ArtifactSyncResult

            # Get ZIP size if not already known
            if zip_bytes_len == 0:
                try:
                    zip_bytes_len = Path(zip_path).stat().st_size
                except OSError:
                    pass

            # Convert ExportUploadResult to ArtifactSyncResult-like object
            artifact_result = None
            if upload_result:
                s3_layout = run_layout.get_run_s3_layout(ep_id, run_id)
                artifact_result = ArtifactSyncResult(
                    success=upload_result.success,
                    uploaded_count=1 if upload_result.success else 0,
                    bytes_uploaded=upload_result.bytes_uploaded,
                    sync_time_ms=upload_result.upload_time_ms,
                    s3_prefix=f"{s3_layout.write_prefix}exports/",
                    s3_bucket=upload_result.s3_bucket,
                    errors=[upload_result.error] if upload_result.error else [],
                    uploaded_files=[upload_result.s3_key] if upload_result.s3_key else [],
                )

            write_export_index(
                ep_id=ep_id,
                run_id=run_id,
                export_type="zip",
                export_key=upload_result.s3_key if upload_result else None,
                export_bytes=zip_bytes_len,
                upload_result=artifact_result,
            )
        except Exception as exc:
            LOGGER.warning("[export] Failed to write export index: %s", exc)

    return zip_path, download_name, upload_result

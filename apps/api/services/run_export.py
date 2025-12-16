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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
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


def build_screentime_run_debug_pdf(
    *,
    ep_id: str,
    run_id: str,
) -> tuple[bytes, str]:
    """Build a Screen Time Run Debug Report PDF.

    Returns:
        (pdf_bytes, download_filename)
    """
    run_id_norm = run_layout.normalize_run_id(run_id)
    run_root = run_layout.run_root(ep_id, run_id_norm)
    if not run_root.exists():
        raise FileNotFoundError(f"Run not found on disk: {run_root}")

    manifests_root = get_path(ep_id, "detections").parent
    body_tracking_dir = run_root / "body_tracking"
    # Fall back to episode-level body_tracking if run-scoped doesn't exist
    if not body_tracking_dir.exists():
        body_tracking_dir = manifests_root / "body_tracking"

    # Load artifact paths
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

    # Load JSON artifact data
    identities_data = _read_json(identities_path) or {}
    track_metrics_data = _read_json(track_metrics_path) or {}
    track_fusion_data = _read_json(track_fusion_path) or {}
    screentime_data = _read_json(screentime_comparison_path) or {}

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
    identity_locks: list[dict[str, Any]] = []
    suggestion_batches: list[dict[str, Any]] = []
    suggestions_rows: list[dict[str, Any]] = []
    suggestion_applies: list[dict[str, Any]] = []
    try:
        from apps.api.services.run_persistence import run_persistence_service

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
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
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

    # =========================================================================
    # RUN HEALTH (Quick Sanity Check)
    # =========================================================================
    # Compute effective body_tracking status - check if artifacts exist even if config says disabled
    body_config_enabled = body_detection_config.get("body_tracking", {}).get("enabled", False)
    body_artifacts_exist = body_detections_path.exists() and _count_jsonl_lines(body_detections_path) > 0
    body_enabled_effective = body_config_enabled or body_artifacts_exist

    # Get fused pairs count (identities with both face AND body tracks)
    fusion_identities = track_fusion_data.get("identities", {})
    actual_fused_pairs = 0
    if isinstance(fusion_identities, dict):
        for identity_data in fusion_identities.values():
            if isinstance(identity_data, dict):
                face_tids = identity_data.get("face_track_ids", [])
                body_tids = identity_data.get("body_track_ids", [])
                if face_tids and body_tids:
                    actual_fused_pairs += 1

    # Get screentime values
    screentime_summary = screentime_data.get("summary", {})
    face_only_duration = screentime_summary.get("total_face_only_duration", 0)
    db_connected = db_error is None

    # Health status helper
    def _health_status(ok: bool) -> str:
        return "✓ Yes" if ok else "✗ No"

    health_data = [
        ["Health Check", "Status", "Details"],
        ["DB Connected", _health_status(db_connected),
         "OK" if db_connected else f"Error: {db_error[:50]}..." if db_error and len(db_error) > 50 else (db_error or "Unknown")],
        ["Body Tracking Ran", _health_status(body_enabled_effective),
         f"Config: {'enabled' if body_config_enabled else 'disabled'} | Artifacts: {'present' if body_artifacts_exist else 'missing'}"],
        ["Face-Body Pairs Fused", _health_status(actual_fused_pairs > 0),
         f"{actual_fused_pairs} pairs" if actual_fused_pairs > 0 else "No fusion occurred"],
        ["Face Duration Tracked", _health_status(face_only_duration > 0),
         f"{face_only_duration:.1f}s" if face_only_duration > 0 else "No face time recorded"],
    ]
    health_table = Table(health_data, colWidths=[1.8 * inch, 0.8 * inch, 3.9 * inch])
    health_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a5568")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            # Color code the status column
            ("TEXTCOLOR", (1, 1), (1, 1), colors.green if db_connected else colors.red),
            ("TEXTCOLOR", (1, 2), (1, 2), colors.green if body_enabled_effective else colors.red),
            ("TEXTCOLOR", (1, 3), (1, 3), colors.green if actual_fused_pairs > 0 else colors.red),
            ("TEXTCOLOR", (1, 4), (1, 4), colors.green if face_only_duration > 0 else colors.red),
        ])
    )
    story.append(Paragraph("Run Health", subsection_style))
    story.append(health_table)

    # Add warning if config/artifact mismatch for body tracking
    if body_artifacts_exist and not body_config_enabled:
        story.append(Paragraph(
            "⚠️ <b>Config Mismatch:</b> body_tracking.enabled=False in config but body artifacts exist. "
            "Artifacts may be from a previous run or the config was overridden at runtime.",
            warning_style
        ))

    story.append(Spacer(1, 12))

    # Executive summary stats
    identities_list = identities_data.get("identities", [])
    cluster_stats = identities_data.get("stats", {})
    # Note: screentime_summary already defined above
    metrics = track_metrics_data.get("metrics", {})

    # Calculate track counts
    num_face_tracks = track_fusion_data.get("num_face_tracks") or _count_jsonl_lines(tracks_path)
    num_body_tracks = track_fusion_data.get("num_body_tracks") or _count_jsonl_lines(body_tracks_path)
    num_fused = track_fusion_data.get("num_fused_identities", 0)

    exec_stats = [
        ["Metric", "Value"],
        ["Total Face Tracks", str(num_face_tracks)],
        ["Total Body Tracks", str(num_body_tracks)],
        ["Face Clusters (identities.json)", str(len(identities_list))],
        ["Total Tracked IDs (face + body)", str(num_fused)],
        ["Screen Time Gain", f"{screentime_summary.get('total_duration_gain', 0):.2f}s"],
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

    # Clarification note for "Total Tracked IDs"
    story.append(Paragraph(
        "<b>Note:</b> 'Total Tracked IDs' is the union of face_tracks + body_tracks after fusion, "
        "NOT the number of actually fused face-body pairs. To see actual fused pairs, check the "
        "'identities' dict in track_fusion.json.",
        note_style
    ))

    # =========================================================================
    # SECTION 0: RUN INPUTS & LINEAGE
    # =========================================================================
    story.append(Paragraph("0. Run Inputs &amp; Lineage", section_style))

    lineage_data = [
        ["Input", "Value"],
        ["Episode ID", ep_id],
        ["Run ID", run_id_norm],
        ["Git SHA", _get_git_sha()],
        ["Generated At", _now_iso()],
    ]

    # Try to get video metadata if available
    video_meta_path = manifests_root / "video_metadata.json"
    video_meta = _read_json(video_meta_path) or {}
    if video_meta:
        lineage_data.extend([
            ["Video Duration", f"{video_meta.get('duration', 'N/A')}s"],
            ["Frame Rate", f"{video_meta.get('fps', 'N/A')} fps"],
            ["Resolution", f"{video_meta.get('width', '?')}x{video_meta.get('height', '?')}"],
        ])

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
    detect_count = _count_jsonl_lines(detections_path)
    story.append(Paragraph(f"Total face detections: <b>{detect_count}</b>", body_style))

    # Configuration used
    story.append(Paragraph("Configuration (detection.yaml):", subsection_style))
    detect_config_rows = [
        ["Setting", "Value"],
        ["model_id", detection_config.get("model_id", "N/A")],
        ["confidence_th", str(detection_config.get("confidence_th", "N/A"))],
        ["min_size", str(detection_config.get("min_size", "N/A"))],
        ["iou_th", str(detection_config.get("iou_th", "N/A"))],
        ["wide_shot_mode", str(detection_config.get("wide_shot_mode", "N/A"))],
        ["wide_shot_confidence_th", str(detection_config.get("wide_shot_confidence_th", "N/A"))],
        ["enable_person_fallback", str(detection_config.get("enable_person_fallback", "N/A"))],
    ]
    detect_config_table = Table(detect_config_rows, colWidths=[2.5 * inch, 2 * inch])
    detect_config_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 1), (-1, -1), "Courier"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(detect_config_table)

    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(Paragraph(f"&bull; detections.jsonl ({_file_size_str(detections_path)}) - {detect_count} records", bullet_style))

    # =========================================================================
    # SECTION 2: FACE TRACK
    # =========================================================================
    story.append(Paragraph("2. Face Track", section_style))
    track_count = _count_jsonl_lines(tracks_path)
    story.append(Paragraph(f"Total face tracks: <b>{track_count}</b>", body_style))

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

    # Configuration used
    story.append(Paragraph("Configuration (tracking.yaml):", subsection_style))
    track_config_rows = [
        ["Setting", "Value"],
        ["track_thresh", str(tracking_config.get("track_thresh", "N/A"))],
        ["match_thresh", str(tracking_config.get("match_thresh", "N/A"))],
        ["track_buffer", str(tracking_config.get("track_buffer", "N/A"))],
        ["new_track_thresh", str(tracking_config.get("new_track_thresh", "N/A"))],
        ["gate_enabled", str(tracking_config.get("gate_enabled", "N/A"))],
    ]
    track_config_table = Table(track_config_rows, colWidths=[2.5 * inch, 2 * inch])
    track_config_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 1), (-1, -1), "Courier"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(track_config_table)

    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(Paragraph(f"&bull; tracks.jsonl ({_file_size_str(tracks_path)}) - {track_count} records", bullet_style))
    story.append(Paragraph(f"&bull; track_metrics.json ({_file_size_str(track_metrics_path)})", bullet_style))

    # =========================================================================
    # SECTION 3: FACE HARVEST / EMBED
    # =========================================================================
    story.append(Paragraph("3. Face Harvest / Embed", section_style))
    faces_count = _count_jsonl_lines(faces_path)
    aligned_count = _count_jsonl_lines(face_alignment_path)

    story.append(Paragraph(f"Harvested faces: <b>{faces_count}</b>", body_style))
    story.append(Paragraph(f"Aligned faces: <b>{aligned_count}</b>", body_style))

    # Diagnostic for alignment drop
    if faces_count > 0 and aligned_count > 0:
        alignment_rate = aligned_count / faces_count * 100
        if alignment_rate < 70:
            story.append(Paragraph(
                f"⚠️ Low alignment rate ({alignment_rate:.1f}%): Many faces rejected by quality gating. "
                "Consider lowering face_alignment.min_alignment_quality in embedding.yaml.",
                warning_style
            ))

    # Configuration used
    story.append(Paragraph("Configuration (embedding.yaml):", subsection_style))
    emb_cfg = embedding_config.get("embedding", {})
    face_align_cfg = embedding_config.get("face_alignment", {})
    embed_config_rows = [
        ["Setting", "Value"],
        ["backend", str(emb_cfg.get("backend", "N/A"))],
        ["face_alignment.enabled", str(face_align_cfg.get("enabled", "N/A"))],
        ["face_alignment.min_alignment_quality", str(face_align_cfg.get("min_alignment_quality", "N/A"))],
        ["output.embedding_dim", str(embedding_config.get("output", {}).get("embedding_dim", "512"))],
    ]
    embed_config_table = Table(embed_config_rows, colWidths=[2.5 * inch, 2 * inch])
    embed_config_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 1), (-1, -1), "Courier"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(embed_config_table)

    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(Paragraph(f"&bull; faces.jsonl ({_file_size_str(faces_path)}) - {faces_count} records", bullet_style))
    story.append(Paragraph(f"&bull; face_alignment/aligned_faces.jsonl ({_file_size_str(face_alignment_path)}) - {aligned_count} records", bullet_style))
    story.append(Paragraph(f"&bull; faces.npy ({_file_size_str(faces_npy)})", bullet_style))

    # =========================================================================
    # SECTION 4: BODY DETECT
    # =========================================================================
    story.append(Paragraph("4. Body Detect", section_style))
    body_detect_count = _count_jsonl_lines(body_detections_path)
    story.append(Paragraph(f"Total body detections: <b>{body_detect_count}</b>", body_style))

    # Configuration used
    story.append(Paragraph("Configuration (body_detection.yaml):", subsection_style))
    person_det_cfg = body_detection_config.get("person_detection", {})

    # Show effective body_tracking status with explanation
    if body_artifacts_exist and not body_config_enabled:
        body_effective_str = "True (effective) - artifacts exist, config=False"
    elif body_config_enabled:
        body_effective_str = "True (config enabled)"
    else:
        body_effective_str = "False"

    body_detect_config_rows = [
        ["Setting", "Value"],
        ["body_tracking.enabled", body_effective_str],
        ["model", str(person_det_cfg.get("model", "N/A"))],
        ["confidence_threshold", str(person_det_cfg.get("confidence_threshold", "N/A"))],
        ["min_height_px", str(person_det_cfg.get("min_height_px", "N/A"))],
        ["detect_every_n_frames", str(person_det_cfg.get("detect_every_n_frames", "N/A"))],
    ]
    body_detect_config_table = Table(body_detect_config_rows, colWidths=[2.5 * inch, 2 * inch])
    body_detect_config_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 1), (-1, -1), "Courier"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(body_detect_config_table)

    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(Paragraph(f"&bull; body_tracking/body_detections.jsonl ({_file_size_str(body_detections_path)}) - {body_detect_count} records", bullet_style))

    # =========================================================================
    # SECTION 5: BODY TRACK
    # =========================================================================
    story.append(Paragraph("5. Body Track", section_style))
    body_track_count = _count_jsonl_lines(body_tracks_path)
    story.append(Paragraph(f"Total body tracks: <b>{body_track_count}</b>", body_style))

    # Configuration used
    story.append(Paragraph("Configuration (body_detection.yaml → person_tracking):", subsection_style))
    person_track_cfg = body_detection_config.get("person_tracking", {})
    body_track_config_rows = [
        ["Setting", "Value"],
        ["tracker", str(person_track_cfg.get("tracker", "N/A"))],
        ["track_thresh", str(person_track_cfg.get("track_thresh", "N/A"))],
        ["new_track_thresh", str(person_track_cfg.get("new_track_thresh", "N/A"))],
        ["match_thresh", str(person_track_cfg.get("match_thresh", "N/A"))],
        ["track_buffer", str(person_track_cfg.get("track_buffer", "N/A"))],
        ["id_offset", str(person_track_cfg.get("id_offset", "N/A"))],
    ]
    body_track_config_table = Table(body_track_config_rows, colWidths=[2.5 * inch, 2 * inch])
    body_track_config_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 1), (-1, -1), "Courier"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(body_track_config_table)

    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(Paragraph(f"&bull; body_tracking/body_tracks.jsonl ({_file_size_str(body_tracks_path)}) - {body_track_count} records", bullet_style))

    # =========================================================================
    # SECTION 6: TRACK FUSION
    # =========================================================================
    story.append(Paragraph("6. Track Fusion", section_style))

    # Get actual fused pair count from identities dict
    fusion_identities = track_fusion_data.get("identities", {})
    actual_fused_pairs = 0
    if isinstance(fusion_identities, dict):
        for identity_data in fusion_identities.values():
            if isinstance(identity_data, dict):
                face_tids = identity_data.get("face_track_ids", [])
                body_tids = identity_data.get("body_track_ids", [])
                if face_tids and body_tids:
                    actual_fused_pairs += 1

    fusion_stats = [
        ["Metric", "Value", "Notes"],
        ["Face Tracks (input)", str(track_fusion_data.get("num_face_tracks", 0)), "From tracks.jsonl"],
        ["Body Tracks (input)", str(track_fusion_data.get("num_body_tracks", 0)), "From body_tracks.jsonl"],
        ["Total Tracked IDs", str(num_fused), "Union of face + body (NOT fused pairs)"],
        ["Actual Fused Pairs", str(actual_fused_pairs), "Identities with both face AND body tracks"],
    ]
    fusion_table = Table(fusion_stats, colWidths=[2 * inch, 1.2 * inch, 2.3 * inch])
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

    # Configuration used
    story.append(Paragraph("Configuration (track_fusion.yaml):", subsection_style))
    iou_cfg = track_fusion_config.get("iou_association", {})
    reid_cfg = track_fusion_config.get("reid_handoff", {})
    fusion_config_rows = [
        ["Setting", "Value"],
        ["track_fusion.enabled", str(track_fusion_config.get("track_fusion", {}).get("enabled", "N/A"))],
        ["iou_association.iou_threshold", str(iou_cfg.get("iou_threshold", "N/A"))],
        ["iou_association.min_overlap_ratio", str(iou_cfg.get("min_overlap_ratio", "N/A"))],
        ["reid_handoff.enabled", str(reid_cfg.get("enabled", "N/A"))],
        ["reid_handoff.similarity_threshold", str(reid_cfg.get("similarity_threshold", "N/A"))],
    ]
    fusion_config_table = Table(fusion_config_rows, colWidths=[2.5 * inch, 2 * inch])
    fusion_config_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 1), (-1, -1), "Courier"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
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

    # Calculate singleton stats
    singleton_count = sum(1 for i in identities_list if len(i.get("track_ids", [])) == 1)
    singleton_frac = singleton_count / len(identities_list) if identities_list else 0

    cluster_table_data = [
        ["Metric", "Value", "Notes"],
        ["Clusters (identities)", str(len(identities_list)), "Unique identity groups"],
        ["Total Faces in Clusters", str(cluster_stats.get("faces", 0)), "Sum of face samples"],
        ["Singleton Clusters", str(singleton_count), f"{singleton_frac:.1%} of total"],
        ["Mixed Tracks", str(cluster_stats.get("mixed_tracks", 0)), "Tracks with multiple people (error)"],
        ["Outlier Tracks", str(cluster_stats.get("outlier_tracks", 0)), "Rejected from clusters"],
        ["Low Cohesion", str(cluster_stats.get("low_cohesion_identities", 0)), "Clusters with poor internal similarity"],
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
    if singleton_frac > 0.5:
        story.append(Paragraph(
            f"⚠️ High singleton fraction ({singleton_frac:.1%}): Over half of clusters have only 1 track. "
            "Consider lowering cluster_thresh in clustering.yaml (currently "
            f"{clustering_config.get('cluster_thresh', 'N/A')}) to merge more aggressively.",
            warning_style
        ))
    mixed_tracks = cluster_stats.get("mixed_tracks", 0)
    if isinstance(mixed_tracks, (int, float)) and mixed_tracks > 5:
        story.append(Paragraph(
            f"⚠️ High mixed tracks ({mixed_tracks}): Some clusters contain tracks from different people. "
            "Consider raising cluster_thresh or increasing min_identity_sim.",
            warning_style
        ))

    # Configuration used
    story.append(Paragraph("Configuration (clustering.yaml):", subsection_style))
    singleton_merge_cfg = clustering_config.get("singleton_merge", {})
    cluster_config_rows = [
        ["Setting", "Value"],
        ["Algorithm", "Agglomerative Clustering"],
        ["Distance Metric", "Cosine (1 - similarity)"],
        ["cluster_thresh", str(clustering_config.get("cluster_thresh", "N/A"))],
        ["min_cluster_size", str(clustering_config.get("min_cluster_size", "N/A"))],
        ["min_identity_sim", str(clustering_config.get("min_identity_sim", "N/A"))],
        ["singleton_merge.enabled", str(singleton_merge_cfg.get("enabled", "N/A"))],
        ["singleton_merge.similarity_thresh", str(singleton_merge_cfg.get("similarity_thresh", "N/A"))],
    ]
    cluster_config_table = Table(cluster_config_rows, colWidths=[2.5 * inch, 2 * inch])
    cluster_config_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 1), (-1, -1), "Courier"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(cluster_config_table)

    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(Paragraph(f"&bull; identities.json ({_file_size_str(identities_path)}) - {len(identities_list)} identities", bullet_style))
    story.append(Paragraph(f"&bull; cluster_centroids.json ({_file_size_str(cluster_centroids_path)})", bullet_style))

    # =========================================================================
    # SECTION 8: FACES REVIEW (DB State)
    # =========================================================================
    story.append(Paragraph("8. Faces Review (DB State)", section_style))
    assigned_count = sum(1 for i in identities_list if i.get("person_id"))

    # Show "unavailable" for DB-sourced data when DB is not connected
    if db_error:
        locked_count_str = "unavailable (DB error)"
    else:
        locked_count = len([lock for lock in identity_locks if lock.get("locked")])
        locked_count_str = str(locked_count)

    review_stats = [
        ["Metric", "Value"],
        ["Total Identities", str(len(identities_list))],
        ["Assigned to Cast", str(assigned_count)],
        ["Locked Identities", locked_count_str],
        ["Unassigned", str(len(identities_list) - assigned_count)],
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
            "<b>Impact:</b> Identity locks and suggestion history may be incomplete. "
            "DB-sourced counts (locks, batches, suggestions) should be treated as approximations.",
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

    # Screen time breakdown table
    face_only_duration = screentime_summary.get("total_face_only_duration", 0)
    combined_duration = screentime_summary.get("total_combined_duration", 0)
    duration_gain = screentime_summary.get("total_duration_gain", 0)
    body_only_duration = combined_duration - face_only_duration if combined_duration > face_only_duration else 0

    screentime_breakdown = [
        ["Source", "Duration", "% of Combined"],
        ["Face-only segments", f"{face_only_duration:.2f}s", f"{(face_only_duration/combined_duration*100) if combined_duration > 0 else 0:.1f}%"],
        ["Body-only segments", f"{body_only_duration:.2f}s", f"{(body_only_duration/combined_duration*100) if combined_duration > 0 else 0:.1f}%"],
        ["Combined Total", f"{combined_duration:.2f}s", "100%"],
        ["Gain from Body Tracking", f"+{duration_gain:.2f}s", f"+{(duration_gain/face_only_duration*100) if face_only_duration > 0 else 0:.1f}%"],
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

    # Adjust wording based on whether fusion actually occurred
    if actual_fused_pairs > 0:
        story.append(Paragraph(
            f"<b>Note:</b> 'Gain from Body Tracking' represents body-only duration gain—additional screen time "
            f"from {actual_fused_pairs} fused face-body pair(s) where faces turned away but bodies remained visible.",
            note_style
        ))
    elif body_enabled_effective and body_track_count > 0:
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
    story.append(Paragraph(f"Total identities analyzed: <b>{screentime_summary.get('total_identities', 0)}</b>", body_style))
    story.append(Paragraph(f"Identities with gain: <b>{screentime_summary.get('identities_with_gain', 0)}</b>", body_style))

    # Configuration used
    story.append(Paragraph("Configuration (screen_time_v2.yaml):", subsection_style))
    preset = screentime_config.get("preset", "bravo_default")
    presets = screentime_config.get("screen_time_presets", {})
    active_preset = presets.get(preset, {})
    screentime_config_rows = [
        ["Setting", "Value"],
        ["Active Preset", preset],
        ["quality_min", str(active_preset.get("quality_min", "N/A"))],
        ["gap_tolerance_s", str(active_preset.get("gap_tolerance_s", "N/A"))],
        ["screen_time_mode", str(active_preset.get("screen_time_mode", "N/A"))],
        ["edge_padding_s", str(active_preset.get("edge_padding_s", "N/A"))],
        ["track_coverage_min", str(active_preset.get("track_coverage_min", "N/A"))],
    ]
    screentime_config_table = Table(screentime_config_rows, colWidths=[2.5 * inch, 2 * inch])
    screentime_config_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 1), (-1, -1), "Courier"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
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
    if detect_count == 0:
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
    if faces_count > 0 and aligned_count > 0:
        alignment_rate = aligned_count / faces_count * 100
        if alignment_rate < 70:
            tuning_suggestions.append((
                "Face Embedding",
                f"Low alignment rate ({alignment_rate:.1f}%)",
                "Lower min_alignment_quality in embedding.yaml (currently "
                f"{embedding_config.get('face_alignment', {}).get('min_alignment_quality', 'N/A')})"
            ))

    # Cluster tuning
    if singleton_frac > 0.5:
        tuning_suggestions.append((
            "Clustering",
            f"High singleton fraction ({singleton_frac:.1%})",
            f"Lower cluster_thresh (currently {clustering_config.get('cluster_thresh', 'N/A')}) "
            "or enable singleton_merge"
        ))
    if isinstance(mixed_tracks, (int, float)) and mixed_tracks > 5:
        tuning_suggestions.append((
            "Clustering",
            f"Mixed tracks ({mixed_tracks})",
            "Raise cluster_thresh or min_identity_sim to separate people better"
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
        tuning_table_data = [["Stage", "Issue", "Suggested Action"]]
        tuning_table_data.extend(tuning_suggestions)
        tuning_table = Table(tuning_table_data, colWidths=[1.3 * inch, 1.7 * inch, 3 * inch])
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

    artifact_data = [
        ["Artifact", "Status", "Size", "Records", "Stage"],
        (*_artifact_row(detections_path), str(detect_count), "Face Detect"),
        (*_artifact_row(tracks_path), str(track_count), "Face Track"),
        (*_artifact_row(track_metrics_path), "-", "Face Track"),
        (*_artifact_row(faces_path), str(faces_count), "Face Harvest"),
        (*_artifact_row(face_alignment_path, "face_alignment/aligned_faces.jsonl"), str(aligned_count), "Face Embed"),
        (*_artifact_row(faces_npy), "-", "Face Embed"),
        (*_artifact_row(identities_path), str(len(identities_list)), "Cluster"),
        (*_artifact_row(cluster_centroids_path), "-", "Cluster"),
        (*_artifact_row(body_detections_path, "body_tracking/body_detections.jsonl"), str(body_detect_count), "Body Detect"),
        (*_artifact_row(body_tracks_path, "body_tracking/body_tracks.jsonl"), str(body_track_count), "Body Track"),
        (*_artifact_row(track_fusion_path, "body_tracking/track_fusion.json"), "-", "Track Fusion"),
        (*_artifact_row(screentime_comparison_path, "body_tracking/screentime_comparison.json"), "-", "Screen Time"),
    ]
    artifact_table = Table(artifact_data, colWidths=[2.8 * inch, 0.7 * inch, 0.7 * inch, 0.6 * inch, 1 * inch])
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

    # Build PDF
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    download_name = f"screenalytics_{ep_id}_{run_id_norm}_debug_report.pdf"
    return pdf_bytes, download_name

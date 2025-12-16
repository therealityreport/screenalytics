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
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
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

    # Load artifact data
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

    # Load JSON files
    identities_data = _read_json(identities_path) or {}
    track_metrics_data = _read_json(track_metrics_path) or {}
    track_fusion_data = _read_json(track_fusion_path) or {}
    screentime_data = _read_json(screentime_comparison_path) or {}

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

    story: list[Any] = []

    # Cover / Executive Summary
    story.append(Paragraph("Screen Time Run Debug Report", title_style))
    story.append(Spacer(1, 12))

    summary_data = [
        ["Episode ID", ep_id],
        ["Run ID", run_id_norm],
        ["Generated At", _now_iso()],
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

    # Executive summary stats
    identities_list = identities_data.get("identities", [])
    cluster_stats = identities_data.get("stats", {})
    screentime_summary = screentime_data.get("summary", {})

    exec_stats = [
        ["Total Face Tracks", str(track_fusion_data.get("num_face_tracks", _count_jsonl_lines(tracks_path)))],
        ["Total Body Tracks", str(track_fusion_data.get("num_body_tracks", _count_jsonl_lines(body_tracks_path)))],
        ["Face Clusters", str(len(identities_list))],
        ["Fused Identities", str(track_fusion_data.get("num_fused_identities", 0))],
        ["Screen Time Gain", f"{screentime_summary.get('total_duration_gain', 0):.2f}s"],
    ]
    exec_table = Table(exec_stats, colWidths=[2 * inch, 2 * inch])
    exec_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7fafc")),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(Paragraph("Executive Summary", subsection_style))
    story.append(exec_table)

    # Section 1: FACE DETECT
    story.append(Paragraph("1. Face Detect", section_style))
    detect_count = _count_jsonl_lines(detections_path)
    story.append(Paragraph(f"Total face detections: <b>{detect_count}</b>", body_style))
    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(Paragraph(f"&bull; detections.jsonl ({_file_size_str(detections_path)})", bullet_style))

    # Section 2: FACE TRACK
    story.append(Paragraph("2. Face Track", section_style))
    metrics = track_metrics_data.get("metrics", {})
    track_count = _count_jsonl_lines(tracks_path)
    story.append(Paragraph(f"Total face tracks: <b>{track_count}</b>", body_style))

    track_stats = [
        ["Tracks Born", str(metrics.get("tracks_born", "N/A"))],
        ["Tracks Lost", str(metrics.get("tracks_lost", "N/A"))],
        ["ID Switches", str(metrics.get("id_switches", "N/A"))],
        ["Forced Splits", str(metrics.get("forced_splits", "N/A"))],
        ["Scene Cuts", str(track_metrics_data.get("scene_cuts", {}).get("count", "N/A"))],
    ]
    track_table = Table(track_stats, colWidths=[2 * inch, 2 * inch])
    track_table.setStyle(
        TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(track_table)
    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(Paragraph(f"&bull; tracks.jsonl ({_file_size_str(tracks_path)})", bullet_style))
    story.append(Paragraph(f"&bull; track_metrics.json ({_file_size_str(track_metrics_path)})", bullet_style))

    # Section 3: FACE HARVEST / EMBED
    story.append(Paragraph("3. Face Harvest / Embed", section_style))
    faces_count = _count_jsonl_lines(faces_path)
    aligned_count = _count_jsonl_lines(face_alignment_path)
    faces_npy = (
        Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
        / "embeds"
        / ep_id
        / "runs"
        / run_id_norm
        / "faces.npy"
    )
    story.append(Paragraph(f"Harvested faces: <b>{faces_count}</b>", body_style))
    story.append(Paragraph(f"Aligned faces: <b>{aligned_count}</b>", body_style))
    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(Paragraph(f"&bull; faces.jsonl ({_file_size_str(faces_path)})", bullet_style))
    story.append(Paragraph(f"&bull; face_alignment/aligned_faces.jsonl ({_file_size_str(face_alignment_path)})", bullet_style))
    story.append(Paragraph(f"&bull; faces.npy ({_file_size_str(faces_npy)})", bullet_style))

    # Section 4: BODY DETECT
    story.append(Paragraph("4. Body Detect", section_style))
    body_detect_count = _count_jsonl_lines(body_detections_path)
    story.append(Paragraph(f"Total body detections: <b>{body_detect_count}</b>", body_style))
    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(Paragraph(f"&bull; body_tracking/body_detections.jsonl ({_file_size_str(body_detections_path)})", bullet_style))

    # Section 5: BODY TRACK
    story.append(Paragraph("5. Body Track", section_style))
    body_track_count = _count_jsonl_lines(body_tracks_path)
    story.append(Paragraph(f"Total body tracks: <b>{body_track_count}</b>", body_style))
    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(Paragraph(f"&bull; body_tracking/body_tracks.jsonl ({_file_size_str(body_tracks_path)})", bullet_style))

    # Section 6: TRACK FUSION
    story.append(Paragraph("6. Track Fusion", section_style))
    fusion_stats = [
        ["Face Tracks", str(track_fusion_data.get("num_face_tracks", 0))],
        ["Body Tracks", str(track_fusion_data.get("num_body_tracks", 0))],
        ["Fused Identities", str(track_fusion_data.get("num_fused_identities", 0))],
    ]
    fusion_table = Table(fusion_stats, colWidths=[2 * inch, 2 * inch])
    fusion_table.setStyle(
        TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(fusion_table)
    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(Paragraph(f"&bull; body_tracking/track_fusion.json ({_file_size_str(track_fusion_path)})", bullet_style))

    # Section 7: CLUSTER
    story.append(Paragraph("7. Cluster", section_style))
    cluster_table_data = [
        ["Clusters", str(len(identities_list))],
        ["Total Faces", str(cluster_stats.get("faces", 0))],
        ["Mixed Tracks", str(cluster_stats.get("mixed_tracks", 0))],
        ["Outlier Tracks", str(cluster_stats.get("outlier_tracks", 0))],
        ["Low Cohesion", str(cluster_stats.get("low_cohesion_identities", 0))],
    ]
    cluster_table = Table(cluster_table_data, colWidths=[2 * inch, 2 * inch])
    cluster_table.setStyle(
        TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(cluster_table)
    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(Paragraph(f"&bull; identities.json ({_file_size_str(identities_path)})", bullet_style))
    cluster_centroids_path = run_root / "cluster_centroids.json"
    story.append(Paragraph(f"&bull; cluster_centroids.json ({_file_size_str(cluster_centroids_path)})", bullet_style))

    # Section 8: FACES REVIEW
    story.append(Paragraph("8. Faces Review", section_style))
    assigned_count = sum(1 for i in identities_list if i.get("person_id"))
    locked_count = len([lock for lock in identity_locks if lock.get("locked")])
    story.append(Paragraph(f"Assigned identities: <b>{assigned_count}</b> / {len(identities_list)}", body_style))
    story.append(Paragraph(f"Locked identities: <b>{locked_count}</b>", body_style))
    if db_error:
        story.append(Paragraph(f"<i>DB Error: {db_error}</i>", body_style))
    story.append(Paragraph("Data Sources:", subsection_style))
    story.append(Paragraph("&bull; identity_locks table (DB)", bullet_style))
    story.append(Paragraph("&bull; identities.json (manual_assignments)", bullet_style))

    # Section 9: SMART SUGGESTIONS
    story.append(Paragraph("9. Smart Suggestions", section_style))
    dismissed_count = sum(1 for s in suggestions_rows if s.get("dismissed"))
    applied_count = len(suggestion_applies)
    story.append(Paragraph(f"Suggestion batches: <b>{len(suggestion_batches)}</b>", body_style))
    story.append(Paragraph(f"Total suggestions: <b>{len(suggestions_rows)}</b>", body_style))
    story.append(Paragraph(f"Dismissed: <b>{dismissed_count}</b>", body_style))
    story.append(Paragraph(f"Applied: <b>{applied_count}</b>", body_style))
    story.append(Paragraph("Data Sources:", subsection_style))
    story.append(Paragraph("&bull; suggestion_batches table (DB)", bullet_style))
    story.append(Paragraph("&bull; suggestions table (DB)", bullet_style))
    story.append(Paragraph("&bull; suggestion_applies table (DB)", bullet_style))

    # Section 10: SCREEN TIME ANALYZE
    story.append(Paragraph("10. Screen Time Analyze", section_style))
    story.append(Paragraph(f"Total identities: <b>{screentime_summary.get('total_identities', 0)}</b>", body_style))
    story.append(Paragraph(f"Identities with gain: <b>{screentime_summary.get('identities_with_gain', 0)}</b>", body_style))
    story.append(Paragraph(f"Face-only duration: <b>{screentime_summary.get('total_face_only_duration', 0):.2f}s</b>", body_style))
    story.append(Paragraph(f"Combined duration: <b>{screentime_summary.get('total_combined_duration', 0):.2f}s</b>", body_style))
    story.append(Paragraph(f"Duration gain: <b>{screentime_summary.get('total_duration_gain', 0):.2f}s</b>", body_style))
    story.append(Paragraph("Artifacts:", subsection_style))
    story.append(Paragraph(f"&bull; body_tracking/screentime_comparison.json ({_file_size_str(screentime_comparison_path)})", bullet_style))

    # Appendix: Artifact Manifest
    story.append(Paragraph("Appendix: Artifact Manifest", section_style))
    story.append(Paragraph("Complete listing of all referenced artifacts and their status:", body_style))
    story.append(Spacer(1, 8))

    artifact_data = [
        ["Artifact", "Status", "Size"],
        _artifact_row(detections_path),
        _artifact_row(tracks_path),
        _artifact_row(track_metrics_path),
        _artifact_row(faces_path),
        _artifact_row(face_alignment_path, "face_alignment/aligned_faces.jsonl"),
        _artifact_row(faces_npy),
        _artifact_row(identities_path),
        _artifact_row(cluster_centroids_path),
        _artifact_row(body_detections_path, "body_tracking/body_detections.jsonl"),
        _artifact_row(body_tracks_path, "body_tracking/body_tracks.jsonl"),
        _artifact_row(track_fusion_path, "body_tracking/track_fusion.json"),
        _artifact_row(screentime_comparison_path, "body_tracking/screentime_comparison.json"),
    ]
    artifact_table = Table(artifact_data, colWidths=[3.5 * inch, 1 * inch, 1 * inch])
    artifact_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
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

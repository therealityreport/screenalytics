"""Run debug bundle export service.

Builds a single zip that captures everything needed to reconstruct and debug a
specific (ep_id, run_id) flow end-to-end.

Also provides PDF report generation for Screen Time Run Debug Reports.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from py_screenalytics import run_layout
from py_screenalytics.artifacts import get_path
from py_screenalytics.episode_status import (
    BlockedReason,
    Stage,
    STAGE_PLAN,
    blocked_update_needed,
    collect_git_state,
    normalize_stage_key,
    read_episode_status,
    stage_artifacts,
    update_episode_status,
    write_stage_blocked,
    write_stage_failed,
    write_stage_finished,
    write_stage_started,
)
from py_screenalytics.run_gates import GateReason, check_prereqs
from py_screenalytics.run_manifests import (
    StageBlockedInfo,
    StageErrorInfo,
    load_stage_manifests,
    read_stage_manifest,
    write_stage_manifest,
)
from py_screenalytics.run_logs import append_log, tail_logs

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


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _json_dumps_sorted(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _hash_snapshot(payload: Any) -> str | None:
    if payload is None:
        return None
    digest = hashlib.sha256(_json_dumps_sorted(payload).encode("utf-8"))
    return digest.hexdigest()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _format_iso(value: str | None) -> str:
    if not value:
        return "—"
    return value


def _read_status_payload(ep_id: str, run_id: str) -> dict[str, Any]:
    path = run_layout.run_root(ep_id, run_layout.normalize_run_id(run_id)) / "episode_status.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _stage_order_from_status(status: Any) -> list[str]:
    raw_plan = list(getattr(status, "stage_plan", []) or [])
    ordered: list[str] = []
    for entry in raw_plan:
        if isinstance(entry, str):
            key = entry
        elif hasattr(entry, "value"):
            key = str(entry.value)
        else:
            key = str(entry)
        if key and key not in ordered:
            ordered.append(key)
    if not ordered:
        ordered = list(STAGE_PLAN)
    for stage in getattr(status, "stages", {}).keys():
        key = stage.value if hasattr(stage, "value") else str(stage)
        if key not in ordered:
            ordered.append(key)
    return ordered


def _sanitize_pdf_text(text: str) -> str:
    """Remove characters that render as black squares or replacement glyphs in PDF fonts.

    Helvetica (and other Type1 fonts) cannot render:
    - U+200B (zero-width space) - renders as black square
    - U+25A0 (black square) - should not appear in output
    - U+FFFD (replacement character) - indicates encoding issues
    """
    return (
        text.replace("\u200b", "")
        .replace("\u25a0", "")
        .replace("\ufffd", "")
    )


def _soft_wrap_text(text: str, *, max_token_len: int = 60, chunk_len: int = 24) -> str:
    """Process long tokens to enable PDF line wrapping without overflow.

    ReportLab wraps on whitespace. For long unbroken tokens (paths, S3 keys, etc.),
    we chunk extremely long spans so wordWrap='CJK' can break at reasonable positions.

    Note: Previously inserted U+200B (zero-width space) but Helvetica renders it as
    black squares. Now relies on wordWrap='CJK' style for breaking at any character.
    """
    raw = str(text or "")
    if not raw:
        return ""

    # Identify natural break positions after separators for internal chunking logic.
    separators = r"/_\-\.=:(),"
    ZWSP = "\u200b"  # Internal marker only, stripped before return
    out_chars: list[str] = []
    for ch in raw:
        out_chars.append(ch)
        if ch in separators:
            out_chars.append(ZWSP)
    softened = "".join(out_chars)

    # Fallback: chunk extremely long spans that still have no breaks.
    tokens = softened.split(" ")
    for i, tok in enumerate(tokens):
        if len(tok) <= max_token_len:
            continue
        parts = tok.split(ZWSP)
        rebuilt: list[str] = []
        for part in parts:
            if len(part) <= max_token_len:
                rebuilt.append(part)
                continue
            # Insert markers to allow breaking in very long segments
            rebuilt.append(ZWSP.join(part[j : j + chunk_len] for j in range(0, len(part), chunk_len)))
        tokens[i] = ZWSP.join(rebuilt)
    result = " ".join(tokens)

    # CRITICAL: Strip zero-width spaces - Helvetica renders them as black squares (■).
    # The wordWrap='CJK' paragraph style handles actual line breaking.
    return _sanitize_pdf_text(result)


def _escape_reportlab_xml(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_wrap_safe_kv_table(
    rows: list[tuple[str, str | list[tuple[str, str]] | None]],
    *,
    width: float,
    header: tuple[str, str] = ("Input", "Value"),
    label_ratio: float = 0.40,
    cell_style: Any,
    header_style: Any,
) -> Any:
    """Build a 2-column key/value table with Paragraph cells and wrap-safe text."""
    from reportlab.lib import colors  # imported lazily to keep module import lightweight
    from reportlab.platypus import Paragraph, Table, TableStyle

    def _p(text: str, *, style: Any, allow_markup: bool = False, soft_wrap: bool = True) -> Paragraph:
        if allow_markup:
            # Markup strings are expected to already have soft-wrap opportunities injected into their
            # data segments; do not run soft-wrap across the markup itself (e.g., "<br/>").
            return Paragraph(text, style)
        payload = _soft_wrap_text(text) if soft_wrap else str(text)
        return Paragraph(_escape_reportlab_xml(payload), style)

    def _value_cell(value: str | list[tuple[str, str]] | None) -> Paragraph:
        if value is None:
            return _p("N/A", style=cell_style)
        if isinstance(value, list):
            lines: list[str] = []
            for k, v in value:
                k_safe = _escape_reportlab_xml(_soft_wrap_text(k))
                v_safe = _escape_reportlab_xml(_soft_wrap_text(v))
                lines.append(f"<b>{k_safe}:</b> {v_safe}")
            return _p("<br/>".join(lines), style=cell_style, allow_markup=True)
        return _p(str(value), style=cell_style)

    label_w = max(width * float(label_ratio), 1)
    value_w = max(width - label_w, 1)
    data: list[list[Any]] = [
        [_p(header[0], style=header_style, soft_wrap=False), _p(header[1], style=header_style, soft_wrap=False)]
    ]
    for label, value in rows:
        data.append([_p(str(label), style=cell_style, soft_wrap=True), _value_cell(value)])

    table = Table(data, colWidths=[label_w, value_w])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
            ]
        )
    )
    return table


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
                pdf_bytes = generate_run_debug_pdf(ep_id, run_id_norm).read_bytes()
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


def _get_git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "N/A"
    except Exception:
        pass
    return "N/A"


def _get_git_dirty() -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return "true" if result.stdout.strip() else "false"
    except Exception:
        pass
    return "N/A"


def _format_git_dirty(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "N/A"


def _cpu_model() -> str:
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine() or "N/A"


def _ram_total_gb() -> str:
    try:
        import psutil  # type: ignore

        return f"{psutil.virtual_memory().total / (1024 ** 3):.1f} GB"
    except Exception:
        pass
    try:
        if hasattr(os, "sysconf"):
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            total = float(page_size) * float(pages)
            return f"{total / (1024 ** 3):.1f} GB"
    except Exception:
        pass
    return "N/A"


def _hw_decode_label() -> str:
    env = os.environ.get("SCREENALYTICS_HW_DECODE")
    if env:
        return env
    if sys.platform == "darwin":
        return "videotoolbox (default)"
    return "unknown"


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


def _resolve_lifecycle_status(stage_key: str, stage_entry: dict[str, Any] | None, artifact_state: dict[str, Any]) -> str:
    """Resolve lifecycle status using episode_status first, then artifact presence."""
    stage_entry = stage_entry if isinstance(stage_entry, dict) else {}
    status_val = str(stage_entry.get("status") or "").strip().lower()
    if status_val in {"success", "completed", "done", "error", "failed", "running", "finalizing", "syncing", "skipped"}:
        return status_val

    if stage_key == "detect":
        detections_ok = bool(artifact_state.get("detections"))
        tracks_ok = bool(artifact_state.get("tracks"))
        if detections_ok and tracks_ok:
            return "success"
        if detections_ok or tracks_ok:
            return "partial"
    elif stage_key == "faces":
        if artifact_state.get("faces"):
            return "success"
    elif stage_key == "cluster":
        if artifact_state.get("identities"):
            return "success"
    elif stage_key == "body_tracking":
        if artifact_state.get("body_tracks"):
            return "success"
        if artifact_state.get("legacy"):
            return "success (legacy)"
    elif stage_key == "track_fusion":
        if artifact_state.get("track_fusion"):
            return "success"
        if artifact_state.get("legacy"):
            return "success (legacy)"
    elif stage_key == "pdf":
        if artifact_state.get("export_index"):
            return "success"
    return status_val or "missing"


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


def _build_run_debug_pdf_bytes(*, ep_id: str, run_id: str) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ModuleNotFoundError as exc:
        raise RuntimeError("reportlab is required to build PDF debug reports (pip install reportlab)") from exc

    run_id_norm = run_layout.normalize_run_id(run_id)
    run_root = run_layout.run_root(ep_id, run_id_norm)
    if not run_root.exists():
        raise FileNotFoundError(f"Run not found on disk: {run_root}")

    status_payload = _read_status_payload(ep_id, run_id_norm)
    status = read_episode_status(ep_id, run_id_norm)
    stage_keys = _stage_order_from_status(status)
    manifests = load_stage_manifests(ep_id, run_id_norm, stage_keys)

    styles = getSampleStyleSheet()
    base_style = ParagraphStyle(
        "RunDebugBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=11,
        wordWrap="CJK",
    )
    header_style = ParagraphStyle(
        "RunDebugHeader",
        parent=base_style,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )
    title_style = ParagraphStyle(
        "RunDebugTitle",
        parent=styles["Heading1"],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#1a202c"),
    )
    section_style = ParagraphStyle(
        "RunDebugSection",
        parent=styles["Heading2"],
        fontSize=12,
        leading=14,
        textColor=colors.HexColor("#1a202c"),
    )

    def _fmt_dt(value: datetime | None) -> str:
        if not value:
            return "N/A"
        if value.tzinfo is None:
            return value.replace(microsecond=0).isoformat()
        return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _fmt_val(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, (float, int)):
            return f"{value:.3f}" if isinstance(value, float) else str(value)
        return str(value)

    def _add_section(title: str, rows: list[tuple[str, Any]]) -> None:
        story.append(Paragraph(title, section_style))
        if rows:
            story.append(
                build_wrap_safe_kv_table(
                    rows,
                    width=doc.width,
                    header=("Field", "Value"),
                    cell_style=base_style,
                    header_style=header_style,
                )
            )
        else:
            story.append(Paragraph("No data available.", base_style))
        story.append(Spacer(1, 0.15 * inch))

    story: list[Any] = []
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)

    story.append(Paragraph("Run Debug Report", title_style))
    story.append(Spacer(1, 0.12 * inch))

    started_times = [state.started_at for state in status.stages.values() if state.started_at]
    finished_times = [state.finished_at for state in status.stages.values() if state.finished_at]
    run_started = min(started_times) if started_times else None
    run_finished = max(finished_times) if finished_times else None

    header_rows = [
        ("episode_id", ep_id),
        ("run_id", run_id_norm),
        ("generated_at", _fmt_val(status_payload.get("generated_at"))),
        ("run_started_at", _fmt_dt(run_started)),
        ("run_finished_at", _fmt_dt(run_finished)),
    ]
    _add_section("Run Header", header_rows)

    env_rows: list[tuple[str, Any]] = []
    for key in ("git_sha", "git_branch", "git_dirty"):
        if key in status_payload:
            env_rows.append((key, _fmt_val(status_payload.get(key))))
    env_payload = status_payload.get("env")
    if isinstance(env_payload, dict):
        for key in sorted(env_payload.keys()):
            env_rows.append((key, _fmt_val(env_payload.get(key))))
    storage_payload = status_payload.get("storage")
    if isinstance(storage_payload, dict):
        for key in sorted(storage_payload.keys()):
            env_rows.append((f"storage.{key}", _fmt_val(storage_payload.get(key))))
    _add_section("Environment Summary", env_rows)

    timeline_rows: list[tuple[str, Any]] = []
    for stage_key in stage_keys:
        stage_enum = Stage.from_key(stage_key)
        state = status.stages.get(stage_enum) if stage_enum else None
        status_value = state.status.value if state else "not_started"
        details: list[tuple[str, str]] = [
            ("status", status_value),
            ("started_at", _fmt_dt(state.started_at) if state else "N/A"),
            ("finished_at", _fmt_dt(state.finished_at) if state else "N/A"),
            ("duration_s", _fmt_val(state.duration_s) if state else "N/A"),
            ("derived", "yes" if state and state.derived else "no"),
        ]
        if state and state.blocked_reason:
            details.append(("reason_code", state.blocked_reason.code))
            details.append(("reason_message", state.blocked_reason.message))
        if state and state.derived and state.derived_from:
            details.append(("derived_from", ", ".join(state.derived_from)))
        timeline_rows.append((stage_key, details))
    _add_section("Stage Timeline", timeline_rows)

    provenance_rows: list[tuple[str, Any]] = []
    for stage_key in stage_keys:
        manifest = manifests.get(stage_key)
        if not isinstance(manifest, dict):
            continue
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            continue
        artifact_rows: list[tuple[str, str]] = []
        for entry in artifacts:
            if not isinstance(entry, dict):
                continue
            label = entry.get("logical_name") or entry.get("label")
            path = entry.get("path")
            sha = entry.get("sha256")
            if isinstance(label, str) and isinstance(path, str):
                digest = f" sha256={sha}" if isinstance(sha, str) else ""
                artifact_rows.append((label, f"{path}{digest}"))
        if artifact_rows:
            artifact_rows.sort(key=lambda item: item[0])
            provenance_rows.append((stage_key, artifact_rows))
    _add_section("Provenance (Artifacts + Digests)", provenance_rows)

    config_rows: list[tuple[str, Any]] = []
    for stage_key in stage_keys:
        manifest = manifests.get(stage_key)
        if not isinstance(manifest, dict):
            continue
        model_versions = manifest.get("model_versions")
        thresholds = manifest.get("thresholds")
        if not isinstance(model_versions, dict) and not isinstance(thresholds, dict):
            continue
        details: list[tuple[str, str]] = []
        if isinstance(model_versions, dict):
            for key in sorted(model_versions.keys()):
                details.append((f"model.{key}", _fmt_val(model_versions.get(key))))
        if isinstance(thresholds, dict):
            for key in sorted(thresholds.keys()):
                details.append((f"threshold.{key}", _fmt_val(thresholds.get(key))))
        if details:
            config_rows.append((stage_key, details))
    _add_section("Config Snapshot", config_rows)

    counts_rows: list[tuple[str, Any]] = []
    for stage_key in stage_keys:
        manifest = manifests.get(stage_key)
        if not isinstance(manifest, dict):
            continue
        counts = manifest.get("counts")
        if not isinstance(counts, dict):
            continue
        details: list[tuple[str, str]] = []
        for key in sorted(counts.keys()):
            details.append((key, _fmt_val(counts.get(key))))
        if details:
            counts_rows.append((stage_key, details))
    _add_section("Quality Counters", counts_rows)

    error_rows: list[tuple[str, Any]] = []
    for stage_key in stage_keys:
        stage_enum = Stage.from_key(stage_key)
        state = status.stages.get(stage_enum) if stage_enum else None
        if not state or state.status.value not in {"failed", "blocked"}:
            continue
        details: list[tuple[str, str]] = []
        if state.blocked_reason:
            details.append(("code", state.blocked_reason.code))
            details.append(("message", state.blocked_reason.message))
        manifest = manifests.get(stage_key)
        if isinstance(manifest, dict):
            error = manifest.get("error")
            if isinstance(error, dict):
                code = error.get("error_code")
                message = error.get("error_message")
                if code and ("code", str(code)) not in details:
                    details.append(("manifest_code", _fmt_val(code)))
                if message and ("message", str(message)) not in details:
                    details.append(("manifest_message", _fmt_val(message)))
        logs = tail_logs(ep_id, run_id_norm, stage_key, n=50)
        log_lines: list[str] = []
        for entry in logs:
            if not isinstance(entry, dict):
                continue
            level = str(entry.get("level", "")).upper()
            if level not in {"WARNING", "ERROR"}:
                continue
            ts = entry.get("ts", "")
            msg = entry.get("msg", "")
            log_lines.append(f"{ts} [{level}] {msg}".strip())
        for idx, line in enumerate(log_lines[-5:], start=1):
            details.append((f"log_{idx}", line))
        if details:
            error_rows.append((stage_key, details))
    _add_section("Error + Warning Digest", error_rows)

    doc.build(story)
    return buffer.getvalue()


def generate_run_debug_pdf(
    episode_id: str,
    run_id: str,
    *,
    output_path: str | Path | None = None,
) -> Path:
    run_id_norm = run_layout.normalize_run_id(run_id)
    path = Path(output_path) if output_path else run_layout.run_root(episode_id, run_id_norm) / "exports" / "run_debug.pdf"
    pdf_bytes = _build_run_debug_pdf_bytes(ep_id=episode_id, run_id=run_id_norm)
    _atomic_write_bytes(path, pdf_bytes)
    return path


def build_screentime_run_debug_pdf(
    *,
    ep_id: str,
    run_id: str,
    include_screentime: bool = True,
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

    include_faces_review = bool(include_screentime)
    s3_layout = run_layout.get_run_s3_layout(ep_id, run_id_norm)

    def _safe_float_opt(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None

    def _safe_int_opt(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.isdigit():
                return int(cleaned)
        return None

    def _sanitize_tensorrt_fallback_reason(
        *,
        configured: str | None,
        effective: str | None,
        reason: str | None,
    ) -> str | None:
        if not reason:
            return reason
        if configured != "tensorrt" or not effective or effective == "tensorrt":
            return reason
        lowered = reason.lower()
        if "pip install" in lowered or "pycuda" in lowered:
            return "TensorRT requires CUDA (NVIDIA) and is not available; using PyTorch backend."
        return reason

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
    faces_embed_marker_path = _resolved_path(run_root / "faces_embed.json", "faces_embed.json")
    env_diagnostics_path = _resolved_path(run_root / "env_diagnostics.json", "env_diagnostics.json")
    episode_status_path = _resolved_path(run_root / "episode_status.json", "episode_status.json")

    # Load JSON artifact data
    identities_payload = _read_json(identities_path)
    identities_data = identities_payload if isinstance(identities_payload, dict) else {}
    track_metrics_payload = _read_json(track_metrics_path)
    track_metrics_data = track_metrics_payload if isinstance(track_metrics_payload, dict) else {}
    episode_status_payload = _read_json(episode_status_path) if episode_status_path.exists() else None
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

    # Load DB data (optional: enabled when DB_URL or SCREENALYTICS_FAKE_DB is configured).
    db_connected: bool | None = None
    db_error: str | None = None
    db_not_configured_reason: str | None = None
    run_row: dict[str, Any] | None = None
    job_runs: list[dict[str, Any]] = []
    identity_locks: list[dict[str, Any]] = []
    suggestion_batches: list[dict[str, Any]] = []
    suggestions_rows: list[dict[str, Any]] = []
    suggestion_applies: list[dict[str, Any]] = []

    db_url = (os.getenv("DB_URL") or "").strip()
    fake_db_enabled = (os.getenv("SCREENALYTICS_FAKE_DB") or "").strip() == "1"
    db_configured = fake_db_enabled or bool(db_url)
    if not db_configured:
        db_connected = None
        db_not_configured_reason = "DB_URL is not set"
    else:
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
            db_connected = False
        else:
            db_connected = True

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
    # Ensure ReportLab breaks long tokens (paths, S3 keys) instead of overflowing.
    # Apply to all styles that may contain long unbroken strings.
    for style in [cell_style, cell_style_small, bullet_style, body_style, note_style]:
        style.wordWrap = "CJK"
        style.splitLongWords = 1

    def _wrap_cell(text: str, style: ParagraphStyle = cell_style) -> Paragraph:
        """Wrap text in a Paragraph for table cell text wrapping."""
        safe_text = _escape_reportlab_xml(_soft_wrap_text(str(text)))
        return Paragraph(safe_text, style)

    def _wrap_row(row: list, style: ParagraphStyle = cell_style) -> list:
        """Wrap all cells in a row for text wrapping."""
        return [_wrap_cell(cell, style) for cell in row]

    story: list[Any] = []

    # =========================================================================
    # COVER / EXECUTIVE SUMMARY
    # =========================================================================
    report_title = "Screen Time Run Debug Report" if include_screentime else "Setup Pipeline Run Debug Report"
    story.append(Paragraph(report_title, title_style))
    story.append(Spacer(1, 12))

    git_sha_value = None
    git_branch_value = None
    git_dirty_value: Any = None
    if isinstance(episode_status_payload, dict):
        git_sha_value = episode_status_payload.get("git_sha")
        git_branch_value = episode_status_payload.get("git_branch")
        git_dirty_value = episode_status_payload.get("git_dirty")
    git_sha_value = git_sha_value or _get_git_sha()
    git_branch_value = git_branch_value or _get_git_branch()
    git_dirty_label = _format_git_dirty(git_dirty_value if git_dirty_value is not None else _get_git_dirty())

    summary_data = [
        ["Episode ID", ep_id],
        ["Run ID", run_id_norm],
        ["Generated At", _now_iso()],
        ["Git SHA", git_sha_value],
        ["Git Branch", git_branch_value],
        ["Git Dirty", git_dirty_label],
        ["Run Root", str(run_root)],
        ["S3 Layout (write)", s3_layout.s3_layout],
        ["S3 Run Prefix (write)", s3_layout.write_prefix],
    ]
    summary_label_style = ParagraphStyle(
        "SummaryLabel",
        parent=cell_style,
        fontName="Helvetica-Bold",
    )
    summary_rows = [[_wrap_cell(k, summary_label_style), _wrap_cell(v, cell_style)] for k, v in summary_data]
    summary_table = Table(summary_rows, colWidths=[1.5 * inch, 5 * inch])
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
            key_safe = _escape_reportlab_xml(_soft_wrap_text(key))
            loc_safe = _escape_reportlab_xml(_soft_wrap_text(location))
            story.append(Paragraph(f"&bull; {key_safe} (artifact_location={loc_safe})", bullet_style))
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

    comparison_label = "Screen Time" if include_screentime else "Fusion Comparison"
    comparison_stage_label = "Screen Time" if include_screentime else "Track Fusion"
    gain_label = "Screen Time Gain" if include_screentime else "Fusion Gain"
    duration_gain: float | None = None
    if screentime_summary is not None:
        duration_gain = _safe_float_opt(
            screentime_summary.get("gain_total_s", screentime_summary.get("total_duration_gain"))
        )

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

    if db_connected is True:
        db_details = "OK"
    elif db_connected is None:
        db_details = f"Not configured ({db_not_configured_reason or 'missing DB_URL'})"
    else:
        db_details = f"Error: {db_error[:50]}..." if db_error and len(db_error) > 50 else (db_error or "Unknown")

    health_data = [
        _wrap_row(["Health Check", "Status", "Details"]),
        _wrap_row(["DB Connected", _health_status(db_connected), db_details]),
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
            f"{comparison_label} Present",
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
        [f"{gain_label} (from comparison)", screentime_gain_exec],
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
        ["Git SHA", git_sha_value],
        ["Git Branch", git_branch_value],
        ["Git Dirty", git_dirty_label],
        ["Generated At", _now_iso()],
        ["Artifact Store", storage_display],
        [
            "DB Connected",
            "Yes"
            if db_connected is True
            else (
                f"Not configured ({db_not_configured_reason or 'missing DB_URL'})"
                if db_connected is None
                else f"No ({db_error[:40]}...)" if db_error and len(db_error) > 40 else f"No ({db_error or 'unknown'})"
            ),
        ],
    ]
    env_payload = episode_status_payload.get("env") if isinstance(episode_status_payload, dict) else {}
    torch_device_value = env_payload.get("torch_device") if isinstance(env_payload, dict) else None
    onnx_provider_value = env_payload.get("onnx_provider") if isinstance(env_payload, dict) else None
    host_lines = [
        ("cpu", _cpu_model()),
        ("ram", _ram_total_gb()),
        ("os", platform.platform()),
        ("hw_decode", _hw_decode_label()),
        ("torch_device", torch_device_value or "N/A"),
        ("onnx_provider", onnx_provider_value or "N/A"),
    ]
    lineage_data.append(["Host & Acceleration", host_lines])

    import_status: dict[str, Any] | None = None
    env_diagnostics = _read_json(env_diagnostics_path) if env_diagnostics_path.exists() else None
    if isinstance(env_diagnostics, dict):
        env_lines: list[tuple[str, str]] = []
        python_version = env_diagnostics.get("python_version")
        if isinstance(python_version, str) and python_version.strip():
            env_lines.append(("python_version", python_version.strip()))
        pip_version = env_diagnostics.get("pip_version")
        if isinstance(pip_version, str) and pip_version.strip():
            env_lines.append(("pip_version", pip_version.strip()))
        venv_active = env_diagnostics.get("venv_active")
        if isinstance(venv_active, bool):
            env_lines.append(("venv_active", "true" if venv_active else "false"))
        sys_executable = env_diagnostics.get("sys_executable")
        if isinstance(sys_executable, str) and sys_executable.strip():
            env_lines.append(("sys.executable", sys_executable.strip()))

        import_status = env_diagnostics.get("import_status") if isinstance(env_diagnostics.get("import_status"), dict) else None

        def _fmt_dep(name: str) -> str:
            if not isinstance(import_status, dict):
                return "unknown"
            entry = import_status.get(name)
            if not isinstance(entry, dict):
                return "unknown"
            status = entry.get("status")
            version = entry.get("version")
            error = entry.get("error")
            status_str = status if isinstance(status, str) and status.strip() else "unknown"
            if isinstance(version, str) and version.strip():
                return f"{status_str} ({version.strip()})"
            if isinstance(error, str) and error.strip() and status_str != "ok":
                trimmed = error.strip()
                if len(trimmed) > 120:
                    trimmed = trimmed[:117] + "..."
                return f"{status_str}: {trimmed}"
            return status_str

        env_lines.append(("supervision", _fmt_dep("supervision")))
        env_lines.append(("torchreid", _fmt_dep("torchreid")))
        lineage_data.append(["Environment (run-scoped preflight)", env_lines])
    else:
        lineage_data.append(["Environment (run-scoped preflight)", _na_artifact(env_diagnostics_path, "env_diagnostics.json")])

    # Video metadata sources (split, labeled, and validated for mismatches).
    video_path = get_path(ep_id, "video")
    ffprobe_meta = _ffprobe_video_metadata(video_path)
    opencv_meta = _opencv_video_metadata(video_path)
    detect_track_marker = _read_json(detect_track_marker_path) if detect_track_marker_path.exists() else None
    faces_embed_marker = _read_json(faces_embed_marker_path) if faces_embed_marker_path.exists() else None

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
    marker_frames_scanned_total: int | None = None
    marker_face_detect_frames_processed: int | None = None
    marker_face_detect_frames_processed_stride: int | None = None
    marker_face_detect_frames_processed_forced_scene_warmup: int | None = None
    marker_stride_effective: int | None = None
    marker_stride_observed_median: int | None = None
    marker_expected_frames_by_stride: int | None = None
    marker_detect_wall_time_s: float | None = None
    marker_effective_fps_processing: float | None = None
    marker_rtf: float | None = None
    marker_scene_detect_wall_time_s: float | None = None
    marker_scene_warmup_dets: int | None = None
    marker_tracker_backend_configured: str | None = None
    marker_tracker_backend_actual: str | None = None
    marker_tracker_fallback_reason: str | None = None
    marker_detect_device: str | None = None
    marker_requested_detect_device: str | None = None
    marker_resolved_detect_device: str | None = None
    marker_onnx_provider_requested: str | None = None
    marker_onnx_provider_resolved: str | None = None
    marker_torch_device_requested: str | None = None
    marker_torch_device_resolved: str | None = None
    marker_torch_device_fallback_reason: str | None = None
    marker_yolo_fallback_enabled: bool | None = None
    marker_yolo_fallback_device: str | None = None
    marker_yolo_fallback_load_status: str | None = None
    marker_yolo_fallback_disabled_reason: str | None = None
    marker_yolo_fallback_invocations: int | None = None
    marker_yolo_fallback_detections_added: int | None = None
    marker_face_detector_model: str | None = None
    marker_embed_requested_device: str | None = None
    marker_embed_resolved_device: str | None = None
    marker_embedding_backend_configured: str | None = None
    marker_embedding_backend_configured_effective: str | None = None
    marker_embedding_backend_actual: str | None = None
    marker_embedding_backend_fallback_reason: str | None = None
    marker_embedding_model_name: str | None = None
    marker_scene_cut_count: int | None = None
    marker_warmup_frames_per_cut_effective: float | None = None
    marker_forced_scene_warmup_ratio: float | None = None
    marker_wall_time_per_processed_frame_s: float | None = None
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

        marker_frames_scanned_total = _safe_int_opt(detect_track_marker.get("frames_scanned_total"))
        marker_face_detect_frames_processed = _safe_int_opt(detect_track_marker.get("face_detect_frames_processed"))
        marker_face_detect_frames_processed_stride = _safe_int_opt(
            detect_track_marker.get("face_detect_frames_processed_stride")
        )
        marker_face_detect_frames_processed_forced_scene_warmup = _safe_int_opt(
            detect_track_marker.get("face_detect_frames_processed_forced_scene_warmup")
        )
        marker_stride_effective = _safe_int_opt(detect_track_marker.get("stride_effective"))
        marker_stride_observed_median = _safe_int_opt(detect_track_marker.get("stride_observed_median"))
        marker_expected_frames_by_stride = _safe_int_opt(detect_track_marker.get("expected_frames_by_stride"))
        marker_detect_wall_time_s = _safe_float_opt(detect_track_marker.get("detect_wall_time_s"))
        marker_effective_fps_processing = _safe_float_opt(detect_track_marker.get("effective_fps_processing"))
        marker_rtf = _safe_float_opt(detect_track_marker.get("rtf"))
        marker_scene_detect_wall_time_s = _safe_float_opt(detect_track_marker.get("scene_detect_wall_time_s"))
        marker_scene_warmup_dets = _safe_int_opt(detect_track_marker.get("scene_warmup_dets"))
        marker_tracker_backend_configured = (
            str(detect_track_marker.get("tracker_backend_configured")).strip()
            if isinstance(detect_track_marker.get("tracker_backend_configured"), str)
            else None
        )
        marker_tracker_backend_actual = (
            str(detect_track_marker.get("tracker_backend_actual")).strip()
            if isinstance(detect_track_marker.get("tracker_backend_actual"), str)
            else None
        )
        marker_tracker_fallback_reason = (
            str(detect_track_marker.get("tracker_fallback_reason")).strip()
            if isinstance(detect_track_marker.get("tracker_fallback_reason"), str)
            else None
        )
        marker_detect_device = (
            str(detect_track_marker.get("device")).strip()
            if isinstance(detect_track_marker.get("device"), str)
            else None
        )
        marker_requested_detect_device = (
            str(detect_track_marker.get("requested_device")).strip()
            if isinstance(detect_track_marker.get("requested_device"), str)
            else None
        )
        marker_resolved_detect_device = (
            str(detect_track_marker.get("resolved_device")).strip()
            if isinstance(detect_track_marker.get("resolved_device"), str)
            else None
        )
        marker_onnx_provider_requested = (
            str(detect_track_marker.get("onnx_provider_requested")).strip()
            if isinstance(detect_track_marker.get("onnx_provider_requested"), str)
            else None
        )
        marker_onnx_provider_resolved = (
            str(detect_track_marker.get("onnx_provider_resolved")).strip()
            if isinstance(detect_track_marker.get("onnx_provider_resolved"), str)
            else None
        )
        marker_torch_device_requested = (
            str(detect_track_marker.get("torch_device_requested")).strip()
            if isinstance(detect_track_marker.get("torch_device_requested"), str)
            else None
        )
        marker_torch_device_resolved = (
            str(detect_track_marker.get("torch_device_resolved")).strip()
            if isinstance(detect_track_marker.get("torch_device_resolved"), str)
            else None
        )
        marker_torch_device_fallback_reason = (
            str(detect_track_marker.get("torch_device_fallback_reason")).strip()
            if isinstance(detect_track_marker.get("torch_device_fallback_reason"), str)
            else None
        )
        yolo_enabled_raw = detect_track_marker.get("yolo_fallback_enabled")
        if isinstance(yolo_enabled_raw, bool):
            marker_yolo_fallback_enabled = yolo_enabled_raw
        marker_yolo_fallback_device = (
            str(detect_track_marker.get("yolo_fallback_device")).strip()
            if isinstance(detect_track_marker.get("yolo_fallback_device"), str)
            else None
        )
        marker_yolo_fallback_load_status = (
            str(detect_track_marker.get("yolo_fallback_load_status")).strip()
            if isinstance(detect_track_marker.get("yolo_fallback_load_status"), str)
            else None
        )
        marker_yolo_fallback_disabled_reason = (
            str(detect_track_marker.get("yolo_fallback_disabled_reason")).strip()
            if isinstance(detect_track_marker.get("yolo_fallback_disabled_reason"), str)
            else None
        )
        marker_yolo_fallback_invocations = _safe_int_opt(detect_track_marker.get("yolo_fallback_invocations"))
        marker_yolo_fallback_detections_added = _safe_int_opt(
            detect_track_marker.get("yolo_fallback_detections_added")
        )
        marker_scene_cut_count = _safe_int_opt(detect_track_marker.get("scene_cut_count"))
        marker_warmup_frames_per_cut_effective = _safe_float_opt(
            detect_track_marker.get("warmup_frames_per_cut_effective")
        )
        marker_forced_scene_warmup_ratio = _safe_float_opt(detect_track_marker.get("forced_scene_warmup_ratio"))
        marker_wall_time_per_processed_frame_s = _safe_float_opt(
            detect_track_marker.get("wall_time_per_processed_frame_s")
        )
        marker_face_detector_model = (
            str(detect_track_marker.get("detector_model_name")).strip()
            if isinstance(detect_track_marker.get("detector_model_name"), str)
            else None
        )
    if isinstance(faces_embed_marker, dict):
        marker_embed_requested_device = (
            str(faces_embed_marker.get("requested_device")).strip()
            if isinstance(faces_embed_marker.get("requested_device"), str)
            else None
        )
        marker_embed_resolved_device = (
            str(faces_embed_marker.get("resolved_device")).strip()
            if isinstance(faces_embed_marker.get("resolved_device"), str)
            else None
        )
        marker_embedding_backend_configured = (
            str(faces_embed_marker.get("embedding_backend_configured")).strip()
            if isinstance(faces_embed_marker.get("embedding_backend_configured"), str)
            else None
        )
        marker_embedding_backend_configured_effective = (
            str(faces_embed_marker.get("embedding_backend_configured_effective")).strip()
            if isinstance(faces_embed_marker.get("embedding_backend_configured_effective"), str)
            else None
        )
        marker_embedding_backend_actual = (
            str(faces_embed_marker.get("embedding_backend_actual")).strip()
            if isinstance(faces_embed_marker.get("embedding_backend_actual"), str)
            else None
        )
        marker_embedding_backend_fallback_reason = (
            str(faces_embed_marker.get("embedding_backend_fallback_reason")).strip()
            if isinstance(faces_embed_marker.get("embedding_backend_fallback_reason"), str)
            else None
        )
        marker_embedding_model_name = (
            str(faces_embed_marker.get("embedding_model_name")).strip()
            if isinstance(faces_embed_marker.get("embedding_model_name"), str)
            else None
        )

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

    onnx_provider_requested = marker_onnx_provider_requested or marker_requested_detect_device
    onnx_provider_resolved = marker_onnx_provider_resolved or marker_resolved_detect_device
    if onnx_provider_requested or onnx_provider_resolved:
        provider_lines: list[tuple[str, str]] = []
        if onnx_provider_requested:
            provider_lines.append(("requested", onnx_provider_requested))
        if marker_detect_device and marker_detect_device != onnx_provider_requested:
            provider_lines.append(("configured", marker_detect_device))
        if onnx_provider_resolved:
            provider_lines.append(("resolved", onnx_provider_resolved))
        lineage_data.append(["Face Detect ONNX Provider", provider_lines])

    torch_requested = marker_torch_device_requested
    torch_resolved = marker_torch_device_resolved
    torch_reason = marker_torch_device_fallback_reason
    if not torch_requested and not torch_resolved and onnx_provider_requested:
        # Legacy runs: infer a safe torch device summary from ONNX provider semantics.
        inferred = onnx_provider_resolved or onnx_provider_requested
        if inferred in {"cpu", "cuda", "mps"}:
            torch_requested = inferred
            torch_resolved = inferred
        elif inferred == "coreml":
            torch_requested = "mps"
            torch_resolved = "mps"
            torch_reason = "legacy_run: onnx_provider=coreml; torch_device inferred as mps"
    if torch_requested or torch_resolved or torch_reason:
        torch_lines: list[tuple[str, str]] = []
        if torch_requested:
            torch_lines.append(("requested", torch_requested))
        if torch_resolved:
            torch_lines.append(("resolved", torch_resolved))
        if torch_reason:
            torch_lines.append(("fallback_reason", torch_reason))
        lineage_data.append(["Torch Device (fallback models)", torch_lines])

    if (
        marker_yolo_fallback_enabled is not None
        or marker_yolo_fallback_device
        or marker_yolo_fallback_load_status
        or marker_yolo_fallback_disabled_reason
    ):
        yolo_lines: list[tuple[str, str]] = []
        if marker_yolo_fallback_enabled is not None:
            yolo_lines.append(("enabled", "true" if marker_yolo_fallback_enabled else "false"))
        if marker_yolo_fallback_device:
            yolo_lines.append(("torch_device", marker_yolo_fallback_device))
        if marker_yolo_fallback_load_status:
            yolo_lines.append(("load_status", marker_yolo_fallback_load_status))
        if marker_yolo_fallback_disabled_reason:
            yolo_lines.append(("disabled_reason", marker_yolo_fallback_disabled_reason))
        if marker_yolo_fallback_invocations is not None:
            yolo_lines.append(("invocations", str(marker_yolo_fallback_invocations)))
        if marker_yolo_fallback_detections_added is not None:
            yolo_lines.append(("detections_added", str(marker_yolo_fallback_detections_added)))
        lineage_data.append(["Person Fallback (YOLO)", yolo_lines])

    if marker_face_detector_model:
        lineage_data.append(["Face Detector Model (runtime)", marker_face_detector_model])
    if marker_embedding_model_name:
        lineage_data.append(["Embedding Model (runtime)", marker_embedding_model_name])
    if marker_embed_requested_device or marker_embed_resolved_device:
        if marker_embed_requested_device:
            lineage_data.append(["Embedding Device (requested)", marker_embed_requested_device])
        if marker_embed_resolved_device:
            lineage_data.append(["Embedding Device (resolved)", marker_embed_resolved_device])
    if marker_embedding_backend_configured_effective or marker_embedding_backend_actual or marker_embedding_backend_configured:
        embedding_backend_effective = (
            marker_embedding_backend_configured_effective
            or marker_embedding_backend_actual
            or marker_embedding_backend_configured
        )
        backend_lines: list[tuple[str, str]] = [("effective", embedding_backend_effective)]
        if (
            marker_embedding_backend_configured
            and embedding_backend_effective
            and marker_embedding_backend_configured != embedding_backend_effective
        ):
            backend_lines.append(("configured", marker_embedding_backend_configured))
        if marker_embedding_backend_actual and embedding_backend_effective and marker_embedding_backend_actual != embedding_backend_effective:
            backend_lines.append(("runtime_actual", marker_embedding_backend_actual))
        backend_reason = _sanitize_tensorrt_fallback_reason(
            configured=marker_embedding_backend_configured,
            effective=embedding_backend_effective,
            reason=marker_embedding_backend_fallback_reason,
        )
        if backend_reason:
            backend_lines.append(("reason", backend_reason))
        lineage_data.append(["Embedding Backend (effective)", backend_lines])
    if marker_stride_effective is not None:
        lineage_data.append(["Face Detection Stride (effective)", str(marker_stride_effective)])
    if marker_stride_observed_median is not None:
        lineage_data.append(["Face Detection Stride (observed median)", str(marker_stride_observed_median)])
    if marker_frames_scanned_total is not None:
        lineage_data.append(["Frames Scanned Total (OpenCV)", str(marker_frames_scanned_total)])
    if marker_face_detect_frames_processed is not None:
        lines: list[tuple[str, str]] = [("total", str(marker_face_detect_frames_processed))]
        if marker_face_detect_frames_processed_stride is not None:
            lines.append(("stride_hits", str(marker_face_detect_frames_processed_stride)))
        if marker_face_detect_frames_processed_forced_scene_warmup is not None:
            lines.append(("forced_scene_warmup", str(marker_face_detect_frames_processed_forced_scene_warmup)))
        if marker_expected_frames_by_stride is not None:
            lines.append(("expected_by_stride", str(marker_expected_frames_by_stride)))
        if marker_forced_scene_warmup_ratio is not None:
            lines.append(("forced_scene_warmup_ratio", f"{marker_forced_scene_warmup_ratio:.3f}"))
        lineage_data.append(["Face Detect Frames Processed", lines])

    if (
        marker_scene_cut_count is not None
        or marker_warmup_frames_per_cut_effective is not None
        or marker_wall_time_per_processed_frame_s is not None
    ):
        warmup_lines: list[tuple[str, str]] = []
        if marker_scene_cut_count is not None:
            warmup_lines.append(("scene_cut_count", str(marker_scene_cut_count)))
        if marker_scene_warmup_dets is not None:
            warmup_lines.append(("scene_warmup_frames_per_cut_configured", str(marker_scene_warmup_dets)))
        if marker_warmup_frames_per_cut_effective is not None:
            warmup_lines.append(("warmup_frames_per_cut_effective", f"{marker_warmup_frames_per_cut_effective:.3f}"))
        if marker_wall_time_per_processed_frame_s is not None:
            warmup_lines.append(("wall_time_per_processed_frame_s", f"{marker_wall_time_per_processed_frame_s:.6f}"))
        lineage_data.append(["Scene Warmup Diagnostics", warmup_lines])
    if marker_scene_detect_wall_time_s is not None:
        lineage_data.append(["Scene Detect Wall Time (wall-clock)", f"{marker_scene_detect_wall_time_s:.1f}s"])
    if marker_detect_wall_time_s is not None:
        lineage_data.append(["Detect Wall Time (wall-clock)", f"{marker_detect_wall_time_s:.1f}s"])
    if marker_rtf is not None:
        lineage_data.append(["Detect Real-time Factor (RTF)", f"{marker_rtf:.3f}x"])
    if marker_effective_fps_processing is not None:
        lineage_data.append(["Detect Effective FPS", f"{marker_effective_fps_processing:.3f} fps"])
    if marker_tracker_backend_actual is not None:
        tracker_lines: list[tuple[str, str]] = [("actual", marker_tracker_backend_actual)]
        if marker_tracker_backend_configured and marker_tracker_backend_configured != marker_tracker_backend_actual:
            tracker_lines.append(("configured", marker_tracker_backend_configured))
        if marker_tracker_fallback_reason:
            tracker_lines.append(("fallback_reason", marker_tracker_fallback_reason))
        lineage_data.append(["Face Tracker Backend (runtime)", tracker_lines])
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

    # Face detection observed frame stats derived from detections.jsonl (frames that emitted ≥1 detection).
    det_stats = _face_detection_frame_stats(detections_path)
    if det_stats.get("ok"):
        lineage_data.append(["Face Detection Frames With Detections Observed", str(det_stats.get("frames_observed"))])
        if marker_stride_observed_median is None:
            stride_median = det_stats.get("stride_median")
            lineage_data.append(
                ["Face Detection Stride (observed)", str(stride_median) if stride_median is not None else "unknown"]
            )
    else:
        lineage_data.append(["Face Detection Frames With Detections Observed", "unknown"])
        if marker_stride_observed_median is None:
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
            total_frames_for_body = marker_frames_scanned_total if marker_frames_scanned_total is not None else resolved_frames
            total_frames_source = "frames_scanned_total" if marker_frames_scanned_total is not None else "resolved_frames"
            if isinstance(total_frames_for_body, int) and body_stride_cfg > 0:
                body_frames_expected = (total_frames_for_body + body_stride_cfg - 1) // body_stride_cfg
                lineage_data.append(
                    ["Body Detect Frames Processed (expected)", f"{body_frames_expected} ({total_frames_source})"]
                )
            else:
                lineage_data.append(["Body Detect Frames Processed (expected)", "unknown"])
        else:
            lineage_data.append(["Body Detect Frames Processed (expected)", "N/A (body tracking not run)"])

        # Stride accounting: compute the union of frames that would be processed by face + body detection.
        # This helps explain why wall time can be high when face/body strides differ and/or scene_warmup_dets is enabled.
        if (
            bool(body_tracking_ran_effective)
            and marker_frames_scanned_total is not None
            and marker_stride_effective is not None
            and marker_scene_warmup_dets is not None
            and body_stride_cfg > 0
        ):
            scene_indices_raw = (
                (track_metrics_data.get("scene_cuts") or {}).get("indices") if isinstance(track_metrics_data, dict) else None
            )
            scene_indices: list[int] = []
            if isinstance(scene_indices_raw, list):
                for value in scene_indices_raw:
                    parsed = _safe_int_opt(value)
                    if parsed is not None:
                        scene_indices.append(parsed)

            frames_scanned_total = int(marker_frames_scanned_total)
            stride_effective = max(int(marker_stride_effective), 1)
            warmup = max(int(marker_scene_warmup_dets), 0)

            face_processed: set[int] = set(range(0, frames_scanned_total, stride_effective))
            if warmup > 0 and scene_indices:
                for cut_idx in scene_indices:
                    if cut_idx < 0:
                        continue
                    for frame in range(cut_idx, min(cut_idx + warmup, frames_scanned_total)):
                        face_processed.add(frame)

            body_processed: set[int] = set(range(0, frames_scanned_total, int(body_stride_cfg)))
            lineage_data.append(["Unique Frames Processed (expected, Face ∪ Body)", str(len(face_processed | body_processed))])
        elif bool(body_tracking_ran_effective):
            lineage_data.append(["Unique Frames Processed (expected, Face ∪ Body)", "unknown"])
        else:
            lineage_data.append(["Unique Frames Processed (expected, Face ∪ Body)", "N/A (body tracking not run)"])

        if isinstance(run_body_tracking_marker, dict):
            actual = run_body_tracking_marker.get("tracker_backend_actual")
            configured = run_body_tracking_marker.get("tracker_backend_configured")
            reason = run_body_tracking_marker.get("tracker_fallback_reason")
            actual_str = str(actual).strip() if isinstance(actual, str) and actual.strip() else None
            configured_str = str(configured).strip() if isinstance(configured, str) and configured.strip() else None
            reason_str = str(reason).strip() if isinstance(reason, str) and reason.strip() else None
            if actual_str:
                tracker_lines: list[tuple[str, str]] = [("actual", actual_str)]
                if configured_str and configured_str != actual_str:
                    tracker_lines.append(("configured", configured_str))
                if reason_str:
                    tracker_lines.append(("fallback_reason", reason_str))
                lineage_data.append(["Body Tracker Backend (runtime)", tracker_lines])

            body_reid = run_body_tracking_marker.get("body_reid")
            if isinstance(body_reid, dict):
                reid_lines: list[tuple[str, str]] = []
                enabled_config = body_reid.get("enabled_config")
                enabled_effective = body_reid.get("enabled_effective")
                embeddings_generated = body_reid.get("reid_embeddings_generated")
                skip_reason = body_reid.get("reid_skip_reason")
                comparisons = body_reid.get("reid_comparisons_performed")
                torchreid_import_ok = body_reid.get("torchreid_import_ok")
                torchreid_version = body_reid.get("torchreid_version")
                torchreid_runtime_ok = body_reid.get("torchreid_runtime_ok")
                torchreid_runtime_error = body_reid.get("torchreid_runtime_error")
                if isinstance(enabled_config, bool):
                    reid_lines.append(("enabled_config", "true" if enabled_config else "false"))
                if isinstance(enabled_effective, bool):
                    reid_lines.append(("enabled_effective", "true" if enabled_effective else "false"))
                if isinstance(embeddings_generated, bool):
                    reid_lines.append(("embeddings_generated", "true" if embeddings_generated else "false"))
                torchreid_env_ok: bool | None = None
                torchreid_env_version: str | None = None
                if isinstance(import_status, dict):
                    torchreid_env = import_status.get("torchreid")
                    if isinstance(torchreid_env, dict):
                        status = torchreid_env.get("status")
                        if isinstance(status, str) and status.strip():
                            torchreid_env_ok = status.strip() == "ok"
                        version = torchreid_env.get("version")
                        if isinstance(version, str) and version.strip():
                            torchreid_env_version = version.strip()
                if torchreid_env_ok is True:
                    installed = "yes"
                    if torchreid_env_version:
                        installed += f" ({torchreid_env_version})"
                    reid_lines.append(("torchreid_installed", installed))
                elif torchreid_env_ok is False:
                    reid_lines.append(("torchreid_installed", "no"))
                if isinstance(import_status, dict):
                    utils_state = import_status.get("torchreid.utils")
                    if isinstance(utils_state, dict):
                        utils_status = utils_state.get("status")
                        if isinstance(utils_status, str) and utils_status.strip():
                            reid_lines.append(
                                ("torchreid_utils_import_ok", "true" if utils_status.strip() == "ok" else "false")
                            )
                        utils_err = utils_state.get("error")
                        if isinstance(utils_err, str) and utils_err.strip():
                            detail = utils_err.strip()
                            if len(detail) > 160:
                                detail = detail[:157] + "..."
                            reid_lines.append(("torchreid_utils_error", detail))
                    torchreid_env = import_status.get("torchreid")
                    if isinstance(torchreid_env, dict):
                        dists = torchreid_env.get("distribution")
                        if isinstance(dists, list) and dists:
                            first = dists[0] if isinstance(dists[0], dict) else None
                            if isinstance(first, dict):
                                name = first.get("name")
                                ver = first.get("version")
                                if isinstance(name, str) and name.strip():
                                    label = name.strip()
                                    if isinstance(ver, str) and ver.strip():
                                        label = f"{label} ({ver.strip()})"
                                    reid_lines.append(("torchreid_distribution", label))
                if isinstance(torchreid_import_ok, bool):
                    reid_lines.append(("torchreid_import_ok", "true" if torchreid_import_ok else "false"))
                if isinstance(torchreid_version, str) and torchreid_version.strip():
                    reid_lines.append(("torchreid_version", torchreid_version.strip()))
                if isinstance(torchreid_runtime_ok, bool):
                    reid_lines.append(("torchreid_runtime_ok", "true" if torchreid_runtime_ok else "false"))
                runtime_error_str: str | None = None
                if isinstance(torchreid_runtime_error, str) and torchreid_runtime_error.strip():
                    runtime_error_str = torchreid_runtime_error.strip()
                elif isinstance(run_body_tracking_marker.get("reid_note"), str) and run_body_tracking_marker.get("reid_note").strip():
                    note = run_body_tracking_marker.get("reid_note").strip()
                    if note.lower().startswith("import_error:"):
                        note = note[len("import_error:") :].strip()
                    if note.lower().startswith("runtime_error:"):
                        note = note[len("runtime_error:") :].strip()
                    runtime_error_str = note or None
                if runtime_error_str and len(runtime_error_str) > 180:
                    runtime_error_str = runtime_error_str[:177] + "..."

                skip_reason_effective = skip_reason.strip() if isinstance(skip_reason, str) and skip_reason.strip() else None
                if (
                    torchreid_env_ok is True
                    and skip_reason_effective == "torchreid_import_error"
                    and torchreid_import_ok is not False
                ):
                    skip_reason_effective = "torchreid_runtime_error"
                if skip_reason_effective:
                    reid_lines.append(("skip_reason", skip_reason_effective))
                if runtime_error_str and (skip_reason_effective == "torchreid_runtime_error" or torchreid_runtime_ok is False):
                    reid_lines.append(("runtime_error", runtime_error_str))
                if isinstance(comparisons, int):
                    reid_lines.append(("comparisons_performed", str(comparisons)))
                if reid_lines:
                    lineage_data.append(["Body Re-ID (runtime)", reid_lines])

    # Model versions
    embedding_backend_display = (
        marker_embedding_backend_configured_effective
        or marker_embedding_backend_actual
        or marker_embedding_backend_configured
        or embedding_config.get("embedding", {}).get("backend", "tensorrt")
    )
    lineage_data.extend([
        ["Face Detector", detection_config.get("model_id", "retinaface_r50")],
        ["Face Tracker", marker_tracker_backend_actual or "ByteTrack"],
        ["Embedding Model", str(embedding_backend_display) + " / ArcFace R100"],
        ["Body Detector", body_detection_config.get("person_detection", {}).get("model", "yolov8n")],
        ["Body Re-ID", body_detection_config.get("person_reid", {}).get("model", "osnet_x1_0")],
    ])

    kv_header_style = ParagraphStyle(
        "KVHeader",
        parent=cell_style,
        fontName="Helvetica-Bold",
    )
    lineage_rows: list[tuple[str, Any]] = []
    for row in lineage_data[1:]:
        if isinstance(row, list) and len(row) == 2:
            lineage_rows.append((str(row[0]), row[1]))

    lineage_table = build_wrap_safe_kv_table(
        lineage_rows,
        width=float(doc.width),
        header=(str(lineage_data[0][0]), str(lineage_data[0][1])),
        label_ratio=0.40,
        cell_style=cell_style,
        header_style=kv_header_style,
    )
    story.append(lineage_table)
    story.append(Spacer(1, 12))

    # =========================================================================
    # SECTION 0.5: RUN-TO-RUN DIFF
    # =========================================================================
    story.append(Paragraph("0.5 Run-to-Run Diff", section_style))

    def _safe_ratio(numer: int | float | None, denom: int | float | None) -> float | None:
        if numer is None or denom in (None, 0):
            return None
        try:
            return float(numer) / float(denom)
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def _singleton_rate_from_metrics(metrics_payload: dict[str, Any] | None) -> float | None:
        if not isinstance(metrics_payload, dict):
            return None
        block = metrics_payload.get("cluster_metrics")
        if not isinstance(block, dict):
            return None
        rate = _safe_float_opt(block.get("singleton_fraction_after") or block.get("singleton_fraction"))
        if rate is not None:
            return rate
        singles = _safe_int_opt(block.get("singleton_count"))
        total = _safe_int_opt(block.get("total_clusters"))
        return _safe_ratio(singles, total)

    def _detect_rtf_for_root(root: Path) -> float | None:
        status_payload = _read_json(root / "episode_status.json")
        if isinstance(status_payload, dict):
            detect_block = status_payload.get("stages", {}).get("detect")
            if isinstance(detect_block, dict):
                metrics = detect_block.get("metrics")
                if isinstance(metrics, dict):
                    rtf_val = _safe_float_opt(metrics.get("rtf"))
                    if rtf_val is not None:
                        return rtf_val
        marker = _read_json(root / "detect_track.json") or {}
        if isinstance(marker, dict):
            return _safe_float_opt(marker.get("rtf"))
        return None

    def _track_count_for_root(root: Path) -> int | None:
        status_payload = _read_json(root / "episode_status.json")
        if isinstance(status_payload, dict):
            detect_block = status_payload.get("stages", {}).get("detect")
            if isinstance(detect_block, dict):
                metrics = detect_block.get("metrics")
                if isinstance(metrics, dict):
                    count = _safe_int_opt(metrics.get("tracks"))
                    if count is not None:
                        return count
        marker = _read_json(root / "detect_track.json") or {}
        if isinstance(marker, dict):
            return _safe_int_opt(marker.get("tracks"))
        return None

    def _forced_splits_share_from_metrics(metrics_payload: dict[str, Any] | None) -> float | None:
        metrics_block = metrics_payload.get("metrics") if isinstance(metrics_payload, dict) else None
        forced = _safe_int_opt(metrics_block.get("forced_splits") if isinstance(metrics_block, dict) else None)
        tracks_total = _safe_int_opt(metrics_block.get("tracks_born") if isinstance(metrics_block, dict) else None)
        return _safe_ratio(forced, tracks_total)

    def _forced_splits_share_for_root(root: Path) -> float | None:
        metrics_payload = _read_json(root / "track_metrics.json")
        share = _forced_splits_share_from_metrics(metrics_payload)
        if share is not None:
            return share
        metrics_block = metrics_payload.get("metrics") if isinstance(metrics_payload, dict) else None
        forced = _safe_int_opt(metrics_block.get("forced_splits") if isinstance(metrics_block, dict) else None)
        tracks_total = _track_count_for_root(root)
        return _safe_ratio(forced, tracks_total)

    def _fused_pairs_for_root(root: Path) -> int | None:
        payload = _read_json(root / "body_tracking" / "track_fusion.json")
        if not isinstance(payload, dict):
            return None
        identities = payload.get("identities")
        if isinstance(identities, dict):
            pairs = 0
            for identity_data in identities.values():
                if not isinstance(identity_data, dict):
                    continue
                if identity_data.get("face_track_ids") and identity_data.get("body_track_ids"):
                    pairs += 1
            return pairs
        diagnostics = payload.get("diagnostics")
        if isinstance(diagnostics, dict):
            return _safe_int_opt(diagnostics.get("final_pairs"))
        return _safe_int_opt(payload.get("num_fused_identities"))

    def _gain_for_root(root: Path) -> float | None:
        payload = _read_json(root / "body_tracking" / "screentime_comparison.json")
        if not isinstance(payload, dict):
            return None
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            return None
        return _safe_float_opt(summary.get("gain_total_s", summary.get("total_duration_gain")))

    current_rtf = marker_rtf
    if current_rtf is None and isinstance(episode_status_payload, dict):
        detect_block = episode_status_payload.get("stages", {}).get("detect")
        if isinstance(detect_block, dict):
            metrics = detect_block.get("metrics")
            if isinstance(metrics, dict):
                current_rtf = _safe_float_opt(metrics.get("rtf"))
    current_forced_share = _forced_splits_share_from_metrics(track_metrics_data if isinstance(track_metrics_data, dict) else None)
    if current_forced_share is None:
        current_forced_share = _forced_splits_share_for_root(run_root)
    current_singleton_rate = _singleton_rate_from_metrics(track_metrics_data if isinstance(track_metrics_data, dict) else None)
    current_fused_pairs = actual_fused_pairs
    current_gain = None
    if isinstance(screentime_summary, dict):
        current_gain = _safe_float_opt(
            screentime_summary.get("gain_total_s", screentime_summary.get("total_duration_gain"))
        )
    if current_gain is None:
        current_gain = _gain_for_root(run_root)

    def _prev_successful_run_id() -> str | None:
        candidates: list[tuple[float, str]] = []
        for run_id in run_layout.list_run_ids(ep_id):
            if run_id == run_id_norm:
                continue
            try:
                mtime = run_layout.run_root(ep_id, run_id).stat().st_mtime
            except (OSError, ValueError):
                mtime = 0.0
            candidates.append((mtime, run_id))
        for _, candidate in sorted(candidates, reverse=True):
            run_root = run_layout.run_root(ep_id, candidate)
            status_payload = _read_json(run_root / "episode_status.json")
            if isinstance(status_payload, dict):
                detect_block = status_payload.get("stages", {}).get("detect")
                if isinstance(detect_block, dict) and detect_block.get("status") == "success":
                    return candidate
            marker = _read_json(run_root / "detect_track.json")
            if isinstance(marker, dict) and str(marker.get("status") or "").lower() == "success":
                return candidate
        return None

    baseline_run_id = _prev_successful_run_id()
    if baseline_run_id:
        baseline_root = run_layout.run_root(ep_id, baseline_run_id)

        baseline_rtf = _detect_rtf_for_root(baseline_root)
        baseline_forced_share = _forced_splits_share_for_root(baseline_root)
        baseline_singleton_rate = _singleton_rate_from_metrics(_read_json(baseline_root / "track_metrics.json"))
        baseline_fused_pairs = _fused_pairs_for_root(baseline_root)
        baseline_gain = _gain_for_root(baseline_root)

        def _delta_text(current: float | int | None, baseline: float | int | None, suffix: str = "") -> str:
            if current is None or baseline is None:
                return "N/A"
            delta = float(current) - float(baseline)
            return f"{delta:+.2f}{suffix}"

        diff_rows = [
            ["Metric", "Current", f"Baseline ({baseline_run_id})", "Delta"],
            [
                "Detect RTF",
                f"{current_rtf:.2f}x" if current_rtf is not None else "N/A",
                f"{baseline_rtf:.2f}x" if baseline_rtf is not None else "N/A",
                _delta_text(current_rtf, baseline_rtf, "x"),
            ],
            [
                "Forced splits share",
                f"{current_forced_share:.2%}" if current_forced_share is not None else "N/A",
                f"{baseline_forced_share:.2%}" if baseline_forced_share is not None else "N/A",
                _delta_text(
                    current_forced_share * 100 if current_forced_share is not None else None,
                    baseline_forced_share * 100 if baseline_forced_share is not None else None,
                    "pp",
                ),
            ],
            [
                "Singleton rate",
                f"{current_singleton_rate:.2%}" if current_singleton_rate is not None else "N/A",
                f"{baseline_singleton_rate:.2%}" if baseline_singleton_rate is not None else "N/A",
                _delta_text(
                    current_singleton_rate * 100 if current_singleton_rate is not None else None,
                    baseline_singleton_rate * 100 if baseline_singleton_rate is not None else None,
                    "pp",
                ),
            ],
            [
                "Fused pairs",
                str(current_fused_pairs) if current_fused_pairs is not None else "N/A",
                str(baseline_fused_pairs) if baseline_fused_pairs is not None else "N/A",
                _delta_text(current_fused_pairs, baseline_fused_pairs, ""),
            ],
            [
                gain_label,
                f"{current_gain:.2f}s" if current_gain is not None else "N/A",
                f"{baseline_gain:.2f}s" if baseline_gain is not None else "N/A",
                _delta_text(current_gain, baseline_gain, "s"),
            ],
        ]

        diff_table = Table(diff_rows, colWidths=[1.6 * inch, 1.3 * inch, 1.7 * inch, 1.0 * inch])
        diff_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
            ])
        )
        story.append(diff_table)
    else:
        story.append(Paragraph("No prior successful run found for baseline comparison.", note_style))

    story.append(Spacer(1, 12))

    # =========================================================================
    # SECTION 0.6: JOB LIFECYCLE
    # =========================================================================
    story.append(Paragraph("0.6 Job Lifecycle", section_style))

    stage_labels = {
        "detect": "Detect/Track",
        "faces": "Faces Harvest",
        "cluster": "Cluster",
        "body_tracking": "Body Tracking",
        "track_fusion": "Track Fusion",
        "pdf": "PDF Export",
    }
    if include_screentime:
        stage_labels["screentime"] = "Screentime"
    raw_plan = (
        [str(item) for item in episode_status_payload.get("stage_plan", []) if isinstance(item, str)]
        if isinstance(episode_status_payload, dict)
        else []
    )
    stage_plan: list[str] = []
    for item in raw_plan:
        normalized = normalize_stage_key(item)
        if normalized and normalized not in stage_plan:
            stage_plan.append(normalized)
    if not include_screentime:
        stage_plan = [stage for stage in stage_plan if stage != "screentime"]
    if not stage_plan:
        stage_plan = ["detect", "faces", "cluster", "body_tracking", "track_fusion", "pdf"]
        if include_screentime:
            stage_plan.insert(-1, "screentime")

    job_records: list[dict[str, Any]] = []
    jobs_dir = get_path(ep_id, "video").parents[2] / "jobs"
    if jobs_dir.exists():
        for path in jobs_dir.glob("*.json"):
            payload = _read_json(path)
            if isinstance(payload, dict) and payload.get("ep_id") == ep_id:
                job_records.append(payload)

    def _record_run_id(record: dict[str, Any]) -> str | None:
        for source_key in ("requested", "summary"):
            source = record.get(source_key)
            if isinstance(source, dict):
                value = source.get("run_id")
                if isinstance(value, str) and value.strip():
                    try:
                        return run_layout.normalize_run_id(value)
                    except ValueError:
                        return value.strip()
        command = record.get("command")
        if isinstance(command, list):
            for idx, token in enumerate(command):
                if token == "--run-id" and idx + 1 < len(command):
                    return str(command[idx + 1])
        if isinstance(command, str) and "--run-id" in command:
            parts = command.split()
            for idx, token in enumerate(parts):
                if token == "--run-id" and idx + 1 < len(parts):
                    return parts[idx + 1]
        return None

    job_type_map = {
        "detect_track": "detect",
        "faces_embed": "faces",
        "cluster": "cluster",
        "body_tracking": "body_tracking",
        "body_tracking_fusion": "track_fusion",
        "screen_time_analyze": "screentime",
        "pdf_export": "pdf",
    }

    job_records_by_stage: dict[str, list[dict[str, Any]]] = {}
    for record in job_records:
        stage_key = job_type_map.get(record.get("job_type"))
        if not stage_key:
            continue
        record_run_id = _record_run_id(record)
        if record_run_id and record_run_id != run_id_norm:
            continue
        job_records_by_stage.setdefault(stage_key, []).append(record)

    job_runs_by_stage: dict[str, list[dict[str, Any]]] = {}
    for job_run in job_runs:
        if not isinstance(job_run, dict):
            continue
        stage_key = job_type_map.get(job_run.get("job_name"))
        if stage_key:
            job_runs_by_stage.setdefault(stage_key, []).append(job_run)

    def _last_log_line(record: dict[str, Any] | None) -> str | None:
        if not isinstance(record, dict):
            return None
        summary = record.get("summary")
        if isinstance(summary, dict):
            phase = summary.get("phase")
            step = summary.get("step")
            message = summary.get("message") or summary.get("status") or summary.get("error")
            parts = [str(val) for val in (phase, step, message) if val]
            if parts:
                return " · ".join(parts)
        error_text = record.get("error")
        if isinstance(error_text, str) and error_text.strip():
            return error_text.strip()
        stderr_log = record.get("stderr_log")
        if isinstance(stderr_log, str):
            try:
                log_path = Path(stderr_log)
                if log_path.exists():
                    content = log_path.read_text(encoding="utf-8", errors="ignore")
                    for line in reversed(content.splitlines()):
                        cleaned = line.strip()
                        if cleaned:
                            return cleaned[:200]
            except OSError:
                return None
        return None

    job_rows = [_wrap_row(["Stage", "Status", "Started", "Ended", "Exit/Retry", "Last Log"], cell_style_small)]
    marker_map = {
        "detect": "detect_track.json",
        "faces": "faces_embed.json",
        "cluster": "cluster.json",
        "body_tracking": "body_tracking.json",
        "track_fusion": "body_tracking_fusion.json",
    }
    export_index_path = run_root / "exports" / "export_index.json"
    artifact_state_by_stage: dict[str, dict[str, Any]] = {
        "detect": {"detections": detections_path.exists(), "tracks": tracks_path.exists()},
        "faces": {"faces": faces_path.exists()},
        "cluster": {"identities": identities_path.exists()},
        "body_tracking": {
            "body_tracks": body_tracks_path.exists(),
            "legacy": legacy_body_artifacts_exist,
        },
        "track_fusion": {
            "track_fusion": track_fusion_path.exists(),
            "legacy": legacy_track_fusion_path.exists(),
        },
        "pdf": {"export_index": export_index_path.exists()},
    }
    for stage_key in stage_plan:
        stage_entry = None
        if isinstance(episode_status_payload, dict):
            stage_entry = episode_status_payload.get("stages", {}).get(stage_key)
        if not isinstance(stage_entry, dict):
            stage_entry = {}
        status_val = _resolve_lifecycle_status(stage_key, stage_entry, artifact_state_by_stage.get(stage_key, {}))
        started_at = stage_entry.get("started_at")
        ended_at = stage_entry.get("ended_at")
        if not started_at and not ended_at:
            marker_name = marker_map.get(stage_key)
            if marker_name:
                marker = _read_json(run_root / marker_name)
                if isinstance(marker, dict):
                    status_val = status_val if status_val != "unknown" else str(marker.get("status") or "unknown")
                    started_at = marker.get("started_at")
                    ended_at = marker.get("finished_at") or marker.get("ended_at")
        started_at = started_at or "—"
        ended_at = ended_at or "—"
        records = job_records_by_stage.get(stage_key, [])
        records.sort(key=lambda r: r.get("started_at") or "")
        latest_record = records[-1] if records else None
        exit_code = None
        if isinstance(latest_record, dict):
            exit_code = latest_record.get("return_code")
        retries = max(len(records) - 1, 0) if records else max(len(job_runs_by_stage.get(stage_key, [])) - 1, 0)
        last_line = _last_log_line(latest_record)
        if last_line is None and job_runs_by_stage.get(stage_key):
            last_error = job_runs_by_stage[stage_key][-1].get("error_text")
            if isinstance(last_error, str) and last_error.strip():
                last_line = last_error.strip()
        job_rows.append(
            _wrap_row(
                [
                    stage_labels.get(stage_key, stage_key),
                    str(status_val),
                    str(started_at),
                    str(ended_at),
                    f"exit={exit_code if exit_code is not None else 'N/A'} retry={retries}",
                    last_line or "—",
                ],
                cell_style_small,
            )
        )

    job_table = Table(job_rows, colWidths=[1.1 * inch, 0.8 * inch, 1.2 * inch, 1.2 * inch, 1.0 * inch, 1.7 * inch])
    job_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ])
    )
    story.append(job_table)
    story.append(Spacer(1, 12))

    # =========================================================================
    # SECTION 0.7: PERFORMANCE & QUALITY
    # =========================================================================
    story.append(Paragraph("0.7 Performance &amp; Quality", section_style))

    detect_eff_fps = marker_effective_fps_processing
    if detect_eff_fps is None and isinstance(episode_status_payload, dict):
        detect_block = episode_status_payload.get("stages", {}).get("detect")
        if isinstance(detect_block, dict):
            metrics = detect_block.get("metrics")
            if isinstance(metrics, dict):
                detect_eff_fps = _safe_float_opt(metrics.get("effective_fps_processing"))

    forced_scene_ratio = marker_forced_scene_warmup_ratio
    embedding_backend_effective = (
        marker_embedding_backend_configured_effective
        or marker_embedding_backend_actual
        or marker_embedding_backend_configured
        or embedding_config.get("embedding", {}).get("backend", "tensorrt")
    )
    embedding_backend_note = (
        _sanitize_tensorrt_fallback_reason(
            configured=marker_embedding_backend_configured,
            effective=embedding_backend_effective,
            reason=marker_embedding_backend_fallback_reason,
        )
        or ""
    )

    reid_cfg_enabled = bool(track_fusion_config.get("reid_handoff", {}).get("enabled", False))
    body_reid = run_body_tracking_marker.get("body_reid") if isinstance(run_body_tracking_marker, dict) else None
    reid_effective = body_reid.get("enabled_effective") if isinstance(body_reid, dict) else None
    reid_skip_reason = body_reid.get("reid_skip_reason") if isinstance(body_reid, dict) else None
    torchreid_error = body_reid.get("torchreid_runtime_error") if isinstance(body_reid, dict) else None

    if not reid_cfg_enabled:
        fusion_mode_value = "IoU-only"
        fusion_mode_note = "reid_handoff.enabled=false"
    elif reid_effective:
        fusion_mode_value = "Hybrid (IoU + Re-ID)"
        fusion_mode_note = "enabled_effective=true"
    else:
        fusion_mode_value = "IoU-only"
        fusion_mode_note = "Re-ID unavailable"
        if isinstance(torchreid_error, str) and torchreid_error.strip():
            fusion_mode_note = f"{fusion_mode_note}; {torchreid_error.strip()}"
        elif isinstance(reid_skip_reason, str) and reid_skip_reason.strip():
            fusion_mode_note = f"{fusion_mode_note}; skip_reason={reid_skip_reason.strip()}"

    reid_value = "available" if reid_effective else "unavailable"
    if not reid_cfg_enabled:
        reid_value = "disabled"
    reid_note = None
    if isinstance(torchreid_error, str) and torchreid_error.strip():
        reid_note = torchreid_error.strip()
        if "missing torchreid.utils" in reid_note:
            reid_note = (
                "missing torchreid.utils (install deep-person-reid; "
                "pip install -r requirements-ml.txt)"
            )
    elif isinstance(reid_skip_reason, str) and reid_skip_reason.strip():
        reid_note = f"skip_reason={reid_skip_reason.strip()}"

    perf_rows = [
        _wrap_row(["Metric", "Value", "Notes"]),
        _wrap_row([
            "Detect RTF",
            f"{current_rtf:.2f}x" if current_rtf is not None else "N/A",
            "detect_track marker / status",
        ]),
        _wrap_row([
            "Detect Effective FPS",
            f"{detect_eff_fps:.2f} fps" if detect_eff_fps is not None else "N/A",
            "detect_track marker / status",
        ]),
        _wrap_row([
            "Forced splits share",
            f"{current_forced_share:.2%}" if current_forced_share is not None else "N/A",
            "track_metrics.json",
        ]),
        _wrap_row([
            "Singleton rate",
            f"{current_singleton_rate:.2%}" if current_singleton_rate is not None else "N/A",
            "cluster metrics",
        ]),
        _wrap_row([
            "Fusion mode",
            fusion_mode_value,
            fusion_mode_note or "",
        ]),
        _wrap_row([
            "Fusion yield (fused pairs)",
            str(current_fused_pairs) if current_fused_pairs is not None else "N/A",
            "track_fusion.json diagnostics",
        ]),
        _wrap_row([
            "Body Re-ID availability",
            reid_value,
            reid_note or "",
        ]),
        _wrap_row([
            "Embedding backend (effective)",
            str(embedding_backend_effective),
            embedding_backend_note,
        ]),
    ]
    if forced_scene_ratio is not None:
        perf_rows.append(
            _wrap_row([
                "Forced scene warmup ratio",
                f"{forced_scene_ratio:.3f}",
                "detect_track marker",
            ])
        )

    perf_table = Table(perf_rows, colWidths=[1.6 * inch, 1.4 * inch, 2.6 * inch])
    perf_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ])
    )
    story.append(perf_table)
    story.append(Spacer(1, 12))

    # =========================================================================
    # SECTION 0.8: DB HEALTH
    # =========================================================================
    story.append(Paragraph("0.8 DB Health", section_style))
    db_rows = [
        _wrap_row(["Signal", "Value", "Notes"]),
        _wrap_row(
            [
                "DB_URL configured",
                "Yes" if db_configured else "No",
                db_not_configured_reason or "",
            ]
        ),
        _wrap_row(
            [
                "DB Connected",
                _health_status(db_connected),
                db_error or ("OK" if db_connected else ""),
            ]
        ),
        _wrap_row(
            [
                "Fake DB enabled",
                "Yes" if fake_db_enabled else "No",
                "SCREENALYTICS_FAKE_DB=1" if fake_db_enabled else "",
            ]
        ),
    ]
    db_table = Table(db_rows, colWidths=[1.6 * inch, 1.0 * inch, 3.0 * inch])
    db_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ])
    )
    story.append(db_table)
    story.append(Spacer(1, 12))

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

    face_tracker_backend_detail: str | None = None
    if isinstance(detect_track_marker, dict):
        backend_actual = detect_track_marker.get("tracker_backend_actual")
        backend_configured = detect_track_marker.get("tracker_backend_configured")
        fallback_reason = detect_track_marker.get("tracker_fallback_reason")
        if isinstance(backend_actual, str) and backend_actual.strip():
            face_tracker_backend_detail = backend_actual.strip()
            if isinstance(fallback_reason, str) and fallback_reason.strip():
                face_tracker_backend_detail += f" (fallback_reason={fallback_reason.strip()})"
            if (
                isinstance(backend_configured, str)
                and backend_configured.strip()
                and backend_configured.strip() != backend_actual.strip()
            ):
                face_tracker_backend_detail += f" (configured={backend_configured.strip()})"

    track_stats = [
        ["Metric", "Value"],
        ["Tracks Born", str(metrics.get("tracks_born", "N/A"))],
        ["Tracks Lost", str(metrics.get("tracks_lost", "N/A"))],
        ["ID Switches", str(metrics.get("id_switches", "N/A"))],
        ["Forced Splits", str(metrics.get("forced_splits", "N/A"))],
        ["Scene Cuts", str(track_metrics_data.get("scene_cuts", {}).get("count", "N/A"))],
    ]
    if face_tracker_backend_detail is not None:
        track_stats.append(["Face Tracker Backend (actual)", face_tracker_backend_detail])
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
    gate_auto_rerun_reason: str | None = None
    gate_splits_share: float | None = None
    gate_splits_share_min: float | None = None
    if isinstance(marker_gate_auto_rerun, dict):
        reason_raw = marker_gate_auto_rerun.get("reason")
        if isinstance(reason_raw, str) and reason_raw.strip():
            gate_auto_rerun_reason = reason_raw.strip()
        decision = marker_gate_auto_rerun.get("decision")
        if isinstance(decision, dict):
            gate_splits_share = _safe_float_opt(decision.get("gate_splits_share"))
            thresholds = decision.get("thresholds")
            if isinstance(thresholds, dict):
                gate_splits_share_min = _safe_float_opt(thresholds.get("min_gate_share"))

    forced_splits = metrics.get("forced_splits", 0)
    id_switches = metrics.get("id_switches", 0)
    if isinstance(forced_splits, (int, float)) and forced_splits > 50:
        gate_splits_share_low = False
        if gate_splits_share is not None and gate_splits_share_min is not None:
            gate_splits_share_low = gate_splits_share < gate_splits_share_min
        if gate_auto_rerun_reason == "gate_share_too_low" or gate_splits_share_low or marker_gate_enabled is False:
            share_str = f"{gate_splits_share:.3f}" if gate_splits_share is not None else "unknown"
            story.append(
                Paragraph(
                    f"⚠️ High forced splits ({forced_splits}): appearance gate is not the primary driver "
                    f"(auto_rerun_reason={gate_auto_rerun_reason or 'unknown'}, gate_splits_share={share_str}). "
                    "Tune track_buffer/match_thresh and scene-cut/warmup settings; disabling gate_enabled is unlikely to help.",
                    warning_style,
                )
            )
        else:
            extra = f" (gate_splits_share={gate_splits_share:.3f})" if gate_splits_share is not None else ""
            story.append(
                Paragraph(
                    f"⚠️ High forced splits ({forced_splits}){extra}: Appearance gate is aggressively splitting tracks. "
                    "Consider disabling gate_enabled in tracking.yaml or adjusting appearance thresholds.",
                    warning_style,
                )
            )
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

    body_tracker_backend_detail: str | None = None
    body_tracker_fallback_reason: str | None = None
    if isinstance(run_body_tracking_marker, dict):
        backend_actual = run_body_tracking_marker.get("tracker_backend_actual")
        backend_configured = run_body_tracking_marker.get("tracker_backend_configured")
        fallback_reason = run_body_tracking_marker.get("tracker_fallback_reason")
        if isinstance(backend_actual, str) and backend_actual.strip():
            body_tracker_backend_detail = backend_actual.strip()
            if isinstance(backend_configured, str) and backend_configured.strip() and backend_configured.strip() != body_tracker_backend_detail:
                body_tracker_backend_detail += f" (configured={backend_configured.strip()})"
            if isinstance(fallback_reason, str) and fallback_reason.strip():
                body_tracker_fallback_reason = fallback_reason.strip()
                body_tracker_backend_detail += f" (fallback_reason={body_tracker_fallback_reason})"
    if body_tracker_backend_detail is not None:
        story.append(Paragraph(f"Body tracker backend (actual): <b>{body_tracker_backend_detail}</b>", body_style))
        if body_tracker_fallback_reason:
            supervision_status: str | None = None
            supervision_error: str | None = None
            if isinstance(run_body_tracking_marker, dict):
                import_status = run_body_tracking_marker.get("import_status")
                if isinstance(import_status, dict):
                    sup = import_status.get("supervision")
                    if isinstance(sup, dict):
                        supervision_status = sup.get("status") if isinstance(sup.get("status"), str) else None
                        supervision_error = sup.get("error") if isinstance(sup.get("error"), str) else None

            if body_tracker_fallback_reason == "supervision_missing":
                warn_msg = (
                    "⚠️ tracking backend fallback activated: supervision is missing. "
                    "Install <b>supervision</b> to use supervision.ByteTrack for body tracking "
                    "(e.g., <font name='Courier'>pip install -r requirements-ml.txt</font>)."
                )
            elif body_tracker_fallback_reason == "supervision_import_error":
                detail = supervision_error or "unknown import error"
                if len(detail) > 160:
                    detail = detail[:157] + "..."
                warn_msg = (
                    "⚠️ tracking backend fallback activated: supervision failed to import. "
                    f"(status={supervision_status or 'import_error'} error={_escape_reportlab_xml(detail)})"
                )
            else:
                warn_msg = (
                    "⚠️ tracking backend fallback activated. "
                    f"(fallback_reason={_escape_reportlab_xml(body_tracker_fallback_reason)})"
                )
            story.append(Paragraph(warn_msg, warning_style))

    body_reid_detail: str | None = None
    if isinstance(run_body_tracking_marker, dict):
        body_reid = run_body_tracking_marker.get("body_reid")
        if isinstance(body_reid, dict):
            reid_lines: list[tuple[str, str]] = []

            torchreid_env_ok: bool | None = None
            torchreid_env_version: str | None = None
            if isinstance(import_status, dict):
                torchreid_env = import_status.get("torchreid")
                if isinstance(torchreid_env, dict):
                    status = torchreid_env.get("status")
                    if isinstance(status, str) and status.strip():
                        torchreid_env_ok = status.strip() == "ok"
                    version = torchreid_env.get("version")
                    if isinstance(version, str) and version.strip():
                        torchreid_env_version = version.strip()
            if torchreid_env_ok is True:
                installed = "yes"
                if torchreid_env_version:
                    installed += f" ({torchreid_env_version})"
                reid_lines.append(("torchreid_installed", installed))
            elif torchreid_env_ok is False:
                reid_lines.append(("torchreid_installed", "no"))
            if isinstance(import_status, dict):
                utils_state = import_status.get("torchreid.utils")
                if isinstance(utils_state, dict):
                    utils_status = utils_state.get("status")
                    if isinstance(utils_status, str) and utils_status.strip():
                        reid_lines.append(
                            ("torchreid_utils_import_ok", "true" if utils_status.strip() == "ok" else "false")
                        )
                    utils_err = utils_state.get("error")
                    if isinstance(utils_err, str) and utils_err.strip():
                        detail = utils_err.strip()
                        if len(detail) > 180:
                            detail = detail[:177] + "..."
                        reid_lines.append(("torchreid_utils_error", detail))
                torchreid_env = import_status.get("torchreid")
                if isinstance(torchreid_env, dict):
                    dists = torchreid_env.get("distribution")
                    if isinstance(dists, list) and dists:
                        first = dists[0] if isinstance(dists[0], dict) else None
                        if isinstance(first, dict):
                            name = first.get("name")
                            ver = first.get("version")
                            if isinstance(name, str) and name.strip():
                                label = name.strip()
                                if isinstance(ver, str) and ver.strip():
                                    label = f"{label} ({ver.strip()})"
                                reid_lines.append(("torchreid_distribution", label))

            enabled_config = body_reid.get("enabled_config")
            enabled_effective = body_reid.get("enabled_effective")
            embeddings_generated = body_reid.get("reid_embeddings_generated")
            skip_reason = body_reid.get("reid_skip_reason")
            comparisons = body_reid.get("reid_comparisons_performed")
            if isinstance(enabled_config, bool):
                reid_lines.append(("enabled_config", "true" if enabled_config else "false"))
            if isinstance(enabled_effective, bool):
                reid_lines.append(("enabled_effective", "true" if enabled_effective else "false"))
            if isinstance(embeddings_generated, bool):
                reid_lines.append(("embeddings_generated", "true" if embeddings_generated else "false"))
            if isinstance(comparisons, int):
                reid_lines.append(("comparisons_performed", str(comparisons)))

            torchreid_import_ok = body_reid.get("torchreid_import_ok")
            torchreid_runtime_ok = body_reid.get("torchreid_runtime_ok")
            torchreid_runtime_error = body_reid.get("torchreid_runtime_error")
            if isinstance(torchreid_import_ok, bool):
                reid_lines.append(("torchreid_import_ok", "true" if torchreid_import_ok else "false"))
            if isinstance(torchreid_runtime_ok, bool):
                reid_lines.append(("torchreid_runtime_ok", "true" if torchreid_runtime_ok else "false"))

            runtime_error_str: str | None = None
            if isinstance(torchreid_runtime_error, str) and torchreid_runtime_error.strip():
                runtime_error_str = torchreid_runtime_error.strip()
            elif isinstance(run_body_tracking_marker.get("reid_note"), str) and run_body_tracking_marker.get("reid_note").strip():
                note = run_body_tracking_marker.get("reid_note").strip()
                if note.lower().startswith("import_error:"):
                    note = note[len("import_error:") :].strip()
                if note.lower().startswith("runtime_error:"):
                    note = note[len("runtime_error:") :].strip()
                runtime_error_str = note or None
            if runtime_error_str and len(runtime_error_str) > 180:
                runtime_error_str = runtime_error_str[:177] + "..."

            skip_reason_effective = skip_reason.strip() if isinstance(skip_reason, str) and skip_reason.strip() else None
            if torchreid_env_ok is True and skip_reason_effective == "torchreid_import_error" and torchreid_import_ok is not False:
                skip_reason_effective = "torchreid_runtime_error"
            if skip_reason_effective:
                reid_lines.append(("skip_reason", skip_reason_effective))
            if runtime_error_str and (skip_reason_effective == "torchreid_runtime_error" or torchreid_runtime_ok is False):
                reid_lines.append(("runtime_error", runtime_error_str))

            if reid_lines:
                body_reid_detail = "<br/>".join(
                    f"<b>{_escape_reportlab_xml(k)}:</b> {_escape_reportlab_xml(_soft_wrap_text(v))}"
                    for k, v in reid_lines
                    if isinstance(v, str) and v.strip()
                )
    if body_reid_detail is not None:
        story.append(Paragraph(f"Body Re-ID (runtime):<br/>{body_reid_detail}", body_style))

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

    fusion_mode_value = "IoU-only"
    fusion_mode_notes = "Re-ID disabled or unavailable"
    try:
        reid_cfg_enabled = bool(track_fusion_config.get("reid_handoff", {}).get("enabled", False))
    except Exception:
        reid_cfg_enabled = False
    body_reid = run_body_tracking_marker.get("body_reid") if isinstance(run_body_tracking_marker, dict) else None
    if not reid_cfg_enabled:
        fusion_mode_value = "IoU-only"
        fusion_mode_notes = "reid_handoff.enabled=false"
    elif isinstance(body_reid, dict) and body_reid.get("enabled_effective") is True:
        fusion_mode_value = "Hybrid (IoU + Re-ID)"
        fusion_mode_notes = "Re-ID enabled_effective=true"
    else:
        skip_reason = body_reid.get("reid_skip_reason") if isinstance(body_reid, dict) else None
        torchreid_err = body_reid.get("torchreid_runtime_error") if isinstance(body_reid, dict) else None
        if isinstance(torchreid_err, str) and "missing torchreid.utils" in torchreid_err:
            fusion_mode_notes = (
                "Re-ID unavailable: missing torchreid.utils "
                "(install deep-person-reid; pip install -r requirements-ml.txt)"
            )
        else:
            parts: list[str] = ["reid_handoff.enabled=true", "enabled_effective=false"]
            if isinstance(skip_reason, str) and skip_reason.strip():
                parts.append(f"skip_reason={skip_reason.strip()}")
            if isinstance(torchreid_err, str) and torchreid_err.strip():
                err_str = torchreid_err.strip()
                if len(err_str) > 120:
                    err_str = err_str[:117] + "..."
                parts.append(f"torchreid_error={err_str}")
            fusion_mode_notes = "; ".join(parts)

    fusion_stats = [
        _wrap_row(["Metric", "Value", "Notes"]),
        _wrap_row(["Fusion Status", fusion_status, "Run-scoped inputs + outputs"]),
        _wrap_row(["Fusion Mode", fusion_mode_value, fusion_mode_notes]),
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
        candidate_overlaps = _safe_int_opt(fusion_diag_payload.get("candidate_overlaps"))
        overlap_ratio_pass = _safe_int_opt(fusion_diag_payload.get("overlap_ratio_pass"))
        iou_pass_count = _safe_int_opt(fusion_diag_payload.get("iou_pass"))
        iou_pairs_count = _safe_int_opt(fusion_diag_payload.get("iou_pairs"))
        reid_pairs_count = _safe_int_opt(fusion_diag_payload.get("reid_pairs"))
        hybrid_pairs_count = _safe_int_opt(fusion_diag_payload.get("hybrid_pairs"))
        final_pairs_count = _safe_int_opt(fusion_diag_payload.get("final_pairs"))
        reid_comparisons = _safe_int_opt(fusion_diag_payload.get("reid_comparisons"))
        reid_pass = _safe_int_opt(fusion_diag_payload.get("reid_pass"))

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
        reid_skip_reason = fusion_diag_payload.get("reid_skip_reason")
        if (
            reid_comparisons == 0
            and isinstance(reid_skip_reason, str)
            and reid_skip_reason.strip()
        ):
            reid_notes = f"{reid_notes}; skip_reason={reid_skip_reason.strip()}"
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
    # SECTION 8: FACES REVIEW (DB State) + SECTION 9: SMART SUGGESTIONS
    # =========================================================================
    if include_faces_review:
        story.append(Paragraph("8. Faces Review (DB State)", section_style))
        assigned_count = (
            sum(1 for identity in identities_list if identity.get("person_id"))
            if identities_count_run is not None
            else None
        )
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
        if db_connected is not True:
            locked_count_str = "unavailable (DB not configured)" if db_connected is None else "unavailable (DB error)"
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

        if db_connected is False:
            story.append(Paragraph(f"⚠️ <b>DB Error:</b> {db_error}", warning_style))
            story.append(
                Paragraph(
                    "<b>Impact:</b> DB-sourced counts (locks, batches, suggestions) are unavailable in this report due to DB connection error.",
                    note_style,
                )
            )
            manual_assignments = identities_data.get("manual_assignments")
            manual_assignments_count = len(manual_assignments) if isinstance(manual_assignments, dict) else 0
            if manual_assignments_count > 0:
                story.append(Paragraph(
                    f"Manual assignments loaded from identities.json fallback: <b>{manual_assignments_count}</b>",
                    note_style
                ))
        elif db_connected is None:
            story.append(
                Paragraph(
                    f"<b>DB Not Configured:</b> {db_not_configured_reason or 'missing DB_URL'} (DB-sourced state is omitted in this report).",
                    note_style,
                )
            )

        story.append(Paragraph("Data Sources:", subsection_style))
        story.append(Paragraph("&bull; identity_locks table (DB)", bullet_style))
        story.append(Paragraph("&bull; identities.json (manual_assignments)", bullet_style))

        story.append(Paragraph("9. Smart Suggestions", section_style))

        # Show "unavailable" for DB-sourced data when DB is not connected
        if db_connected is not True:
            suggestion_stats = [
                ["Metric", "Value"],
                [
                    "Suggestion Batches",
                    "unavailable (DB not configured)" if db_connected is None else "unavailable (DB error)",
                ],
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
    else:
        story.append(Paragraph("8. Out-of-Scope Sections", section_style))
        story.append(
            Paragraph(
                "Faces Review (DB state) and Smart Suggestions are out of scope for setup-only exports.",
                note_style,
            )
        )
        out_of_scope_items: list[str] = []
        if screentime_comparison_path.exists():
            out_of_scope_items.append("body_tracking/screentime_comparison.json")
        analytics_dir = run_root / "analytics"
        for filename in ("screentime.json", "screentime.csv"):
            if (analytics_dir / filename).exists():
                out_of_scope_items.append(f"analytics/{filename}")
        if out_of_scope_items:
            story.append(Paragraph("Out-of-scope artifacts present:", subsection_style))
            for item in out_of_scope_items:
                story.append(Paragraph(f"&bull; {item}", bullet_style))

    # =========================================================================
    if include_screentime:
        # SECTION 10: SCREEN TIME ANALYZE
        # =========================================================================
        story.append(Paragraph("10. Screen Time Analyze", section_style))
    
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
    
            breakdowns = screentime_payload.get("breakdowns") if isinstance(screentime_payload, dict) else None
            if isinstance(breakdowns, list) and breakdowns:
                contributors: list[dict[str, Any]] = []
                for entry in breakdowns:
                    if not isinstance(entry, dict):
                        continue
                    delta = entry.get("delta") if isinstance(entry.get("delta"), dict) else {}
                    gain = _safe_float_opt(delta.get("duration_gain"))
                    if gain is None or gain <= 0:
                        continue
                    contributors.append(entry)
    
                contributors.sort(
                    key=lambda item: _safe_float_opt((item.get("delta") or {}).get("duration_gain")) or 0.0,
                    reverse=True,
                )
                contributors = contributors[:5]
    
                if contributors:
                    story.append(Spacer(1, 8))
                    story.append(Paragraph("Top Contributors to Screen Time Gain", subsection_style))
                    contrib_rows = [_wrap_row(["Identity", "Gain", "Body-only segments"], cell_style_small)]
                    for entry in contributors:
                        identity_id = str(entry.get("identity_id") or "unknown")
                        delta = entry.get("delta") if isinstance(entry.get("delta"), dict) else {}
                        gain = _safe_float_opt(delta.get("duration_gain")) or 0.0
                        body_only_segments = entry.get("body_only_segments")
                        if not isinstance(body_only_segments, list):
                            segments_block = entry.get("segments") if isinstance(entry.get("segments"), dict) else {}
                            body_only_segments = segments_block.get("body_only") if isinstance(segments_block, dict) else None
                        segment_texts: list[str] = []
                        if isinstance(body_only_segments, list):
                            for seg in body_only_segments[:3]:
                                if not isinstance(seg, dict):
                                    continue
                                start_time = _safe_float_opt(seg.get("start_time"))
                                end_time = _safe_float_opt(seg.get("end_time"))
                                duration = _safe_float_opt(seg.get("duration"))
                                if start_time is None or end_time is None:
                                    continue
                                if duration is None:
                                    duration = max(end_time - start_time, 0.0)
                                segment_texts.append(f"{start_time:.2f}-{end_time:.2f}s ({duration:.2f}s)")
                        segments_label = ", ".join(segment_texts) if segment_texts else "N/A"
                        if isinstance(body_only_segments, list) and len(body_only_segments) > 3:
                            segments_label = f"{segments_label} (+{len(body_only_segments) - 3} more)"
                        contrib_rows.append(
                            _wrap_row([identity_id, f"+{gain:.2f}s", segments_label], cell_style_small)
                        )
    
                    contrib_table = Table(contrib_rows, colWidths=[1.2 * inch, 0.9 * inch, 4.4 * inch])
                    contrib_table.setStyle(
                        TableStyle([
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, -1), 8),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("LEFTPADDING", (0, 0), (-1, -1), 4),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
                        ])
                    )
                    story.append(contrib_table)
    
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
        gate_splits_share_low = False
        if gate_splits_share is not None and gate_splits_share_min is not None:
            gate_splits_share_low = gate_splits_share < gate_splits_share_min
        if gate_auto_rerun_reason == "gate_share_too_low" or gate_splits_share_low or marker_gate_enabled is False:
            share_str = f"{gate_splits_share:.3f}" if gate_splits_share is not None else "unknown"
            tuning_suggestions.append((
                "Face Tracking",
                f"High forced splits ({forced_splits})",
                f"Auto-rerun skipped (reason={gate_auto_rerun_reason or 'unknown'}, gate_splits_share={share_str}); "
                "tune track_buffer/match_thresh and scene-cut/warmup settings (not gate_enabled).",
            ))
        else:
            tuning_suggestions.append((
                "Face Tracking",
                f"High forced splits ({forced_splits})",
                "Disable gate_enabled in tracking.yaml or adjust appearance thresholds to reduce false splits",
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

    # Comparison tuning
    if duration_gain == 0 and num_body_tracks > 0:
        tuning_stage = "Screen Time" if include_screentime else "Track Fusion"
        tuning_suggestions.append((
            tuning_stage,
            "No duration gain from fusion comparison",
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

    story.append(Paragraph("Artifact Provenance (run-scoped)", subsection_style))
    provenance_targets: list[tuple[str, str]] = [
        ("episode_status.json", "episode_status.json"),
        ("detections.jsonl", "detections.jsonl"),
        ("tracks.jsonl", "tracks.jsonl"),
        ("track_metrics.json", "track_metrics.json"),
        ("faces.jsonl", "faces.jsonl"),
        ("face_alignment/aligned_faces.jsonl", "face_alignment/aligned_faces.jsonl"),
        ("identities.json", "identities.json"),
        ("cluster_centroids.json", "cluster_centroids.json"),
        ("body_tracking/body_detections.jsonl", "body_tracking/body_detections.jsonl"),
        ("body_tracking/body_tracks.jsonl", "body_tracking/body_tracks.jsonl"),
        ("body_tracking/track_fusion.json", "body_tracking/track_fusion.json"),
        ("body_tracking/screentime_comparison.json", "body_tracking/screentime_comparison.json"),
        ("exports/export_index.json", "exports/export_index.json"),
    ]
    if include_screentime:
        provenance_targets.insert(-1, ("analytics/screentime.json", "analytics/screentime.json"))
    provenance_rows = [_wrap_row(["Artifact", "Scope", "Source", "Local mtime", "Hydrated mtime", "S3 key"], cell_style_small)]
    for label, rel_name in provenance_targets:
        local_path = run_root / rel_name
        hydrated_path = hydrated_paths.get(rel_name)
        source = "hydrated" if hydrated_path else ("local" if local_path.exists() else "missing")
        local_mtime = _format_mtime(local_path) if local_path.exists() else "N/A"
        hydrated_mtime = _format_mtime(hydrated_path) if hydrated_path else "—"
        s3_key = hydrated_s3_keys.get(rel_name) or "—"
        provenance_rows.append(
            _wrap_row([label, "run", source, local_mtime, hydrated_mtime, s3_key], cell_style_small)
        )

    provenance_table = Table(
        provenance_rows,
        colWidths=[2.1 * inch, 0.6 * inch, 0.7 * inch, 1.1 * inch, 1.1 * inch, 1.4 * inch],
    )
    provenance_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ])
    )
    story.append(provenance_table)
    story.append(Spacer(1, 8))
    if not include_screentime:
        out_of_scope_paths: list[tuple[str, Path]] = []
        run_analytics_dir = run_root / "analytics"
        for filename in ("screentime.json", "screentime.csv"):
            candidate = run_analytics_dir / filename
            if candidate.exists():
                out_of_scope_paths.append((f"run analytics/{filename}", candidate))
        data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
        legacy_analytics_dir = data_root / "analytics" / ep_id
        for filename in ("screentime.json", "screentime.csv"):
            candidate = legacy_analytics_dir / filename
            if candidate.exists():
                out_of_scope_paths.append((f"legacy analytics/{filename}", candidate))
        story.append(Paragraph("Out-of-scope artifacts present", subsection_style))
        if out_of_scope_paths:
            for label, path in out_of_scope_paths:
                story.append(Paragraph(f"&bull; {label} ({path})", bullet_style))
        else:
            story.append(Paragraph("None detected.", note_style))
        story.append(Spacer(1, 8))
    story.append(Paragraph("Legacy artifacts are listed in the section below as out-of-scope references.", note_style))
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
        (*_artifact_row(screentime_comparison_path, "body_tracking/screentime_comparison.json"), "-", comparison_stage_label, _bundle_status(screentime_comparison_path, in_allowlist=True)),
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
            (*_artifact_row(
                legacy_screentime_comparison_path,
                "legacy/body_tracking/screentime_comparison.json",
            ), "-", f"Legacy {comparison_label}", _bundle_status(legacy_screentime_comparison_path, in_allowlist=False)),
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
    include_screentime: bool = True,
    upload_to_s3: bool = True,
    fail_on_s3_error: bool = False,
    write_index: bool = True,
) -> tuple[bytes, str, ExportUploadResult | None]:
    """Build debug PDF and optionally upload to S3.

    Args:
        ep_id: Episode ID
        run_id: Run ID
        include_screentime: Whether to include screentime sections in the report
        upload_to_s3: Whether to upload to S3 (if configured)
        fail_on_s3_error: If True, raise on S3 upload failure
        write_index: If True, write export_index.json marker file

    Returns:
        (pdf_bytes, download_filename, upload_result or None)
    """
    started_at = _now_iso()
    run_id_norm = run_layout.normalize_run_id(run_id)
    heartbeat_interval = 5.0
    last_tick = 0.0
    frames_done_at: str | None = None
    finalize_started_at: str | None = None
    ended_at: str | None = None
    total_steps = 1
    gate = check_prereqs("pdf", ep_id, run_id)
    if not gate.ok:
        reasons = gate.reasons or []
        primary = reasons[0] if reasons else None
        blocked_reason = BlockedReason(
            code=primary.code if primary else "blocked",
            message=primary.message if primary else "Stage blocked by prerequisites",
            details={
                "reasons": [reason.as_dict() for reason in reasons],
                "suggested_actions": list(gate.suggested_actions),
            },
        )
        blocked_info = StageBlockedInfo(reasons=list(reasons), suggested_actions=list(gate.suggested_actions))
        should_block = blocked_update_needed(ep_id, run_id, "pdf", blocked_reason)
        if should_block:
            try:
                write_stage_blocked(ep_id, run_id, "pdf", blocked_reason)
            except Exception as exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[export] Failed to mark PDF blocked: %s", exc)
            try:
                append_log(
                    ep_id,
                    run_id,
                    "pdf",
                    "WARNING",
                    "stage blocked",
                    progress=0.0,
                    meta={
                        "reason_code": blocked_reason.code,
                        "reason_message": blocked_reason.message,
                        "suggested_actions": list(gate.suggested_actions),
                    },
                )
            except Exception as exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to log PDF blocked: %s", exc)
            try:
                write_stage_manifest(
                    ep_id,
                    run_id,
                    "pdf",
                    "BLOCKED",
                    started_at=started_at,
                    finished_at=_now_iso(),
                    duration_s=None,
                    blocked=blocked_info,
                )
            except Exception as exc:  # pragma: no cover - best effort manifest write
                LOGGER.warning("[export] Failed to write PDF blocked manifest: %s", exc)
        raise RuntimeError(blocked_reason.message)
    try:
        try:
            append_log(ep_id, run_id, "pdf", "INFO", "stage started", progress=0.0)
        except Exception as exc:  # pragma: no cover - best effort log write
            LOGGER.debug("[run_logs] Failed to log PDF start: %s", exc)
        write_stage_started(
            ep_id,
            run_id,
            "pdf",
            started_at=datetime.fromisoformat(started_at.replace("Z", "+00:00")),
        )
    except Exception as exc:  # pragma: no cover - best effort status update
        LOGGER.warning("[export] Failed to mark PDF start: %s", exc)

    def _emit_pdf_progress(
        *,
        done: int,
        phase: str,
        message: str,
        force: bool = False,
        mark_frames_done: bool = False,
        mark_finalize_start: bool = False,
        mark_end: bool = False,
    ) -> None:
        nonlocal last_tick, frames_done_at, finalize_started_at, ended_at
        now = time.time()
        if not force and (now - last_tick) < heartbeat_interval:
            return
        stamp = _now_iso()
        if mark_frames_done and frames_done_at is None:
            frames_done_at = stamp
        if mark_finalize_start and finalize_started_at is None:
            finalize_started_at = stamp
        if mark_end:
            ended_at = stamp
        progress_payload = {
            "done": max(int(done), 0),
            "total": total_steps,
            "pct": min(max(float(done) / max(total_steps, 1), 0.0), 1.0),
            "phase": phase,
            "message": message,
            "last_update_at": stamp,
        }
        timestamps_payload = {
            "started_at": started_at,
            "frames_done_at": frames_done_at,
            "finalize_started_at": finalize_started_at,
            "ended_at": ended_at,
        }
        stage_update = {
            "progress": progress_payload,
            "timestamps": timestamps_payload,
        }
        try:
            update_episode_status(
                ep_id,
                run_id,
                stage_key="pdf",
                stage_update=stage_update,
                git_info=collect_git_state(),
            )
        except Exception as exc:
            LOGGER.warning("[export] Failed to update PDF heartbeat status: %s", exc)
        last_tick = now

    def _run_with_heartbeat(action, *, phase: str, message: str, done: int, mark_finalize: bool = False):
        def _heartbeat():
            _emit_pdf_progress(
                done=done,
                phase=phase,
                message=message,
                mark_frames_done=mark_finalize,
                mark_finalize_start=mark_finalize,
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(action)
            while True:
                try:
                    return future.result(timeout=heartbeat_interval)
                except TimeoutError:
                    _heartbeat()

    _emit_pdf_progress(done=0, phase="running", message="Building debug PDF", force=True)
    try:
        pdf_bytes, download_name = _run_with_heartbeat(
            lambda: (
                generate_run_debug_pdf(ep_id, run_id_norm).read_bytes(),
                f"screenalytics_{ep_id}_{run_id_norm}_debug_report.pdf",
            ),
            phase="running",
            message="Building debug PDF",
            done=0,
        )
    except Exception as exc:
        _emit_pdf_progress(
            done=0,
            phase="error",
            message=f"PDF export failed: {exc}",
            force=True,
            mark_end=True,
        )
        try:
            write_stage_failed(
                ep_id,
                run_id,
                "pdf",
                error_code=type(exc).__name__,
                error_message=str(exc),
                artifact_paths={
                    entry["label"]: entry["path"]
                    for entry in stage_artifacts(ep_id, run_id, "pdf")
                    if isinstance(entry, dict) and entry.get("exists")
                },
            )
        except Exception as status_exc:
            LOGGER.warning("[export] Failed to update episode_status.json for PDF error: %s", status_exc)
        try:
            append_log(
                ep_id,
                run_id,
                "pdf",
                "ERROR",
                "stage failed",
                progress=100.0,
                meta={"error_code": type(exc).__name__, "error_message": str(exc)},
            )
        except Exception as log_exc:
            LOGGER.debug("[run_logs] Failed to log PDF failure: %s", log_exc)
        try:
            write_stage_manifest(
                ep_id,
                run_id,
                "pdf",
                "FAILED",
                started_at=started_at,
                finished_at=_now_iso(),
                duration_s=None,
                error=StageErrorInfo(code=type(exc).__name__, message=str(exc)),
                artifacts={
                    entry["label"]: entry["path"]
                    for entry in stage_artifacts(ep_id, run_id, "pdf")
                    if isinstance(entry, dict) and entry.get("exists")
                },
            )
        except Exception as manifest_exc:
            LOGGER.warning("[export] Failed to write PDF failed manifest: %s", manifest_exc)
        raise

    upload_result = None
    _emit_pdf_progress(
        done=1,
        phase="finalizing",
        message="Uploading debug PDF",
        force=True,
        mark_frames_done=True,
        mark_finalize_start=True,
    )
    if upload_to_s3:
        upload_result = _run_with_heartbeat(
            lambda: upload_export_to_s3(
                ep_id=ep_id,
                run_id=run_id,
                file_bytes=pdf_bytes,
                filename="debug_report.pdf",
                content_type="application/pdf",
                fail_on_error=fail_on_s3_error,
            ),
            phase="finalizing",
            message="Uploading debug PDF",
            done=1,
            mark_finalize=True,
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

    _emit_pdf_progress(
        done=1,
        phase="done",
        message="PDF export complete",
        force=True,
        mark_end=True,
    )

    try:
        write_stage_finished(
            ep_id,
            run_id,
            "pdf",
            artifact_paths={
                entry["label"]: entry["path"]
                for entry in stage_artifacts(ep_id, run_id, "pdf")
                if isinstance(entry, dict) and entry.get("exists")
            },
            metrics={
                "export_bytes": len(pdf_bytes),
                "export_key": upload_result.s3_key if upload_result else None,
                "upload_success": upload_result.success if upload_result else None,
            },
        )
    except Exception as exc:
        LOGGER.warning("[export] Failed to update episode_status.json for PDF success: %s", exc)
    try:
        append_log(
            ep_id,
            run_id,
            "pdf",
            "INFO",
            "stage finished",
            progress=100.0,
            meta={"export_bytes": len(pdf_bytes)},
        )
    except Exception as log_exc:
        LOGGER.debug("[run_logs] Failed to log PDF success: %s", log_exc)
    try:
        write_stage_manifest(
            ep_id,
            run_id,
            "pdf",
            "SUCCESS",
            started_at=started_at,
            finished_at=_now_iso(),
            duration_s=None,
            counts={"export_bytes": len(pdf_bytes)},
            artifacts={
                entry["label"]: entry["path"]
                for entry in stage_artifacts(ep_id, run_id, "pdf")
                if isinstance(entry, dict) and entry.get("exists")
            },
        )
    except Exception as exc:
        LOGGER.warning("[export] Failed to write PDF success manifest: %s", exc)

    return pdf_bytes, download_name, upload_result


def _segments_source_path(ep_id: str, run_id: str) -> Path:
    run_root = run_layout.run_root(ep_id, run_layout.normalize_run_id(run_id))
    return run_root / "body_tracking" / "screentime_comparison.json"


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _build_segments_dataframe(ep_id: str, run_id: str) -> tuple[Any, dict[str, Any]]:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas is required for segments.parquet export (pip install pandas pyarrow)") from exc

    run_id_norm = run_layout.normalize_run_id(run_id)
    source_path = _segments_source_path(ep_id, run_id_norm)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing screentime comparison: {source_path}")
    payload = _read_json(source_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Unreadable screentime comparison payload: {source_path}")

    breakdowns = payload.get("breakdowns") or []
    if not isinstance(breakdowns, list):
        raise ValueError(f"Invalid screentime comparison breakdowns: {source_path}")

    manifest = (
        read_stage_manifest(ep_id, run_id_norm, "screentime")
        or read_stage_manifest(ep_id, run_id_norm, "track_fusion")
        or {}
    )
    model_versions = manifest.get("model_versions") if isinstance(manifest.get("model_versions"), dict) else {}
    thresholds = manifest.get("thresholds") if isinstance(manifest.get("thresholds"), dict) else {}
    model_versions_json = _json_dumps_sorted(model_versions) if model_versions else None
    thresholds_hash = _hash_snapshot(thresholds)

    rows: list[dict[str, Any]] = []
    identity_ids: set[str] = set()
    for breakdown in breakdowns:
        if not isinstance(breakdown, dict):
            continue
        identity_id = breakdown.get("identity_id")
        identity_label = breakdown.get("identity")
        if identity_id is not None:
            identity_ids.add(str(identity_id))
        segments = breakdown.get("body_only_segments") or []
        if not isinstance(segments, list):
            raise ValueError(f"Invalid body_only_segments for identity {identity_id}")

        observed_duration = 0.0
        valid_segments = 0
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            start_time = _coerce_float(seg.get("start_time"))
            end_time = _coerce_float(seg.get("end_time"))
            duration = _coerce_float(seg.get("duration"))
            if duration is None and start_time is not None and end_time is not None:
                duration = end_time - start_time
            if start_time is None or end_time is None or duration is None:
                raise ValueError(f"Invalid segment timing for identity {identity_id}")
            source = seg.get("segment_type") or "body_only"
            rows.append(
                {
                    "run_id": run_id_norm,
                    "episode_id": ep_id,
                    "model_versions": model_versions_json,
                    "identity": identity_label,
                    "identity_id": str(identity_id) if identity_id is not None else None,
                    "track_id": None,
                    "segment_start": float(start_time),
                    "segment_end": float(end_time),
                    "duration_s": float(duration),
                    "confidence": None,
                    "source": str(source),
                    "thresholds_snapshot_hash": thresholds_hash,
                }
            )
            observed_duration += float(duration)
            valid_segments += 1

        expected = None
        breakdown_block = breakdown.get("breakdown")
        if isinstance(breakdown_block, dict):
            expected = _coerce_float(breakdown_block.get("body_only_duration"))
        if expected is not None:
            tolerance = 0.1 * max(valid_segments, 1)
            if abs(observed_duration - expected) > tolerance:
                raise ValueError(
                    f"Segment duration mismatch for {identity_id}: observed={observed_duration:.3f}s "
                    f"expected={expected:.3f}s tolerance={tolerance:.3f}s"
                )

    columns = [
        "run_id",
        "episode_id",
        "model_versions",
        "identity",
        "identity_id",
        "track_id",
        "segment_start",
        "segment_end",
        "duration_s",
        "confidence",
        "source",
        "thresholds_snapshot_hash",
    ]
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df = df.sort_values(
            ["identity_id", "track_id", "segment_start", "segment_end"],
            na_position="last",
            kind="mergesort",
        )

    meta = {
        "segment_count": len(rows),
        "identity_count": len(identity_ids),
        "model_versions": model_versions,
        "thresholds": thresholds,
        "thresholds_snapshot_hash": thresholds_hash,
        "source_path": str(source_path),
    }
    return df, meta


def export_segments_parquet(
    episode_id: str,
    run_id: str,
    *,
    output_path: str | Path | None = None,
) -> Path:
    df, _meta = _build_segments_dataframe(episode_id, run_id)
    run_id_norm = run_layout.normalize_run_id(run_id)
    path = (
        Path(output_path)
        if output_path
        else run_layout.run_root(episode_id, run_id_norm) / "exports" / "segments.parquet"
    )
    return _write_segments_parquet(df, path)


def _write_segments_parquet(df: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def run_segments_export(
    *,
    ep_id: str,
    run_id: str,
) -> Path | None:
    started_at = _now_iso()
    run_id_norm = run_layout.normalize_run_id(run_id)
    try:
        append_log(ep_id, run_id_norm, "segments", "INFO", "stage started", progress=0.0)
    except Exception as exc:  # pragma: no cover - best effort log write
        LOGGER.debug("[run_logs] Failed to log segments start: %s", exc)

    try:
        write_stage_started(ep_id, run_id_norm, "segments")
    except Exception as exc:  # pragma: no cover - best effort status update
        LOGGER.warning("[export] Failed to mark segments start: %s", exc)

    try:
        df, meta = _build_segments_dataframe(ep_id, run_id_norm)
    except FileNotFoundError as exc:
        blocked_reason = BlockedReason(
            code="missing_artifact",
            message="segments source missing",
            details={"expected_path": str(_segments_source_path(ep_id, run_id_norm))},
        )
        blocked_info = StageBlockedInfo(
            reasons=[
                GateReason(
                    code=blocked_reason.code,
                    message=blocked_reason.message,
                    details=blocked_reason.details,
                )
            ],
            suggested_actions=["Run track_fusion to generate screentime_comparison.json."],
        )
        should_block = blocked_update_needed(ep_id, run_id_norm, "segments", blocked_reason)
        if should_block:
            try:
                write_stage_blocked(ep_id, run_id_norm, "segments", blocked_reason)
            except Exception as status_exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[export] Failed to mark segments blocked: %s", status_exc)
            try:
                append_log(
                    ep_id,
                    run_id_norm,
                    "segments",
                    "WARNING",
                    "stage blocked",
                    progress=0.0,
                    meta={"reason_code": blocked_reason.code, "reason_message": blocked_reason.message},
                )
            except Exception as log_exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to log segments blocked: %s", log_exc)
            try:
                write_stage_manifest(
                    ep_id,
                    run_id_norm,
                    "segments",
                    "BLOCKED",
                    started_at=started_at,
                    finished_at=_now_iso(),
                    duration_s=None,
                    blocked=blocked_info,
                )
            except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                LOGGER.warning("[export] Failed to write segments blocked manifest: %s", manifest_exc)
        LOGGER.info("[export] segments.parquet blocked: %s", exc)
        return None
    except Exception as exc:
        try:
            write_stage_failed(
                ep_id,
                run_id_norm,
                "segments",
                error_code=type(exc).__name__,
                error_message=str(exc),
            )
        except Exception as status_exc:  # pragma: no cover - best effort status update
            LOGGER.warning("[export] Failed to mark segments failed: %s", status_exc)
        try:
            append_log(
                ep_id,
                run_id_norm,
                "segments",
                "ERROR",
                "stage failed",
                progress=100.0,
                meta={"error_code": type(exc).__name__, "error_message": str(exc)},
            )
        except Exception as log_exc:  # pragma: no cover - best effort log write
            LOGGER.debug("[run_logs] Failed to log segments failed: %s", log_exc)
        try:
            write_stage_manifest(
                ep_id,
                run_id_norm,
                "segments",
                "FAILED",
                started_at=started_at,
                finished_at=_now_iso(),
                duration_s=None,
                error=StageErrorInfo(code=type(exc).__name__, message=str(exc)),
            )
        except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
            LOGGER.warning("[export] Failed to write segments failed manifest: %s", manifest_exc)
        raise

    try:
        output_path = _write_segments_parquet(
            df,
            run_layout.run_root(ep_id, run_id_norm) / "exports" / "segments.parquet",
        )
    except Exception as exc:
        try:
            write_stage_failed(
                ep_id,
                run_id_norm,
                "segments",
                error_code=type(exc).__name__,
                error_message=str(exc),
            )
        except Exception as status_exc:  # pragma: no cover - best effort status update
            LOGGER.warning("[export] Failed to mark segments failed: %s", status_exc)
        try:
            append_log(
                ep_id,
                run_id_norm,
                "segments",
                "ERROR",
                "stage failed",
                progress=100.0,
                meta={"error_code": type(exc).__name__, "error_message": str(exc)},
            )
        except Exception as log_exc:  # pragma: no cover - best effort log write
            LOGGER.debug("[run_logs] Failed to log segments failed: %s", log_exc)
        try:
            write_stage_manifest(
                ep_id,
                run_id_norm,
                "segments",
                "FAILED",
                started_at=started_at,
                finished_at=_now_iso(),
                duration_s=None,
                error=StageErrorInfo(code=type(exc).__name__, message=str(exc)),
            )
        except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
            LOGGER.warning("[export] Failed to write segments failed manifest: %s", manifest_exc)
        raise

    counts = {
        "segments": int(meta.get("segment_count", 0) or 0),
        "identities": int(meta.get("identity_count", 0) or 0),
    }
    try:
        write_stage_finished(
            ep_id,
            run_id_norm,
            "segments",
            artifact_paths={"segments.parquet": str(output_path)},
            counts=counts,
        )
    except Exception as status_exc:  # pragma: no cover - best effort status update
        LOGGER.warning("[export] Failed to mark segments success: %s", status_exc)
    try:
        append_log(
            ep_id,
            run_id_norm,
            "segments",
            "INFO",
            "stage finished",
            progress=100.0,
            meta={"segments": counts["segments"], "identities": counts["identities"]},
        )
    except Exception as log_exc:  # pragma: no cover - best effort log write
        LOGGER.debug("[run_logs] Failed to log segments success: %s", log_exc)
    try:
        write_stage_manifest(
            ep_id,
            run_id_norm,
            "segments",
            "SUCCESS",
            started_at=started_at,
            finished_at=_now_iso(),
            duration_s=None,
            counts=counts,
            artifacts={"exports/segments.parquet": str(output_path)},
            model_versions=meta.get("model_versions"),
            thresholds=meta.get("thresholds"),
        )
    except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
        LOGGER.warning("[export] Failed to write segments success manifest: %s", manifest_exc)

    return output_path


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

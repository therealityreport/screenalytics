from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from py_screenalytics import run_layout

PIPELINE_STAGE_PLAN: tuple[str, ...] = (
    "detect",
    "faces",
    "cluster",
    "body_tracking",
    "track_fusion",
    "pdf",
)

PRIMARY_STAGE_CARD_KEYS: tuple[str, ...] = (
    "detect",
    "faces",
    "cluster",
    "body_tracking",
    "track_fusion",
    "pdf",
)

ACTIVE_STAGE_KEYS: tuple[str, ...] = PIPELINE_STAGE_PLAN

STAGE_LABELS: dict[str, str] = {
    "detect": "Detect/Track",
    "faces": "Faces Harvest",
    "cluster": "Cluster",
    "body_tracking": "Body Tracking",
    "track_fusion": "Track Fusion",
    "pdf": "PDF Export",
}

_STAGE_KEY_ALIASES: dict[str, str] = {
    "detect": "detect",
    "detect/track": "detect",
    "detect track": "detect",
    "detect_track": "detect",
    "faces": "faces",
    "faces harvest": "faces",
    "faces_embed": "faces",
    "face harvest": "faces",
    "cluster": "cluster",
    "clustering": "cluster",
    "body tracking": "body_tracking",
    "body_tracking": "body_tracking",
    "body tracking fusion": "track_fusion",
    "body_tracking_fusion": "track_fusion",
    "track fusion": "track_fusion",
    "track_fusion": "track_fusion",
    "pdf": "pdf",
    "pdf export": "pdf",
    "export pdf": "pdf",
}


@dataclass(frozen=True)
class StageCardLayout:
    primary_stage_keys: tuple[str, ...]
    show_active_downstream_panel: bool
    active_downstream_stage: str | None


@dataclass(frozen=True)
class ArtifactPresence:
    local: bool
    remote: bool
    path: str | None = None
    s3_key: str | None = None


@dataclass(frozen=True)
class RunDebugPdfInfo:
    episode_id: str
    run_id: str
    local_path: Path
    exists: bool
    export_index: dict[str, Any] | None = None
    s3_key: str | None = None


def normalize_stage_key(raw: str | None) -> str | None:
    if not raw:
        return None
    label = raw.strip().lower()
    for delim in ("(", ":", "-", "|"):
        if delim in label:
            label = label.split(delim, 1)[0].strip()
    label = label.replace("_", " ")
    label = " ".join(label.split())
    return _STAGE_KEY_ALIASES.get(label)


def resolve_stage_key(stage_key: str | None, available: Iterable[str]) -> str | None:
    if not stage_key:
        return None
    normalized = normalize_stage_key(stage_key) or stage_key
    for candidate in available:
        if candidate == normalized:
            return candidate
    for candidate in available:
        if normalize_stage_key(candidate) == normalized:
            return candidate
    return None


def canonical_status_from_entry(entry: Mapping[str, Any] | None) -> str | None:
    if not entry:
        return None
    status_value = entry.get("status")
    derived = bool(entry.get("derived") or entry.get("is_derived"))
    derived_paths = entry.get("derived_from") or entry.get("artifact_paths")
    has_evidence = bool(derived_paths)
    if not status_value:
        if derived and has_evidence:
            return "success"
        return None
    normalized = str(status_value).strip().lower()
    if normalized in {"not_started", "missing", "unknown"}:
        if derived and has_evidence:
            return "success"
        return "missing"
    if normalized in {"error"}:
        return "failed"
    return normalized


def resolve_run_debug_pdf(
    ep_id: str,
    run_id: str,
    stage_entry: Mapping[str, Any] | None = None,
) -> RunDebugPdfInfo:
    run_id_norm = run_layout.normalize_run_id(run_id)
    run_root = run_layout.run_root(ep_id, run_id_norm)
    local_path = run_root / "exports" / "run_debug.pdf"
    export_index: dict[str, Any] | None = None
    s3_key: str | None = None

    if stage_entry:
        artifact_paths = stage_entry.get("artifact_paths")
        if isinstance(artifact_paths, Mapping):
            candidate = artifact_paths.get("exports/run_debug.pdf")
            if isinstance(candidate, str) and candidate.strip():
                local_path = Path(candidate)

    index_path = run_root / "exports" / "export_index.json"
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            export_index = payload
            export_upload = payload.get("export_upload")
            if isinstance(export_upload, dict):
                s3_key = export_upload.get("s3_key")
            if not s3_key:
                s3_key = payload.get("export_s3_key")

    return RunDebugPdfInfo(
        episode_id=ep_id,
        run_id=run_id_norm,
        local_path=local_path,
        exists=local_path.exists(),
        export_index=export_index,
        s3_key=s3_key if isinstance(s3_key, str) and s3_key.strip() else None,
    )

def stage_label(stage_key: str | None) -> str:
    if not stage_key:
        return "Unknown"
    return STAGE_LABELS.get(stage_key, stage_key.replace("_", " ").title())


def normalize_completed_stages(stages: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for entry in stages:
        key = normalize_stage_key(entry)
        if key and key not in normalized:
            normalized.append(key)
    return tuple(normalized)


def progress_counts(
    completed_stages: Iterable[str],
    stage_plan: Sequence[str] = PIPELINE_STAGE_PLAN,
) -> tuple[int, int]:
    completed_keys = set(normalize_completed_stages(completed_stages))
    completed_in_plan = [stage for stage in stage_plan if stage in completed_keys]
    return len(completed_in_plan), len(stage_plan)


def artifact_available(presence: ArtifactPresence | None) -> bool:
    if not presence:
        return False
    return bool(presence.local or presence.remote)


def track_fusion_prereq_state(
    faces: ArtifactPresence | None,
    body_tracks: ArtifactPresence | None,
) -> tuple[bool, tuple[str, ...]]:
    missing: list[str] = []
    if not artifact_available(faces):
        missing.append("faces.jsonl")
    if not artifact_available(body_tracks):
        missing.append("body_tracking/body_tracks.jsonl")
    return (not missing, tuple(missing))


def downstream_stage_allows_advance(status: str, error_reason: str | None = None) -> bool:
    if status != "success":
        return False
    if error_reason in {"run_id_mismatch", "missing_artifacts"}:
        return False
    return True


def get_stage_card_layout(autorun_phase: str | None) -> StageCardLayout:
    autorun_key = normalize_stage_key(autorun_phase)
    show_panel = bool(autorun_key in ACTIVE_STAGE_KEYS)
    active_stage = autorun_key if show_panel else None
    return StageCardLayout(
        primary_stage_keys=PRIMARY_STAGE_CARD_KEYS,
        show_active_downstream_panel=show_panel,
        active_downstream_stage=active_stage,
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

PIPELINE_STAGE_PLAN: tuple[str, ...] = (
    "detect",
    "faces",
    "cluster",
    "body_tracking",
    "track_fusion",
    "screentime",
    "pdf",
)

PRIMARY_STAGE_CARD_KEYS: tuple[str, ...] = (
    "detect",
    "faces",
    "cluster",
    "body_tracking",
    "track_fusion",
)

DOWNSTREAM_STAGE_KEYS: tuple[str, ...] = (
    "body_tracking",
    "track_fusion",
    "screentime",
    "pdf",
)

STAGE_LABELS: dict[str, str] = {
    "detect": "Detect/Track",
    "faces": "Faces Harvest",
    "cluster": "Cluster",
    "body_tracking": "Body Tracking",
    "track_fusion": "Track Fusion",
    "screentime": "Screentime Analyze",
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
    "screentime": "screentime",
    "screentime analyze": "screentime",
    "screen time": "screentime",
    "screen time analyze": "screentime",
    "screen_time_analyze": "screentime",
    "screen_time": "screentime",
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
    show_panel = bool(autorun_key in DOWNSTREAM_STAGE_KEYS)
    active_stage = autorun_key if show_panel else None
    return StageCardLayout(
        primary_stage_keys=PRIMARY_STAGE_CARD_KEYS,
        show_active_downstream_panel=show_panel,
        active_downstream_stage=active_stage,
    )

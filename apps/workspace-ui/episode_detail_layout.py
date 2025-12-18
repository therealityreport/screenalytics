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
    "cluster": "Clustering",
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


def get_stage_card_layout(autorun_phase: str | None) -> StageCardLayout:
    autorun_key = normalize_stage_key(autorun_phase)
    show_panel = bool(autorun_key in DOWNSTREAM_STAGE_KEYS)
    active_stage = autorun_key if show_panel else None
    return StageCardLayout(
        primary_stage_keys=PRIMARY_STAGE_CARD_KEYS,
        show_active_downstream_panel=show_panel,
        active_downstream_stage=active_stage,
    )

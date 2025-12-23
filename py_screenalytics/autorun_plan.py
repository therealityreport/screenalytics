from __future__ import annotations

from typing import Mapping, Sequence


def build_autorun_stage_plan(
    *,
    body_tracking_enabled: bool | None = None,
    track_fusion_enabled: bool | None = None,
) -> list[str]:
    """Return the ordered Auto-Run stage plan for Episode Details setup."""
    _ = body_tracking_enabled, track_fusion_enabled
    return ["detect", "faces", "cluster", "body_tracking", "track_fusion", "pdf"]


def autorun_complete(stage_plan: Sequence[str], done_map: Mapping[str, bool]) -> bool:
    """Return True only when every stage in stage_plan is done."""
    return all(bool(done_map.get(stage)) for stage in stage_plan)

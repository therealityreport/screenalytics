from __future__ import annotations

from typing import Mapping, Sequence


def build_autorun_stage_plan() -> list[str]:
    """Return the ordered Auto-Run stage plan for Episode Details setup."""
    return ["detect", "faces", "cluster", "body_tracking", "track_fusion", "pdf"]


def autorun_complete(stage_plan: Sequence[str], done_map: Mapping[str, bool]) -> bool:
    """Return True only when every stage in stage_plan is done."""
    return all(bool(done_map.get(stage)) for stage in stage_plan)

from __future__ import annotations

from typing import Mapping, Sequence


def build_autorun_stage_plan(*, body_tracking_enabled: bool, track_fusion_enabled: bool) -> list[str]:
    """Return the ordered Auto-Run stage plan.

    Stage keys match Episode Details Auto-Run phases:
      detect -> faces -> cluster -> (body_tracking) -> (track_fusion) -> screentime -> pdf
    """
    stages = ["detect", "faces", "cluster"]
    if body_tracking_enabled:
        stages.append("body_tracking")
    if track_fusion_enabled:
        stages.append("track_fusion")
    stages.extend(["screentime", "pdf"])
    return stages


def autorun_complete(stage_plan: Sequence[str], done_map: Mapping[str, bool]) -> bool:
    """Return True only when every stage in stage_plan is done."""
    return all(bool(done_map.get(stage)) for stage in stage_plan)


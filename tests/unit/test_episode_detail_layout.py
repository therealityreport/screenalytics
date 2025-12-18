from __future__ import annotations

import sys
from pathlib import Path


def _load_layout_module():
    repo_root = Path(__file__).resolve().parents[2]
    workspace_ui = repo_root / "apps" / "workspace-ui"
    if str(workspace_ui) not in sys.path:
        sys.path.append(str(workspace_ui))
    import episode_detail_layout

    return episode_detail_layout


def test_stage_card_layout_includes_downstream_when_autorun_phase_downstream() -> None:
    layout = _load_layout_module()
    card_layout = layout.get_stage_card_layout("body_tracking")

    assert "body_tracking" in card_layout.primary_stage_keys
    assert "track_fusion" in card_layout.primary_stage_keys
    assert card_layout.show_active_downstream_panel is True
    assert card_layout.active_downstream_stage == "body_tracking"


def test_progress_counts_use_normalized_stage_keys_and_plan_order() -> None:
    layout = _load_layout_module()
    completed = [
        "Detect/Track (120 detections)",
        "Faces Harvest (80 faces)",
        "Clustering (12 identities)",
        "Screen Time Analyze",
        "Detect/Track (duplicate)",
    ]

    completed_count, total_count = layout.progress_counts(completed, layout.PIPELINE_STAGE_PLAN)

    assert completed_count == 4
    assert total_count == len(layout.PIPELINE_STAGE_PLAN)

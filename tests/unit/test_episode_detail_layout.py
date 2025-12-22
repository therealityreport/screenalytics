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


def test_track_fusion_prereqs_accept_remote_artifacts() -> None:
    layout = _load_layout_module()
    faces = layout.ArtifactPresence(local=True, remote=False)
    body_tracks = layout.ArtifactPresence(local=True, remote=False)

    ready, missing = layout.track_fusion_prereq_state(faces, body_tracks)

    assert ready is True
    assert missing == ()

    faces_remote = layout.ArtifactPresence(local=False, remote=True)
    body_tracks_remote = layout.ArtifactPresence(local=False, remote=True)
    ready_remote, missing_remote = layout.track_fusion_prereq_state(faces_remote, body_tracks_remote)

    assert ready_remote is True
    assert missing_remote == ()


def test_downstream_stage_gate_blocks_failed_or_wrong_run() -> None:
    layout = _load_layout_module()

    assert layout.downstream_stage_allows_advance("error") is False
    assert layout.downstream_stage_allows_advance("success", "run_id_mismatch") is False
    assert layout.downstream_stage_allows_advance("success") is True

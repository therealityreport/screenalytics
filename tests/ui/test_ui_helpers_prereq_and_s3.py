from __future__ import annotations

from tests.ui._helpers_loader import load_ui_helpers_module


def test_describe_prereq_state_waiting() -> None:
    helpers = load_ui_helpers_module()
    state, msg = helpers.describe_prereq_state(
        ["faces.jsonl", "body_tracking/body_tracks.jsonl"],
        upstream_complete=False,
    )

    assert state == "waiting"
    assert "Waiting for" in msg


def test_describe_prereq_state_error() -> None:
    helpers = load_ui_helpers_module()
    state, msg = helpers.describe_prereq_state(
        ["faces.jsonl"],
        upstream_complete=True,
    )

    assert state == "error"
    assert "Missing prerequisites" in msg


def test_run_artifact_s3_keys_include_run_id() -> None:
    helpers = load_ui_helpers_module()
    run_id = "Attempt7_2025-01-01_000000EST"

    canonical, legacy = helpers.run_artifact_s3_keys(
        "rhoslc-s06e11",
        run_id,
        "detections.jsonl",
    )

    assert f"/{run_id}/" in canonical
    assert legacy is None or legacy.endswith(f"{run_id}/detections.jsonl")

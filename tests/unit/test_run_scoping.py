"""Unit tests for run-scoped promotion/validation helpers."""

from __future__ import annotations

import json
from pathlib import Path

from py_screenalytics.run_scoping import (
    json_file_matches_run_id,
    marker_run_id_matches,
    marker_status_success,
    should_synthesize_detect_track_success,
)


def test_marker_status_success_requires_success_and_no_error() -> None:
    assert marker_status_success({"status": "success"}) is True
    assert marker_status_success({"status": "SUCCESS"}) is True
    assert marker_status_success({"status": "completed"}) is False
    assert marker_status_success({"status": "success", "error": "boom"}) is False
    assert marker_status_success(None) is False


def test_marker_run_id_matches() -> None:
    assert marker_run_id_matches({"run_id": "r1"}, None) is True
    assert marker_run_id_matches({"run_id": "r1"}, "r1") is True
    assert marker_run_id_matches({"run_id": "r1"}, "r2") is False
    assert marker_run_id_matches({"run_id": None}, "r1") is False


def test_should_synthesize_detect_track_success_is_strict() -> None:
    assert (
        should_synthesize_detect_track_success(
            run_id="r1",
            marker_payload={"status": "success", "run_id": "r1"},
            detections_exists=True,
            tracks_exists=True,
        )
        is True
    )
    assert (
        should_synthesize_detect_track_success(
            run_id="r1",
            marker_payload={"status": "error", "run_id": "r1"},
            detections_exists=True,
            tracks_exists=True,
        )
        is False
    )
    assert (
        should_synthesize_detect_track_success(
            run_id="r1",
            marker_payload={"status": "success", "run_id": "r2"},
            detections_exists=True,
            tracks_exists=True,
        )
        is False
    )
    assert (
        should_synthesize_detect_track_success(
            run_id="r1",
            marker_payload={"status": "success", "run_id": "r1"},
            detections_exists=False,
            tracks_exists=True,
        )
        is False
    )


def test_json_file_matches_run_id(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    path.write_text(json.dumps({"run_id": "r1"}), encoding="utf-8")
    assert json_file_matches_run_id(path, "r1") is True
    assert json_file_matches_run_id(path, "r2") is False


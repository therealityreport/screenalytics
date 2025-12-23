from __future__ import annotations

import json

from py_screenalytics import run_layout
from py_screenalytics.run_gates import GateReason
from py_screenalytics.run_manifests import StageBlockedInfo, StageErrorInfo, write_stage_manifest


def test_manifest_success_written(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "rhobh-s01e01"
    run_id = run_layout.get_or_create_run_id(ep_id, None)

    artifact_path = run_layout.run_root(ep_id, run_id) / "tracks.jsonl"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("ok", encoding="utf-8")

    manifest_path = write_stage_manifest(
        ep_id,
        run_id,
        "detect",
        "SUCCESS",
        started_at="2025-01-01T00:00:00Z",
        finished_at="2025-01-01T00:00:05Z",
        duration_s=None,
        counts={"tracks": 1},
        thresholds={"det_thresh": 0.5},
        model_versions={"detector": "retinaface"},
        artifacts={"tracks": str(artifact_path)},
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["stage"] == "detect"
    assert payload["status"] == "SUCCESS"
    assert payload["counts"]["tracks"] == 1
    assert any(entry["logical_name"] == "tracks" for entry in payload["artifacts"])


def test_manifest_failed_includes_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "rhobh-s01e01"
    run_id = run_layout.get_or_create_run_id(ep_id, None)

    manifest_path = write_stage_manifest(
        ep_id,
        run_id,
        "faces",
        "FAILED",
        started_at="2025-01-01T00:00:00Z",
        finished_at="2025-01-01T00:00:02Z",
        duration_s=None,
        error=StageErrorInfo(code="boom", message="bad"),
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "FAILED"
    assert payload["error"]["error_code"] == "boom"


def test_manifest_blocked_includes_reasons(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "rhobh-s01e01"
    run_id = run_layout.get_or_create_run_id(ep_id, None)

    blocked = StageBlockedInfo(
        reasons=[GateReason(code="missing_artifact", message="missing tracks", details=None)],
        suggested_actions=["Run detect"],
    )

    manifest_path = write_stage_manifest(
        ep_id,
        run_id,
        "cluster",
        "BLOCKED",
        started_at="2025-01-01T00:00:00Z",
        finished_at="2025-01-01T00:00:01Z",
        duration_s=None,
        blocked=blocked,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED"
    assert payload["blocked"]["blocked_reasons"][0]["code"] == "missing_artifact"


def test_manifest_run_scoped_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "rhobh-s01e01"
    run_id_a = run_layout.get_or_create_run_id(ep_id, None)
    run_id_b = run_layout.get_or_create_run_id(ep_id, None)

    path_a = write_stage_manifest(
        ep_id,
        run_id_a,
        "detect",
        "SUCCESS",
        started_at="2025-01-01T00:00:00Z",
        finished_at="2025-01-01T00:00:01Z",
        duration_s=None,
    )
    path_b = write_stage_manifest(
        ep_id,
        run_id_b,
        "detect",
        "SUCCESS",
        started_at="2025-01-01T00:00:00Z",
        finished_at="2025-01-01T00:00:01Z",
        duration_s=None,
    )

    assert path_a != path_b
    assert json.loads(path_a.read_text(encoding="utf-8"))["run_id"] == run_id_a
    assert json.loads(path_b.read_text(encoding="utf-8"))["run_id"] == run_id_b

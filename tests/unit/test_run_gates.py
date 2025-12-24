from __future__ import annotations

from pathlib import Path

from py_screenalytics import run_layout
from py_screenalytics.episode_status import write_stage_failed, write_stage_finished
from py_screenalytics.run_gates import check_prereqs


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok", encoding="utf-8")


def test_gate_missing_artifact(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "rhobh-s01e01"
    run_id = run_layout.get_or_create_run_id(ep_id, None)

    write_stage_finished(ep_id, run_id, "detect")

    gate = check_prereqs("faces", ep_id, run_id)
    assert not gate.ok
    assert any(reason.code == "missing_artifact" for reason in gate.reasons)
    assert gate.suggested_actions


def test_gate_upstream_failed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "rhobh-s01e01"
    run_id = run_layout.get_or_create_run_id(ep_id, None)

    run_root = run_layout.run_root(ep_id, run_id)
    _touch(run_root / "faces.jsonl")
    _touch(run_root / "tracks.jsonl")

    write_stage_failed(ep_id, run_id, "faces", error_code="failed", error_message="boom")

    gate = check_prereqs("cluster", ep_id, run_id)
    assert not gate.ok
    assert any(reason.code == "upstream_failed" for reason in gate.reasons)


def test_gate_stage_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "rhobh-s01e01"
    run_id = run_layout.get_or_create_run_id(ep_id, None)

    gate = check_prereqs("body_tracking", ep_id, run_id, config={"body_tracking": {"enabled": False}})
    assert not gate.ok
    assert any(reason.code == "stage_disabled" for reason in gate.reasons)
    assert gate.suggested_actions


def test_gate_reconciles_upstream_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "rhobh-s01e02"
    run_id = run_layout.get_or_create_run_id(ep_id, None)

    write_stage_finished(ep_id, run_id, "detect")

    run_root = run_layout.run_root(ep_id, run_id)
    _touch(run_root / "faces.jsonl")
    _touch(run_root / "body_tracking" / "body_tracks.jsonl")

    gate = check_prereqs("track_fusion", ep_id, run_id, config={"body_tracking": {"enabled": True}})
    assert gate.ok

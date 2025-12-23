from __future__ import annotations

import json
from pathlib import Path

from py_screenalytics import run_layout
from py_screenalytics.episode_status import (
    Stage,
    StageStatus,
    read_episode_status,
    write_stage_finished,
    write_stage_started,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_episode_status_round_trip(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "ep-roundtrip"
    run_id = "attempt-1"

    write_stage_started(ep_id, run_id, "detect")
    write_stage_finished(ep_id, run_id, "detect", counts={"detections": 2, "tracks": 1})

    status = read_episode_status(ep_id, run_id)
    assert status.episode_id == ep_id
    assert status.run_id == run_id

    detect_state = status.stages[Stage.DETECT]
    assert detect_state.status == StageStatus.SUCCESS
    assert detect_state.counts.get("detections") == 2
    assert detect_state.counts.get("tracks") == 1
    assert detect_state.started_at is not None
    assert detect_state.finished_at is not None


def test_episode_status_atomic_write(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "ep-atomic"
    run_id = "attempt-2"

    write_stage_started(ep_id, run_id, "detect")
    status_path = run_layout.run_root(ep_id, run_id) / "episode_status.json"

    assert status_path.exists()
    payload = _read_json(status_path)
    assert payload.get("episode_id") == ep_id
    assert payload.get("run_id") == run_id
    assert isinstance(payload.get("stages"), dict)


def test_episode_status_derivation_fallback(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "ep-derive"
    run_id = "attempt-3"
    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    (run_root / "faces.jsonl").write_text('{"f":1}\n', encoding="utf-8")

    status = read_episode_status(ep_id, run_id)
    faces_state = status.stages[Stage.FACES]
    assert faces_state.status == StageStatus.SUCCESS
    assert faces_state.derived is True
    detect_state = status.stages[Stage.DETECT]
    assert detect_state.status == StageStatus.NOT_STARTED
    assert detect_state.derived is True

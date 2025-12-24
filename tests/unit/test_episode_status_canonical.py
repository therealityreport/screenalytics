from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from py_screenalytics import run_layout
import pytest

from py_screenalytics.episode_status import (
    BlockedReason,
    Stage,
    StageStatus,
    StageTransitionError,
    blocked_update_needed,
    read_episode_status,
    write_stage_blocked,
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
    assert faces_state.derived_from
    assert faces_state.derivation_reason
    detect_state = status.stages[Stage.DETECT]
    assert detect_state.status == StageStatus.NOT_STARTED
    assert detect_state.derived is True


def test_pdf_reconciles_running_when_exports_exist(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "ep-pdf-reconcile"
    run_id = "attempt-8"
    run_root = run_layout.run_root(ep_id, run_id)
    (run_root / "exports").mkdir(parents=True, exist_ok=True)
    (run_root / "exports" / "run_debug.pdf").write_bytes(b"%PDF-1.4 test")
    (run_root / "exports" / "export_index.json").write_text("{}", encoding="utf-8")

    write_stage_started(ep_id, run_id, "pdf")
    status = read_episode_status(ep_id, run_id)
    pdf_state = status.stages[Stage.PDF]
    assert pdf_state.status == StageStatus.SUCCESS
    assert pdf_state.derived is True
    assert pdf_state.finished_at is not None


def test_pdf_reconciles_blocked_when_exports_exist(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "ep-pdf-blocked"
    run_id = "attempt-9"
    run_root = run_layout.run_root(ep_id, run_id)
    (run_root / "exports").mkdir(parents=True, exist_ok=True)
    (run_root / "exports" / "run_debug.pdf").write_bytes(b"%PDF-1.4 test")
    (run_root / "exports" / "export_index.json").write_text("{}", encoding="utf-8")

    blocked_reason = BlockedReason(code="missing", message="missing prereq", details=None)
    write_stage_blocked(ep_id, run_id, "pdf", blocked_reason)

    status = read_episode_status(ep_id, run_id)
    pdf_state = status.stages[Stage.PDF]
    assert pdf_state.status == StageStatus.SUCCESS
    assert pdf_state.blocked_reason is None


def test_episode_status_lost_update_protection(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "ep-concurrent"
    run_id = "attempt-4"

    def _worker(stage_key: str) -> None:
        write_stage_started(ep_id, run_id, stage_key)

    threads = [
        threading.Thread(target=_worker, args=("detect",)),
        threading.Thread(target=_worker, args=("faces",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    status = read_episode_status(ep_id, run_id)
    assert status.stages[Stage.DETECT].status == StageStatus.RUNNING
    assert status.stages[Stage.FACES].status == StageStatus.RUNNING


def test_episode_status_monotonic_transition(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "ep-monotonic"
    run_id = "attempt-5"

    write_stage_finished(ep_id, run_id, "detect")
    with pytest.raises(StageTransitionError) as excinfo:
        write_stage_started(ep_id, run_id, "detect")

    assert excinfo.value.code == "status_regression"


def test_stage_start_resets_terminal_fields(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "ep-reset"
    run_id = "attempt-10"
    blocked_reason = BlockedReason(code="missing", message="missing input", details=None)

    write_stage_blocked(ep_id, run_id, "detect", blocked_reason)
    status_blocked = read_episode_status(ep_id, run_id)
    assert status_blocked.stages[Stage.DETECT].status == StageStatus.BLOCKED
    assert status_blocked.stages[Stage.DETECT].finished_at is not None

    write_stage_started(ep_id, run_id, "detect")
    status_running = read_episode_status(ep_id, run_id)
    detect_state = status_running.stages[Stage.DETECT]
    assert detect_state.status == StageStatus.RUNNING
    assert detect_state.finished_at is None
    assert detect_state.blocked_reason is None

    finished_at = datetime.now(timezone.utc) + timedelta(seconds=5)
    write_stage_finished(ep_id, run_id, "detect", finished_at=finished_at)
    status_done = read_episode_status(ep_id, run_id)
    detect_done = status_done.stages[Stage.DETECT]
    assert detect_done.started_at is not None
    assert detect_done.finished_at is not None
    assert detect_done.started_at <= detect_done.finished_at


def test_pdf_timestamps_monotonic(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "ep-pdf-times"
    run_id = "attempt-11"
    blocked_reason = BlockedReason(code="missing", message="missing prereq", details=None)

    write_stage_blocked(ep_id, run_id, "pdf", blocked_reason)
    write_stage_started(ep_id, run_id, "pdf")
    finished_at = datetime.now(timezone.utc) + timedelta(seconds=10)
    write_stage_finished(ep_id, run_id, "pdf", finished_at=finished_at)

    status = read_episode_status(ep_id, run_id)
    pdf_state = status.stages[Stage.PDF]
    assert pdf_state.started_at is not None
    assert pdf_state.finished_at is not None
    assert pdf_state.started_at <= pdf_state.finished_at


def test_stage_blocked_idempotent(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "ep-blocked"
    run_id = "attempt-6"
    blocked_reason = BlockedReason(code="missing_artifact", message="Missing tracks.jsonl", details=None)

    assert blocked_update_needed(ep_id, run_id, "detect", blocked_reason)
    write_stage_blocked(ep_id, run_id, "detect", blocked_reason)
    assert not blocked_update_needed(ep_id, run_id, "detect", blocked_reason)


def test_blocked_does_not_downgrade_success(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "ep-blocked-success"
    run_id = "attempt-7"
    blocked_reason = BlockedReason(code="missing_artifact", message="Missing faces.jsonl", details=None)

    write_stage_finished(ep_id, run_id, "detect")
    write_stage_blocked(ep_id, run_id, "detect", blocked_reason)

    status = read_episode_status(ep_id, run_id)
    assert status.stages[Stage.DETECT].status == StageStatus.SUCCESS

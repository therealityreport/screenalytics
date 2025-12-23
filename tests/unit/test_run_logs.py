from __future__ import annotations

from py_screenalytics import run_layout
from py_screenalytics.run_logs import append_log, read_stage_progress, tail_logs


def test_append_and_tail_logs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "rhobh-s01e01"
    run_id = run_layout.get_or_create_run_id(ep_id, None)

    append_log(ep_id, run_id, "detect", "INFO", "start", progress=0)
    append_log(ep_id, run_id, "detect", "INFO", "mid", progress=50)
    append_log(ep_id, run_id, "detect", "INFO", "done", progress=100)

    events = tail_logs(ep_id, run_id, "detect", n=2)
    assert len(events) == 2
    assert events[0]["msg"] == "mid"
    assert events[1]["msg"] == "done"
    assert read_stage_progress(ep_id, run_id, "detect") == 100.0


def test_run_scoped_logs_separate(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "rhobh-s01e01"
    run_a = run_layout.get_or_create_run_id(ep_id, None)
    run_b = run_layout.get_or_create_run_id(ep_id, None)

    append_log(ep_id, run_a, "faces", "INFO", "run_a")
    append_log(ep_id, run_b, "faces", "INFO", "run_b")

    events_a = tail_logs(ep_id, run_a, "faces", n=10)
    events_b = tail_logs(ep_id, run_b, "faces", n=10)
    assert len(events_a) == 1
    assert len(events_b) == 1
    assert events_a[0]["msg"] == "run_a"
    assert events_b[0]["msg"] == "run_b"


def test_missing_log_file_returns_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "rhobh-s01e01"
    run_id = run_layout.get_or_create_run_id(ep_id, None)

    assert tail_logs(ep_id, run_id, "cluster", n=5) == []

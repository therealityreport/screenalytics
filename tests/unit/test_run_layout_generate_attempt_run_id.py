from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from py_screenalytics import run_layout


def test_generate_attempt_run_id_increments_from_existing_attempt_numbers(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "rhoslc-s06e11"

    runs_root = run_layout.runs_root(ep_id)
    runs_root.mkdir(parents=True, exist_ok=True)

    # Existing non-attempt run ids should not break numbering.
    (runs_root / "abc123").mkdir(parents=True, exist_ok=True)
    (runs_root / "Attempt3_2025-01-01_000000EST").mkdir(parents=True, exist_ok=True)

    now = datetime(2025, 1, 2, 3, 4, 5, tzinfo=ZoneInfo("America/New_York"))
    run_id = run_layout.generate_attempt_run_id(ep_id, now=now)
    assert run_id == "Attempt4_2025-01-02_030405EST"


def test_generate_attempt_run_id_falls_back_to_count_when_no_attempt_prefix(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "rhoslc-s06e11"

    runs_root = run_layout.runs_root(ep_id)
    runs_root.mkdir(parents=True, exist_ok=True)

    (runs_root / "a").mkdir(parents=True, exist_ok=True)
    (runs_root / "b").mkdir(parents=True, exist_ok=True)

    now = datetime(2025, 1, 2, 3, 4, 5, tzinfo=ZoneInfo("America/New_York"))
    run_id = run_layout.generate_attempt_run_id(ep_id, now=now)
    assert run_id == "Attempt3_2025-01-02_030405EST"


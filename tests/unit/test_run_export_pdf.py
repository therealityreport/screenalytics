"""Unit tests for canonical run_debug.pdf generation."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_generate_run_debug_pdf_failed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("reportlab")
    sys.path.insert(0, str(PROJECT_ROOT))
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))

    from py_screenalytics import run_layout
    from py_screenalytics.episode_status import write_stage_failed
    from py_screenalytics.run_manifests import StageErrorInfo, write_stage_manifest
    from py_screenalytics.run_logs import append_log
    from apps.api.services.run_export import generate_run_debug_pdf

    ep_id = "ep-pdf-failed"
    run_id = "run123"
    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    write_stage_failed(ep_id, run_id, "detect", error_code="boom", error_message="fail")
    now = datetime.now(timezone.utc)
    write_stage_manifest(
        ep_id,
        run_id,
        "detect",
        "FAILED",
        started_at=now,
        finished_at=now,
        duration_s=0.1,
        error=StageErrorInfo(code="boom", message="fail"),
    )
    append_log(ep_id, run_id, "detect", "ERROR", "stage failed", progress=100.0)

    pdf_path = generate_run_debug_pdf(ep_id, run_id)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


def test_generate_run_debug_pdf_missing_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("reportlab")
    sys.path.insert(0, str(PROJECT_ROOT))
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))

    from py_screenalytics import run_layout
    from apps.api.services.run_export import generate_run_debug_pdf

    ep_id = "ep-pdf-missing"
    run_id = "run456"
    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    pdf_path = generate_run_debug_pdf(ep_id, run_id)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0

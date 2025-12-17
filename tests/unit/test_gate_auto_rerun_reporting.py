"""Unit tests for adaptive appearance-gate rerun decision + PDF reporting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ascii_strings(data: bytes, *, min_len: int = 4) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    for b in data:
        if 32 <= b <= 126:
            buf.append(chr(b))
        else:
            if len(buf) >= min_len:
                out.append("".join(buf))
            buf = []
    if len(buf) >= min_len:
        out.append("".join(buf))
    return out


def test_gate_auto_rerun_decision_triggers_for_extreme_forced_splits() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))

    from tools.episode_run import (
        _gate_auto_rerun_decision,
        GATE_AUTO_RERUN_ENABLED,
        GATE_AUTO_RERUN_FORCED_SPLITS_THRESHOLD,
        GATE_AUTO_RERUN_MIN_GATE_SPLITS_SHARE,
    )

    if not GATE_AUTO_RERUN_ENABLED:
        pytest.skip("gate auto rerun disabled via env")

    forced_splits = max(GATE_AUTO_RERUN_FORCED_SPLITS_THRESHOLD + 1, 1)
    gate_splits = int(forced_splits * GATE_AUTO_RERUN_MIN_GATE_SPLITS_SHARE) + 1
    triggered, reason, snapshot = _gate_auto_rerun_decision(
        {
            "forced_splits": forced_splits,
            "id_switches": 0,
            "appearance_gate": {"splits": {"total": gate_splits}},
        }
    )
    assert triggered is True
    assert reason == "forced_splits_high_id_switches_low"
    assert snapshot.get("forced_splits") == forced_splits


def test_pdf_reports_gate_auto_rerun_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("reportlab")

    sys.path.insert(0, str(PROJECT_ROOT))
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SCREENALYTICS_PDF_NO_COMPRESSION", "1")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    from py_screenalytics import run_layout
    from apps.api.services.run_export import build_screentime_run_debug_pdf

    ep_id = "ep-test"
    run_id = "run123"
    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    # Minimal tracks.jsonl so the report can render.
    (run_root / "tracks.jsonl").write_text(
        json.dumps({"track_id": 1, "first_ts": 0.0, "last_ts": 1.0}) + "\n",
        encoding="utf-8",
    )

    # Minimal track_metrics.json with effective gate disabled.
    (run_root / "track_metrics.json").write_text(
        json.dumps(
            {
                "ep_id": ep_id,
                "generated_at": "now",
                "metrics": {"forced_splits": 300, "id_switches": 0, "tracking_gate": {"enabled": False}},
                "scene_cuts": {"count": 0},
                "tracking_gate": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )

    # detect_track.json carries the auto-rerun decision for PDF lineage.
    (run_root / "detect_track.json").write_text(
        json.dumps(
            {
                "phase": "detect_track",
                "status": "success",
                "run_id": run_id,
                "tracking_gate": {
                    "enabled": False,
                    "auto_rerun": {"triggered": True, "selected": "rerun", "reason": "forced_splits_high_id_switches_low"},
                },
            }
        ),
        encoding="utf-8",
    )

    pdf_bytes, _name = build_screentime_run_debug_pdf(ep_id=ep_id, run_id=run_id)
    combined = "\n".join(_ascii_strings(pdf_bytes))

    assert "Appearance Gate Enabled" in combined
    assert "Appearance Gate Auto-Rerun" in combined
    assert "true (selected=rerun" in combined


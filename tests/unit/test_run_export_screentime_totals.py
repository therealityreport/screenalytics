"""Unit tests for screentime_comparison totals rendering in the PDF report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ascii_strings(data: bytes, *, min_len: int = 4) -> list[str]:
    """Extract printable ASCII runs (similar to the `strings` utility)."""
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


def test_pdf_renders_face_baseline_total_when_present(
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

    # Minimal tracks.jsonl so the report has face-track context.
    (run_root / "tracks.jsonl").write_text(
        json.dumps({"track_id": 1, "first_ts": 0.0, "last_ts": 10.0}) + "\n",
        encoding="utf-8",
    )

    body_dir = run_root / "body_tracking"
    body_dir.mkdir(parents=True, exist_ok=True)
    (body_dir / "screentime_comparison.json").write_text(
        json.dumps(
            {
                "summary": {
                    "total_identities": 1,
                    "identities_with_gain": 1,
                    "face_total_s": 12.34,
                    "body_total_s": 20.0,
                    "fused_total_s": 5.0,
                    "combined_total_s": 25.0,
                    "gain_total_s": 12.66,
                    # Back-compat fields (still supported)
                    "total_face_only_duration": 12.34,
                    "total_body_duration": 20.0,
                    "total_fused_duration": 5.0,
                    "total_combined_duration": 25.0,
                    "total_duration_gain": 12.66,
                },
                "breakdowns": [],
            }
        ),
        encoding="utf-8",
    )

    pdf_bytes, _name = build_screentime_run_debug_pdf(ep_id=ep_id, run_id=run_id)
    strings = _ascii_strings(pdf_bytes)
    combined = "\n".join(strings)

    assert "Face baseline total" in combined
    assert "12.34s" in combined


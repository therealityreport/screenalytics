"""Unit tests for missing-artifact handling in the PDF run debug report."""

from __future__ import annotations

import json
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


def test_pdf_missing_body_artifacts_renders_na_and_keeps_face_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("reportlab")
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SCREENALYTICS_PDF_NO_COMPRESSION", "1")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    from py_screenalytics import run_layout
    from apps.api.services.run_export import build_screentime_run_debug_pdf

    ep_id = "ep-test"
    run_id = "run123"
    face_track_count = 278

    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    tracks_path = run_root / "tracks.jsonl"
    with tracks_path.open("w", encoding="utf-8") as handle:
        for track_id in range(face_track_count):
            handle.write(json.dumps({"track_id": track_id, "first_ts": 0.0, "last_ts": 1.0}) + "\n")

    pdf_bytes, _name = build_screentime_run_debug_pdf(ep_id=ep_id, run_id=run_id)
    strings = _ascii_strings(pdf_bytes)
    combined = "\n".join(strings)

    assert "missing body_tracking/body_tracks.jsonl" in combined
    assert "missing body_tracking/track_fusion.json" in combined
    assert "Screen Time Analyze is unavailable" in combined
    assert "missing body_tracking/screentime_comparison.json" in combined
    assert "No face time recorded" not in combined

    # Track Fusion must use run_root/tracks.jsonl for face track inputs even when body artifacts are missing.
    label_idx = next(idx for idx, s in enumerate(strings) if "Face Tracks \\(input\\)" in s)
    lookahead = "\n".join(strings[label_idx : label_idx + 250])
    assert f"({face_track_count})" in lookahead

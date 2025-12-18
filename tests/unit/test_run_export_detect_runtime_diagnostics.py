"""Unit tests for detect runtime diagnostics + stride accounting in the PDF run debug report."""

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


def test_pdf_reports_detect_wall_time_rtf_and_stride_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SCREENALYTICS_PDF_NO_COMPRESSION", "1")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    from py_screenalytics import run_layout
    from apps.api.services.run_export import build_screentime_run_debug_pdf

    ep_id = "ep-test"
    run_id = "run-detect-diagnostics"

    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    (run_root / "tracks.jsonl").write_text(
        json.dumps({"track_id": 1, "first_ts": 0.0, "last_ts": 1.0}) + "\n",
        encoding="utf-8",
    )

    (run_root / "detect_track.json").write_text(
        json.dumps(
            {
                "phase": "detect_track",
                "status": "success",
                "run_id": run_id,
                "video_duration_sec": 105.0,
                "fps": 24.0,
                "frames_total": 2525,
                "frames_scanned_total": 2525,
                "stride": 6,
                "stride_effective": 6,
                "stride_observed_median": 6,
                "expected_frames_by_stride": 421,
                "face_detect_frames_processed": 616,
                "face_detect_frames_processed_stride": 421,
                "face_detect_frames_processed_forced_scene_warmup": 195,
                "scene_warmup_dets": 3,
                "detect_wall_time_s": 749.4,
                "rtf": 7.137,
                "effective_fps_processing": 0.822,
                "scene_detect_wall_time_s": 10.1,
                "tracker_backend_configured": "bytetrack",
                "tracker_backend_actual": "ultralytics.bytetrack",
                "tracker_fallback_reason": None,
                "requested_device": "cuda",
                "resolved_device": "cuda",
                "detector_model_name": "retinaface_r50",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pdf_bytes, _name = build_screentime_run_debug_pdf(ep_id=ep_id, run_id=run_id)
    combined = "\n".join(_ascii_strings(pdf_bytes))

    assert "Detect Wall Time" in combined
    assert "RTF" in combined
    assert "Detect Effective FPS" in combined

    assert "Frames Scanned Total" in combined
    assert "Face Detect Frames Processed" in combined
    assert "stride_" in combined
    assert "warmup" in combined
    assert "expected_" in combined

    assert "Face Tracker Backend" in combined
    assert "ultralytics.bytetrack" in combined

"""Unit tests for OpenCV fallback in resolved_frames computation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

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


def test_pdf_uses_opencv_frame_count_when_ffprobe_and_marker_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that resolved_frames falls back to opencv.frame_count when ffprobe and marker are unavailable."""
    pytest.importorskip("reportlab")
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SCREENALYTICS_PDF_NO_COMPRESSION", "1")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    from py_screenalytics import run_layout
    from apps.api.services import run_export

    ep_id = "ep-opencv-fallback"
    run_id = "run-fallback"
    opencv_frame_count = 1500
    opencv_fps = 24.0

    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    # Create minimal tracks.jsonl
    tracks_path = run_root / "tracks.jsonl"
    with tracks_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"track_id": 1, "first_ts": 0.0, "last_ts": 1.0}) + "\n")

    # Mock ffprobe to return failure (no nb_frames)
    def mock_ffprobe_failure(video_path: Path) -> dict:
        return {"ok": False, "error": "ffprobe_not_found", "duration_s": None, "avg_fps": None, "nb_frames": None}

    # Mock opencv to return valid frame count
    def mock_opencv_success(video_path: Path) -> dict:
        return {"ok": True, "fps": opencv_fps, "frame_count": opencv_frame_count, "width": 1920, "height": 1080}

    with patch.object(run_export, "_ffprobe_video_metadata", mock_ffprobe_failure):
        with patch.object(run_export, "_opencv_video_metadata", mock_opencv_success):
            pdf_bytes, _name = run_export.build_screentime_run_debug_pdf(ep_id=ep_id, run_id=run_id)

    strings = _ascii_strings(pdf_bytes)
    combined = "\n".join(strings)

    # Verify opencv was used as the source for resolved frames
    assert f"{opencv_frame_count}" in combined, "OpenCV frame count should appear in PDF"
    # PDF escapes parentheses, so we check for the frame count followed by opencv source indicator
    # The PDF may render as "(opencv)" or "\\(opencv\\)" depending on encoding
    has_opencv_label = "opencv" in combined.lower() and "resolved" in combined.lower()
    assert has_opencv_label, "Source should be labeled with 'opencv' as fallback source"


def test_pdf_prefers_ffprobe_over_opencv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that ffprobe is preferred over opencv when both are available."""
    pytest.importorskip("reportlab")
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SCREENALYTICS_PDF_NO_COMPRESSION", "1")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    from py_screenalytics import run_layout
    from apps.api.services import run_export

    ep_id = "ep-ffprobe-preferred"
    run_id = "run-prefer"
    ffprobe_frame_count = 2000
    opencv_frame_count = 1500  # Different value to distinguish

    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    # Create minimal tracks.jsonl
    tracks_path = run_root / "tracks.jsonl"
    with tracks_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"track_id": 1, "first_ts": 0.0, "last_ts": 1.0}) + "\n")

    # Mock ffprobe to return valid data
    def mock_ffprobe_success(video_path: Path) -> dict:
        return {"ok": True, "duration_s": 83.33, "avg_fps": 24.0, "nb_frames": ffprobe_frame_count}

    # Mock opencv to return different valid frame count
    def mock_opencv_success(video_path: Path) -> dict:
        return {"ok": True, "fps": 24.0, "frame_count": opencv_frame_count, "width": 1920, "height": 1080}

    with patch.object(run_export, "_ffprobe_video_metadata", mock_ffprobe_success):
        with patch.object(run_export, "_opencv_video_metadata", mock_opencv_success):
            pdf_bytes, _name = run_export.build_screentime_run_debug_pdf(ep_id=ep_id, run_id=run_id)

    strings = _ascii_strings(pdf_bytes)
    combined = "\n".join(strings)

    # Verify ffprobe was used (its value appears with ffprobe label)
    assert f"{ffprobe_frame_count}" in combined, "ffprobe frame count should appear in PDF"
    # PDF escapes parentheses, so we check for ffprobe appearing as a source
    # The PDF may render as "(ffprobe)" or "\\(ffprobe\\)" depending on encoding
    has_ffprobe_label = "ffprobe" in combined.lower() and "resolved" in combined.lower()
    assert has_ffprobe_label, "Source should be labeled with 'ffprobe' as preferred source"

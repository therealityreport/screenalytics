"""Unit tests for video lineage resolution and fallback logic in PDF export.

Tests cover:
- Fallback priority: ffprobe > detect_track.marker > opencv
- Source attribution labels in PDF output
- Mismatch warning triggers when sources disagree by >10%
- "In Bundle" column in artifact manifest
"""

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


def test_pdf_uses_marker_when_ffprobe_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that detect_track.marker is used when ffprobe fails but marker exists."""
    pytest.importorskip("reportlab")
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SCREENALYTICS_PDF_NO_COMPRESSION", "1")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    from py_screenalytics import run_layout
    from apps.api.services import run_export

    ep_id = "ep-marker-fallback"
    run_id = "run-marker"
    marker_frame_count = 1800
    marker_fps = 23.976
    opencv_frame_count = 1500  # Different to distinguish

    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    # Create tracks.jsonl
    tracks_path = run_root / "tracks.jsonl"
    with tracks_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"track_id": 1, "first_ts": 0.0, "last_ts": 1.0}) + "\n")

    # Create detect_track.json with video metadata (this is the marker file name used by run_export)
    marker_path = run_root / "detect_track.json"
    marker_data = {
        "run_id": run_id,
        "video_duration_sec": marker_frame_count / marker_fps,  # Note: field name is video_duration_sec
        "fps": marker_fps,
        "frames_total": marker_frame_count,
        "stride": 3,  # Note: field name is stride, not detect_every_n_frames
    }
    with marker_path.open("w", encoding="utf-8") as handle:
        json.dump(marker_data, handle)

    # Mock ffprobe to fail
    def mock_ffprobe_failure(video_path: Path) -> dict:
        return {"ok": False, "error": "ffprobe_not_found", "duration_s": None, "avg_fps": None, "nb_frames": None}

    # Mock opencv with different values
    def mock_opencv_success(video_path: Path) -> dict:
        return {"ok": True, "fps": 24.0, "frame_count": opencv_frame_count, "width": 1920, "height": 1080}

    with patch.object(run_export, "_ffprobe_video_metadata", mock_ffprobe_failure):
        with patch.object(run_export, "_opencv_video_metadata", mock_opencv_success):
            pdf_bytes, _name = run_export.build_screentime_run_debug_pdf(ep_id=ep_id, run_id=run_id)

    strings = _ascii_strings(pdf_bytes)
    combined = "\n".join(strings)

    # Verify marker was used (its value appears)
    assert f"{marker_frame_count}" in combined, "Marker frame count should appear in PDF"
    # Verify marker source is labeled
    has_marker_label = "detect_track" in combined.lower() or "marker" in combined.lower()
    assert has_marker_label, "Source should reference detect_track.marker when used as fallback"


def test_pdf_mismatch_warning_triggers_on_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that mismatch warning appears when sources disagree by >10%."""
    pytest.importorskip("reportlab")
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SCREENALYTICS_PDF_NO_COMPRESSION", "1")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    from py_screenalytics import run_layout
    from apps.api.services import run_export

    ep_id = "ep-mismatch"
    run_id = "run-mismatch"

    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    # Create tracks.jsonl
    tracks_path = run_root / "tracks.jsonl"
    with tracks_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"track_id": 1, "first_ts": 0.0, "last_ts": 1.0}) + "\n")

    # Mock ffprobe with one value
    def mock_ffprobe(video_path: Path) -> dict:
        return {"ok": True, "duration_s": 100.0, "avg_fps": 24.0, "nb_frames": 2400}

    # Mock opencv with significantly different value (>10% diff)
    # 2400 vs 2000 = 400/2400 = 16.7% diff
    def mock_opencv(video_path: Path) -> dict:
        return {"ok": True, "fps": 20.0, "frame_count": 2000, "width": 1920, "height": 1080}

    with patch.object(run_export, "_ffprobe_video_metadata", mock_ffprobe):
        with patch.object(run_export, "_opencv_video_metadata", mock_opencv):
            pdf_bytes, _name = run_export.build_screentime_run_debug_pdf(ep_id=ep_id, run_id=run_id)

    strings = _ascii_strings(pdf_bytes)
    combined = "\n".join(strings)

    # Verify mismatch warning appears
    has_mismatch_warning = "mismatch" in combined.lower()
    assert has_mismatch_warning, "Mismatch warning should appear when sources disagree by >10%"


def test_pdf_no_mismatch_warning_when_sources_agree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that no mismatch warning appears when sources agree within 10%."""
    pytest.importorskip("reportlab")
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SCREENALYTICS_PDF_NO_COMPRESSION", "1")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    from py_screenalytics import run_layout
    from apps.api.services import run_export

    ep_id = "ep-no-mismatch"
    run_id = "run-agree"

    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    # Create tracks.jsonl
    tracks_path = run_root / "tracks.jsonl"
    with tracks_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"track_id": 1, "first_ts": 0.0, "last_ts": 1.0}) + "\n")

    # Mock ffprobe and opencv with similar values (<10% diff)
    # 2400 vs 2350 = 50/2400 = 2.1% diff
    def mock_ffprobe(video_path: Path) -> dict:
        return {"ok": True, "duration_s": 100.0, "avg_fps": 24.0, "nb_frames": 2400}

    def mock_opencv(video_path: Path) -> dict:
        return {"ok": True, "fps": 23.5, "frame_count": 2350, "width": 1920, "height": 1080}

    with patch.object(run_export, "_ffprobe_video_metadata", mock_ffprobe):
        with patch.object(run_export, "_opencv_video_metadata", mock_opencv):
            pdf_bytes, _name = run_export.build_screentime_run_debug_pdf(ep_id=ep_id, run_id=run_id)

    strings = _ascii_strings(pdf_bytes)
    combined = "\n".join(strings)

    # Verify mismatch warning does NOT appear
    # Note: The word "mismatch" might appear in other contexts, so we check for the warning pattern
    has_mismatch_warning = "metadata mismatch" in combined.lower()
    assert not has_mismatch_warning, "Mismatch warning should NOT appear when sources agree within 10%"


def test_pdf_contains_in_bundle_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the artifact manifest includes 'In Bundle' column."""
    pytest.importorskip("reportlab")
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SCREENALYTICS_PDF_NO_COMPRESSION", "1")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    from py_screenalytics import run_layout
    from apps.api.services import run_export

    ep_id = "ep-bundle-col"
    run_id = "run-bundle"

    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    # Create tracks.jsonl
    tracks_path = run_root / "tracks.jsonl"
    with tracks_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"track_id": 1, "first_ts": 0.0, "last_ts": 1.0}) + "\n")

    def mock_ffprobe(video_path: Path) -> dict:
        return {"ok": True, "duration_s": 100.0, "avg_fps": 24.0, "nb_frames": 2400}

    def mock_opencv(video_path: Path) -> dict:
        return {"ok": True, "fps": 24.0, "frame_count": 2400, "width": 1920, "height": 1080}

    with patch.object(run_export, "_ffprobe_video_metadata", mock_ffprobe):
        with patch.object(run_export, "_opencv_video_metadata", mock_opencv):
            pdf_bytes, _name = run_export.build_screentime_run_debug_pdf(ep_id=ep_id, run_id=run_id)

    strings = _ascii_strings(pdf_bytes)
    combined = "\n".join(strings)

    # Verify "In Bundle" column header appears
    has_in_bundle = "in bundle" in combined.lower()
    assert has_in_bundle, "Artifact manifest should contain 'In Bundle' column"

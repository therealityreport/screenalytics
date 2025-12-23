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
    # Note: PDF text may include zero-width wrap hints, so avoid matching the full label as a single ASCII run.
    label_idx = next(idx for idx, s in enumerate(strings) if "Face Tracks" in s)
    lookahead = "\n".join(strings[label_idx : label_idx + 250])
    assert f"({face_track_count})" in lookahead


def test_pdf_missing_face_artifacts_render_na_not_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: missing face artifacts must render as N/A (not '0')."""
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SCREENALYTICS_PDF_NO_COMPRESSION", "1")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    import numpy as np

    from py_screenalytics import run_layout
    from apps.api.services.run_export import build_screentime_run_debug_pdf

    ep_id = "ep-test"
    run_id = "run-missing-faces"

    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    # Simulate "embed outputs exist" while upstream face artifacts are missing.
    (run_root / "face_alignment").mkdir(parents=True, exist_ok=True)
    (run_root / "face_alignment" / "aligned_faces.jsonl").write_text(
        json.dumps({"track_id": 1, "frame_idx": 0, "alignment_quality": 0.9}) + "\n",
        encoding="utf-8",
    )
    embeds_dir = tmp_path / "embeds" / ep_id / "runs" / run_id
    embeds_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeds_dir / "faces.npy", np.zeros((1, 512), dtype=np.float32))

    pdf_bytes, _name = build_screentime_run_debug_pdf(ep_id=ep_id, run_id=run_id)
    strings = _ascii_strings(pdf_bytes)
    combined = "\n".join(strings)

    det_idx = next(idx for idx, s in enumerate(strings) if "Total face detections:" in s)
    det_window = "\n".join(strings[det_idx : det_idx + 80])
    assert "N/A \\(missing detections.jsonl\\)" in det_window
    assert "Total face detections: <b>0</b>" not in det_window

    trk_idx = next(idx for idx, s in enumerate(strings) if "Total face tracks:" in s)
    trk_window = "\n".join(strings[trk_idx : trk_idx + 80])
    assert "N/A \\(missing tracks.jsonl\\)" in trk_window
    assert "Total face tracks: <b>0</b>" not in trk_window

    assert "missing identities.json" in combined


def test_pdf_body_tracking_ran_effective_requires_run_scoped_artifacts_even_if_legacy_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: legacy body artifacts must not imply run-scoped execution."""
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SCREENALYTICS_PDF_NO_COMPRESSION", "1")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    from py_screenalytics import run_layout
    from apps.api.services.run_export import build_screentime_run_debug_pdf

    ep_id = "ep-test"
    run_id = "run-legacy-body-only"

    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    # Only legacy body artifacts exist (episode-level); run-scoped body artifacts are missing.
    legacy_body_dir = tmp_path / "manifests" / ep_id / "body_tracking"
    legacy_body_dir.mkdir(parents=True, exist_ok=True)
    (legacy_body_dir / "body_detections.jsonl").write_text(json.dumps({"frame_idx": 0}) + "\n", encoding="utf-8")
    (legacy_body_dir / "body_tracks.jsonl").write_text(json.dumps({"track_id": 100000}) + "\n", encoding="utf-8")

    legacy_runs_dir = tmp_path / "manifests" / ep_id / "runs"
    legacy_runs_dir.mkdir(parents=True, exist_ok=True)
    (legacy_runs_dir / "body_tracking.json").write_text(
        json.dumps({"phase": "body_tracking", "status": "success", "run_id": run_id}, indent=2),
        encoding="utf-8",
    )

    pdf_bytes, _name = build_screentime_run_debug_pdf(ep_id=ep_id, run_id=run_id)
    strings = _ascii_strings(pdf_bytes)
    combined = "\n".join(strings)

    ran_idx = next(idx for idx, s in enumerate(strings) if "body_tracking.ran_effective \\(run-scoped\\)" in s)
    ran_window = "\n".join(strings[ran_idx : ran_idx + 80])
    assert "False" in ran_window

    legacy_idx = next(idx for idx, s in enumerate(strings) if "legacy_body_tracking.out_of_scope" in s)
    legacy_window = "\n".join(strings[legacy_idx : legacy_idx + 80])
    assert "yes" in legacy_window

    assert "out_of_scope=yes" in combined


def test_pdf_reports_body_tracker_backend_fallback_when_supervision_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SCREENALYTICS_PDF_NO_COMPRESSION", "1")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    from py_screenalytics import run_layout
    from apps.api.services.run_export import build_screentime_run_debug_pdf

    ep_id = "ep-test"
    run_id = "run-body-fallback"

    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    # Minimal face tracks so the report can render.
    (run_root / "tracks.jsonl").write_text(
        json.dumps({"track_id": 1, "first_ts": 0.0, "last_ts": 1.0}) + "\n",
        encoding="utf-8",
    )

    # Simulate a body-tracking run marker that recorded a fallback backend due to missing supervision.
    (run_root / "body_tracking.json").write_text(
        json.dumps(
            {
                "phase": "body_tracking",
                "status": "success",
                "run_id": run_id,
                "tracker_backend_configured": "bytetrack",
                "tracker_backend_actual": "iou_fallback",
                "tracker_fallback_reason": "supervision_missing",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pdf_bytes, _name = build_screentime_run_debug_pdf(ep_id=ep_id, run_id=run_id)
    combined = "\n".join(_ascii_strings(pdf_bytes))

    assert "Body tracker backend" in combined
    assert "iou_fallback" in combined
    assert "tracking backend fallback activated" in combined
    assert "Install" in combined


def test_pdf_setup_scope_excludes_screentime_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Episode Details PDF should omit screentime sections when include_screentime=False."""
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SCREENALYTICS_PDF_NO_COMPRESSION", "1")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    from py_screenalytics import run_layout
    from apps.api.services.run_export import build_screentime_run_debug_pdf

    ep_id = "ep-setup-only"
    run_id = "run-setup"

    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    pdf_bytes, _name = build_screentime_run_debug_pdf(
        ep_id=ep_id,
        run_id=run_id,
        include_screentime=False,
    )
    combined = "\n".join(_ascii_strings(pdf_bytes))

    assert "Setup Pipeline Run Debug Report" in combined
    assert "Screen Time Analyze" not in combined

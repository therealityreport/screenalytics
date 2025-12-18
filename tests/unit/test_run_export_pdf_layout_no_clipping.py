"""Regression tests for PDF layout (no clipping/overflow) in the lineage/runtime diagnostics table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_run_export_pdf_no_text_overflows_page_bounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fitz = pytest.importorskip("fitz")  # PyMuPDF

    sys.path.insert(0, str(PROJECT_ROOT))
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SCREENALYTICS_PDF_NO_COMPRESSION", "1")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    from py_screenalytics import run_layout
    from apps.api.services import run_export

    # Assert the lineage table is built via the wrap-safe Paragraph-based helper.
    original_builder = run_export.build_wrap_safe_kv_table
    builder_called: dict[str, bool] = {"called": False}

    def _wrapped_builder(*args, **kwargs):
        table = original_builder(*args, **kwargs)
        from reportlab.platypus import Paragraph

        assert hasattr(table, "_cellvalues")
        assert all(isinstance(cell, Paragraph) for row in table._cellvalues for cell in row)
        builder_called["called"] = True
        return table

    monkeypatch.setattr(run_export, "build_wrap_safe_kv_table", _wrapped_builder)

    ep_id = "ep-test"
    run_id = "run-pdf-layout"
    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    (run_root / "tracks.jsonl").write_text(
        json.dumps({"track_id": 1, "first_ts": 0.0, "last_ts": 1.0}) + "\n",
        encoding="utf-8",
    )

    unbreakable = "X" * 320
    long_reason = (
        "TensorRT backend requires CUDA-enabled runtime; "
        f"fallback_reason=missing_cuda unbreakable={unbreakable} "
        "model_path=models/insightface/arcface_r100_v1.onnx "
        "s3_key=s3://bucket/runs/ep-test/run-pdf-layout/artifacts/detect_track.json"
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
                "scene_cut_count": 65,
                "warmup_cuts_applied": 65,
                "warmup_frames_per_cut_effective": 3.0,
                "forced_scene_warmup_ratio": 0.463,
                "detect_wall_time_s": 749.4,
                "wall_time_per_processed_frame_s": 1.216883,
                "rtf": 7.137,
                "effective_fps_processing": 0.822,
                "scene_detect_wall_time_s": 10.1,
                "tracker_backend_configured": "bytetrack",
                "tracker_backend_actual": "ultralytics.bytetrack",
                "tracker_fallback_reason": None,
                "requested_device": "coreml",
                "resolved_device": "coreml",
                "onnx_provider_requested": "coreml",
                "onnx_provider_resolved": "coreml",
                "torch_device_requested": "mps",
                "torch_device_resolved": "mps",
                "torch_device_fallback_reason": long_reason,
                "detector_model_name": "retinaface_r50",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (run_root / "faces_embed.json").write_text(
        json.dumps(
            {
                "phase": "faces_embed",
                "status": "success",
                "run_id": run_id,
                "requested_device": "coreml",
                "resolved_device": "coreml",
                "embedding_backend_configured": "tensorrt",
                "embedding_backend_actual": "pytorch",
                "embedding_backend_fallback_reason": long_reason,
                "embedding_model_name": "arcface_r100_v1",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pdf_bytes, _name = run_export.build_screentime_run_debug_pdf(ep_id=ep_id, run_id=run_id)
    assert builder_called["called"] is True

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for page in doc:
            rect = page.rect
            blocks = page.get_text("blocks")
            for x0, y0, x1, y1, text, *_rest in blocks:
                if not str(text).strip():
                    continue
                assert x0 >= -1.0
                assert x1 <= rect.width + 1.0
    finally:
        doc.close()


def test_run_export_pdf_no_black_squares(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test: PDF must not contain black squares (■) or replacement chars (�).

    These appear when Helvetica font cannot render certain Unicode characters like:
    - U+200B (zero-width space) - renders as black square
    - U+25A0 (black square) - should not appear in output
    - U+FFFD (replacement character) - indicates encoding issues
    """
    fitz = pytest.importorskip("fitz")  # PyMuPDF

    sys.path.insert(0, str(PROJECT_ROOT))
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SCREENALYTICS_PDF_NO_COMPRESSION", "1")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    from py_screenalytics import run_layout
    from apps.api.services import run_export

    ep_id = "rhoslc-s06e11"
    run_id = "test-black-squares"
    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    # Create test data with many separators that trigger soft-wrap logic
    long_s3_key = "s3://screenalytics-test/manifests/rhoslc-s06e11/runs/test-black-squares/analytics/screentime.json"
    long_path = "/Volumes/HardDrive/SCREENALYTICS/data/manifests/rhoslc-s06e11/runs/test-black-squares/tracks.jsonl"

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
                "torch_device_fallback_reason": f"path={long_path} s3_key={long_s3_key}",
                "detector_model_name": "retinaface_r50",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (run_root / "faces_embed.json").write_text(
        json.dumps(
            {
                "phase": "faces_embed",
                "status": "success",
                "run_id": run_id,
                "embedding_backend_fallback_reason": f"fallback_path={long_path}",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pdf_bytes, _name = run_export.build_screentime_run_debug_pdf(ep_id=ep_id, run_id=run_id)

    # Characters that should NOT appear in PDF text
    forbidden_chars = {
        "\u25a0": "BLACK SQUARE (■)",
        "\u200b": "ZERO-WIDTH SPACE",
        "\ufffd": "REPLACEMENT CHARACTER (�)",
    }

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    violations: list[str] = []
    try:
        for page_num, page in enumerate(doc, 1):
            full_text = page.get_text("text")
            for char, description in forbidden_chars.items():
                if char in full_text:
                    # Find context around the forbidden char
                    idx = full_text.find(char)
                    context_start = max(0, idx - 20)
                    context_end = min(len(full_text), idx + 20)
                    context = full_text[context_start:context_end].replace("\n", "\\n")
                    violations.append(
                        f"Page {page_num}: Found {description} at position {idx}, context: ...{context}..."
                    )
    finally:
        doc.close()

    assert not violations, f"PDF contains forbidden characters:\n" + "\n".join(violations)


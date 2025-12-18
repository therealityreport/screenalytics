"""Regression tests for PDF messaging: Re-ID runtime errors, embedding backend hints, and gate tuning guidance."""

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


def test_pdf_rewrites_torchreid_import_error_to_runtime_error_when_env_ok(
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
    run_id = "run-reid-false-negative"
    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    (run_root / "tracks.jsonl").write_text(json.dumps({"track_id": 1, "first_ts": 0.0, "last_ts": 1.0}) + "\n", encoding="utf-8")
    (run_root / "env_diagnostics.json").write_text(
        json.dumps({"python_version": "3.11", "import_status": {"torchreid": {"status": "ok", "version": "0.2.5"}}}, indent=2),
        encoding="utf-8",
    )
    (run_root / "body_tracking.json").write_text(
        json.dumps(
            {
                "phase": "body_tracking",
                "status": "success",
                "run_id": run_id,
                "reid_note": "import_error: No module named 'torchvision'",
                "body_reid": {
                    "enabled_config": True,
                    "enabled_effective": False,
                    "reid_embeddings_generated": False,
                    "reid_skip_reason": "torchreid_import_error",
                    "reid_comparisons_performed": 0,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pdf_bytes, _name = build_screentime_run_debug_pdf(ep_id=ep_id, run_id=run_id)
    combined = "\n".join(_ascii_strings(pdf_bytes))

    # When torchreid is installed, treat ImportError during FeatureExtractor init as a runtime error.
    assert "runtime_error" in combined
    assert "torchreid_installed" in combined


def test_pdf_suppresses_tensorrt_install_advice_without_cuda(
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
    run_id = "run-embed-hints"
    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    (run_root / "tracks.jsonl").write_text(json.dumps({"track_id": 1, "first_ts": 0.0, "last_ts": 1.0}) + "\n", encoding="utf-8")
    (run_root / "faces_embed.json").write_text(
        json.dumps(
            {
                "phase": "faces_embed",
                "status": "success",
                "run_id": run_id,
                "requested_device": "cpu",
                "resolved_device": "cpu",
                "embedding_backend_configured": "tensorrt",
                "embedding_backend_actual": "pytorch",
                "embedding_backend_fallback_reason": "ImportError: please `pip install tensorrt pycuda` to enable TensorRT backend",
                "embedding_model_name": "arcface_r100_v1",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pdf_bytes, _name = build_screentime_run_debug_pdf(ep_id=ep_id, run_id=run_id)
    combined = "\n".join(_ascii_strings(pdf_bytes))

    # Never show NVIDIA-only install advice when CUDA isn't available.
    assert "pycuda" not in combined.lower()


def test_pdf_does_not_recommend_disabling_gate_when_gate_share_too_low(
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
    run_id = "run-gate-share-low"
    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    (run_root / "tracks.jsonl").write_text(json.dumps({"track_id": 1, "first_ts": 0.0, "last_ts": 1.0}) + "\n", encoding="utf-8")
    (run_root / "track_metrics.json").write_text(
        json.dumps(
            {
                "ep_id": ep_id,
                "generated_at": "now",
                "metrics": {"forced_splits": 300, "id_switches": 0},
                "scene_cuts": {"count": 0},
                "tracking_gate": {"enabled": True},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_root / "detect_track.json").write_text(
        json.dumps(
            {
                "phase": "detect_track",
                "status": "success",
                "run_id": run_id,
                "tracking_gate": {
                    "enabled": True,
                    "auto_rerun": {
                        "triggered": False,
                        "reason": "gate_share_too_low",
                        "decision": {"gate_splits_share": 0.1, "thresholds": {"min_gate_share": 0.6}},
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pdf_bytes, _name = build_screentime_run_debug_pdf(ep_id=ep_id, run_id=run_id)
    combined = "\n".join(_ascii_strings(pdf_bytes))

    assert "Disable gate_enabled in tracking.yaml" not in combined
    assert "Consider disabling gate_enabled in tracking.yaml" not in combined

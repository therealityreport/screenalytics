"""Unit tests for body Re-ID runtime diagnostics (torchreid import ok but runtime init fails)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_body_reid_marks_runtime_error_when_torchreid_import_ok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))

    from py_screenalytics import run_layout
    from tools import episode_run

    monkeypatch.setattr(episode_run, "_load_body_tracking_config", lambda: {"body_tracking": {"enabled": True}})

    ep_id = "ep-test"
    run_id = "run-body-reid-runtime"
    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    (run_root / "tracks.jsonl").write_text(json.dumps({"track_id": 1}) + "\n", encoding="utf-8")

    # Inject a minimal BodyTrackingRunner that fails during embedding initialization.
    runner_mod = types.ModuleType("FEATURES.body_tracking.src.body_tracking_runner")

    class _DummyBodyTrackingRunner:
        def __init__(
            self,
            episode_id: str,
            config_path=None,
            fusion_config_path=None,
            video_path=None,
            output_dir: Path | None = None,
            skip_existing: bool = True,
        ) -> None:
            assert output_dir is not None
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.config = types.SimpleNamespace(reid_enabled=True, reid_embedding_dim=256, reid_model="osnet_x1_0")
            self.embeddings_path = self.output_dir / "body_embeddings.npy"
            self.embeddings_meta_path = self.output_dir / "body_embeddings_meta.json"
            self.metrics_path = self.output_dir / "body_metrics.json"
            self.tracker_backend_configured = "bytetrack"
            self.tracker_backend_actual = "bytetrack"
            self.tracker_fallback_reason = None

        def run_detection(self) -> Path:
            path = self.output_dir / "body_detections.jsonl"
            path.write_text(json.dumps({"frame_idx": 0}) + "\n", encoding="utf-8")
            return path

        def run_tracking(self) -> Path:
            path = self.output_dir / "body_tracks.jsonl"
            path.write_text(json.dumps({"track_id": 100000}) + "\n", encoding="utf-8")
            return path

        def run_embedding(self) -> Path:
            raise ImportError("No module named 'torchvision'")

    runner_mod.BodyTrackingRunner = _DummyBodyTrackingRunner

    # Ensure the package hierarchy exists for `from FEATURES.body_tracking.src.body_tracking_runner import BodyTrackingRunner`.
    features_pkg = types.ModuleType("FEATURES")
    features_pkg.__path__ = []
    body_pkg = types.ModuleType("FEATURES.body_tracking")
    body_pkg.__path__ = []
    src_pkg = types.ModuleType("FEATURES.body_tracking.src")
    src_pkg.__path__ = []

    monkeypatch.setitem(sys.modules, "FEATURES", features_pkg)
    monkeypatch.setitem(sys.modules, "FEATURES.body_tracking", body_pkg)
    monkeypatch.setitem(sys.modules, "FEATURES.body_tracking.src", src_pkg)
    monkeypatch.setitem(sys.modules, "FEATURES.body_tracking.src.body_tracking_runner", runner_mod)

    payload = episode_run._maybe_run_body_tracking(
        ep_id=ep_id,
        run_id=run_id,
        effective_run_id=run_id,
        video_path=tmp_path / "video.mp4",
        import_status={"torchreid": {"status": "ok", "version": "0.2.5"}},
    )
    assert isinstance(payload, dict)
    assert payload.get("status") == "success"

    body_reid = payload.get("body_reid")
    assert isinstance(body_reid, dict)
    assert body_reid.get("torchreid_import_ok") is True
    assert body_reid.get("torchreid_runtime_ok") is False
    assert body_reid.get("reid_skip_reason") == "torchreid_runtime_error"
    assert "torchvision" in str(body_reid.get("torchreid_runtime_error"))

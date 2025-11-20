from __future__ import annotations

import sys
import types
from pathlib import Path

from apps.api.services.jobs import JobService
from py_screenalytics.artifacts import ensure_dirs, get_path
from tools import episode_run


class _FakeMPS:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


def _install_fake_torch(monkeypatch, available: bool) -> None:
    fake_torch = types.SimpleNamespace(backends=types.SimpleNamespace(mps=_FakeMPS(available)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_coreml_prefers_mps(monkeypatch):
    _install_fake_torch(monkeypatch, True)
    assert episode_run.resolve_device("coreml") == "mps"


def test_coreml_falls_back_to_cpu(monkeypatch, tmp_path):
    _install_fake_torch(monkeypatch, False)
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    service = JobService(data_root=data_root)
    ep_id = "device-resolve"
    ensure_dirs(ep_id)
    video_path = get_path(ep_id, "video")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"\x00\x01video")

    captured: dict = {}

    def _fake_launch_job(self, **kwargs):
        captured["command"] = kwargs.get("command")
        captured["requested"] = kwargs.get("requested")
        return {
            "job_id": "fake",
            "state": "running",
            "progress_file": str(kwargs.get("progress_path")),
            "requested": kwargs.get("requested"),
            "command": kwargs.get("command"),
        }

    monkeypatch.setattr(JobService, "_launch_job", _fake_launch_job)
    monkeypatch.setattr(service, "ensure_retinaface_ready", lambda *a, **k: "cpu")

    job = service.start_detect_track_job(
        ep_id=ep_id,
        stride=4,
        fps=None,
        device="coreml",
        video_path=video_path,
        save_frames=False,
        save_crops=False,
        jpeg_quality=85,
        detector="retinaface",
        tracker="bytetrack",
        max_gap=30,
        det_thresh=0.5,
        scene_detector="pyscenedetect",
        scene_threshold=27.0,
        scene_min_len=12,
        scene_warmup_dets=3,
        track_high_thresh=None,
        new_track_thresh=None,
        track_buffer=None,
        min_box_area=None,
        profile=None,
        cpu_threads=None,
    )

    assert job["requested"]["device"] == "coreml"
    assert job["requested"]["device_resolved"] == "cpu"
    device_index = captured["command"].index("--device") + 1
    assert captured["command"][device_index] == "coreml"

from __future__ import annotations

import types
from pathlib import Path

import tools.episode_run as episode_run


class _FakeProgress:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.target_frames = 0
        self.static_fields: dict = {}

    def set_static_fields(self, fields: dict) -> None:
        # Store static fields so emit can merge them, mirroring ProgressEmitter behavior
        self.static_fields = dict(fields)

    def emit(self, frames_done: int, *, phase: str, summary: dict | None = None, extra=None, force: bool | None = None):
        combined_summary = dict(self.static_fields)
        if summary:
            combined_summary.update(summary)
        self.events.append({"phase": phase, "summary": combined_summary, "frames_done": frames_done})


def test_scene_fallback_on_pyscenedetect_failure(monkeypatch, tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(episode_run, "detect_scene_cuts_pyscenedetect", _boom)
    progress = _FakeProgress()

    monkeypatch.setattr(episode_run, "_detect_scene_cuts_histogram", lambda *a, **k: [])
    cuts = episode_run.detect_scene_cuts(video_path, detector="pyscenedetect", progress=progress)
    assert cuts == []
    assert progress.events, "progress events should be emitted"
    summary = progress.events[-1]["summary"]
    assert summary["detector_resolved"] == "internal"
    assert summary["scene_mode_resolved"] == "internal"


def test_scene_fallback_when_opencv_cannot_open(monkeypatch, tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")

    monkeypatch.setattr(episode_run, "_opencv_can_open", lambda path: False)
    monkeypatch.setattr(episode_run, "_opencv_has_ffmpeg", lambda: False)
    monkeypatch.setattr(episode_run, "_detect_scene_cuts_histogram", lambda *a, **k: [])
    progress = _FakeProgress()

    cuts = episode_run.detect_scene_cuts(video_path, detector="pyscenedetect", progress=progress)
    assert cuts == []
    summary = progress.events[-1]["summary"]
    assert summary["detector_resolved"] == "internal"
    assert summary["scene_mode_resolved"] == "internal"


def test_scene_requested_resolved_populated(monkeypatch, tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")

    monkeypatch.setattr(episode_run, "_opencv_has_ffmpeg", lambda: False)
    # Force internal to avoid real decode
    monkeypatch.setattr(episode_run, "_detect_scene_cuts_histogram", lambda *a, **k: [])
    progress = _FakeProgress()
    cuts = episode_run.detect_scene_cuts(video_path, detector="internal", progress=progress)
    assert cuts == []
    summary = progress.events[-1]["summary"]
    assert summary["detector_requested"] == "internal"
    assert summary["detector_resolved"] == "internal"
    assert summary["scene_mode_resolved"] == "internal"
    assert summary["decode_backend"] == "opencv"

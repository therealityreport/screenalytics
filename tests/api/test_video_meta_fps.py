from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from py_screenalytics.artifacts import ensure_dirs, get_path


def _make_video(path: Path, fps: int = 5, frames: int = 10) -> None:
    import cv2  # type: ignore

    width, height = 64, 64
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():  # pragma: no cover - guard
        raise RuntimeError("Failed to create test video")
    for idx in range(frames):
        frame = np.full((height, width, 3), idx * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.mark.timeout(30)
def test_video_meta_reports_fps(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    ep_id = "test-s01e01"
    ensure_dirs(ep_id)
    video_path = get_path(ep_id, "video")
    _make_video(video_path, fps=5, frames=25)

    client = TestClient(app)
    resp = client.get(f"/episodes/{ep_id}/video_meta")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["local_exists"] is True
    assert abs(payload["fps_detected"] - 5.0) < 0.5
    assert payload["frames"] == 25
    assert payload["duration_sec"] == pytest.approx(5.0)

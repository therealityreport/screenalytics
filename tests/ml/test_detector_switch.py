from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional ML dependency
    pytest.skip("numpy is required for ML detector tests", allow_module_level=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ML_TESTS = os.environ.get("RUN_ML_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_ML_TESTS, reason="set RUN_ML_TESTS=1 to run ML integration tests"
)


def _make_sample_video(target: Path, frame_count: int = 8, size: tuple[int, int] = (64, 64)) -> Path:
    import cv2  # type: ignore

    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(target), fourcc, 10, (width, height))
    if not writer.isOpened():  # pragma: no cover
        raise RuntimeError("Unable to create synthetic video for ML test")
    for idx in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        offset = 5 + idx * 2
        cv2.rectangle(frame, (offset, offset), (offset + 12, offset + 16), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    return target


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_progress(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.timeout(120)
@pytest.mark.parametrize("detector", ["retinaface", "yolov8face"])
def test_detector_switch_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detector: str) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    video_path = _make_sample_video(tmp_path / "sample.mp4")
    ep_id = f"det-switch-{detector}"

    cmd = [
        sys.executable,
        "tools/episode_run.py",
        "--ep-id",
        ep_id,
        "--video",
        str(video_path),
        "--stride",
        "5",
        "--device",
        "cpu",
        "--detector",
        detector,
        "--stub",
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.returncode == 0

    manifest_dir = data_root / "manifests" / ep_id
    detections_path = manifest_dir / "detections.jsonl"
    tracks_path = manifest_dir / "tracks.jsonl"
    progress_path = manifest_dir / "progress.json"

    assert detections_path.exists()
    assert tracks_path.exists()
    assert progress_path.exists()

    det_rows = _read_jsonl(detections_path)
    assert det_rows, "expected stub detections"
    assert all(row["class"] == "face" for row in det_rows)
    assert all(row["detector"] == detector for row in det_rows)

    progress_payload = _read_progress(progress_path)
    assert progress_payload.get("detector") == detector
    assert progress_payload.get("tracker") == "bytetrack"
    summary = progress_payload.get("summary") or {}
    assert summary.get("detector") == detector
    assert summary.get("tracker") == "bytetrack"

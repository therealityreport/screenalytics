from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ML_TESTS = os.environ.get("RUN_ML_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_ML_TESTS, reason="set RUN_ML_TESTS=1 to run ML integration tests"
)


def _make_sample_video(target: Path, frame_count: int = 12, size: tuple[int, int] = (96, 96)) -> Path:
    import cv2  # type: ignore

    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(target), fourcc, 10, (width, height))
    if not writer.isOpened():  # pragma: no cover - sanity guard
        raise RuntimeError("Unable to create synthetic video for ML test")
    for idx in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        offset = 5 + idx * 2
        cv2.rectangle(frame, (offset, offset), (offset + 20, offset + 30), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    return target


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@pytest.mark.timeout(120)
def test_detect_track_real_pipeline(tmp_path: Path) -> None:

    data_root = tmp_path / "data"
    video_path = _make_sample_video(tmp_path / "sample.mp4")
    ep_id = "ml-test-episode"

    cmd = [
        sys.executable,
        "tools/episode_run.py",
        "--ep-id",
        ep_id,
        "--video",
        str(video_path),
        "--stride",
        "1",
        "--out-root",
        str(data_root),
    ]
    env = os.environ.copy()
    completed = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    manifest_root = data_root / "manifests" / ep_id
    detections_path = manifest_root / "detections.jsonl"
    tracks_path = manifest_root / "tracks.jsonl"

    assert detections_path.exists(), completed.stderr
    assert tracks_path.exists(), completed.stderr

    det_rows = _read_jsonl(detections_path)
    track_rows = _read_jsonl(tracks_path)

    assert det_rows, "Expected at least one detection from YOLOv8 pipeline"
    assert track_rows, "Expected at least one track from ByteTrack pipeline"

    assert det_rows[0]["model"].startswith("yolov8")
    assert det_rows[0]["tracker"] == "bytetrack"
    assert any(row.get("track_id") is not None for row in det_rows), "Detections lack track IDs"
    assert track_rows[0]["frame_count"] >= 1
    assert track_rows[0]["pipeline_ver"]

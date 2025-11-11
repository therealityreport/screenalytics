import json
import sys
from pathlib import Path

import cv2  # type: ignore
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "FEATURES" / "detection" / "src"
sys.path.append(str(SRC_DIR))

from run_retinaface import load_config, run_detection  # noqa: E402


def _make_dummy_video(path: Path, frames: int = 4, size: tuple[int, int] = (48, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        size,
    )
    for idx in range(frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        cv2.rectangle(
            frame,
            (5 + idx, 5 + idx),
            (size[0] - 5, size[1] - 5),
            (idx * 40 % 255, 128, 255 - idx * 40 % 255),
            -1,
        )
        writer.write(frame)
    writer.release()


def test_detection_pipeline_emits_det_v1(tmp_path, monkeypatch):
    monkeypatch.setenv("SCREENALYTICS_VISION_STUB", "1")
    video_path = tmp_path / "sample.mp4"
    _make_dummy_video(video_path)

    cfg = load_config(Path("config/pipeline/detection.yaml"))
    cfg["force_stub"] = True
    out_path = tmp_path / "detections.jsonl"

    count = run_detection(
        ep_id="ep_demo",
        video_path=video_path,
        output_path=out_path,
        cfg=cfg,
        frame_plan=None,
        sample_stride=1,
    )

    assert count > 0
    detections = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert all(det["schema_version"] == "det_v1" for det in detections)
    assert all(len(det["bbox"]) == 4 for det in detections)
    assert all(len(det["landmarks"]) == 10 for det in detections)
    frame_indices = [det["frame_idx"] for det in detections]
    assert all(isinstance(idx, int) and idx >= 0 for idx in frame_indices)

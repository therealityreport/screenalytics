import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "FEATURES" / "detection" / "src"
sys.path.append(str(SRC_DIR))

from run_retinaface import run_detection  # noqa: E402


def test_detection_config_keys_present():
    cfg_path = REPO_ROOT / "config" / "pipeline" / "detection.yaml"
    data = (
        yaml.safe_load(cfg_path.read_text())
        if yaml
        else _fallback_parse(cfg_path.read_text())
    )
    assert {"model_id", "min_size", "confidence_th", "iou_th"}.issubset(data.keys())


def _fallback_parse(content: str):
    data = {}
    for line in content.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        try:
            data[key.strip()] = float(value)
        except ValueError:
            data[key.strip()] = value
    return data


def test_run_detection_with_plan(tmp_path):
    import cv2  # type: ignore
    import numpy as np

    # fabricate frame images referenced by the plan
    frames = []
    for idx in range(2):
        img = np.full((32, 32, 3), idx * 60, dtype=np.uint8)
        frame_path = tmp_path / f"frame_{idx}.png"
        cv2.imwrite(str(frame_path), img)
        frames.append({"ep_id": "ep1", "frame_idx": idx, "ts_s": idx * 0.33, "image_path": str(frame_path)})

    manifest = tmp_path / "frames.jsonl"
    manifest.write_text("\n".join(json.dumps(fr) for fr in frames))

    out_path = tmp_path / "detections.jsonl"
    cfg = {"model_id": "retinaface_r50", "force_stub": True}
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"not-a-video")  # never consumed because plan has image_path
    count = run_detection(
        ep_id="ep1",
        video_path=dummy_video,
        output_path=out_path,
        cfg=cfg,
        frame_plan=manifest,
        sample_stride=1,
    )

    lines = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert count == len(frames) == len(lines)
    sample = lines[0]
    assert sample["schema_version"] == "det_v1"
    assert sample["model_id"] == "retinaface_r50"
    assert len(sample["bbox"]) == 4
    assert len(sample["landmarks"]) == 10

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

from run_retinaface import detect_frames  # noqa: E402


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


def test_detect_frames_emits_schema(tmp_path):
    manifest = tmp_path / "frames.jsonl"
    frames = [
        {"ep_id": "ep1", "ts_s": 0.0},
        {"ep_id": "ep1", "ts_s": 0.5, "bbox": [0.1, 0.2, 0.8, 0.9]},
    ]
    manifest.write_text("\n".join(json.dumps(fr) for fr in frames))

    out_path = tmp_path / "detections.jsonl"
    cfg = {"model_id": "retinaface_r50", "confidence_th": 0.9}
    count = detect_frames(manifest, out_path, cfg)

    lines = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert count == len(frames) == len(lines)
    sample = lines[0]
    assert sample["schema_version"] == "det_v1"
    assert sample["model_id"] == "retinaface_r50"
    assert len(sample["bbox"]) == 4
    assert len(sample["landmarks"]) == 10

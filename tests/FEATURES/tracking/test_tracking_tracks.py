import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "FEATURES" / "tracking" / "src"
sys.path.append(str(SRC_DIR))

from bytetrack_runner import load_config, run_tracking  # noqa: E402


def test_tracking_produces_consistent_track_ids(tmp_path):
    detections_path = tmp_path / "detections.jsonl"
    detections = [
        {
            "ep_id": "ep_demo",
            "frame_idx": 0,
            "ts_s": 0.0,
            "bbox": [0.1, 0.1, 0.3, 0.3],
            "conf": 0.95,
        },
        {
            "ep_id": "ep_demo",
            "frame_idx": 1,
            "ts_s": 0.5,
            "bbox": [0.11, 0.11, 0.31, 0.31],
            "conf": 0.96,
        },
        {
            "ep_id": "ep_demo",
            "frame_idx": 0,
            "ts_s": 0.0,
            "bbox": [0.6, 0.6, 0.8, 0.8],
            "conf": 0.9,
        },
        {
            "ep_id": "ep_demo",
            "frame_idx": 1,
            "ts_s": 0.5,
            "bbox": [0.62, 0.62, 0.82, 0.82],
            "conf": 0.88,
        },
    ]
    detections_path.write_text("\n".join(json.dumps(det) for det in detections))

    cfg = load_config(Path("config/pipeline/tracking.yaml"))
    cfg["track_thresh"] = 0.0  # ensure all detections are kept for the test
    cfg["match_thresh"] = 0.5
    out_path = tmp_path / "tracks.jsonl"

    count = run_tracking(detections_path, out_path, cfg)
    records = [json.loads(line) for line in out_path.read_text().splitlines()]

    assert count == len(records) == 2
    track_ids = {rec["track_id"] for rec in records}
    assert track_ids == {1, 2}
    track_labels = {rec.get("track_label") for rec in records}
    assert track_labels == {"track_0001", "track_0002"}
    for rec in records:
        assert rec["schema_version"] == "track_v1"
        assert rec["stats"]["detections"] == 2
        assert rec["stats"]["avg_conf"] > 0.0

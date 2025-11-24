import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "FEATURES" / "tracking" / "src"
sys.path.append(str(SRC_DIR))

from bytetrack_runner import build_tracks, main  # noqa: E402


def test_build_tracks_generates_ids():
    detections = [
        {
            "ep_id": "ep1",
            "ts_s": 0.2,
            "frame_idx": 0,
            "conf": 0.9,
            "bbox": [0.1, 0.1, 0.3, 0.3],
        },
        {
            "ep_id": "ep1",
            "ts_s": 1.0,
            "frame_idx": 1,
            "conf": 0.95,
            "bbox": [0.1, 0.1, 0.3, 0.3],
        },
    ]
    cfg = {"track_thresh": 0.0, "match_thresh": 0.1, "track_buffer": 1}
    tracks = list(build_tracks(detections, cfg))
    assert len(tracks) == 1
    track = tracks[0]
    assert isinstance(track["track_id"], int)
    assert track["track_id"] > 0
    # track_label is the formatted string used for artifacts
    assert track.get("track_label", "").startswith("track_")
    assert tracks[0]["schema_version"] == "track_v1"


def test_cli_writes_tracks(tmp_path):
    detections_file = tmp_path / "detections.jsonl"
    rows = [
        {"ep_id": "ep1", "ts_s": 0.0, "conf": 0.9},
        {"ep_id": "ep2", "ts_s": 3.5, "conf": 0.92},
    ]
    detections_file.write_text("\n".join(json.dumps(r) for r in rows))

    output_path = tmp_path / "tracks.jsonl"
    main(
        [
            "--detections",
            str(detections_file),
            "--output",
            str(output_path),
            "--config",
            str(REPO_ROOT / "config" / "pipeline" / "tracking.yaml"),
        ]
    )

    lines = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(lines) == len(rows)
    assert all("track_id" in line for line in lines)
    assert all(line["schema_version"] == "track_v1" for line in lines)

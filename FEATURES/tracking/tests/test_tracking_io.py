import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "FEATURES" / "tracking" / "src"
sys.path.append(str(SRC_DIR))

from bytetrack_runner import build_tracks, main  # noqa: E402


def test_build_tracks_generates_ids():
    detections = [{"ep_id": "ep1", "ts_s": 0.2}, {"ep_id": "ep1", "ts_s": 1.0}]
    tracks = build_tracks(detections)
    assert len(tracks) == 2
    assert tracks[0]["track_id"].startswith("track-")
    assert tracks[0]["schema_version"] == "track_v1"


def test_cli_writes_tracks(tmp_path):
    detections_file = tmp_path / "detections.jsonl"
    rows = [{"ep_id": "ep1", "ts_s": 0.0}, {"ep_id": "ep2", "ts_s": 3.5}]
    detections_file.write_text("\n".join(json.dumps(r) for r in rows))

    output_path = tmp_path / "tracks.jsonl"
    main(["--detections", str(detections_file), "--output", str(output_path)])

    lines = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(lines) == len(rows)
    assert all("track_id" in line for line in lines)
    assert all(line["schema_version"] == "track_v1" for line in lines)

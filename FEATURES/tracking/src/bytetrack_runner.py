"""Tracking scaffolding for ByteTrack stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

SCHEMA_VERSION = "track_v1"


def load_detections(path: Path) -> List[Dict[str, Any]]:
    detections = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            detections.append(json.loads(line))
    return detections


def build_tracks(detections: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tracks = []
    for idx, det in enumerate(detections, start=1):
        ts = det.get("ts_s", 0.0)
        track = {
            "track_id": f"track-{idx:05d}",
            "ep_id": det.get("ep_id", "unknown-episode"),
            "start_s": ts,
            "end_s": ts,
            "frame_span": det.get("frame_span", [ts, ts]),
            "sample_thumbs": det.get("sample_thumbs", []),
            "schema_version": SCHEMA_VERSION,
        }
        tracks.append(track)
    return tracks


def write_tracks(tracks: Iterable[Dict[str, Any]], out_path: Path) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for track in tracks:
            fh.write(json.dumps(track) + "\n")
            count += 1
    return count


def run(detections_path: Path, output_path: Path) -> int:
    detections = load_detections(detections_path)
    tracks = build_tracks(detections)
    return write_tracks(tracks, output_path)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ByteTrack stub runner")
    parser.add_argument("--detections", required=True, help="Path to detections.jsonl")
    parser.add_argument("--output", required=True, help="Path to write tracks.jsonl")
    args = parser.parse_args(argv)
    written = run(Path(args.detections), Path(args.output))
    print(f"Wrote {written} tracks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

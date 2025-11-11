"""ByteTrack-inspired tracker that consumes det_v1 JSONL records.

The implementation favors deterministic behaviour suitable for CI:

* Detections are filtered by `track_thresh`.
* Tracks are matched via IoU with a configurable `match_thresh`.
* Tracks remain active for `track_buffer` frames of inactivity.
* Output adheres to `track_v1` (DATA_SCHEMA.md) and adds basic stats.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

SCHEMA_VERSION = "track_v1"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from py_screenalytics.artifacts import ensure_dirs, get_path


def load_config(path: Path) -> Dict[str, float]:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text())
    except ModuleNotFoundError:  # pragma: no cover
        cfg: Dict[str, float] = {}
        for line in path.read_text().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            cfg[key.strip()] = float(value.strip())
        return cfg


def load_detections(path: Path) -> List[Dict[str, object]]:
    detections: List[Dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            detections.append(json.loads(raw))
    return detections


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0 else 0.0


@dataclass
class Track:
    track_id: int
    ep_id: str
    start_s: float
    end_s: float
    start_frame: int
    end_frame: int
    bbox: List[float]
    last_frame_idx: int
    hits: int = 0
    conf_sum: float = 0.0

    def update(self, detection: Dict[str, object]) -> None:
        frame_idx = int(detection.get("frame_idx", self.last_frame_idx))
        ts_s = float(detection.get("ts_s", self.end_s))
        bbox = list(detection.get("bbox", self.bbox))
        conf = float(detection.get("conf", 0.0))

        self.end_s = ts_s
        self.end_frame = frame_idx
        self.last_frame_idx = frame_idx
        self.bbox = bbox
        self.hits += 1
        self.conf_sum += conf

    def to_record(self) -> Dict[str, object]:
        avg_conf = self.conf_sum / self.hits if self.hits else 0.0
        return {
            "track_id": f"track-{self.track_id:05d}",
            "ep_id": self.ep_id,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "frame_span": [self.start_frame, self.end_frame],
            "sample_thumbs": [],
            "stats": {"detections": self.hits, "avg_conf": avg_conf},
            "schema_version": SCHEMA_VERSION,
        }


class ByteTrackLite:
    def __init__(self, match_thresh: float, track_buffer: int):
        self.match_thresh = match_thresh
        self.track_buffer = int(track_buffer)
        self._tracks: List[Track] = []
        self._next_id = 1

    def _expire_tracks(self, current_frame: int) -> List[Track]:
        alive: List[Track] = []
        expired: List[Track] = []
        for track in self._tracks:
            if current_frame - track.last_frame_idx > self.track_buffer:
                expired.append(track)
            else:
                alive.append(track)
        self._tracks = alive
        return expired

    def update(self, detection: Dict[str, object]) -> List[Track]:
        frame_idx = int(detection.get("frame_idx", 0))
        expired = self._expire_tracks(frame_idx)
        candidates = list(self._tracks)
        best_track = None
        best_iou = 0.0
        bbox = detection.get("bbox")
        if isinstance(bbox, list):
            for track in candidates:
                overlap = iou(track.bbox, bbox)
                if overlap > self.match_thresh and overlap > best_iou:
                    best_track = track
                    best_iou = overlap

        if best_track:
            best_track.update(detection)
        else:
            track = Track(
                track_id=self._next_id,
                ep_id=str(detection.get("ep_id", "unknown")),
                start_s=float(detection.get("ts_s", 0.0)),
                end_s=float(detection.get("ts_s", 0.0)),
                start_frame=int(detection.get("frame_idx", 0)),
                end_frame=int(detection.get("frame_idx", 0)),
                bbox=list(detection.get("bbox", [0, 0, 0, 0])),
                last_frame_idx=int(detection.get("frame_idx", 0)),
                hits=0,
                conf_sum=0.0,
            )
            track.update(detection)
            self._tracks.append(track)
            self._next_id += 1
        return expired

    def flush(self) -> List[Track]:
        expired = list(self._tracks)
        self._tracks = []
        return expired


def build_tracks(
    detections: Iterable[Dict[str, object]], cfg: Dict[str, float]
) -> Iterator[Dict[str, object]]:
    track_thresh = float(cfg.get("track_thresh", 0.5))
    tracker = ByteTrackLite(
        match_thresh=float(cfg.get("match_thresh", 0.8)),
        track_buffer=int(cfg.get("track_buffer", 30)),
    )
    ordered = sorted(
        detections,
        key=lambda det: (int(det.get("frame_idx", 0)), float(det.get("ts_s", 0.0))),
    )
    for det in ordered:
        if float(det.get("conf", 0.0)) < track_thresh:
            continue
        expired = tracker.update(det)
        for track in expired:
            yield track.to_record()
    for track in tracker.flush():
        yield track.to_record()


def write_tracks(records: Iterable[Dict[str, object]], output_path: Path) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row) + "\n")
            count += 1
    return count


def run_tracking(
    detections_path: Path,
    output_path: Path,
    cfg: Dict[str, float],
) -> int:
    detections = load_detections(detections_path)
    records = list(build_tracks(detections, cfg))
    return write_tracks(records, output_path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ByteTrack-lite runner")
    parser.add_argument("--ep-id", help="Episode id (used for default paths)")
    parser.add_argument("--detections", help="Path to det_v1 JSONL (defaults via resolver)")
    parser.add_argument("--output", help="Path for track_v1 JSONL (defaults via resolver)")
    parser.add_argument(
        "--config",
        default="config/pipeline/tracking.yaml",
        help="Tracking YAML config path",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(Path(args.config))

    if args.detections:
        detections_path = Path(args.detections)
    else:
        if not args.ep_id:
            raise SystemExit("--ep-id is required when --detections is omitted")
        ensure_dirs(args.ep_id)
        detections_path = get_path(args.ep_id, "detections")

    if args.output:
        output_path = Path(args.output)
    else:
        if not args.ep_id:
            raise SystemExit("--ep-id is required when --output is omitted")
        ensure_dirs(args.ep_id)
        output_path = get_path(args.ep_id, "tracks")

    written = run_tracking(detections_path, output_path, cfg)
    print(f"Wrote {written} tracks to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

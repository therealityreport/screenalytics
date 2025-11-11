#!/usr/bin/env python
"""Dev-only CLI to run detection → tracking for a single episode."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_screenalytics.artifacts import ensure_dirs, get_path

PIPELINE_VERSION = os.environ.get("SCREENALYTICS_PIPELINE_VERSION", "2025-11-11")
YOLO_MODEL_NAME = os.environ.get("SCREENALYTICS_YOLO_MODEL", "yolov8n.pt")
TRACKER_CONFIG = os.environ.get("SCREENALYTICS_TRACKER_CONFIG", "bytetrack.yaml")
TRACKER_NAME = Path(TRACKER_CONFIG).stem if TRACKER_CONFIG else "bytetrack"
YOLO_IMAGE_SIZE = int(os.environ.get("SCREENALYTICS_YOLO_IMGSZ", 640))
YOLO_CONF_THRESHOLD = float(os.environ.get("SCREENALYTICS_YOLO_CONF", 0.25))
YOLO_DEVICE = os.environ.get("SCREENALYTICS_YOLO_DEVICE", "auto")
TRACK_SAMPLE_LIMIT = int(os.environ.get("SCREENALYTICS_TRACK_SAMPLE_LIMIT", 5))


@dataclass
class TrackAccumulator:
    track_id: int
    class_id: int
    first_ts: float
    last_ts: float
    frame_count: int = 0
    samples: List[dict] = field(default_factory=list)

    def add(self, ts: float, frame_idx: int, bbox_xyxy: List[float]) -> None:
        self.frame_count += 1
        self.last_ts = ts
        if len(self.samples) < TRACK_SAMPLE_LIMIT:
            self.samples.append(
                {
                    "frame_idx": frame_idx,
                    "ts": round(float(ts), 4),
                    "bbox_xyxy": [round(float(coord), 4) for coord in bbox_xyxy],
                }
            )

    def to_row(self) -> dict:
        row = {
            "track_id": self.track_id,
            "class": self.class_id,
            "first_ts": round(float(self.first_ts), 4),
            "last_ts": round(float(self.last_ts), 4),
            "frame_count": self.frame_count,
            "pipeline_ver": PIPELINE_VERSION,
        }
        if self.samples:
            row["bboxes_sampled"] = self.samples
        return row


def _copy_video(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dest.resolve():
        return
    shutil.copy2(src, dest)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detection + tracking locally.")
    parser.add_argument("--ep-id", required=True, help="Episode identifier")
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--stride", type=int, default=5, help="Frame stride for detection sampling")
    parser.add_argument(
        "--fps",
        type=float,
        help="Optional target FPS for downsampling before detection",
    )
    parser.add_argument(
        "--out-root",
        help="Data root override (defaults to SCREENALYTICS_DATA_ROOT or ./data)",
    )
    parser.add_argument("--stub", action="store_true", help="Use stub detections (fast, no ML)")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = (
        Path(args.out_root).expanduser()
        if args.out_root
        else Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
    )
    os.environ["SCREENALYTICS_DATA_ROOT"] = str(data_root)
    ensure_dirs(args.ep_id)

    video_src = Path(args.video)
    if not video_src.exists():
        raise FileNotFoundError(f"Video not found: {video_src}")
    video_dest = get_path(args.ep_id, "video")
    _copy_video(video_src, video_dest)

    if args.stub:
        det_count, track_count, frames_sampled = _run_stub_pipeline(args.ep_id)
    else:
        det_count, track_count, frames_sampled = _run_full_pipeline(
            args,
            video_dest,
        )

    summary = {
        "ep_id": args.ep_id,
        "detections": det_count,
        "tracks": track_count,
        "frames_sampled": frames_sampled,
        "data_root": str(data_root),
    }
    print("[episode_run]", summary)
    return 0


def _run_stub_pipeline(ep_id: str) -> Tuple[int, int, int]:
    det_path = get_path(ep_id, "detections")
    track_path = get_path(ep_id, "tracks")
    det_rows = []
    for idx in range(3):
        ts = idx * 0.5
        det_rows.append(
            {
                "ep_id": ep_id,
                "ts": round(ts, 4),
                "frame_idx": idx,
                "class": 0,
                "conf": 0.99,
                "bbox_xyxy": [
                    round(50 + idx * 5, 1),
                    round(60 + idx * 5, 1),
                    round(150 + idx * 5, 1),
                    round(160 + idx * 5, 1),
                ],
                "track_id": 1,
                "model": YOLO_MODEL_NAME,
                "tracker": TRACKER_NAME,
                "pipeline_ver": PIPELINE_VERSION,
            }
        )
    track_rows = [
        {
            "ep_id": ep_id,
            "track_id": 1,
            "class": 0,
            "first_ts": 0.0,
            "last_ts": round((len(det_rows) - 1) * 0.5, 4),
            "frame_count": len(det_rows),
            "bboxes_sampled": [
                {
                    "frame_idx": row["frame_idx"],
                    "ts": row["ts"],
                    "bbox_xyxy": row["bbox_xyxy"],
                }
                for row in det_rows
            ],
            "pipeline_ver": PIPELINE_VERSION,
        }
    ]
    _write_jsonl(det_path, det_rows)
    _write_jsonl(track_path, track_rows)
    return len(det_rows), len(track_rows), len(det_rows)


def _effective_stride(stride: int, target_fps: float | None, source_fps: float) -> int:
    stride = max(stride, 1)
    if target_fps and target_fps > 0 and source_fps > 0:
        fps_stride = max(int(round(source_fps / target_fps)), 1)
        stride = max(stride, fps_stride)
    return stride


def _probe_video(video_path: Path) -> Tuple[float, int]:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0:
        fps = 30.0
    return fps, frame_count


def _run_full_pipeline(args: argparse.Namespace, video_dest: Path) -> Tuple[int, int, int]:
    from ultralytics import YOLO  # type: ignore

    source_fps, _ = _probe_video(video_dest)
    frame_stride = _effective_stride(args.stride, args.fps, source_fps)
    ts_fps = source_fps if source_fps > 0 else max(args.fps or 30.0, 1.0)
    det_path = get_path(args.ep_id, "detections")
    track_path = get_path(args.ep_id, "tracks")
    track_acc: Dict[int, TrackAccumulator] = {}
    det_count = 0
    frames_sampled = 0

    def detection_rows() -> Iterator[dict]:
        nonlocal det_count, frames_sampled
        model = YOLO(YOLO_MODEL_NAME)
        results = model.track(
            source=str(video_dest),
            stream=True,
            tracker=TRACKER_CONFIG,
            imgsz=YOLO_IMAGE_SIZE,
            conf=YOLO_CONF_THRESHOLD,
            device=YOLO_DEVICE,
            vid_stride=frame_stride,
            persist=True,
        )
        for processed_idx, result in enumerate(results):
            frames_sampled = processed_idx + 1
            frame_idx = int(processed_idx * frame_stride)
            ts = frame_idx / ts_fps
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.data is None:
                continue
            num_boxes = len(boxes)
            if num_boxes == 0:
                continue
            track_ids = getattr(boxes, "id", None)
            for box_idx in range(num_boxes):
                bbox_xyxy = boxes.xyxy[box_idx].tolist()
                conf = float(boxes.conf[box_idx].item())
                class_id = int(boxes.cls[box_idx].item())
                track_id_val = None
                if track_ids is not None and len(track_ids) > box_idx:
                    tid_float = float(track_ids[box_idx].item())
                    if not math.isnan(tid_float):
                        track_id_val = int(tid_float)
                row = {
                    "ep_id": args.ep_id,
                    "ts": round(float(ts), 4),
                    "frame_idx": frame_idx,
                    "class": class_id,
                    "conf": conf,
                    "bbox_xyxy": [round(float(coord), 4) for coord in bbox_xyxy],
                    "track_id": track_id_val,
                    "model": YOLO_MODEL_NAME,
                    "tracker": TRACKER_NAME,
                    "pipeline_ver": PIPELINE_VERSION,
                }
                det_count += 1
                if track_id_val is not None:
                    track = track_acc.get(track_id_val)
                    if track is None:
                        track = TrackAccumulator(
                            track_id=track_id_val,
                            class_id=class_id,
                            first_ts=ts,
                            last_ts=ts,
                        )
                        track_acc[track_id_val] = track
                    track.add(ts, frame_idx, bbox_xyxy)
                yield row

    _write_jsonl(det_path, detection_rows())
    track_rows = []
    for track in sorted(track_acc.values(), key=lambda t: t.track_id):
        row = track.to_row()
        row["ep_id"] = args.ep_id
        track_rows.append(row)
    _write_jsonl(track_path, track_rows)
    return det_count, len(track_rows), frames_sampled


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

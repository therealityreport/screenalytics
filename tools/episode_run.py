#!/usr/bin/env python
"""Dev-only CLI to run detection → tracking for a single episode."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Tuple

import cv2  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_screenalytics.artifacts import ensure_dirs, get_path
from FEATURES.detection.src.run_retinaface import (
    load_config as load_det_config,
    run_detection,
)
from FEATURES.tracking.src.bytetrack_runner import (
    load_config as load_track_config,
    run_tracking,
)


def _copy_video(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dest.resolve():
        return
    shutil.copy2(src, dest)


def _extract_frames(video_path: Path, frames_root: Path, target_fps: float) -> int:
    frames_root.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    if src_fps == 0:
        src_fps = target_fps
    step = max(int(round(src_fps / target_fps)), 1)
    saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frame_path = frames_root / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved += 1
        frame_idx += 1
    cap.release()
    return saved


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detection + tracking locally.")
    parser.add_argument("--ep-id", required=True, help="Episode identifier")
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--stride", type=int, default=5, help="Frame stride for detection sampling")
    parser.add_argument("--fps", type=float, help="Optional frame export FPS")
    parser.add_argument(
        "--out-root",
        help="Data root override (defaults to SCREENALYTICS_DATA_ROOT or ./data)",
    )
    parser.add_argument("--stub", action="store_true", help="Force RetinaFace stub mode")
    parser.add_argument(
        "--detection-config",
        default="config/pipeline/detection.yaml",
        help="Detection YAML path",
    )
    parser.add_argument(
        "--tracking-config",
        default="config/pipeline/tracking.yaml",
        help="Tracking YAML path",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = (
        Path(args.out_root).expanduser()
        if args.out_root
        else Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
    )
    os.environ["SCREENALYTICS_DATA_ROOT"] = str(data_root)
    if args.stub:
        os.environ["SCREENALYTICS_VISION_STUB"] = "1"

    ensure_dirs(args.ep_id)

    video_src = Path(args.video)
    if not video_src.exists():
        raise FileNotFoundError(f"Video not found: {video_src}")
    video_dest = get_path(args.ep_id, "video")
    _copy_video(video_src, video_dest)

    frames_exported = 0
    if args.fps:
        frames_root = get_path(args.ep_id, "frames_root")
        frames_exported = _extract_frames(video_dest, frames_root, args.fps)

    det_cfg = load_det_config(Path(args.detection_config))
    det_path = get_path(args.ep_id, "detections")
    det_count = run_detection(
        ep_id=args.ep_id,
        video_path=video_dest,
        output_path=det_path,
        cfg=det_cfg,
        frame_plan=None,
        sample_stride=args.stride,
    )

    track_cfg = load_track_config(Path(args.tracking_config))
    track_path = get_path(args.ep_id, "tracks")
    track_count = run_tracking(det_path, track_path, track_cfg)

    summary = {
        "ep_id": args.ep_id,
        "detections": det_count,
        "tracks": track_count,
        "frames_exported": frames_exported,
        "data_root": str(data_root),
    }
    print("[episode_run]", summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

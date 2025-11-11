"""RetinaFace detection runner for Screenalytics.

This module bridges config-driven RetinaFace inference with the JSONL artifacts
described in DATA_SCHEMA.md (`det_v1`).  It supports two execution modes:

1. A real InsightFace-backed RetinaFace detector (default) that runs on CPU or
   GPU depending on `ctx_id`.
2. A deterministic simulated used by tests/CI via `SCREENALYTICS_VISION_SIM=1` or
   `cfg["force_simulated"]=True`.  The simulated still exercises real I/O: frames are
   sampled with OpenCV and detections are emitted in `det_v1`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import cv2  # type: ignore
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from py_screenalytics.artifacts import ensure_dirs, get_path

SCHEMA_VERSION = "det_v1"
DEFAULT_SAMPLE_STRIDE = 5
LOGGER = logging.getLogger("screenalytics.detection")


def _ensure_float_list(values: Sequence[float], length: int) -> List[float]:
    buf = list(values)[:length]
    while len(buf) < length:
        buf.append(0.0)
    return buf


@dataclass
class FrameSample:
    """Frame metadata + pixel payload used by the detector."""

    ep_id: str
    frame_idx: int
    ts_s: float
    image: np.ndarray


def _read_plan(path: Path) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _capture_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return fps or 30.0


def _iter_video_samples(video_path: Path, ep_id: str, stride: int) -> Iterator[FrameSample]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps == 0:
        fps = 30.0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            ts_s = frame_idx / fps
            yield FrameSample(ep_id=ep_id, frame_idx=frame_idx, ts_s=ts_s, image=frame)
        frame_idx += 1
    cap.release()


class _VideoRandomAccess:
    """Lightweight helper that allows looking up arbitrary frame indices."""

    def __init__(self, video_path: Path):
        self._path = str(video_path)
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read(self, frame_idx: int) -> np.ndarray:
        if frame_idx < 0:
            raise ValueError("frame_idx must be >= 0")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {self._path}")
        return frame

    def close(self) -> None:
        self._cap.release()


def _iter_plan_samples(
    video_path: Path | None,
    plan_entries: List[Dict[str, object]],
    default_ep_id: str,
) -> Iterator[FrameSample]:
    reader: _VideoRandomAccess | None = None
    fps = 30.0
    try:
        for entry in plan_entries:
            ep_id = str(entry.get("ep_id") or default_ep_id)
            frame_idx = int(entry.get("frame_idx") or entry.get("frame_id") or 0)
            image = None
            image_path = entry.get("image_path")
            if image_path:
                image = cv2.imread(str(image_path))
            elif video_path:
                if reader is None:
                    reader = _VideoRandomAccess(video_path)
                    fps = reader.fps
                image = reader.read(frame_idx)
            if image is None:
                raise RuntimeError(
                    "Frame plan entry missing image data and video_path is unavailable"
                )
            ts_s = float(entry.get("ts_s") or frame_idx / fps)
            yield FrameSample(ep_id=ep_id, frame_idx=frame_idx, ts_s=ts_s, image=image)
    finally:
        if reader:
            reader.close()


class RetinaFaceDetector:
    """Wrap InsightFace RetinaFace with a deterministic simulated fallback."""

    def __init__(self, cfg: Dict[str, object]):
        self.cfg = cfg
        self.model_id = str(cfg.get("model_id", "retinaface_r50"))
        self.simulated = bool(
            cfg.get("force_simulated") or os.getenv("SCREENALYTICS_VISION_SIM") == "1"
        )
        self._model = None
        if not self.simulated:
            self._init_model()

    def _init_model(self) -> None:
        try:
            from insightface.app import FaceAnalysis  # type: ignore

            profile = str(self.cfg.get("insightface_profile", "antelopev2"))
            det_size = self.cfg.get("det_size", (640, 640))
            if isinstance(det_size, list):
                det_size = tuple(det_size)
            providers = self.cfg.get("providers")
            kwargs = {"name": profile}
            if providers:
                kwargs["providers"] = providers
            self._model = FaceAnalysis(**kwargs)
            ctx_id = int(self.cfg.get("ctx_id", -1))
            self._model.prepare(ctx_id=ctx_id, det_size=det_size)
        except Exception as exc:  # pragma: no cover - exercised in integration
            LOGGER.warning("Falling back to simulated RetinaFace detector: %s", exc)
            self.simulated = True
            self._model = None

    def __call__(self, frame: np.ndarray) -> List[Dict[str, object]]:
        if self.simulated or not self._model:
            return [self._simulated_detection(frame)]

        faces = self._model.get(frame)
        h, w = frame.shape[:2]
        detections: List[Dict[str, object]] = []
        for face in faces:
            bbox = face.bbox.tolist()
            rel_bbox = [
                bbox[0] / w,
                bbox[1] / h,
                bbox[2] / w,
                bbox[3] / h,
            ]
            landmarks = face.landmark.reshape(-1, 2)
            rel_landmarks: List[float] = []
            for point in landmarks:
                rel_landmarks.extend([point[0] / w, point[1] / h])
            detections.append(
                {
                    "bbox": _ensure_float_list(rel_bbox, 4),
                    "landmarks": _ensure_float_list(rel_landmarks, 10),
                    "conf": float(face.det_score),
                }
            )
        return detections

    def _simulated_detection(self, frame: np.ndarray) -> Dict[str, object]:
        """Generate a deterministic box centered on the brightest pixels."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        min_val, max_val, _, max_loc = cv2.minMaxLoc(gray)
        span = max(0.05, min(0.25, (max_val - min_val) / 255.0))
        cx = max_loc[0] / w
        cy = max_loc[1] / h
        half_w = span / 2
        half_h = span / 2
        bbox = [
            max(0.0, cx - half_w),
            max(0.0, cy - half_h),
            min(1.0, cx + half_w),
            min(1.0, cy + half_h),
        ]
        landmarks = [cx, cy] * 5
        return {
            "bbox": _ensure_float_list(bbox, 4),
            "landmarks": _ensure_float_list(landmarks, 10),
            "conf": float(span + 0.5),
        }


def iter_samples(
    ep_id: str,
    video_path: Path,
    frame_plan: Path | None,
    stride: int,
) -> Iterator[FrameSample]:
    if frame_plan and frame_plan.exists():
        plan_entries = _read_plan(frame_plan)
        yield from _iter_plan_samples(video_path, plan_entries, ep_id)
    else:
        yield from _iter_video_samples(video_path, ep_id, stride)


def run_detection(
    ep_id: str,
    video_path: Path,
    output_path: Path,
    cfg: Dict[str, object],
    frame_plan: Path | None = None,
    sample_stride: int = DEFAULT_SAMPLE_STRIDE,
) -> int:
    detector = RetinaFaceDetector(cfg)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for sample in iter_samples(ep_id, video_path, frame_plan, sample_stride):
            detections = detector(sample.image)
            for det in detections:
                record = {
                    "ep_id": sample.ep_id,
                    "frame_idx": sample.frame_idx,
                    "ts_s": sample.ts_s,
                    "bbox": det["bbox"],
                    "landmarks": det["landmarks"],
                    "conf": float(det["conf"]),
                    "model_id": detector.model_id,
                    "schema_version": SCHEMA_VERSION,
                }
                out_f.write(json.dumps(record) + "\n")
                total += 1
    return total


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RetinaFace detection stage.")
    parser.add_argument("--ep-id", required=True, help="Episode identifier")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument(
        "--output",
        help="detections.jsonl output path (defaults to resolver path)",
    )
    parser.add_argument(
        "--frame-plan",
        help="Optional frame plan JSONL (falls back to sampling every N frames)",
    )
    parser.add_argument(
        "--config",
        default="config/pipeline/detection.yaml",
        help="Detection YAML config",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=DEFAULT_SAMPLE_STRIDE,
        help="Frame sampling stride when no frame plan is provided",
    )
    return parser.parse_args(argv)


def load_config(config_path: Path) -> Dict[str, object]:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(config_path.read_text())
    except ModuleNotFoundError:  # pragma: no cover
        cfg: Dict[str, object] = {}
        for line in config_path.read_text().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            cfg[key.strip()] = value.strip()
        return cfg


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(Path(args.config))
    video_path = Path(args.video)
    if args.output:
        output_path = Path(args.output)
    else:
        ensure_dirs(args.ep_id)
        output_path = get_path(args.ep_id, "detections")
    frame_plan = Path(args.frame_plan) if args.frame_plan else None
    count = run_detection(
        ep_id=args.ep_id,
        video_path=video_path,
        output_path=output_path,
        cfg=cfg,
        frame_plan=frame_plan,
        sample_stride=args.sample_stride,
    )
    LOGGER.info("Wrote %d detections to %s", count, output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())

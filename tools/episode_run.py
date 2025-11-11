#!/usr/bin/env python
"""Dev-only CLI to run detection → tracking for a single episode."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import logging

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.api.services.storage import (
    EpisodeContext,
    StorageService,
    artifact_prefixes,
    episode_context_from_id,
)
from py_screenalytics.artifacts import ensure_dirs, get_path

PIPELINE_VERSION = os.environ.get("SCREENALYTICS_PIPELINE_VERSION", "2025-11-11")
YOLO_MODEL_NAME = os.environ.get("SCREENALYTICS_YOLO_MODEL", "yolov8n.pt")
TRACKER_CONFIG = os.environ.get("SCREENALYTICS_TRACKER_CONFIG", "bytetrack.yaml")
TRACKER_NAME = Path(TRACKER_CONFIG).stem if TRACKER_CONFIG else "bytetrack"
YOLO_IMAGE_SIZE = int(os.environ.get("SCREENALYTICS_YOLO_IMGSZ", 640))
YOLO_CONF_THRESHOLD = float(os.environ.get("SCREENALYTICS_YOLO_CONF", 0.25))
YOLO_IOU_THRESHOLD = float(os.environ.get("SCREENALYTICS_YOLO_IOU", 0.45))
TRACK_SAMPLE_LIMIT = int(os.environ.get("SCREENALYTICS_TRACK_SAMPLE_LIMIT", 5))
PROGRESS_FRAME_STEP = int(os.environ.get("SCREENALYTICS_PROGRESS_FRAME_STEP", 25))
LOGGER = logging.getLogger("episode_run")


def pick_device(explicit: str | None = None) -> str:
    """Return the safest device available.

    Order of preference: explicit override → CUDA → MPS → CPU.
    Values returned are what Ultralytics expects ("cpu", "mps", "cuda"/"0").
    """

    if explicit and explicit not in {"auto", ""}:
        return explicit

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():  # pragma: no cover - depends on env
            return "0"
        mps_available = getattr(torch.backends, "mps", None)
        if mps_available is not None and mps_available.is_available():  # pragma: no cover - mac only
            return "mps"
    except Exception:  # pragma: no cover - torch import/runtime guard
        pass

    return "cpu"


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


class ProgressEmitter:
    """Emit structured progress to stdout + optional file for SSE/polling."""

    VERSION = 2

    def __init__(
        self,
        ep_id: str,
        file_path: str | Path | None,
        *,
        frames_total: int,
        secs_total: float | None,
        stride: int,
        fps_detected: float | None,
        fps_requested: float | None,
        frame_interval: int = PROGRESS_FRAME_STEP,
    ) -> None:
        self.ep_id = ep_id
        self.path = Path(file_path).expanduser() if file_path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frames_total = max(int(frames_total or 0), 0)
        self.secs_total = float(secs_total) if secs_total else None
        self.stride = max(int(stride), 1)
        self.fps_detected = float(fps_detected) if fps_detected else None
        self.fps_requested = float(fps_requested) if fps_requested else None
        self._frame_interval = max(int(frame_interval), 1)
        self._start_ts = time.time()
        self._last_frames = 0
        self._last_phase: str | None = None
        self._device: str | None = None

    def _now(self) -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def _should_emit(self, frames_done: int, phase: str, force: bool) -> bool:
        if force:
            return True
        if phase != self._last_phase:
            return True
        return (frames_done - self._last_frames) >= self._frame_interval

    def _compose_payload(
        self,
        frames_done: int,
        phase: str,
        device: str | None,
        summary: Dict[str, object] | None,
        error: str | None,
    ) -> Dict[str, object]:
        secs_done = time.time() - self._start_ts
        fps_infer = None
        if secs_done > 0 and frames_done >= 0:
            fps_infer = frames_done / secs_done
        payload: Dict[str, object] = {
            "progress_version": self.VERSION,
            "ep_id": self.ep_id,
            "phase": phase,
            "frames_done": frames_done,
            "frames_total": self.frames_total,
            "secs_done": round(float(secs_done), 3),
            "secs_total": round(float(self.secs_total), 3) if self.secs_total else None,
            "device": device or self._device,
            "fps_infer": round(float(fps_infer), 3) if fps_infer else None,
            "fps_detected": round(float(self.fps_detected), 3) if self.fps_detected else None,
            "fps_requested": round(float(self.fps_requested), 3) if self.fps_requested else None,
            "stride": self.stride,
            "updated_at": self._now(),
        }
        if summary:
            payload["summary"] = summary
        if error:
            payload["error"] = error
        return payload

    def _write_payload(self, payload: Dict[str, object]) -> None:
        line = json.dumps(payload, sort_keys=True)
        print(line, flush=True)
        if self.path:
            tmp_path = self.path.with_suffix(".tmp")
            tmp_path.write_text(line, encoding="utf-8")
            tmp_path.replace(self.path)

    def emit(
        self,
        frames_done: int,
        *,
        phase: str,
        device: str | None = None,
        summary: Dict[str, object] | None = None,
        error: str | None = None,
        force: bool = False,
    ) -> None:
        frames_done = max(int(frames_done), 0)
        if self.frames_total and frames_done > self.frames_total:
            frames_done = self.frames_total
        if not self._should_emit(frames_done, phase, force):
            return
        if device is not None:
            self._device = device
        payload = self._compose_payload(frames_done, phase, device, summary, error)
        self._write_payload(payload)
        self._last_frames = frames_done
        self._last_phase = phase

    def complete(self, summary: Dict[str, object], device: str | None = None) -> None:
        final_frames = self.frames_total or summary.get("frames_sampled") or self._last_frames
        final_frames = int(final_frames or 0)
        self.emit(final_frames, phase="done", device=device, summary=summary, force=True)

    def fail(self, error: str) -> None:
        self.emit(self._last_frames, phase="error", error=error, force=True)

    @property
    def target_frames(self) -> int:
        return self.frames_total or 0


class FrameExporter:
    """Handles optional frame + crop JPEG exports for S3 sync."""

    def __init__(
        self,
        ep_id: str,
        *,
        save_frames: bool,
        save_crops: bool,
        jpeg_quality: int,
    ) -> None:
        self.ep_id = ep_id
        self.save_frames = save_frames
        self.save_crops = save_crops
        self.jpeg_quality = max(1, min(int(jpeg_quality or 85), 100))
        self.root_dir = get_path(ep_id, "frames_root")
        self.frames_dir = self.root_dir / "frames"
        self.crops_dir = self.root_dir / "crops"
        if self.save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        if self.save_crops:
            self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.frames_written = 0
        self.crops_written = 0
        self._cv2 = None

    def _ensure_cv2(self):
        if self._cv2 is None:
            import cv2  # type: ignore

            self._cv2 = cv2

    def _write_jpeg(self, path: Path, image) -> None:
        self._ensure_cv2()
        cv2 = self._cv2
        params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        ok, buffer = cv2.imencode(".jpg", image, params)
        if not ok:
            raise RuntimeError(f"Failed to encode JPEG for {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buffer.tobytes())

    def _clamp_bbox(self, image, bbox: List[float]) -> tuple[int, int, int, int] | None:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(int(math.floor(x1)), 0)
        y1 = max(int(math.floor(y1)), 0)
        x2 = min(int(math.ceil(x2)), w)
        y2 = min(int(math.ceil(y2)), h)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def export(self, frame_idx: int, image, crops: List[Tuple[int, List[float]]]) -> None:
        if not (self.save_frames or self.save_crops):
            return
        if self.save_frames:
            frame_path = self.frames_dir / f"frame_{frame_idx:06d}.jpg"
            try:
                self._write_jpeg(frame_path, image)
                self.frames_written += 1
            except Exception as exc:  # pragma: no cover - best effort
                LOGGER.warning("Failed to save frame %s: %s", frame_path, exc)
        if self.save_crops and crops:
            for track_id, bbox in crops:
                if track_id is None:
                    continue
                clamp = self._clamp_bbox(image, bbox)
                if clamp is None:
                    continue
                x1, y1, x2, y2 = clamp
                crop_img = image[y1:y2, x1:x2]
                if crop_img.size == 0:
                    continue
                crop_path = self.crop_abs_path(track_id, frame_idx)
                try:
                    self._write_jpeg(crop_path, crop_img)
                    self.crops_written += 1
                except Exception as exc:  # pragma: no cover - best effort
                    LOGGER.warning("Failed to save crop %s: %s", crop_path, exc)

    def crop_component(self, track_id: int, frame_idx: int) -> str:
        return f"track_{track_id:04d}/frame_{frame_idx:06d}.jpg"

    def crop_rel_path(self, track_id: int, frame_idx: int) -> str:
        return f"crops/{self.crop_component(track_id, frame_idx)}"

    def crop_abs_path(self, track_id: int, frame_idx: int) -> Path:
        return self.crops_dir / self.crop_component(track_id, frame_idx)


class FrameDecoder:
    """Random-access video frame reader."""

    def __init__(self, video_path: Path) -> None:
        import cv2  # type: ignore

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Unable to open video {video_path}")

    def read(self, frame_idx: int):
        self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, max(int(frame_idx), 0))
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError(f"Failed to decode frame {frame_idx}")
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self) -> None:  # pragma: no cover - defensive
        try:
            self.close()
        except Exception:
            pass


def _copy_video(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dest.resolve():
        return
    shutil.copy2(src, dest)


def _estimate_duration(frame_count: int, fps: float) -> float | None:
    if frame_count > 0 and fps > 0:
        return frame_count / fps
    return None


def _estimate_frame_budget(
    *,
    stride: int,
    target_fps: float | None,
    detected_fps: float,
    duration_sec: float | None,
    frame_count: int,
) -> int:
    stride = max(stride, 1)
    fps_source = target_fps if target_fps and target_fps > 0 else detected_fps
    if fps_source and fps_source > 0 and duration_sec:
        value = int(math.ceil((fps_source * duration_sec) / stride))
    elif frame_count > 0:
        value = int(math.ceil(frame_count / stride))
    else:
        value = 0
    return max(value, 1)


def _episode_ctx(ep_id: str) -> EpisodeContext | None:
    try:
        return episode_context_from_id(ep_id)
    except ValueError:
        LOGGER.warning("Unable to parse episode id '%s'; artifact prefixes unavailable", ep_id)
        return None
 

def _storage_context(ep_id: str) -> tuple[StorageService | None, EpisodeContext | None, Dict[str, str] | None]:
    storage_backend = os.environ.get("STORAGE_BACKEND", "local").lower()
    storage: StorageService | None = None
    if storage_backend in {"s3", "minio"}:
        try:
            storage = StorageService()
        except Exception as exc:  # pragma: no cover - best effort init
            LOGGER.warning("Storage init failed (%s); disabling uploads", exc)
            storage = None
    ep_ctx = _episode_ctx(ep_id)
    prefixes = artifact_prefixes(ep_ctx) if ep_ctx else None
    return storage, ep_ctx, prefixes


def _sync_artifacts_to_s3(
    ep_id: str,
    storage: StorageService | None,
    ep_ctx: EpisodeContext | None,
    exporter: FrameExporter | None,
) -> Dict[str, int]:
    stats = {"manifests": 0, "frames": 0, "crops": 0}
    if storage is None or ep_ctx is None:
        return stats
    prefixes = artifact_prefixes(ep_ctx)
    manifests_dir = get_path(ep_id, "detections").parent
    manifest_files = ["detections.jsonl", "tracks.jsonl", "faces.jsonl", "identities.json"]
    for name in manifest_files:
        path = manifests_dir / name
        if not path.exists():
            continue
        uploaded = storage.put_artifact(ep_ctx, "manifests", path, name)
        if uploaded:
            stats["manifests"] += 1
    if exporter and exporter.save_frames and exporter.frames_dir.exists():
        stats["frames"] = storage.sync_tree_to_s3(ep_ctx, exporter.frames_dir, prefixes["frames"])
    if exporter and exporter.save_crops and exporter.crops_dir.exists():
        stats["crops"] = storage.sync_tree_to_s3(ep_ctx, exporter.crops_dir, prefixes["crops"])
    return stats


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detection + tracking locally.")
    parser.add_argument("--ep-id", required=True, help="Episode identifier")
    parser.add_argument("--video", help="Path to source video (required for detect/track runs)")
    parser.add_argument("--stride", type=int, default=5, help="Frame stride for detection sampling")
    parser.add_argument(
        "--fps",
        type=float,
        help="Optional target FPS for downsampling before detection",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Execution device override (auto→CUDA/MPS/CPU)",
    )
    parser.add_argument(
        "--out-root",
        help="Data root override (defaults to SCREENALYTICS_DATA_ROOT or ./data)",
    )
    parser.add_argument("--stub", action="store_true", help="Use stub detections (fast, no ML)")
    parser.add_argument("--progress-file", help="Progress JSON file to update during processing")
    parser.add_argument("--save-frames", action="store_true", help="Save sampled frame JPGs under data/frames/{ep_id}")
    parser.add_argument("--save-crops", action="store_true", help="Save per-track crops (requires --save-frames or track IDs)")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality for frame exports (1-100)")
    parser.add_argument("--faces-embed", action="store_true", help="Run faces embedding stage only")
    parser.add_argument("--cluster", action="store_true", help="Run clustering stage only")
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
    storage, ep_ctx, s3_prefixes = _storage_context(args.ep_id)

    phase_flags = [flag for flag in (args.faces_embed, args.cluster) if flag]
    if len(phase_flags) > 1:
        raise ValueError("Specify at most one of --faces-embed/--cluster per run")

    if args.faces_embed:
        summary = _run_faces_embed_stage(args, storage, ep_ctx, s3_prefixes)
    elif args.cluster:
        summary = _run_cluster_stage(args, storage, ep_ctx, s3_prefixes)
    else:
        summary = _run_detect_track_stage(args, storage, ep_ctx, s3_prefixes)

    stage = summary.get("stage", "detect_track")
    device_label = summary.get("device")
    analyzed_fps = summary.get("analyzed_fps")
    log_msg = f"stage={stage}"
    if device_label:
        log_msg += f" device={device_label}"
    if analyzed_fps:
        log_msg += f" analyzed_fps={analyzed_fps:.3f}"
    print(f"[episode_run] {log_msg}", file=sys.stderr)
    print("[episode_run] summary", summary, file=sys.stderr)
    return 0


def _run_stub_pipeline(
    ep_id: str,
    *,
    progress: ProgressEmitter | None = None,
    analyzed_fps: float | None = None,
) -> Tuple[int, int, int]:
    det_path = get_path(ep_id, "detections")
    track_path = get_path(ep_id, "tracks")
    det_rows = []
    if progress:
        progress.emit(0, phase="detect", device="cpu", force=True)
    stub_frames = 3
    for idx in range(stub_frames):
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
        if progress:
            total = max(progress.target_frames, 1)
            ratio = (idx + 1) / stub_frames
            scale = int(round(total * ratio)) or (idx + 1)
            progress.emit(scale, phase="detect", device="cpu")
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
    if progress:
        progress.emit(progress.target_frames, phase="track", device="cpu", force=True)
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


def _detect_fps(video_path: Path) -> float:
    fps, _ = _probe_video(video_path)
    if fps <= 0:
        fps = 24.0
    return fps


def _run_full_pipeline(
    args: argparse.Namespace,
    video_dest: Path,
    *,
    source_fps: float,
    progress: ProgressEmitter | None = None,
    target_fps: float | None = None,
    frame_exporter: FrameExporter | None = None,
) -> Tuple[int, int, int, str, float | None]:
    from ultralytics import YOLO  # type: ignore

    analyzed_fps = target_fps or source_fps
    if not analyzed_fps or analyzed_fps <= 0:
        analyzed_fps = _detect_fps(video_dest)
    frame_stride = _effective_stride(args.stride, target_fps or analyzed_fps, source_fps)
    ts_fps = analyzed_fps if analyzed_fps > 0 else max(args.fps or 30.0, 1.0)
    device = pick_device(args.device)
    det_path = get_path(args.ep_id, "detections")
    track_path = get_path(args.ep_id, "tracks")
    track_acc: Dict[int, TrackAccumulator] = {}
    det_count = 0
    frames_sampled = 0
    if progress:
        progress.emit(0, phase="detect", device=device, force=True)

    def detection_rows() -> Iterator[dict]:
        nonlocal det_count, frames_sampled
        model = YOLO(YOLO_MODEL_NAME)
        results = model.track(
            source=str(video_dest),
            stream=True,
            tracker=TRACKER_CONFIG,
            imgsz=YOLO_IMAGE_SIZE,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            device=device,
            vid_stride=frame_stride,
            persist=True,
        )
        for processed_idx, result in enumerate(results):
            frames_sampled = processed_idx + 1
            frame_idx = int(processed_idx * frame_stride)
            ts = frame_idx / ts_fps
            if progress:
                emit_frames = min(frames_sampled, progress.target_frames or frames_sampled)
                progress.emit(emit_frames, phase="detect", device=device)
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.data is None:
                if frame_exporter and frame_exporter.save_frames:
                    orig = getattr(result, "orig_img", None)
                    if orig is not None:
                        frame_exporter.export(frame_idx, orig, [])
                continue
            num_boxes = len(boxes)
            if num_boxes == 0:
                if frame_exporter and frame_exporter.save_frames:
                    orig = getattr(result, "orig_img", None)
                    if orig is not None:
                        frame_exporter.export(frame_idx, orig, [])
                continue
            track_ids = getattr(boxes, "id", None)
            crop_records: List[Tuple[int, List[float]]] = []
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
                    "fps": round(float(analyzed_fps), 4) if analyzed_fps else None,
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
                if track_id_val is not None and frame_exporter and frame_exporter.save_crops:
                    crop_records.append((track_id_val, bbox_xyxy))
                yield row
            if frame_exporter and (frame_exporter.save_frames or crop_records):
                orig = getattr(result, "orig_img", None)
                if orig is not None:
                    frame_exporter.export(frame_idx, orig, crop_records)

    _write_jsonl(det_path, detection_rows())
    track_rows = []
    for track in sorted(track_acc.values(), key=lambda t: t.track_id):
        row = track.to_row()
        row["ep_id"] = args.ep_id
        track_rows.append(row)
    _write_jsonl(track_path, track_rows)
    if progress:
        progress.emit(progress.target_frames, phase="track", device=device, force=True)
    return det_count, len(track_rows), frames_sampled, device, analyzed_fps


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _run_detect_track_stage(
    args: argparse.Namespace,
    storage: StorageService | None,
    ep_ctx: EpisodeContext | None,
    s3_prefixes: Dict[str, str] | None,
) -> Dict[str, Any]:
    if not args.video:
        raise ValueError("--video is required for detect/track runs")
    video_src = Path(args.video)
    if not video_src.exists():
        raise FileNotFoundError(f"Video not found: {video_src}")
    video_dest = get_path(args.ep_id, "video")
    _copy_video(video_src, video_dest)

    source_fps, frame_count = _probe_video(video_dest)
    target_fps = args.fps if args.fps and args.fps > 0 else None
    duration_sec = _estimate_duration(frame_count, source_fps)
    if duration_sec is None and frame_count > 0:
        fallback_fps = target_fps or source_fps or 30.0
        if fallback_fps > 0:
            duration_sec = frame_count / fallback_fps
    frames_total = _estimate_frame_budget(
        stride=args.stride,
        target_fps=target_fps,
        detected_fps=source_fps,
        duration_sec=duration_sec,
        frame_count=frame_count,
    )

    progress = ProgressEmitter(
        args.ep_id,
        args.progress_file,
        frames_total=frames_total,
        secs_total=duration_sec,
        stride=args.stride,
        fps_detected=source_fps,
        fps_requested=target_fps,
    )

    save_frames = bool(args.save_frames)
    save_crops = bool(args.save_crops)
    jpeg_quality = max(1, min(int(args.jpeg_quality or 85), 100))
    frame_exporter = (
        FrameExporter(
            args.ep_id,
            save_frames=save_frames,
            save_crops=save_crops,
            jpeg_quality=jpeg_quality,
        )
        if (save_frames or save_crops)
        else None
    )

    try:
        if args.stub:
            det_count, track_count, frames_sampled = _run_stub_pipeline(
                args.ep_id,
                progress=progress,
                analyzed_fps=source_fps,
            )
            resolved_device = "cpu"
            analyzed_fps = source_fps
        else:
            (
                det_count,
                track_count,
                frames_sampled,
                resolved_device,
                analyzed_fps,
            ) = _run_full_pipeline(
                args,
                video_dest,
                source_fps=source_fps,
                progress=progress,
                target_fps=target_fps,
                frame_exporter=frame_exporter,
            )

        manifests_dir = get_path(args.ep_id, "detections").parent
        s3_stats = _sync_artifacts_to_s3(args.ep_id, storage, ep_ctx, frame_exporter)
        summary: Dict[str, Any] = {
            "stage": "detect_track",
            "ep_id": args.ep_id,
            "detections": det_count,
            "tracks": track_count,
            "frames_sampled": frames_sampled,
            "frames_total": progress.target_frames,
            "frames_exported": frame_exporter.frames_written if frame_exporter else 0,
            "crops_exported": frame_exporter.crops_written if frame_exporter else 0,
            "device": resolved_device,
            "analyzed_fps": analyzed_fps,
            "artifacts": {
                "local": {
                    "detections": str(get_path(args.ep_id, "detections")),
                    "tracks": str(get_path(args.ep_id, "tracks")),
                    "manifests_dir": str(manifests_dir),
                    "frames_dir": str(frame_exporter.frames_dir) if frame_exporter and frame_exporter.save_frames else None,
                    "crops_dir": str(frame_exporter.crops_dir) if frame_exporter and frame_exporter.save_crops else None,
                },
                "s3_prefixes": s3_prefixes,
                "s3_uploads": s3_stats,
            },
        }
        progress.complete(summary, device=resolved_device)
        return summary
    except Exception as exc:
        progress.fail(str(exc))
        raise


def _run_faces_embed_stage(
    args: argparse.Namespace,
    storage: StorageService | None,
    ep_ctx: EpisodeContext | None,
    s3_prefixes: Dict[str, str] | None,
) -> Dict[str, Any]:
    track_path = get_path(args.ep_id, "tracks")
    if not track_path.exists():
        raise FileNotFoundError("tracks.jsonl not found; run detect/track first")
    samples = _load_track_samples(track_path)
    if not samples:
        raise RuntimeError("No track samples available for faces embedding")

    progress = ProgressEmitter(
        args.ep_id,
        args.progress_file,
        frames_total=len(samples),
        secs_total=None,
        stride=1,
        fps_detected=None,
        fps_requested=None,
    )
    device = pick_device(args.device)
    save_crops = bool(args.save_crops)
    jpeg_quality = max(1, min(int(args.jpeg_quality or 85), 100))
    exporter = FrameExporter(
        args.ep_id,
        save_frames=False,
        save_crops=save_crops,
        jpeg_quality=jpeg_quality,
    ) if save_crops else None

    manifests_dir = get_path(args.ep_id, "detections").parent
    faces_path = manifests_dir / "faces.jsonl"
    video_path = get_path(args.ep_id, "video")
    frame_decoder: FrameDecoder | None = None
    blank_image = _blank_image() if exporter and args.stub else None

    try:
        progress.emit(0, phase="faces_embed", device=device, force=True)
        rows: List[Dict[str, Any]] = []
        for idx, sample in enumerate(samples, start=1):
            crop_rel_path = None
            crop_s3_key = None
            if exporter:
                image = None
                if args.stub:
                    image = blank_image
                else:
                    if not video_path.exists():
                        raise FileNotFoundError("Local video not found for crop export")
                    if frame_decoder is None:
                        frame_decoder = FrameDecoder(video_path)
                    image = frame_decoder.read(sample["frame_idx"])
                if image is not None:
                    exporter.export(sample["frame_idx"], image, [(sample["track_id"], sample["bbox_xyxy"])])
                    crop_rel_path = exporter.crop_rel_path(sample["track_id"], sample["frame_idx"])
                    if s3_prefixes and s3_prefixes.get("crops"):
                        crop_s3_key = f"{s3_prefixes['crops']}{exporter.crop_component(sample['track_id'], sample['frame_idx'])}"

            face_row = {
                "ep_id": args.ep_id,
                "face_id": f"face_{sample['track_id']:04d}_{sample['frame_idx']:06d}",
                "track_id": sample["track_id"],
                "frame_idx": sample["frame_idx"],
                "ts": round(float(sample["ts"]), 4),
                "bbox_xyxy": [round(float(val), 4) for val in sample["bbox_xyxy"]],
                "embedding": _fake_embedding(sample["track_id"], sample["frame_idx"]),
                "pipeline_ver": PIPELINE_VERSION,
            }
            if crop_rel_path:
                face_row["crop_rel_path"] = crop_rel_path
            if crop_s3_key:
                face_row["crop_s3_key"] = crop_s3_key
            rows.append(face_row)
            progress.emit(idx, phase="faces_embed", device=device)

        _write_jsonl(faces_path, rows)
        s3_stats = _sync_artifacts_to_s3(args.ep_id, storage, ep_ctx, exporter)
        summary: Dict[str, Any] = {
            "stage": "faces_embed",
            "ep_id": args.ep_id,
            "faces": len(rows),
            "device": device,
            "crops_exported": exporter.crops_written if exporter else 0,
            "artifacts": {
                "local": {
                    "faces": str(faces_path),
                    "tracks": str(track_path),
                    "manifests_dir": str(manifests_dir),
                },
                "s3_prefixes": s3_prefixes,
                "s3_uploads": s3_stats,
            },
            "stats": {"faces": len(rows)},
        }
        progress.complete(summary, device=device)
        return summary
    except Exception as exc:
        progress.fail(str(exc))
        raise
    finally:
        if frame_decoder:
            frame_decoder.close()


def _run_cluster_stage(
    args: argparse.Namespace,
    storage: StorageService | None,
    ep_ctx: EpisodeContext | None,
    s3_prefixes: Dict[str, str] | None,
) -> Dict[str, Any]:
    manifests_dir = get_path(args.ep_id, "detections").parent
    faces_path = manifests_dir / "faces.jsonl"
    if not faces_path.exists():
        raise FileNotFoundError("faces.jsonl not found; run faces embedding first")
    faces_rows = list(_iter_jsonl(faces_path))
    if not faces_rows:
        raise RuntimeError("faces.jsonl is empty; cannot cluster")

    progress = ProgressEmitter(
        args.ep_id,
        args.progress_file,
        frames_total=len(faces_rows),
        secs_total=None,
        stride=1,
        fps_detected=None,
        fps_requested=None,
    )
    device = pick_device(args.device)
    progress.emit(0, phase="cluster", device=device, force=True)

    faces_by_track: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(faces_rows, start=1):
        track_id = int(row.get("track_id", -1))
        faces_by_track[track_id].append(row)
        progress.emit(idx, phase="cluster", device=device)

    identities = _build_identity_clusters(faces_by_track, s3_prefixes)
    identities_path = manifests_dir / "identities.json"
    payload = {
        "ep_id": args.ep_id,
        "pipeline_ver": PIPELINE_VERSION,
        "stats": {
            "faces": len(faces_rows),
            "clusters": len(identities),
        },
        "identities": identities,
    }
    identities_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    s3_stats = _sync_artifacts_to_s3(args.ep_id, storage, ep_ctx, exporter=None)

    summary: Dict[str, Any] = {
        "stage": "cluster",
        "ep_id": args.ep_id,
        "identities_count": len(identities),
        "faces_count": len(faces_rows),
        "device": device,
        "artifacts": {
            "local": {
                "faces": str(faces_path),
                "identities": str(identities_path),
                "manifests_dir": str(manifests_dir),
            },
            "s3_prefixes": s3_prefixes,
            "s3_uploads": s3_stats,
        },
        "stats": payload["stats"],
    }
    progress.complete(summary, device=device)
    return summary


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _load_track_samples(track_path: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for row in _iter_jsonl(track_path):
        track_id = int(row.get("track_id", -1))
        bbox_samples = row.get("bboxes_sampled") or []
        if not bbox_samples:
            fallback = {
                "frame_idx": int(row.get("first_frame_idx") or 0),
                "ts": float(row.get("first_ts") or 0.0),
                "bbox_xyxy": row.get("bbox_xyxy") or [0, 0, 10, 10],
            }
            bbox_samples = [fallback]
        for sample in bbox_samples:
            frame_idx = int(sample.get("frame_idx") or 0)
            ts = float(sample.get("ts") or 0.0)
            bbox = sample.get("bbox_xyxy") or [0, 0, 10, 10]
            if not isinstance(bbox, list) or len(bbox) != 4:
                bbox = [0, 0, 10, 10]
            samples.append(
                {
                    "track_id": track_id,
                    "frame_idx": frame_idx,
                    "ts": ts,
                    "bbox_xyxy": [float(val) for val in bbox],
                }
            )
    return samples


def _fake_embedding(track_id: int, frame_idx: int, length: int = 8) -> List[float]:
    rnd = random.Random(track_id * 100000 + frame_idx)
    return [round(rnd.uniform(-1.0, 1.0), 4) for _ in range(length)]


def _blank_image(width: int = 160, height: int = 160, color: tuple[int, int, int] = (180, 180, 180)):
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("numpy is required when saving face crops") from exc
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = color
    return image


def _build_identity_clusters(
    faces_by_track: Dict[int, List[Dict[str, Any]]],
    s3_prefixes: Dict[str, str] | None,
) -> List[Dict[str, Any]]:
    track_ids = sorted(track_id for track_id in faces_by_track.keys() if track_id is not None)
    clusters: List[List[int]] = []
    current: List[int] = []
    for track_id in track_ids:
        current.append(track_id)
        if len(current) == 2:
            clusters.append(current)
            current = []
    if current:
        clusters.append(current)
    if not clusters and faces_by_track:
        for track_id in track_ids:
            clusters.append([track_id])

    if not clusters:
        return []

    identities: List[Dict[str, Any]] = []
    for idx, track_group in enumerate(clusters, start=1):
        track_faces: List[Dict[str, Any]] = []
        for track_id in track_group:
            track_faces.extend(faces_by_track.get(track_id, []))
        track_faces.sort(key=lambda face: face.get("ts", 0.0))
        count = len(track_faces)
        rep_face = track_faces[0] if track_faces else None
        identity = {
            "identity_id": f"id_{idx:04d}",
            "label": f"Identity {idx:02d}",
            "track_ids": track_group,
            "count": count,
            "samples": [face.get("face_id") for face in track_faces[:3] if face.get("face_id")],
        }
        rep_payload = _rep_payload(rep_face, s3_prefixes)
        if rep_payload:
            identity["rep"] = rep_payload
        identities.append(identity)
    return identities


def _rep_payload(face: Dict[str, Any] | None, s3_prefixes: Dict[str, str] | None) -> Dict[str, Any] | None:
    if not face:
        return None
    rep: Dict[str, Any] = {
        "track_id": face.get("track_id"),
        "frame_idx": face.get("frame_idx"),
        "ts": face.get("ts"),
    }
    if face.get("crop_rel_path"):
        rep["crop_rel_path"] = face["crop_rel_path"]
    s3_key = face.get("crop_s3_key")
    if not s3_key and s3_prefixes and s3_prefixes.get("crops"):
        track_id = face.get("track_id")
        frame_idx = face.get("frame_idx")
        if track_id is not None and frame_idx is not None:
            s3_key = f"{s3_prefixes['crops']}track_{int(track_id):04d}/frame_{int(frame_idx):06d}.jpg"
    if s3_key:
        rep["s3_key"] = s3_key
    return rep


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

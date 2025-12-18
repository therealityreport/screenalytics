"""
Body Detection using YOLO.

Detects persons in video frames and outputs bounding boxes.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class BodyDetection:
    """Single body detection result."""

    frame_idx: int
    timestamp: float
    bbox: List[float]  # [x1, y1, x2, y2]
    score: float
    class_id: int = 0  # COCO person class

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "bbox": self.bbox,
            "score": self.score,
            "class_id": self.class_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BodyDetection":
        return cls(
            frame_idx=d["frame_idx"],
            timestamp=d["timestamp"],
            bbox=d["bbox"],
            score=d["score"],
            class_id=d.get("class_id", 0),
        )


class BodyDetector:
    """YOLO-based person detector."""

    # COCO person class ID
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name: str = "yolov8n",
        confidence_threshold: float = 0.50,
        nms_iou_threshold: float = 0.45,
        device: str = "auto",
        min_height_px: int = 50,
        min_width_px: int = 25,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device
        self.min_height_px = min_height_px
        self.min_width_px = min_width_px

        self._model = None

    def _load_model(self):
        """Lazy-load YOLO model."""
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package required for body detection. "
                "Install with: pip install ultralytics"
            )

        logger.info(f"Loading YOLO model: {self.model_name}")

        from .device_routing import resolve_torch_device_request

        _requested, device, reason = resolve_torch_device_request(self.device)
        if reason:
            logger.info("[device] resolved=%s (reason=%s requested=%s)", device, reason, self.device)

        # Load model
        model_path = f"{self.model_name}.pt"
        self._model = YOLO(model_path)
        self._model.to(device)

        logger.info(f"YOLO model loaded on device: {device}")

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
    ) -> List[BodyDetection]:
        """Detect persons in a single frame."""
        self._load_model()

        # Run inference
        results = self._model(
            frame,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            classes=[self.PERSON_CLASS_ID],  # Only detect persons
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                # Get box coordinates
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy.tolist()

                # Check minimum size
                width = x2 - x1
                height = y2 - y1
                if width < self.min_width_px or height < self.min_height_px:
                    continue

                score = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())

                detections.append(BodyDetection(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    bbox=[x1, y1, x2, y2],
                    score=score,
                    class_id=class_id,
                ))

        return detections

    def detect_batch(
        self,
        frames: List[np.ndarray],
        frame_indices: List[int],
        timestamps: List[float],
    ) -> List[List[BodyDetection]]:
        """Detect persons in a batch of frames."""
        self._load_model()

        # Run batch inference
        results = self._model(
            frames,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
        )

        all_detections = []
        for idx, result in enumerate(results):
            frame_idx = frame_indices[idx]
            timestamp = timestamps[idx]

            frame_detections = []
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy.tolist()

                    width = x2 - x1
                    height = y2 - y1
                    if width < self.min_width_px or height < self.min_height_px:
                        continue

                    score = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())

                    frame_detections.append(BodyDetection(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        bbox=[x1, y1, x2, y2],
                        score=score,
                        class_id=class_id,
                    ))

            all_detections.append(frame_detections)

        return all_detections


def _iter_video_frames(
    video_path: Path,
    sample_every_n: int = 1,
) -> Iterator[tuple]:
    """Iterate over video frames with sampling."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video: {video_path}")
    logger.info(f"  FPS: {fps:.2f}")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Sampling every {sample_every_n} frame(s)")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every_n == 0:
                timestamp = frame_idx / fps if fps > 0 else 0.0
                yield frame, frame_idx, timestamp

            frame_idx += 1
    finally:
        cap.release()


def detect_bodies(
    detector: BodyDetector,
    video_path: Path,
    output_path: Path,
    sample_every_n: int = 1,
    batch_size: int = 4,
) -> int:
    """
    Detect bodies in video and write to JSONL.

    Args:
        detector: BodyDetector instance
        video_path: Path to input video
        output_path: Path to output JSONL file
        sample_every_n: Process every Nth frame
        batch_size: Number of frames per batch

    Returns:
        Total number of detections
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_detections = 0
    frames_processed = 0

    # Batch processing
    batch_frames = []
    batch_indices = []
    batch_timestamps = []

    with open(output_path, "w") as f:
        for frame, frame_idx, timestamp in _iter_video_frames(video_path, sample_every_n):
            batch_frames.append(frame)
            batch_indices.append(frame_idx)
            batch_timestamps.append(timestamp)

            if len(batch_frames) >= batch_size:
                # Process batch
                batch_detections = detector.detect_batch(
                    batch_frames, batch_indices, batch_timestamps
                )

                for frame_dets in batch_detections:
                    for det in frame_dets:
                        f.write(json.dumps(det.to_dict()) + "\n")
                        total_detections += 1

                frames_processed += len(batch_frames)
                if frames_processed % 1000 == 0:
                    logger.info(f"  Processed {frames_processed} frames, {total_detections} detections")

                # Clear batch
                batch_frames = []
                batch_indices = []
                batch_timestamps = []

        # Process remaining frames
        if batch_frames:
            batch_detections = detector.detect_batch(
                batch_frames, batch_indices, batch_timestamps
            )
            for frame_dets in batch_detections:
                for det in frame_dets:
                    f.write(json.dumps(det.to_dict()) + "\n")
                    total_detections += 1
            frames_processed += len(batch_frames)

    logger.info(f"Detection complete: {frames_processed} frames, {total_detections} detections")
    return total_detections

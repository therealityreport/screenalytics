"""
Body Tracking using ByteTrack.

Associates body detections across frames into tracks.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .detect_bodies import BodyDetection


logger = logging.getLogger(__name__)


@dataclass
class BodyTrack:
    """A tracked body across frames."""

    track_id: int
    detections: List[BodyDetection] = field(default_factory=list)

    @property
    def start_frame(self) -> int:
        return self.detections[0].frame_idx if self.detections else -1

    @property
    def end_frame(self) -> int:
        return self.detections[-1].frame_idx if self.detections else -1

    @property
    def start_time(self) -> float:
        return self.detections[0].timestamp if self.detections else 0.0

    @property
    def end_time(self) -> float:
        return self.detections[-1].timestamp if self.detections else 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def frame_count(self) -> int:
        return len(self.detections)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "frame_count": self.frame_count,
            "detections": [d.to_dict() for d in self.detections],
        }

    def to_summary_dict(self) -> dict:
        """Return summary without full detection list."""
        return {
            "track_id": self.track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "frame_count": self.frame_count,
        }


class BodyTracker:
    """ByteTrack-based body tracker."""

    def __init__(
        self,
        tracker_type: str = "bytetrack",
        track_thresh: float = 0.50,
        new_track_thresh: float = 0.55,
        match_thresh: float = 0.70,
        track_buffer: int = 120,
        id_offset: int = 100000,
    ):
        self.tracker_type = tracker_type
        self.track_thresh = track_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.id_offset = id_offset

        self._tracker = None

    def _init_tracker(self):
        """Initialize the tracker."""
        if self._tracker is not None:
            return

        if self.tracker_type == "bytetrack":
            self._tracker = self._create_bytetrack()
        else:
            raise ValueError(f"Unknown tracker type: {self.tracker_type}")

    def _create_bytetrack(self):
        """Create ByteTrack tracker instance."""
        try:
            # Try to use supervision's ByteTrack
            import supervision as sv
            return sv.ByteTrack(
                track_activation_threshold=self.track_thresh,
                lost_track_buffer=self.track_buffer,
                minimum_matching_threshold=self.match_thresh,
                frame_rate=24,  # Will be updated per video
            )
        except ImportError:
            pass

        # Fallback to custom simple tracker
        logger.warning("supervision not found, using simple IoU tracker")
        return SimpleIoUTracker(
            iou_threshold=self.match_thresh,
            max_age=self.track_buffer,
            id_offset=self.id_offset,
        )

    def track_detections(
        self,
        detections_by_frame: Dict[int, List[BodyDetection]],
    ) -> Dict[int, BodyTrack]:
        """
        Track detections across frames.

        Args:
            detections_by_frame: Dict mapping frame_idx to list of detections

        Returns:
            Dict mapping track_id to BodyTrack
        """
        self._init_tracker()

        tracks: Dict[int, BodyTrack] = {}
        sorted_frames = sorted(detections_by_frame.keys())

        for frame_idx in sorted_frames:
            frame_dets = detections_by_frame[frame_idx]

            if not frame_dets:
                # Update tracker with empty detections
                self._update_tracker_empty(frame_idx)
                continue

            # Convert to format expected by tracker
            bboxes = np.array([d.bbox for d in frame_dets])
            scores = np.array([d.score for d in frame_dets])

            # Run tracking
            track_ids = self._update_tracker(frame_idx, bboxes, scores)

            # Associate tracks with detections
            for det, track_id in zip(frame_dets, track_ids):
                if track_id < 0:
                    continue  # Untracked detection

                # Apply ID offset to avoid collision with face track IDs
                track_id = track_id + self.id_offset

                if track_id not in tracks:
                    tracks[track_id] = BodyTrack(track_id=track_id)

                tracks[track_id].detections.append(det)

        logger.info(f"Tracking complete: {len(tracks)} body tracks")
        return tracks

    def _update_tracker(
        self,
        frame_idx: int,
        bboxes: np.ndarray,
        scores: np.ndarray,
    ) -> List[int]:
        """Update tracker with detections and return track IDs."""
        if hasattr(self._tracker, "update_with_detections"):
            # supervision ByteTrack
            import supervision as sv
            detections = sv.Detections(
                xyxy=bboxes,
                confidence=scores,
            )
            tracked = self._tracker.update_with_detections(detections)
            return tracked.tracker_id.tolist() if tracked.tracker_id is not None else [-1] * len(bboxes)
        else:
            # Custom tracker
            return self._tracker.update(frame_idx, bboxes, scores)

    def _update_tracker_empty(self, frame_idx: int):
        """Update tracker with no detections (for age management)."""
        if hasattr(self._tracker, "update_with_detections"):
            import supervision as sv
            detections = sv.Detections.empty()
            self._tracker.update_with_detections(detections)
        elif hasattr(self._tracker, "update"):
            self._tracker.update(frame_idx, np.array([]).reshape(0, 4), np.array([]))


class SimpleIoUTracker:
    """Simple IoU-based tracker as fallback."""

    def __init__(
        self,
        iou_threshold: float = 0.5,
        max_age: int = 30,
        id_offset: int = 0,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.id_offset = id_offset

        self._next_id = 1
        self._active_tracks: Dict[int, dict] = {}  # track_id -> {bbox, age, last_frame}

    def update(
        self,
        frame_idx: int,
        bboxes: np.ndarray,
        scores: np.ndarray,
    ) -> List[int]:
        """Update tracks with new detections."""
        if len(bboxes) == 0:
            # Age out tracks
            self._age_tracks(frame_idx)
            return []

        # Calculate IoU matrix between active tracks and new detections
        track_ids = list(self._active_tracks.keys())
        track_bboxes = np.array([self._active_tracks[tid]["bbox"] for tid in track_ids]) if track_ids else np.array([]).reshape(0, 4)

        assigned_track_ids = [-1] * len(bboxes)

        if len(track_bboxes) > 0:
            iou_matrix = self._compute_iou_matrix(track_bboxes, bboxes)

            # Greedy assignment
            while True:
                if iou_matrix.size == 0:
                    break
                max_iou = np.max(iou_matrix)
                if max_iou < self.iou_threshold:
                    break

                track_idx, det_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                track_id = track_ids[track_idx]

                assigned_track_ids[det_idx] = track_id
                self._active_tracks[track_id]["bbox"] = bboxes[det_idx].tolist()
                self._active_tracks[track_id]["last_frame"] = frame_idx
                self._active_tracks[track_id]["age"] = 0

                # Mark as used
                iou_matrix[track_idx, :] = -1
                iou_matrix[:, det_idx] = -1

        # Create new tracks for unassigned detections
        for i, track_id in enumerate(assigned_track_ids):
            if track_id < 0:
                new_id = self._next_id
                self._next_id += 1
                assigned_track_ids[i] = new_id
                self._active_tracks[new_id] = {
                    "bbox": bboxes[i].tolist(),
                    "last_frame": frame_idx,
                    "age": 0,
                }

        # Age out old tracks
        self._age_tracks(frame_idx)

        return assigned_track_ids

    def _age_tracks(self, current_frame: int):
        """Remove tracks that are too old."""
        to_remove = []
        for track_id, track in self._active_tracks.items():
            age = current_frame - track["last_frame"]
            if age > self.max_age:
                to_remove.append(track_id)
            else:
                track["age"] = age

        for tid in to_remove:
            del self._active_tracks[tid]

    def _compute_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU between two sets of boxes."""
        # boxes: [N, 4] in x1, y1, x2, y2 format
        n1, n2 = len(boxes1), len(boxes2)
        iou_matrix = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                iou_matrix[i, j] = self._compute_iou(boxes1[i], boxes2[j])

        return iou_matrix

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area == 0:
            return 0.0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0


def track_bodies(
    tracker: BodyTracker,
    detections_path: Path,
    output_path: Path,
) -> int:
    """
    Track bodies from detections JSONL and write tracks.

    Args:
        tracker: BodyTracker instance
        detections_path: Path to body_detections.jsonl
        output_path: Path to output body_tracks.jsonl

    Returns:
        Number of tracks created
    """
    detections_path = Path(detections_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load detections grouped by frame
    detections_by_frame: Dict[int, List[BodyDetection]] = defaultdict(list)

    logger.info(f"Loading detections from: {detections_path}")
    with open(detections_path) as f:
        for line in f:
            det = BodyDetection.from_dict(json.loads(line))
            detections_by_frame[det.frame_idx].append(det)

    logger.info(f"Loaded detections from {len(detections_by_frame)} frames")

    # Run tracking
    tracks = tracker.track_detections(detections_by_frame)

    # Write tracks to JSONL
    with open(output_path, "w") as f:
        for track_id in sorted(tracks.keys()):
            track = tracks[track_id]
            f.write(json.dumps(track.to_dict()) + "\n")

    logger.info(f"Wrote {len(tracks)} tracks to: {output_path}")
    return len(tracks)

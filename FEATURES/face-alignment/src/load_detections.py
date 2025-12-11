"""
Load face detections and tracks from main manifests.

Helper functions to read existing pipeline outputs for alignment.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def load_face_detections(
    detections_path: Path,
    min_confidence: float = 0.5,
    min_size: int = 20,
) -> List[Dict]:
    """
    Load face detections from JSONL file.

    Args:
        detections_path: Path to detections.jsonl
        min_confidence: Minimum detection confidence
        min_size: Minimum face box size in pixels

    Returns:
        List of detection dicts with frame_idx, bbox, score, etc.
    """
    detections_path = Path(detections_path)

    if not detections_path.exists():
        raise FileNotFoundError(f"Detections not found: {detections_path}")

    detections = []
    skipped_confidence = 0
    skipped_size = 0

    with open(detections_path) as f:
        for line in f:
            det = json.loads(line)

            # Filter by confidence
            score = det.get("score", det.get("confidence", 1.0))
            if score < min_confidence:
                skipped_confidence += 1
                continue

            # Filter by size
            bbox = det.get("bbox", det.get("box", []))
            if len(bbox) >= 4:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                if width < min_size or height < min_size:
                    skipped_size += 1
                    continue

            # Normalize detection format
            normalized = {
                "frame_idx": det.get("frame_idx", det.get("frame", 0)),
                "bbox": bbox,
                "score": score,
                "track_id": det.get("track_id"),
            }

            # Copy optional fields
            for key in ["timestamp", "landmarks", "pose"]:
                if key in det:
                    normalized[key] = det[key]

            detections.append(normalized)

    logger.info(f"Loaded {len(detections)} detections from {detections_path}")
    if skipped_confidence > 0:
        logger.info(f"  Skipped {skipped_confidence} low-confidence detections")
    if skipped_size > 0:
        logger.info(f"  Skipped {skipped_size} small detections")

    return detections


def load_face_tracks(
    tracks_path: Path,
    min_track_length: int = 3,
) -> Dict[int, Dict]:
    """
    Load face tracks from JSONL file.

    Args:
        tracks_path: Path to tracks.jsonl or faces.jsonl
        min_track_length: Minimum number of detections per track

    Returns:
        Dict mapping track_id to track data with detections list
    """
    tracks_path = Path(tracks_path)

    if not tracks_path.exists():
        raise FileNotFoundError(f"Tracks not found: {tracks_path}")

    tracks = {}
    skipped_short = 0

    with open(tracks_path) as f:
        for line in f:
            track = json.loads(line)
            track_id = track.get("track_id")

            if track_id is None:
                continue

            # Check track length
            detections = track.get("detections", [])
            if len(detections) < min_track_length:
                skipped_short += 1
                continue

            tracks[track_id] = track

    logger.info(f"Loaded {len(tracks)} tracks from {tracks_path}")
    if skipped_short > 0:
        logger.info(f"  Skipped {skipped_short} short tracks")

    return tracks


def get_representative_frames(
    track: Dict,
    max_frames: int = 5,
    strategy: str = "uniform",
) -> List[int]:
    """
    Select representative frame indices from a track.

    Args:
        track: Track dict with detections list
        max_frames: Maximum number of frames to select
        strategy: Selection strategy ("uniform", "quality", "first")

    Returns:
        List of detection indices within the track
    """
    detections = track.get("detections", [])
    n = len(detections)

    if n <= max_frames:
        return list(range(n))

    if strategy == "uniform":
        # Evenly spaced
        import numpy as np
        indices = np.linspace(0, n - 1, max_frames, dtype=int)
        return indices.tolist()

    elif strategy == "quality":
        # Sort by quality/confidence, take top
        scored = [
            (i, det.get("quality", det.get("score", 0)))
            for i, det in enumerate(detections)
        ]
        scored.sort(key=lambda x: -x[1])
        return [i for i, _ in scored[:max_frames]]

    elif strategy == "first":
        return list(range(max_frames))

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def group_detections_by_frame(
    detections: List[Dict],
) -> Dict[int, List[Dict]]:
    """
    Group detections by frame index.

    Args:
        detections: List of detection dicts

    Returns:
        Dict mapping frame_idx to list of detections
    """
    from collections import defaultdict

    by_frame = defaultdict(list)
    for det in detections:
        frame_idx = det.get("frame_idx", 0)
        by_frame[frame_idx].append(det)

    return dict(by_frame)

"""
Test fixtures for body tracking pipeline.

Provides synthetic data for testing without requiring ML model loads.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest


# ============================================================================
# Synthetic Detection Data
# ============================================================================

def create_synthetic_detections(
    num_frames: int = 100,
    num_persons: int = 2,
    fps: float = 24.0,
    frame_width: int = 1920,
    frame_height: int = 1080,
) -> List[Dict]:
    """
    Create synthetic body detections with realistic movement patterns.

    Returns list of detection dicts that can be written to JSONL.
    """
    detections = []

    # Initialize person positions
    persons = []
    for i in range(num_persons):
        # Start positions spread across frame
        x = 200 + i * 400
        y = 200
        width = 150
        height = 400
        velocity_x = (np.random.random() - 0.5) * 5
        velocity_y = (np.random.random() - 0.5) * 2

        persons.append({
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "vx": velocity_x,
            "vy": velocity_y,
        })

    for frame_idx in range(num_frames):
        timestamp = frame_idx / fps

        for person_idx, person in enumerate(persons):
            # Add some noise to position
            noise_x = np.random.normal(0, 2)
            noise_y = np.random.normal(0, 1)

            # Update position
            person["x"] += person["vx"] + noise_x
            person["y"] += person["vy"] + noise_y

            # Bounce off edges
            if person["x"] < 50 or person["x"] + person["width"] > frame_width - 50:
                person["vx"] *= -1
            if person["y"] < 50 or person["y"] + person["height"] > frame_height - 50:
                person["vy"] *= -1

            # Clamp to frame
            person["x"] = max(0, min(frame_width - person["width"], person["x"]))
            person["y"] = max(0, min(frame_height - person["height"], person["y"]))

            # Skip some frames randomly (simulates occlusion/detection misses)
            if np.random.random() < 0.05:
                continue

            x1 = person["x"]
            y1 = person["y"]
            x2 = x1 + person["width"]
            y2 = y1 + person["height"]

            detections.append({
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": 0.80 + np.random.random() * 0.15,
                "class_id": 0,
            })

    return detections


def create_synthetic_face_detections(
    body_detections: List[Dict],
    face_visibility_prob: float = 0.7,
) -> List[Dict]:
    """
    Create synthetic face detections corresponding to body detections.

    Face boxes are placed in upper portion of body boxes.
    """
    face_detections = []

    for det in body_detections:
        # Skip some faces (simulates turned away, occlusion)
        if np.random.random() > face_visibility_prob:
            continue

        body_bbox = det["bbox"]
        x1, y1, x2, y2 = body_bbox
        body_width = x2 - x1
        body_height = y2 - y1

        # Face is in upper 1/4 of body, centered horizontally
        face_width = body_width * 0.6
        face_height = body_height * 0.2
        face_x1 = x1 + (body_width - face_width) / 2
        face_y1 = y1 + body_height * 0.05
        face_x2 = face_x1 + face_width
        face_y2 = face_y1 + face_height

        face_detections.append({
            "frame_idx": det["frame_idx"],
            "timestamp": det["timestamp"],
            "bbox": [float(face_x1), float(face_y1), float(face_x2), float(face_y2)],
            "score": 0.85 + np.random.random() * 0.10,
            "track_id": None,  # Will be assigned by tracker
        })

    return face_detections


# ============================================================================
# Synthetic Track Data
# ============================================================================

def create_synthetic_tracks(
    detections: List[Dict],
    id_offset: int = 100000,
) -> List[Dict]:
    """
    Create synthetic tracks from detections using simple IoU tracking.

    This is a simplified version for testing purposes.
    """
    from collections import defaultdict

    # Group detections by frame
    by_frame = defaultdict(list)
    for det in detections:
        by_frame[det["frame_idx"]].append(det)

    tracks = {}
    next_id = 1
    active_tracks = {}  # track_id -> last_bbox

    sorted_frames = sorted(by_frame.keys())

    for frame_idx in sorted_frames:
        frame_dets = by_frame[frame_idx]
        assigned = set()

        # Try to match to existing tracks
        for track_id, last_bbox in list(active_tracks.items()):
            best_det_idx = -1
            best_iou = 0.3  # Threshold

            for i, det in enumerate(frame_dets):
                if i in assigned:
                    continue
                iou = _compute_iou(last_bbox, det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i

            if best_det_idx >= 0:
                det = frame_dets[best_det_idx]
                assigned.add(best_det_idx)
                tracks[track_id]["detections"].append(det)
                active_tracks[track_id] = det["bbox"]
            else:
                # Track lost
                del active_tracks[track_id]

        # Create new tracks for unassigned detections
        for i, det in enumerate(frame_dets):
            if i in assigned:
                continue
            track_id = next_id + id_offset
            next_id += 1
            tracks[track_id] = {
                "track_id": track_id,
                "detections": [det],
            }
            active_tracks[track_id] = det["bbox"]

    # Post-process tracks
    result = []
    for track_id, track in tracks.items():
        dets = track["detections"]
        if len(dets) < 2:
            continue  # Skip very short tracks

        result.append({
            "track_id": track_id,
            "start_frame": dets[0]["frame_idx"],
            "end_frame": dets[-1]["frame_idx"],
            "start_time": dets[0]["timestamp"],
            "end_time": dets[-1]["timestamp"],
            "duration": dets[-1]["timestamp"] - dets[0]["timestamp"],
            "frame_count": len(dets),
            "detections": dets,
        })

    return result


def _compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter)


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def synthetic_body_detections() -> List[Dict]:
    """Fixture providing synthetic body detections."""
    return create_synthetic_detections(num_frames=50, num_persons=3)


@pytest.fixture
def synthetic_body_tracks(synthetic_body_detections) -> List[Dict]:
    """Fixture providing synthetic body tracks."""
    return create_synthetic_tracks(synthetic_body_detections)


@pytest.fixture
def synthetic_face_detections(synthetic_body_detections) -> List[Dict]:
    """Fixture providing synthetic face detections."""
    return create_synthetic_face_detections(synthetic_body_detections, face_visibility_prob=0.6)


@pytest.fixture
def synthetic_face_tracks(synthetic_face_detections) -> List[Dict]:
    """Fixture providing synthetic face tracks."""
    return create_synthetic_tracks(synthetic_face_detections, id_offset=0)


@pytest.fixture
def temp_detections_jsonl(synthetic_body_detections) -> Path:
    """Fixture providing temp JSONL file with body detections."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for det in synthetic_body_detections:
            f.write(json.dumps(det) + "\n")
        path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def temp_tracks_jsonl(synthetic_body_tracks) -> Path:
    """Fixture providing temp JSONL file with body tracks."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for track in synthetic_body_tracks:
            f.write(json.dumps(track) + "\n")
        path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def temp_manifest_dir(
    synthetic_body_detections,
    synthetic_body_tracks,
    synthetic_face_detections,
    synthetic_face_tracks,
) -> Path:
    """Fixture providing a temp manifest directory with all artifacts."""
    import tempfile
    import os

    tmpdir = Path(tempfile.mkdtemp())

    # Write body detections
    body_det_path = tmpdir / "body_tracking" / "body_detections.jsonl"
    body_det_path.parent.mkdir(parents=True, exist_ok=True)
    with open(body_det_path, "w") as f:
        for det in synthetic_body_detections:
            f.write(json.dumps(det) + "\n")

    # Write body tracks
    body_tracks_path = tmpdir / "body_tracking" / "body_tracks.jsonl"
    with open(body_tracks_path, "w") as f:
        for track in synthetic_body_tracks:
            f.write(json.dumps(track) + "\n")

    # Write face detections (as faces.jsonl in main dir)
    faces_path = tmpdir / "faces.jsonl"
    with open(faces_path, "w") as f:
        for track in synthetic_face_tracks:
            f.write(json.dumps(track) + "\n")

    yield tmpdir

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Mock Embeddings
# ============================================================================

def create_mock_embeddings(num_embeddings: int, dim: int = 256) -> Tuple[np.ndarray, List[Dict]]:
    """
    Create mock Re-ID embeddings with metadata.

    Returns (embeddings_array, metadata_list).
    """
    # Create normalized random embeddings
    embeddings = np.random.randn(num_embeddings, dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Create metadata
    meta = []
    for i in range(num_embeddings):
        meta.append({
            "track_id": 100001 + (i // 3),  # ~3 embeddings per track
            "frame_idx": i * 10,
        })

    return embeddings, meta


@pytest.fixture
def mock_body_embeddings() -> Tuple[np.ndarray, List[Dict]]:
    """Fixture providing mock body embeddings."""
    return create_mock_embeddings(15, dim=256)

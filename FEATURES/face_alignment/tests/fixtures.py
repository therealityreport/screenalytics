"""
Test fixtures for face alignment sandbox.

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

def create_synthetic_face_detections(
    num_frames: int = 50,
    num_faces: int = 2,
    fps: float = 24.0,
    frame_width: int = 1920,
    frame_height: int = 1080,
) -> List[Dict]:
    """
    Create synthetic face detections with realistic movement patterns.

    Returns list of detection dicts that can be written to JSONL.
    """
    detections = []

    # Initialize face positions
    faces = []
    for i in range(num_faces):
        # Start positions spread across frame
        x = 300 + i * 500
        y = 200
        width = 120
        height = 150
        velocity_x = (np.random.random() - 0.5) * 3
        velocity_y = (np.random.random() - 0.5) * 1.5

        faces.append({
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "vx": velocity_x,
            "vy": velocity_y,
        })

    detection_id = 0
    for frame_idx in range(num_frames):
        timestamp = frame_idx / fps

        for face_idx, face in enumerate(faces):
            # Add some noise to position
            noise_x = np.random.normal(0, 1.5)
            noise_y = np.random.normal(0, 1.0)

            # Update position
            face["x"] += face["vx"] + noise_x
            face["y"] += face["vy"] + noise_y

            # Bounce off edges
            if face["x"] < 50 or face["x"] + face["width"] > frame_width - 50:
                face["vx"] *= -1
            if face["y"] < 50 or face["y"] + face["height"] > frame_height - 50:
                face["vy"] *= -1

            # Clamp to frame
            face["x"] = max(0, min(frame_width - face["width"], face["x"]))
            face["y"] = max(0, min(frame_height - face["height"], face["y"]))

            # Skip some frames randomly (simulates occlusion/detection misses)
            if np.random.random() < 0.03:
                continue

            x1 = face["x"]
            y1 = face["y"]
            x2 = x1 + face["width"]
            y2 = y1 + face["height"]

            detections.append({
                "detection_id": detection_id,
                "frame_idx": frame_idx,
                "timestamp": round(timestamp, 4),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": 0.85 + np.random.random() * 0.10,
                "track_id": face_idx + 1,
            })
            detection_id += 1

    return detections


def create_synthetic_tracks(detections: List[Dict]) -> List[Dict]:
    """
    Group detections into tracks by track_id.
    """
    from collections import defaultdict

    by_track = defaultdict(list)
    for det in detections:
        track_id = det.get("track_id")
        if track_id is not None:
            by_track[track_id].append(det)

    tracks = []
    for track_id, dets in sorted(by_track.items()):
        if len(dets) < 2:
            continue

        sorted_dets = sorted(dets, key=lambda d: d["frame_idx"])
        tracks.append({
            "track_id": track_id,
            "start_frame": sorted_dets[0]["frame_idx"],
            "end_frame": sorted_dets[-1]["frame_idx"],
            "start_time": sorted_dets[0]["timestamp"],
            "end_time": sorted_dets[-1]["timestamp"],
            "duration": sorted_dets[-1]["timestamp"] - sorted_dets[0]["timestamp"],
            "frame_count": len(sorted_dets),
            "detections": sorted_dets,
        })

    return tracks


# ============================================================================
# Synthetic Landmark Data
# ============================================================================

def create_synthetic_68_landmarks(
    bbox: List[float],
    add_noise: bool = True,
) -> List[List[float]]:
    """
    Create synthetic 68-point landmarks for a face bbox.

    Follows standard 68-point layout:
    - 0-16: Jaw contour
    - 17-21: Left eyebrow
    - 22-26: Right eyebrow
    - 27-35: Nose
    - 36-41: Left eye
    - 42-47: Right eye
    - 48-67: Mouth
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    landmarks = []

    # Jaw contour (0-16) - arc from left to right
    for i in range(17):
        angle = np.pi * (0.1 + 0.8 * i / 16)
        lx = cx + 0.45 * w * np.cos(angle)
        ly = cy + 0.45 * h * (0.2 + 0.8 * np.sin(angle))
        landmarks.append([lx, ly])

    # Left eyebrow (17-21)
    for i in range(5):
        lx = cx - 0.30 * w + i * 0.12 * w
        ly = cy - 0.20 * h - 0.02 * w * np.sin(np.pi * i / 4)
        landmarks.append([lx, ly])

    # Right eyebrow (22-26)
    for i in range(5):
        lx = cx + 0.06 * w + i * 0.12 * w
        ly = cy - 0.20 * h - 0.02 * w * np.sin(np.pi * i / 4)
        landmarks.append([lx, ly])

    # Nose bridge + tip (27-35)
    for i in range(4):  # Bridge
        lx = cx
        ly = cy - 0.15 * h + i * 0.08 * h
        landmarks.append([lx, ly])
    for i in range(5):  # Tip/nostrils
        angle = np.pi * (0.3 + 0.4 * i / 4)
        lx = cx + 0.08 * w * np.cos(angle)
        ly = cy + 0.10 * h + 0.02 * h * np.sin(angle)
        landmarks.append([lx, ly])

    # Left eye (36-41)
    eye_cx = cx - 0.18 * w
    eye_cy = cy - 0.10 * h
    for i in range(6):
        angle = 2 * np.pi * i / 6
        lx = eye_cx + 0.06 * w * np.cos(angle)
        ly = eye_cy + 0.03 * h * np.sin(angle)
        landmarks.append([lx, ly])

    # Right eye (42-47)
    eye_cx = cx + 0.18 * w
    for i in range(6):
        angle = 2 * np.pi * i / 6
        lx = eye_cx + 0.06 * w * np.cos(angle)
        ly = eye_cy + 0.03 * h * np.sin(angle)
        landmarks.append([lx, ly])

    # Mouth (48-67)
    mouth_cy = cy + 0.25 * h
    # Outer lip (48-59)
    for i in range(12):
        angle = 2 * np.pi * i / 12
        lx = cx + 0.12 * w * np.cos(angle)
        ly = mouth_cy + 0.04 * h * np.sin(angle)
        landmarks.append([lx, ly])
    # Inner lip (60-67)
    for i in range(8):
        angle = 2 * np.pi * i / 8
        lx = cx + 0.08 * w * np.cos(angle)
        ly = mouth_cy + 0.025 * h * np.sin(angle)
        landmarks.append([lx, ly])

    # Add noise if requested
    if add_noise:
        noise_scale = min(w, h) * 0.01
        for lm in landmarks:
            lm[0] += np.random.normal(0, noise_scale)
            lm[1] += np.random.normal(0, noise_scale)

    return landmarks


def create_synthetic_aligned_faces(
    detections: List[Dict],
    include_landmarks: bool = True,
) -> List[Dict]:
    """
    Create synthetic aligned face records.
    """
    aligned_faces = []

    for det in detections:
        bbox = det["bbox"]
        landmarks = create_synthetic_68_landmarks(bbox) if include_landmarks else None

        aligned = {
            "frame_idx": det["frame_idx"],
            "bbox": bbox,
            "landmarks_68": landmarks,
            "confidence": det.get("score", 0.9),
            "detection_id": det.get("detection_id"),
            "track_id": det.get("track_id"),
            "alignment_quality": 0.70 + np.random.random() * 0.25,
            "pose_yaw": np.random.uniform(-30, 30),
            "pose_pitch": np.random.uniform(-20, 20),
            "pose_roll": np.random.uniform(-15, 15),
        }
        aligned_faces.append(aligned)

    return aligned_faces


# ============================================================================
# Test Image Generation
# ============================================================================

def create_test_image(
    width: int = 640,
    height: int = 480,
    face_regions: List[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    Create a simple test image with optional face-like regions.

    Returns RGB numpy array.
    """
    # Create gray background
    img = np.full((height, width, 3), 128, dtype=np.uint8)

    # Add face-like regions (slightly different color)
    if face_regions:
        for x1, y1, x2, y2 in face_regions:
            # Skin-tone-ish color
            img[y1:y2, x1:x2] = [180, 150, 140]
            # Darker "eye" regions
            eye_y = y1 + (y2 - y1) // 3
            eye_h = (y2 - y1) // 8
            left_eye_x = x1 + (x2 - x1) // 4
            right_eye_x = x1 + 3 * (x2 - x1) // 4
            eye_w = (x2 - x1) // 6
            img[eye_y:eye_y + eye_h, left_eye_x - eye_w:left_eye_x + eye_w] = [60, 60, 60]
            img[eye_y:eye_y + eye_h, right_eye_x - eye_w:right_eye_x + eye_w] = [60, 60, 60]

    return img


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def synthetic_face_detections() -> List[Dict]:
    """Fixture providing synthetic face detections."""
    return create_synthetic_face_detections(num_frames=30, num_faces=2)


@pytest.fixture
def synthetic_face_tracks(synthetic_face_detections) -> List[Dict]:
    """Fixture providing synthetic face tracks."""
    return create_synthetic_tracks(synthetic_face_detections)


@pytest.fixture
def synthetic_aligned_faces(synthetic_face_detections) -> List[Dict]:
    """Fixture providing synthetic aligned face records."""
    return create_synthetic_aligned_faces(synthetic_face_detections)


@pytest.fixture
def synthetic_landmarks() -> List[List[float]]:
    """Fixture providing synthetic 68-point landmarks."""
    return create_synthetic_68_landmarks([100, 100, 220, 250], add_noise=False)


@pytest.fixture
def temp_detections_jsonl(synthetic_face_detections) -> Path:
    """Fixture providing temp JSONL file with face detections."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for det in synthetic_face_detections:
            f.write(json.dumps(det) + "\n")
        path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def temp_manifest_dir(synthetic_face_detections, synthetic_face_tracks) -> Path:
    """Fixture providing a temp manifest directory with face artifacts."""
    import shutil

    tmpdir = Path(tempfile.mkdtemp())

    # Write detections
    det_path = tmpdir / "detections.jsonl"
    with open(det_path, "w") as f:
        for det in synthetic_face_detections:
            f.write(json.dumps(det) + "\n")

    # Write tracks
    tracks_path = tmpdir / "tracks.jsonl"
    with open(tracks_path, "w") as f:
        for track in synthetic_face_tracks:
            f.write(json.dumps(track) + "\n")

    # Create face_alignment output dir
    (tmpdir / "face_alignment").mkdir(exist_ok=True)

    yield tmpdir

    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def test_image() -> np.ndarray:
    """Fixture providing a simple test image."""
    return create_test_image(
        width=640,
        height=480,
        face_regions=[(200, 100, 320, 280), (400, 150, 500, 300)],
    )

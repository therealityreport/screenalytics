"""
FAN-based Face Alignment.

Uses face-alignment library (1adrianb/face-alignment) for 68-point landmark extraction.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .alignment_quality import compute_alignment_quality


logger = logging.getLogger(__name__)


# Standard 68-point landmark indices for facial regions
LANDMARK_REGIONS = {
    "jaw": list(range(0, 17)),
    "left_eyebrow": list(range(17, 22)),
    "right_eyebrow": list(range(22, 27)),
    "nose_bridge": list(range(27, 31)),
    "nose_tip": list(range(31, 36)),
    "left_eye": list(range(36, 42)),
    "right_eye": list(range(42, 48)),
    "outer_lip": list(range(48, 60)),
    "inner_lip": list(range(60, 68)),
}

# 5-point reference for ArcFace alignment
ARCFACE_SRC = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


@dataclass
class AlignedFace:
    """Result of face alignment."""

    frame_idx: int
    bbox: List[float]  # Original detection bbox [x1, y1, x2, y2]
    landmarks_68: List[List[float]]  # 68 x 2 landmark coordinates
    confidence: float  # Detection confidence

    # Optional fields
    track_id: Optional[int] = None
    detection_id: Optional[int] = None  # Links to original detection
    crop: Optional[np.ndarray] = None
    crop_path: Optional[str] = None

    # Quality metrics (for future LUVLi integration)
    alignment_quality: Optional[float] = None
    landmark_confidences: Optional[List[float]] = None

    # Pose estimates (for future 3DDFA_V2 integration)
    pose_yaw: Optional[float] = None
    pose_pitch: Optional[float] = None
    pose_roll: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            "frame_idx": self.frame_idx,
            "bbox": self.bbox,
            "landmarks_68": self.landmarks_68,
            "confidence": self.confidence,
        }

        if self.track_id is not None:
            d["track_id"] = self.track_id
        if self.detection_id is not None:
            d["detection_id"] = self.detection_id
        if self.crop_path is not None:
            d["crop_path"] = self.crop_path
        if self.alignment_quality is not None:
            d["alignment_quality"] = self.alignment_quality
        if self.pose_yaw is not None:
            d["pose_yaw"] = self.pose_yaw
        if self.pose_pitch is not None:
            d["pose_pitch"] = self.pose_pitch
        if self.pose_roll is not None:
            d["pose_roll"] = self.pose_roll

        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AlignedFace":
        return cls(
            frame_idx=d["frame_idx"],
            bbox=d["bbox"],
            landmarks_68=d["landmarks_68"],
            confidence=d["confidence"],
            track_id=d.get("track_id"),
            detection_id=d.get("detection_id"),
            crop_path=d.get("crop_path"),
            alignment_quality=d.get("alignment_quality"),
            pose_yaw=d.get("pose_yaw"),
            pose_pitch=d.get("pose_pitch"),
            pose_roll=d.get("pose_roll"),
        )


class FANAligner:
    """FAN-based face alignment using face-alignment library."""

    def __init__(
        self,
        model_type: str = "2d",
        landmarks_type: str = "2D",
        device: str = "auto",
        flip_input: bool = False,
    ):
        """
        Initialize FAN aligner.

        Args:
            model_type: "2d" or "3d" (3D includes depth)
            landmarks_type: "2D", "3D", or "2.5D"
            device: "auto", "cuda", or "cpu"
            flip_input: Whether to flip input for ensemble
        """
        self.model_type = model_type
        self.landmarks_type = landmarks_type
        self.device = device
        self.flip_input = flip_input

        self._fa = None

    def _load_model(self):
        """Lazy-load FAN model."""
        if self._fa is not None:
            return

        try:
            import face_alignment
        except ImportError:
            raise ImportError(
                "face-alignment package required. "
                "Install with: pip install face-alignment"
            )

        # Determine device
        device = self.device
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        # Map landmarks type
        landmarks_map = {
            "2D": face_alignment.LandmarksType.TWO_D,
            "3D": face_alignment.LandmarksType.THREE_D,
            "2.5D": face_alignment.LandmarksType.TWO_HALF_D,
        }
        landmarks_type = landmarks_map.get(
            self.landmarks_type,
            face_alignment.LandmarksType.TWO_D
        )

        logger.info(f"Loading FAN model ({self.landmarks_type}) on {device}")

        self._fa = face_alignment.FaceAlignment(
            landmarks_type,
            device=device,
            flip_input=self.flip_input,
        )

        logger.info("FAN model loaded")

    def align_face(
        self,
        image: np.ndarray,
        bbox: List[float],
        detect_faces: bool = False,
    ) -> Optional[List[List[float]]]:
        """
        Get 68-point landmarks for a face.

        Args:
            image: BGR image (OpenCV format)
            bbox: Face bounding box [x1, y1, x2, y2]
            detect_faces: If True, use FAN's face detector; else use provided bbox

        Returns:
            68 x 2 landmark coordinates, or None if detection failed
        """
        self._load_model()

        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if detect_faces:
            # Let FAN detect faces
            landmarks = self._fa.get_landmarks(rgb)
        else:
            # Use provided bbox
            x1, y1, x2, y2 = [int(v) for v in bbox]
            detected_faces = [[x1, y1, x2, y2]]
            landmarks = self._fa.get_landmarks(rgb, detected_faces=detected_faces)

        if landmarks is None or len(landmarks) == 0:
            return None

        # Return first face's landmarks
        return landmarks[0].tolist()

    def align_faces_batch(
        self,
        images: List[np.ndarray],
        bboxes: List[List[float]],
    ) -> List[Optional[List[List[float]]]]:
        """
        Get landmarks for multiple faces (processed sequentially for now).

        Args:
            images: List of BGR images
            bboxes: List of face bounding boxes

        Returns:
            List of 68x2 landmarks or None for each face
        """
        results = []
        for img, bbox in zip(images, bboxes):
            landmarks = self.align_face(img, bbox)
            results.append(landmarks)
        return results


def get_5_point_landmarks(landmarks_68: List[List[float]]) -> np.ndarray:
    """
    Extract 5-point landmarks from 68-point for ArcFace alignment.

    5 points: left eye center, right eye center, nose tip, left mouth corner, right mouth corner

    Args:
        landmarks_68: 68 x 2 landmark coordinates

    Returns:
        5 x 2 numpy array of key landmark positions
    """
    if len(landmarks_68) < 68:
        raise ValueError(f"Expected 68 landmarks, got {len(landmarks_68)}")

    lm = np.array(landmarks_68)

    # Left eye center
    left_eye = lm[36:42].mean(axis=0)
    # Right eye center
    right_eye = lm[42:48].mean(axis=0)
    # Nose tip
    nose = lm[30]
    # Mouth corners
    left_mouth = lm[48]
    right_mouth = lm[54]

    return np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)


def compute_similarity_transform(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> np.ndarray:
    """
    Compute similarity transform matrix from src to dst points.

    Returns 2x3 affine matrix.
    """
    # Use cv2.estimateAffinePartial2D for similarity transform
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    return M


def align_face_crop(
    image: np.ndarray,
    landmarks_68: List[List[float]],
    crop_size: int = 112,
    margin: float = 0.0,
) -> np.ndarray:
    """
    Generate aligned face crop using 68-point landmarks.

    Args:
        image: BGR image
        landmarks_68: 68 x 2 landmarks
        crop_size: Output crop size (default 112 for ArcFace)
        margin: Additional margin around face

    Returns:
        Aligned face crop (crop_size x crop_size x 3)
    """
    # Get 5-point landmarks
    src_pts = get_5_point_landmarks(landmarks_68)

    # Scale reference points to crop size
    scale = crop_size / 112.0
    dst_pts = ARCFACE_SRC * scale

    # Add margin
    if margin > 0:
        center = dst_pts.mean(axis=0)
        dst_pts = center + (dst_pts - center) * (1 + margin)

    # Compute transform
    M = compute_similarity_transform(src_pts, dst_pts)

    # Apply transform
    aligned = cv2.warpAffine(
        image, M, (crop_size, crop_size),
        borderMode=cv2.BORDER_REPLICATE
    )

    return aligned


def run_fan_alignment(
    aligner: FANAligner,
    video_path: Path,
    detections: List[Dict],
    batch_size: int = 16,
    crop_size: int = 112,
    crop_margin: float = 0.0,
) -> List[AlignedFace]:
    """
    Run FAN alignment on detections from a video.

    Args:
        aligner: FANAligner instance
        video_path: Path to video file
        detections: List of detection dicts with frame_idx and bbox
        batch_size: Number of faces to process per batch
        crop_size: Output crop size
        crop_margin: Margin around aligned face

    Returns:
        List of AlignedFace results
    """
    video_path = Path(video_path)

    # Group detections by frame
    from .load_detections import group_detections_by_frame
    by_frame = group_detections_by_frame(detections)

    logger.info(f"Running FAN alignment on {len(detections)} faces from {len(by_frame)} frames")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []
    processed = 0

    sorted_frames = sorted(by_frame.keys())
    current_frame = 0

    for frame_idx in sorted_frames:
        # Seek to frame
        if current_frame != frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            current_frame = frame_idx

        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Could not read frame {frame_idx}")
            continue
        current_frame += 1

        frame_dets = by_frame[frame_idx]

        for det in frame_dets:
            bbox = det["bbox"]

            # Run FAN alignment
            landmarks = aligner.align_face(frame, bbox)

            if landmarks is None:
                logger.debug(f"FAN failed for frame {frame_idx}, bbox {bbox}")
                continue

            # Generate aligned crop
            crop = align_face_crop(frame, landmarks, crop_size, crop_margin)

            aligned = AlignedFace(
                frame_idx=frame_idx,
                bbox=bbox,
                landmarks_68=landmarks,
                confidence=det.get("score", 1.0),
                track_id=det.get("track_id"),
                detection_id=det.get("detection_id"),
                crop=crop,
            )

            # Compute alignment quality (heuristic-based for now)
            # TODO: Replace with LUVLi model output once integrated
            aligned.alignment_quality = compute_alignment_quality(
                bbox=bbox,
                landmarks_68=landmarks,
            )

            results.append(aligned)
            processed += 1

        if processed % 100 == 0:
            logger.info(f"  Processed {processed}/{len(detections)} faces")

    cap.release()
    logger.info(f"Alignment complete: {len(results)}/{len(detections)} successful")

    return results

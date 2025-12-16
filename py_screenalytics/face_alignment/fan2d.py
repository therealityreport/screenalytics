from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


ARCFACE_SRC = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _coerce_bbox_xyxy(bbox_xyxy: Sequence[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy[:4]
    return float(x1), float(y1), float(x2), float(y2)


def get_5pt_from_68(landmarks_68: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert 68-point landmarks into ArcFace-style 5 point landmarks."""
    lm = np.asarray(landmarks_68, dtype=np.float32)
    if lm.ndim != 2 or lm.shape[0] < 68 or lm.shape[1] < 2:
        raise ValueError(f"Expected 68x2 landmarks, got shape {lm.shape}")

    left_eye = lm[36:42].mean(axis=0)
    right_eye = lm[42:48].mean(axis=0)
    nose = lm[30]
    left_mouth = lm[48]
    right_mouth = lm[54]
    return np.asarray([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)


def align_face_crop(
    image_bgr: np.ndarray,
    landmarks_68: Sequence[Sequence[float]],
    *,
    crop_size: int = 112,
    margin: float = 0.0,
) -> np.ndarray:
    """Return an ArcFace-style aligned crop from 68-point landmarks."""
    import cv2  # type: ignore

    src_pts = get_5pt_from_68(landmarks_68)
    scale = float(crop_size) / 112.0
    dst_pts = ARCFACE_SRC * scale

    if margin and margin > 0:
        center = dst_pts.mean(axis=0)
        dst_pts = center + (dst_pts - center) * (1.0 + float(margin))

    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is None:
        raise ValueError("Failed to estimate alignment transform")

    aligned = cv2.warpAffine(
        image_bgr,
        M,
        (int(crop_size), int(crop_size)),
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned


def compute_alignment_quality(
    bbox_xyxy: Sequence[float],
    landmarks_68: Sequence[Sequence[float]] | None = None,
    *,
    min_face_size: int = 20,
    ideal_aspect_ratio: float = 0.85,
) -> float:
    """Deterministic heuristic quality score in [0, 1]."""
    x1, y1, x2, y2 = _coerce_bbox_xyxy(bbox_xyxy)
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)

    face_size = min(width, height)
    size_score = float(np.clip(face_size / max(min_face_size * 5.0, 1.0), 0.0, 1.0))

    aspect_ratio = width / max(height, 1.0)
    aspect_deviation = abs(aspect_ratio - float(ideal_aspect_ratio))
    aspect_score = float(np.clip(1.0 - aspect_deviation * 2.0, 0.0, 1.0))

    landmark_spread_score = 0.5
    if landmarks_68 is not None:
        lm = np.asarray(landmarks_68, dtype=np.float32)
        if lm.ndim == 2 and lm.shape[0] >= 68 and lm.shape[1] >= 2:
            lm_width = float(lm[:, 0].max() - lm[:, 0].min())
            lm_height = float(lm[:, 1].max() - lm[:, 1].min())
            width_coverage = lm_width / max(width, 1.0)
            height_coverage = lm_height / max(height, 1.0)
            width_score = float(np.clip(width_coverage / 0.7, 0.0, 1.0) * np.clip(2.0 - width_coverage, 0.0, 1.0))
            height_score = float(np.clip(height_coverage / 0.7, 0.0, 1.0) * np.clip(2.0 - height_coverage, 0.0, 1.0))

            left_eye_center = lm[36:42].mean(axis=0)
            right_eye_center = lm[42:48].mean(axis=0)
            eye_height_diff = float(abs(left_eye_center[1] - right_eye_center[1]))
            symmetry_score = float(np.clip(1.0 - eye_height_diff / max(height * 0.1, 1e-6), 0.0, 1.0))
            landmark_spread_score = float((width_score + height_score + symmetry_score) / 3.0)

    quality = 0.3 * size_score + 0.3 * aspect_score + 0.4 * landmark_spread_score
    return float(np.clip(quality, 0.0, 1.0))


@dataclass
class Fan2dAligner:
    """Lazy FAN (face-alignment) wrapper for 68-point landmark extraction."""

    landmarks_type: str = "2D"
    device: str = "auto"
    flip_input: bool = False

    _model: Any = None
    _resolved_device: str | None = None

    def _lazy_model(self) -> Any:
        """Load face-alignment model on first use."""
        if self._model is not None:
            return self._model

        try:
            import face_alignment  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency in CI
            raise ImportError(
                "face-alignment package required for FAN 68-point landmarks. "
                "Install with: pip install face-alignment"
            ) from exc

        device = (self.device or "auto").lower()
        if device == "auto":
            try:
                import torch  # type: ignore

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        landmarks_map = {
            "2d": face_alignment.LandmarksType.TWO_D,
            "3d": face_alignment.LandmarksType.THREE_D,
            "2.5d": face_alignment.LandmarksType.TWO_HALF_D,
        }
        landmarks_type = landmarks_map.get(str(self.landmarks_type or "2D").lower(), face_alignment.LandmarksType.TWO_D)

        LOGGER.info("Loading FAN landmarks=%s device=%s", self.landmarks_type, device)
        self._resolved_device = device
        self._model = face_alignment.FaceAlignment(landmarks_type, device=device, flip_input=bool(self.flip_input))
        return self._model

    @property
    def resolved_device(self) -> str:
        if self._resolved_device is None:
            return str(self.device or "auto")
        return self._resolved_device

    def align_face(self, image_bgr: np.ndarray, bbox_xyxy: Sequence[float]) -> list[list[float]] | None:
        """Return 68x2 landmarks for bbox in the image, or None if FAN fails."""
        import cv2  # type: ignore

        model = self._lazy_model()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = _coerce_bbox_xyxy(bbox_xyxy)
        detected_faces = [[int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]]
        landmarks = model.get_landmarks(rgb, detected_faces=detected_faces)
        if not landmarks:
            return None
        return landmarks[0].tolist()


def aligned_face_row(
    *,
    track_id: int,
    frame_idx: int,
    bbox_xyxy: Sequence[float],
    confidence: float | None,
    landmarks_68: Sequence[Sequence[float]] | None,
    alignment_quality: float | None,
    alignment_quality_source: str = "heuristic",
    pose_yaw: float | None = None,
    pose_pitch: float | None = None,
    pose_roll: float | None = None,
    pose_reprojection_error_px: float | None = None,
    pose_source: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable aligned face row for `aligned_faces.jsonl`."""
    row: dict[str, Any] = {
        "track_id": int(track_id),
        "frame_idx": int(frame_idx),
        "bbox": [round(float(v), 4) for v in bbox_xyxy[:4]],
    }
    if confidence is not None:
        row["confidence"] = round(float(confidence), 4)
    if landmarks_68 is not None:
        row["landmarks_68"] = landmarks_68
    if alignment_quality is not None:
        row["alignment_quality"] = round(float(alignment_quality), 4)
        row["alignment_quality_source"] = alignment_quality_source
    if pose_yaw is not None:
        row["pose_yaw"] = round(float(pose_yaw), 3)
    if pose_pitch is not None:
        row["pose_pitch"] = round(float(pose_pitch), 3)
    if pose_roll is not None:
        row["pose_roll"] = round(float(pose_roll), 3)
    if pose_reprojection_error_px is not None:
        row["pose_reprojection_error_px"] = round(float(pose_reprojection_error_px), 3)
    if pose_source is not None:
        row["pose_source"] = str(pose_source)
    return row


def coerce_landmarks_68(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or not value:
        return None
    points: list[list[float]] = []
    for pt in value:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            return None
        try:
            x = float(pt[0])
            y = float(pt[1])
        except (TypeError, ValueError):
            return None
        points.append([x, y])
    if len(points) < 68:
        return None
    return points

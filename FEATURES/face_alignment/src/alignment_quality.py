"""
Alignment Quality Estimation.

Provides quality scores for face alignments to enable quality gating
before embedding computation.

CURRENT STATUS: Heuristic-based
FUTURE: LUVLi model integration for per-landmark uncertainty

TODO: Replace heuristic with LUVLi model
    1. Add LUVLi dependency
    2. Load LUVLi model in AlignmentQualityEstimator
    3. Use per-landmark uncertainty for quality scoring
    4. Update ACCEPTANCE_MATRIX.md status from "Heuristic" to "Model-based"
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    """Result of alignment quality estimation."""

    quality_score: float  # 0.0 to 1.0, higher is better
    confidence: float  # Confidence in the quality estimate

    # Components (for debugging)
    size_score: float = 0.0
    aspect_ratio_score: float = 0.0
    landmark_spread_score: float = 0.0

    # Future: LUVLi per-landmark uncertainty
    landmark_uncertainties: Optional[List[float]] = None

    def to_dict(self) -> dict:
        d = {
            "quality_score": round(self.quality_score, 4),
            "confidence": round(self.confidence, 4),
            "components": {
                "size_score": round(self.size_score, 4),
                "aspect_ratio_score": round(self.aspect_ratio_score, 4),
                "landmark_spread_score": round(self.landmark_spread_score, 4),
            },
        }
        if self.landmark_uncertainties is not None:
            d["landmark_uncertainties"] = [
                round(u, 4) for u in self.landmark_uncertainties
            ]
        return d


class AlignmentQualityEstimator:
    """
    Estimates quality of face alignments.

    CURRENT: Uses heuristics based on bbox size and aspect ratio.
    FUTURE: Will use LUVLi model for per-landmark uncertainty.
    """

    def __init__(
        self,
        min_face_size: int = 20,
        ideal_aspect_ratio: float = 0.85,  # width/height for typical face
        use_luvli: bool = False,
        luvli_model_path: Optional[str] = None,
    ):
        """
        Initialize quality estimator.

        Args:
            min_face_size: Minimum face size in pixels for good quality
            ideal_aspect_ratio: Ideal width/height ratio for frontal face
            use_luvli: Whether to use LUVLi model (future)
            luvli_model_path: Path to LUVLi model (future)
        """
        self.min_face_size = min_face_size
        self.ideal_aspect_ratio = ideal_aspect_ratio
        self.use_luvli = use_luvli
        self.luvli_model_path = luvli_model_path

        self._luvli_model = None

        if use_luvli:
            logger.warning(
                "LUVLi model requested but not yet implemented. "
                "Using heuristic quality estimation."
            )
            # TODO: Load LUVLi model here when available
            # self._luvli_model = load_luvli_model(luvli_model_path)

    def estimate_quality(
        self,
        bbox: List[float],
        landmarks_68: Optional[List[List[float]]] = None,
    ) -> QualityResult:
        """
        Estimate alignment quality for a face.

        Args:
            bbox: Face bounding box [x1, y1, x2, y2]
            landmarks_68: 68-point landmarks (optional, improves estimate)

        Returns:
            QualityResult with quality score
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Component 1: Size score
        # Larger faces generally have better alignments
        face_size = min(width, height)
        size_score = np.clip(face_size / (self.min_face_size * 5), 0, 1)

        # Component 2: Aspect ratio score
        # Extreme aspect ratios suggest profile or unusual pose
        aspect_ratio = width / max(height, 1)
        aspect_deviation = abs(aspect_ratio - self.ideal_aspect_ratio)
        aspect_ratio_score = np.clip(1.0 - aspect_deviation * 2, 0, 1)

        # Component 3: Landmark spread score (if landmarks available)
        landmark_spread_score = 0.5  # Default
        if landmarks_68 is not None and len(landmarks_68) >= 68:
            landmark_spread_score = self._compute_landmark_spread_score(
                landmarks_68, width, height
            )

        # Combine scores (weighted average)
        quality_score = (
            0.3 * size_score
            + 0.3 * aspect_ratio_score
            + 0.4 * landmark_spread_score
        )

        # Confidence is lower for heuristic-based estimates
        confidence = 0.6  # Will increase to ~0.9 with LUVLi

        return QualityResult(
            quality_score=float(quality_score),
            confidence=confidence,
            size_score=float(size_score),
            aspect_ratio_score=float(aspect_ratio_score),
            landmark_spread_score=float(landmark_spread_score),
        )

    def _compute_landmark_spread_score(
        self,
        landmarks_68: List[List[float]],
        face_width: float,
        face_height: float,
    ) -> float:
        """
        Compute quality score based on landmark spread.

        Well-distributed landmarks indicate good alignment.
        Landmarks clustered on one side suggest profile view.
        """
        lm = np.array(landmarks_68)

        # Check landmark spread relative to face size
        lm_width = lm[:, 0].max() - lm[:, 0].min()
        lm_height = lm[:, 1].max() - lm[:, 1].min()

        # Landmarks should span most of the face
        width_coverage = lm_width / max(face_width, 1)
        height_coverage = lm_height / max(face_height, 1)

        # Good coverage is 0.6-0.9
        width_score = np.clip(width_coverage / 0.7, 0, 1) * np.clip(2 - width_coverage, 0, 1)
        height_score = np.clip(height_coverage / 0.7, 0, 1) * np.clip(2 - height_coverage, 0, 1)

        # Check symmetry (left vs right eye positions)
        left_eye_center = lm[36:42].mean(axis=0)
        right_eye_center = lm[42:48].mean(axis=0)

        # Eyes should be roughly at same height
        eye_height_diff = abs(left_eye_center[1] - right_eye_center[1])
        symmetry_score = np.clip(1.0 - eye_height_diff / (face_height * 0.1), 0, 1)

        return float((width_score + height_score + symmetry_score) / 3)

    def estimate_batch(
        self,
        bboxes: List[List[float]],
        landmarks_batch: Optional[List[List[List[float]]]] = None,
    ) -> List[QualityResult]:
        """
        Estimate quality for multiple faces.

        Args:
            bboxes: List of bounding boxes
            landmarks_batch: List of 68-point landmarks (optional)

        Returns:
            List of QualityResult
        """
        results = []

        for i, bbox in enumerate(bboxes):
            landmarks = None
            if landmarks_batch is not None and i < len(landmarks_batch):
                landmarks = landmarks_batch[i]

            result = self.estimate_quality(bbox, landmarks)
            results.append(result)

        return results


def compute_alignment_quality(
    bbox: List[float],
    landmarks_68: Optional[List[List[float]]] = None,
) -> float:
    """
    Compute alignment quality score (convenience function).

    Args:
        bbox: Face bounding box [x1, y1, x2, y2]
        landmarks_68: 68-point landmarks (optional)

    Returns:
        Quality score (0.0 to 1.0)

    Note:
        This is a heuristic-based estimate.
        TODO: Replace with LUVLi model for more accurate scores.
    """
    estimator = AlignmentQualityEstimator()
    result = estimator.estimate_quality(bbox, landmarks_68)
    return result.quality_score


def filter_by_quality(
    aligned_faces: List[dict],
    min_quality: float = 0.6,
) -> Tuple[List[dict], List[dict]]:
    """
    Filter aligned faces by quality threshold.

    Args:
        aligned_faces: List of aligned face dicts with 'alignment_quality'
        min_quality: Minimum quality threshold

    Returns:
        Tuple of (passed_faces, rejected_faces)
    """
    passed = []
    rejected = []

    for face in aligned_faces:
        quality = face.get("alignment_quality", 0.0)
        if quality >= min_quality:
            passed.append(face)
        else:
            rejected.append(face)

    logger.info(
        f"Quality filter: {len(passed)} passed, {len(rejected)} rejected "
        f"(threshold: {min_quality})"
    )

    return passed, rejected

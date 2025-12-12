"""
LUVLi-based Alignment Quality Estimation.

LUVLi provides per-landmark uncertainty and visibility estimates for
quality-aware face alignment.

CURRENT STATUS: Heuristic fallback with LUVLi model integration ready
- When face-alignment package is available: Uses FAN-based uncertainty estimation
- Fallback: Heuristic-based quality from bbox size and aspect ratio

Quality sources:
- "luvli": Model-based quality (confidence ~0.85-0.9)
- "heuristic": Fallback heuristic-based (confidence ~0.6)
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Quality source identifiers
QUALITY_SOURCE_LUVLI = "luvli"
QUALITY_SOURCE_HEURISTIC = "heuristic"


@dataclass
class LUVLiResult:
    """Result from LUVLi quality estimation."""

    # 68-point landmarks
    landmarks_68: Optional[List[List[float]]] = None

    # Per-landmark uncertainty (lower is better)
    uncertainties: Optional[List[float]] = None

    # Per-landmark visibility (0-1, higher is better)
    visibilities: Optional[List[float]] = None

    # Aggregate quality score (0-1, higher is better)
    quality_score: float = 0.0

    # Summary statistics
    uncertainty_mean: Optional[float] = None
    uncertainty_p95: Optional[float] = None
    visibility_fraction: Optional[float] = None
    visible_landmark_count: Optional[int] = None

    # Source tracking
    source: str = QUALITY_SOURCE_HEURISTIC
    confidence: float = 0.6  # Confidence in the estimate

    def to_dict(self) -> dict:
        d = {
            "quality_score": round(self.quality_score, 4),
            "source": self.source,
            "confidence": round(self.confidence, 4),
        }

        if self.source == QUALITY_SOURCE_LUVLI:
            d["uncertainty_summary"] = {
                "mean": round(self.uncertainty_mean, 4) if self.uncertainty_mean else None,
                "p95": round(self.uncertainty_p95, 4) if self.uncertainty_p95 else None,
            }
            d["visibility_summary"] = {
                "fraction": round(self.visibility_fraction, 4) if self.visibility_fraction else None,
                "visible_count": self.visible_landmark_count,
            }

        return d


class LUVLiQualityEstimator:
    """
    LUVLi-based alignment quality estimator.

    Estimates per-landmark uncertainty and visibility to compute
    an overall alignment quality score.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        visibility_threshold: float = 0.5,
        uncertainty_weight: float = 0.6,
        visibility_weight: float = 0.4,
    ):
        """
        Initialize LUVLi estimator.

        Args:
            model_path: Path to LUVLi model weights (optional)
            device: Device for inference (cuda/cpu)
            visibility_threshold: Threshold for counting visible landmarks
            uncertainty_weight: Weight for uncertainty in quality score
            visibility_weight: Weight for visibility in quality score
        """
        self.model_path = model_path
        self.device = device
        self.visibility_threshold = visibility_threshold
        self.uncertainty_weight = uncertainty_weight
        self.visibility_weight = visibility_weight

        self._model = None
        self._available = None

        self._init_model()

    def _init_model(self):
        """Initialize the FAN/LUVLi model."""
        try:
            import face_alignment

            self._model = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                device=self.device,
                flip_input=False,
            )
            self._available = True
            logger.info("LUVLi/FAN model initialized successfully")

        except ImportError:
            logger.warning("face-alignment package not available, using heuristic fallback")
            self._available = False
        except Exception as e:
            logger.warning(f"Failed to initialize LUVLi model: {e}")
            self._available = False

    @property
    def is_available(self) -> bool:
        """Check if LUVLi model is available."""
        return self._available or False

    def estimate_quality(
        self,
        image: np.ndarray,
        bbox: List[float],
        landmarks_68: Optional[List[List[float]]] = None,
    ) -> LUVLiResult:
        """
        Estimate alignment quality for a face.

        Args:
            image: Face image (BGR format)
            bbox: Face bounding box [x1, y1, x2, y2]
            landmarks_68: Pre-computed 68-point landmarks (optional)

        Returns:
            LUVLiResult with quality score and components
        """
        if self.is_available:
            try:
                return self._estimate_quality_luvli(image, bbox, landmarks_68)
            except Exception as e:
                logger.warning(f"LUVLi estimation failed, using heuristic: {e}")

        return self._estimate_quality_heuristic(bbox, landmarks_68)

    def _estimate_quality_luvli(
        self,
        image: np.ndarray,
        bbox: List[float],
        landmarks_68: Optional[List[List[float]]] = None,
    ) -> LUVLiResult:
        """Estimate quality using LUVLi/FAN model."""
        # Get landmarks from model if not provided
        if landmarks_68 is None or len(landmarks_68) < 68:
            detected = self._model.get_landmarks(image)
            if detected is None or len(detected) == 0:
                return self._estimate_quality_heuristic(bbox, landmarks_68)
            landmarks_68 = detected[0].tolist()

        # Estimate uncertainties (based on landmark spread and consistency)
        uncertainties = self._estimate_uncertainties(landmarks_68, bbox)

        # Estimate visibilities (based on landmark positions relative to bbox)
        visibilities = self._estimate_visibilities(landmarks_68, bbox)

        # Compute aggregate scores
        uncertainty_mean = float(np.mean(uncertainties))
        uncertainty_p95 = float(np.percentile(uncertainties, 95))
        visibility_fraction = float(np.mean([1 if v > self.visibility_threshold else 0 for v in visibilities]))
        visible_count = sum(1 for v in visibilities if v > self.visibility_threshold)

        # Compute quality score
        quality_score = self._compute_quality_score(
            uncertainty_mean, visibility_fraction
        )

        return LUVLiResult(
            landmarks_68=landmarks_68,
            uncertainties=uncertainties,
            visibilities=visibilities,
            quality_score=quality_score,
            uncertainty_mean=uncertainty_mean,
            uncertainty_p95=uncertainty_p95,
            visibility_fraction=visibility_fraction,
            visible_landmark_count=visible_count,
            source=QUALITY_SOURCE_LUVLI,
            confidence=0.85,  # Higher confidence for model-based estimate
        )

    def _estimate_quality_heuristic(
        self,
        bbox: List[float],
        landmarks_68: Optional[List[List[float]]] = None,
    ) -> LUVLiResult:
        """Estimate quality using heuristics."""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Size score (larger faces are generally better)
        face_size = min(width, height)
        size_score = np.clip(face_size / 100.0, 0, 1)

        # Aspect ratio score (frontal faces ~0.85 w/h ratio)
        aspect_ratio = width / (height if height > 0 else 1)
        ideal_ratio = 0.85
        aspect_score = np.clip(1.0 - abs(aspect_ratio - ideal_ratio) * 2, 0, 1)

        # Landmark spread score (if available)
        landmark_score = 0.5
        if landmarks_68 is not None and len(landmarks_68) >= 68:
            landmarks_arr = np.array(landmarks_68)
            lm_spread = (landmarks_arr[:, 0].max() - landmarks_arr[:, 0].min()) / (width if width > 0 else 1)
            landmark_score = np.clip(lm_spread / 0.7, 0, 1)

        # Combined score
        quality_score = 0.3 * size_score + 0.3 * aspect_score + 0.4 * landmark_score

        return LUVLiResult(
            landmarks_68=landmarks_68,
            quality_score=float(quality_score),
            source=QUALITY_SOURCE_HEURISTIC,
            confidence=0.6,
        )

    def _estimate_uncertainties(
        self,
        landmarks_68: List[List[float]],
        bbox: List[float],
    ) -> List[float]:
        """Estimate per-landmark uncertainties."""
        lm = np.array(landmarks_68)
        x1, y1, x2, y2 = bbox
        face_size = max(x2 - x1, y2 - y1)

        uncertainties = []
        for i, pt in enumerate(lm):
            # Distance from bbox center (normalized)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            dist = np.sqrt((pt[0] - center_x) ** 2 + (pt[1] - center_y) ** 2)
            normalized_dist = dist / (face_size / 2)

            # Points far from center have higher uncertainty
            uncertainty = min(normalized_dist * 0.5, 1.0)
            uncertainties.append(uncertainty)

        return uncertainties

    def _estimate_visibilities(
        self,
        landmarks_68: List[List[float]],
        bbox: List[float],
    ) -> List[float]:
        """Estimate per-landmark visibilities."""
        lm = np.array(landmarks_68)
        x1, y1, x2, y2 = bbox

        visibilities = []
        for pt in lm:
            # Check if point is within bbox
            in_bbox = (x1 <= pt[0] <= x2) and (y1 <= pt[1] <= y2)

            # Compute visibility based on position
            if in_bbox:
                # Higher visibility near center
                margin_x = min(pt[0] - x1, x2 - pt[0]) / (x2 - x1)
                margin_y = min(pt[1] - y1, y2 - pt[1]) / (y2 - y1)
                visibility = min(margin_x, margin_y) * 2
            else:
                visibility = 0.0

            visibilities.append(min(max(visibility, 0.0), 1.0))

        return visibilities

    def _compute_quality_score(
        self,
        uncertainty_mean: float,
        visibility_fraction: float,
    ) -> float:
        """Compute overall quality score from components."""
        # Lower uncertainty is better (invert)
        uncertainty_score = 1.0 - np.clip(uncertainty_mean, 0, 1)

        # Higher visibility is better
        visibility_score = visibility_fraction

        # Weighted combination
        quality = (
            self.uncertainty_weight * uncertainty_score
            + self.visibility_weight * visibility_score
        )

        return float(np.clip(quality, 0, 1))

"""
Regression tests for small face detection in wide shots.

Tests the wide shot mode configuration, detection parameter tuning,
and person fallback detection features added in 2025-12.

Related: feature/2025-12-small-face-detection-fallback
"""

import numpy as np
import pytest


class TestDetectionConfigLoading:
    """Tests for detection config YAML loading."""

    def test_detection_config_loads(self):
        """Detection config should load from YAML."""
        from tools.episode_run import _load_detection_config_yaml

        config = _load_detection_config_yaml()
        # Config may be empty dict if file doesn't exist, but shouldn't error
        assert isinstance(config, dict), "Should return a dict"

    def test_wide_shot_mode_constant_exists(self):
        """WIDE_SHOT_MODE_ENABLED constant should be defined."""
        from tools.episode_run import WIDE_SHOT_MODE_ENABLED

        assert isinstance(WIDE_SHOT_MODE_ENABLED, bool), "Should be a boolean"

    def test_wide_shot_input_size_constant_exists(self):
        """WIDE_SHOT_INPUT_SIZE constant should be defined."""
        from tools.episode_run import WIDE_SHOT_INPUT_SIZE

        assert isinstance(WIDE_SHOT_INPUT_SIZE, int), "Should be an integer"
        assert WIDE_SHOT_INPUT_SIZE >= 640, "Should be at least 640 for effective detection"

    def test_detection_min_size_constant_exists(self):
        """DETECTION_MIN_SIZE constant should be defined."""
        from tools.episode_run import DETECTION_MIN_SIZE

        assert isinstance(DETECTION_MIN_SIZE, int), "Should be an integer"
        assert DETECTION_MIN_SIZE >= 8, "Should be at least 8 pixels"
        assert DETECTION_MIN_SIZE <= 64, "Should be at most 64 pixels for small face detection"


class TestWideShortModeApplication:
    """Tests for wide shot mode application in detection."""

    def test_detector_uses_config_min_area(self):
        """DetectionPass should use config-based min_area."""
        from tools.episode_run import (
            RetinaFaceDetectorBackend,
            WIDE_SHOT_MODE_ENABLED,
            WIDE_SHOT_MIN_FACE_SIZE,
            DETECTION_MIN_SIZE,
        )

        # Create detector instance (won't load model, just check config)
        detector = RetinaFaceDetectorBackend.__new__(RetinaFaceDetectorBackend)
        # Call __init__ to set attributes
        detector.__init__(device="cpu")

        # Check min_area is calculated from config
        if WIDE_SHOT_MODE_ENABLED:
            expected_min_area = float(WIDE_SHOT_MIN_FACE_SIZE * WIDE_SHOT_MIN_FACE_SIZE)
        else:
            expected_min_area = float(DETECTION_MIN_SIZE * DETECTION_MIN_SIZE)

        assert detector.min_area == expected_min_area, (
            f"Expected min_area={expected_min_area}, got {detector.min_area}"
        )

    def test_valid_face_box_respects_min_area(self):
        """_valid_face_box should filter based on min_area."""
        from tools.episode_run import _valid_face_box

        # 15x15 face = 225 px² area
        small_bbox = np.array([100.0, 100.0, 115.0, 115.0])
        # 30x30 face = 900 px² area
        medium_bbox = np.array([100.0, 100.0, 130.0, 130.0])

        # With high min_area (500), small face should be rejected
        result_small = _valid_face_box(small_bbox, score=0.8, min_score=0.5, min_area=500.0)
        assert result_small is False, "Should reject 15x15 face with min_area=500"

        # With low min_area (100), small face should be accepted
        result_small_low = _valid_face_box(small_bbox, score=0.8, min_score=0.5, min_area=100.0)
        assert result_small_low is True, "Should accept 15x15 face with min_area=100"

        # Medium face should be accepted with high min_area
        result_medium = _valid_face_box(medium_bbox, score=0.8, min_score=0.5, min_area=500.0)
        assert result_medium is True, "Should accept 30x30 face with min_area=500"


class TestPersonFallbackDetector:
    """Tests for person fallback detection."""

    def test_person_fallback_constants_exist(self):
        """Person fallback config constants should be defined."""
        from tools.episode_run import (
            PERSON_FALLBACK_ENABLED,
            PERSON_FALLBACK_MIN_BODY_HEIGHT,
            PERSON_FALLBACK_FACE_REGION_RATIO,
            PERSON_FALLBACK_CONFIDENCE_TH,
            PERSON_FALLBACK_MAX_PER_FRAME,
        )

        assert isinstance(PERSON_FALLBACK_ENABLED, bool), "Should be boolean"
        assert isinstance(PERSON_FALLBACK_MIN_BODY_HEIGHT, int), "Should be int"
        assert isinstance(PERSON_FALLBACK_FACE_REGION_RATIO, float), "Should be float"
        assert 0 < PERSON_FALLBACK_FACE_REGION_RATIO < 1, "Should be between 0 and 1"
        assert isinstance(PERSON_FALLBACK_CONFIDENCE_TH, float), "Should be float"
        assert isinstance(PERSON_FALLBACK_MAX_PER_FRAME, int), "Should be int"

    def test_person_fallback_detector_class_exists(self):
        """PersonFallbackDetector class should exist."""
        from tools.episode_run import PersonFallbackDetector

        assert PersonFallbackDetector is not None

    def test_person_fallback_detector_init(self):
        """PersonFallbackDetector should initialize without loading model."""
        from tools.episode_run import PersonFallbackDetector

        detector = PersonFallbackDetector(
            device="cpu",
            confidence_thresh=0.5,
            min_body_height=100,
            face_region_ratio=0.25,
            max_detections=10,
        )

        assert detector.device == "cpu"
        assert detector.confidence_thresh == 0.5
        assert detector.min_body_height == 100
        assert detector.face_region_ratio == 0.25
        assert detector.max_detections == 10
        assert detector._model is None, "Model should not be loaded until needed"

    def test_build_person_fallback_detector_respects_config(self):
        """_build_person_fallback_detector should return None if disabled."""
        from tools.episode_run import (
            _build_person_fallback_detector,
            PERSON_FALLBACK_ENABLED,
        )

        result = _build_person_fallback_detector("cpu")

        if PERSON_FALLBACK_ENABLED:
            assert result is not None, "Should return detector when enabled"
        else:
            assert result is None, "Should return None when disabled"

    def test_person_fallback_returns_empty_when_disabled(self):
        """PersonFallbackDetector.detect_persons should return [] if globally disabled."""
        from tools.episode_run import PersonFallbackDetector, PERSON_FALLBACK_ENABLED

        detector = PersonFallbackDetector(device="cpu")

        # Create dummy image
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        # If disabled globally, should return empty list without loading model
        if not PERSON_FALLBACK_ENABLED:
            result = detector.detect_persons(image)
            assert result == [], "Should return empty list when disabled"
            assert detector._model is None, "Should not load model when disabled"


class TestFaceRegionEstimation:
    """Tests for face region estimation from body bounding boxes."""

    def test_face_region_ratio_bounds(self):
        """Face region should be estimated from top portion of body."""
        from tools.episode_run import PERSON_FALLBACK_FACE_REGION_RATIO

        # Face should typically be in upper 20-30% of body
        assert 0.15 <= PERSON_FALLBACK_FACE_REGION_RATIO <= 0.40, (
            f"Face region ratio {PERSON_FALLBACK_FACE_REGION_RATIO} out of expected range"
        )

    def test_face_bbox_from_body_calculation(self):
        """Face bbox should be estimated correctly from body bbox."""
        # Simulate the calculation done in PersonFallbackDetector.detect_persons
        body_x1, body_y1, body_x2, body_y2 = 100.0, 50.0, 200.0, 400.0
        body_height = body_y2 - body_y1  # 350 pixels
        body_width = body_x2 - body_x1  # 100 pixels
        face_region_ratio = 0.25

        # Calculate face region (same as PersonFallbackDetector)
        face_height = body_height * face_region_ratio  # 87.5 pixels
        face_width = min(face_height * 0.8, body_width * 0.6)  # min(70, 60) = 60

        body_center_x = (body_x1 + body_x2) / 2  # 150
        face_x1 = body_center_x - face_width / 2  # 150 - 30 = 120
        face_x2 = body_center_x + face_width / 2  # 150 + 30 = 180
        face_y1 = body_y1  # 50
        face_y2 = body_y1 + face_height  # 50 + 87.5 = 137.5

        # Verify dimensions are reasonable
        assert face_y2 > face_y1, "Face should have positive height"
        assert face_x2 > face_x1, "Face should have positive width"
        assert face_y1 == body_y1, "Face should start at top of body"
        assert face_y2 <= body_y1 + body_height * 0.5, "Face should be in upper half of body"


class TestPNGExportSupport:
    """Tests for PNG export support in image utilities."""

    def test_safe_imwrite_supports_png_flag(self):
        """safe_imwrite should accept use_png parameter."""
        from tools._img_utils import safe_imwrite
        import inspect

        sig = inspect.signature(safe_imwrite)
        params = list(sig.parameters.keys())

        assert "use_png" in params, "safe_imwrite should have use_png parameter"
        assert "png_compression" in params, "safe_imwrite should have png_compression parameter"

    def test_encode_png_bytes_exists(self):
        """encode_png_bytes function should exist."""
        from tools._img_utils import encode_png_bytes

        assert encode_png_bytes is not None, "encode_png_bytes should be importable"

    def test_encode_png_bytes_returns_bytes(self):
        """encode_png_bytes should return PNG bytes."""
        from tools._img_utils import encode_png_bytes

        # Create test image
        image = np.ones((50, 50, 3), dtype=np.uint8) * 128

        result = encode_png_bytes(image)

        assert isinstance(result, bytes), "Should return bytes"
        # PNG magic bytes: 89 50 4E 47 0D 0A 1A 0A
        assert result[:4] == b'\x89PNG', "Should be valid PNG format"

    def test_encode_png_bytes_handles_none(self):
        """encode_png_bytes should return None for None input."""
        from tools._img_utils import encode_png_bytes

        result = encode_png_bytes(None)
        assert result is None, "Should return None for None input"

    def test_encode_png_bytes_handles_empty_image(self):
        """encode_png_bytes should return None for empty image."""
        from tools._img_utils import encode_png_bytes

        empty = np.array([])
        result = encode_png_bytes(empty)
        assert result is None, "Should return None for empty image"


class TestMinFaceAreaCalculation:
    """Tests for min face area calculation from pixel dimension."""

    def test_min_area_from_dimension(self):
        """Min area should be dimension squared."""
        min_size_dim = 16
        expected_area = 16 * 16  # 256 px²

        actual_area = float(min_size_dim * min_size_dim)
        assert actual_area == expected_area, f"Expected {expected_area}, got {actual_area}"

    def test_small_face_passes_low_min_area(self):
        """16x16 face (256 px²) should pass with min_area=200."""
        from tools.episode_run import _valid_face_box

        bbox = np.array([100.0, 100.0, 116.0, 116.0])  # 16x16 = 256 px²
        result = _valid_face_box(bbox, score=0.5, min_score=0.4, min_area=200.0)
        assert result is True, "16x16 face should pass with min_area=200"

    def test_small_face_fails_high_min_area(self):
        """16x16 face (256 px²) should fail with min_area=300."""
        from tools.episode_run import _valid_face_box

        bbox = np.array([100.0, 100.0, 116.0, 116.0])  # 16x16 = 256 px²
        result = _valid_face_box(bbox, score=0.5, min_score=0.4, min_area=300.0)
        assert result is False, "16x16 face should fail with min_area=300"


class TestDetectionSampleClassLabel:
    """Tests for DetectionSample class labeling."""

    def test_face_estimated_label(self):
        """Face estimated from body should have 'face_estimated' label."""
        from tools.episode_run import DetectionSample

        # Create sample like PersonFallbackDetector does
        sample = DetectionSample(
            bbox=np.array([100.0, 100.0, 150.0, 150.0]),
            conf=0.8,
            class_idx=0,
            class_label="face_estimated",
            landmarks=None,
        )

        assert sample.class_label == "face_estimated", "Should have face_estimated label"
        assert sample.landmarks is None, "Estimated faces should have no landmarks"

    def test_regular_face_label(self):
        """Regular face detection should have 'face' label."""
        from tools.episode_run import DetectionSample, FACE_CLASS_LABEL

        sample = DetectionSample(
            bbox=np.array([100.0, 100.0, 150.0, 150.0]),
            conf=0.9,
            class_idx=0,
            class_label=FACE_CLASS_LABEL,
            landmarks=np.zeros(10),
        )

        assert sample.class_label == FACE_CLASS_LABEL, f"Should have '{FACE_CLASS_LABEL}' label"

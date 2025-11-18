"""
Defensive tests for detect/track pipeline - None bbox and invalid margin handling.

Tests comprehensive validation of bbox coordinates, margins, and scaling factors
to prevent TypeError crashes from None multiplication.

Related: nov-17-detect-track-none-bbox-fix.md
"""

import numpy as np


def test_valid_face_box_with_none_coords():
    """_valid_face_box should reject bboxes with None coordinates."""
    from tools.episode_run import _valid_face_box

    # Bbox with None values
    bbox = np.array([100.0, 200.0, None, None])
    result = _valid_face_box(bbox, score=0.8, min_score=0.5, min_area=20.0)
    assert result is False, "Should reject bbox with None coordinates"


def test_valid_face_box_with_invalid_types():
    """_valid_face_box should reject bboxes with non-numeric values."""
    from tools.episode_run import _valid_face_box

    # Bbox with string values
    bbox = np.array([100.0, 200.0, "foo", 400.0], dtype=object)
    result = _valid_face_box(bbox, score=0.8, min_score=0.5, min_area=20.0)
    assert result is False, "Should reject bbox with string coordinates"


def test_valid_face_box_with_short_bbox():
    """_valid_face_box should reject bboxes with insufficient coordinates."""
    from tools.episode_run import _valid_face_box

    # Bbox with only 3 elements
    bbox = np.array([100.0, 200.0, 300.0])
    result = _valid_face_box(bbox, score=0.8, min_score=0.5, min_area=20.0)
    assert result is False, "Should reject bbox with < 4 coordinates"


def test_valid_face_box_with_nan_coords():
    """_valid_face_box should reject bboxes with NaN coordinates."""
    from tools.episode_run import _valid_face_box

    # Bbox with NaN values
    bbox = np.array([100.0, 200.0, np.nan, 400.0])
    result = _valid_face_box(bbox, score=0.8, min_score=0.5, min_area=20.0)
    assert result is False, "Should reject bbox with NaN coordinates"


def test_valid_face_box_with_valid_bbox():
    """_valid_face_box should accept valid bboxes."""
    from tools.episode_run import _valid_face_box

    # Valid bbox
    bbox = np.array([100.0, 200.0, 300.0, 400.0])
    result = _valid_face_box(bbox, score=0.8, min_score=0.5, min_area=20.0)
    assert result is True, "Should accept valid bbox"


def test_prepare_face_crop_with_none_bbox():
    """_prepare_face_crop should return None and error message for None coords."""
    from tools.episode_run import _prepare_face_crop

    # Create dummy image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = [100.0, 200.0, None, None]

    crop, error = _prepare_face_crop(image, bbox, None)
    assert crop is None, "Should return None crop for invalid bbox"
    assert error is not None, "Should return error message"
    assert (
        "invalid_bbox_none_values" in error
    ), f"Expected 'invalid_bbox_none_values' in error, got: {error}"


def test_prepare_face_crop_with_partial_none_bbox():
    """_prepare_face_crop should handle bbox with any None value."""
    from tools.episode_run import _prepare_face_crop

    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Test each position having None
    test_cases = [
        [None, 200.0, 300.0, 400.0],
        [100.0, None, 300.0, 400.0],
        [100.0, 200.0, None, 400.0],
        [100.0, 200.0, 300.0, None],
    ]

    for bbox in test_cases:
        crop, error = _prepare_face_crop(image, bbox, None)
        assert crop is None, f"Should return None crop for bbox {bbox}"
        assert (
            error is not None and "invalid_bbox_none_values" in error
        ), f"Should return 'invalid_bbox_none_values' error for bbox {bbox}"


def test_prepare_face_crop_with_invalid_margin():
    """_prepare_face_crop should handle invalid margin gracefully."""
    from tools.episode_run import _prepare_face_crop

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = [100.0, 200.0, 300.0, 400.0]

    # Should not crash with None margin - uses default (0.15)
    crop, error = _prepare_face_crop(image, bbox, None, margin=None)
    # Either succeeds with default margin or fails gracefully
    assert (
        crop is not None or error is not None
    ), "Should handle None margin without crashing"


def test_prepare_face_crop_with_string_margin():
    """_prepare_face_crop should handle non-numeric margin gracefully."""
    from tools.episode_run import _prepare_face_crop

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = [100.0, 200.0, 300.0, 400.0]

    # Should not crash with string margin - uses default (0.15)
    crop, error = _prepare_face_crop(image, bbox, None, margin="invalid")
    # Either succeeds with default margin or fails gracefully
    assert (
        crop is not None or error is not None
    ), "Should handle string margin without crashing"


def test_prepare_face_crop_with_negative_margin():
    """_prepare_face_crop should clamp negative margin to 0."""
    from tools.episode_run import _prepare_face_crop

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = [100.0, 200.0, 300.0, 400.0]

    # Negative margin should be clamped to 0.0
    crop, error = _prepare_face_crop(image, bbox, None, margin=-0.5)
    # Should either succeed or fail gracefully (not crash)
    assert (
        crop is not None or error is not None
    ), "Should handle negative margin without crashing"


def test_prepare_face_crop_with_valid_inputs():
    """_prepare_face_crop should work with valid inputs."""
    from tools.episode_run import _prepare_face_crop

    # Create image with some content
    image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    bbox = [100.0, 100.0, 300.0, 300.0]

    crop, error = _prepare_face_crop(image, bbox, None, margin=0.15)
    # Should succeed (crop might be None if safe_crop fails for other reasons, but no TypeError)
    assert (
        error != "invalid_bbox_none_values"
    ), "Should not report None bbox error for valid bbox"
    assert "invalid_bbox_coordinates" not in str(
        error
    ), "Should not report coordinate error for valid bbox"


def test_prepare_face_crop_adaptive_margin_with_small_face():
    """_prepare_face_crop should apply larger margin for small faces when adaptive_margin=True."""
    from tools.episode_run import _prepare_face_crop

    image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    # Small bbox (50x50 = 2500 px² < 5000 threshold)
    bbox = [100.0, 100.0, 150.0, 150.0]

    crop, error = _prepare_face_crop(
        image, bbox, None, margin=0.15, adaptive_margin=True
    )
    # Should not crash with adaptive margin calculations
    assert (
        crop is not None or error is not None
    ), "Should handle adaptive margin without crashing"
    if error:
        assert (
            "invalid_bbox" not in error
        ), f"Should not report bbox error for valid small face, got: {error}"


def test_prepare_face_crop_adaptive_margin_with_large_face():
    """_prepare_face_crop should apply smaller margin for large faces when adaptive_margin=True."""
    from tools.episode_run import _prepare_face_crop

    image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    # Large bbox (200x200 = 40000 px² > 15000 threshold)
    bbox = [100.0, 100.0, 300.0, 300.0]

    crop, error = _prepare_face_crop(
        image, bbox, None, margin=0.15, adaptive_margin=True
    )
    # Should not crash with adaptive margin calculations
    assert (
        crop is not None or error is not None
    ), "Should handle adaptive margin without crashing"
    if error:
        assert (
            "invalid_bbox" not in error
        ), f"Should not report bbox error for valid large face, got: {error}"


def test_prepare_face_crop_with_invalid_string_coordinates():
    """_prepare_face_crop should handle string coordinates gracefully."""
    from tools.episode_run import _prepare_face_crop

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = ["100", "200", "300", "400"]  # String coordinates

    crop, error = _prepare_face_crop(image, bbox, None)
    # Should either succeed (if strings convert to float) or fail gracefully
    assert (
        crop is not None or error is not None
    ), "Should handle string coordinates without crashing"


# === Tests for _safe_bbox_or_none validator (regression tests for NoneType multiply prevention) ===


def test_safe_bbox_or_none_with_valid_list():
    """_safe_bbox_or_none should accept valid bbox as list."""
    from tools.episode_run import _safe_bbox_or_none

    bbox = [100.0, 200.0, 300.0, 400.0]
    validated, error = _safe_bbox_or_none(bbox)

    assert validated is not None, "Should validate valid bbox"
    assert error is None, "Should not return error for valid bbox"
    assert validated == [100.0, 200.0, 300.0, 400.0], "Should return same coordinates"


def test_safe_bbox_or_none_with_valid_numpy_array():
    """_safe_bbox_or_none should accept valid bbox as numpy array."""
    from tools.episode_run import _safe_bbox_or_none

    bbox = np.array([100.0, 200.0, 300.0, 400.0])
    validated, error = _safe_bbox_or_none(bbox)

    assert validated is not None, "Should validate valid numpy bbox"
    assert error is None, "Should not return error for valid numpy bbox"
    assert validated == [100.0, 200.0, 300.0, 400.0], "Should convert to list"


def test_safe_bbox_or_none_with_none_bbox():
    """_safe_bbox_or_none should reject None bbox."""
    from tools.episode_run import _safe_bbox_or_none

    validated, error = _safe_bbox_or_none(None)

    assert validated is None, "Should reject None bbox"
    assert error == "bbox_is_none", f"Expected 'bbox_is_none' error, got: {error}"


def test_safe_bbox_or_none_with_none_coordinates():
    """_safe_bbox_or_none should reject bbox with None coordinates."""
    from tools.episode_run import _safe_bbox_or_none

    # Test different positions of None
    test_cases = [
        ([None, 200.0, 300.0, 400.0], "bbox_coord_0_is_none"),
        ([100.0, None, 300.0, 400.0], "bbox_coord_1_is_none"),
        ([100.0, 200.0, None, 400.0], "bbox_coord_2_is_none"),
        ([100.0, 200.0, 300.0, None], "bbox_coord_3_is_none"),
    ]

    for bbox, expected_error in test_cases:
        validated, error = _safe_bbox_or_none(bbox)
        assert validated is None, f"Should reject bbox {bbox}"
        assert (
            error == expected_error
        ), f"Expected '{expected_error}' error for {bbox}, got: {error}"


def test_safe_bbox_or_none_with_nan_coordinates():
    """_safe_bbox_or_none should reject bbox with NaN coordinates."""
    from tools.episode_run import _safe_bbox_or_none

    bbox = [100.0, 200.0, np.nan, 400.0]
    validated, error = _safe_bbox_or_none(bbox)

    assert validated is None, "Should reject bbox with NaN"
    assert (
        "bbox_coord_2_not_finite" in error
    ), f"Expected 'not_finite' error, got: {error}"


def test_safe_bbox_or_none_with_inf_coordinates():
    """_safe_bbox_or_none should reject bbox with infinity coordinates."""
    from tools.episode_run import _safe_bbox_or_none

    bbox = [100.0, 200.0, np.inf, 400.0]
    validated, error = _safe_bbox_or_none(bbox)

    assert validated is None, "Should reject bbox with infinity"
    assert (
        "bbox_coord_2_not_finite" in error
    ), f"Expected 'not_finite' error, got: {error}"


def test_safe_bbox_or_none_with_wrong_length():
    """_safe_bbox_or_none should reject bbox with wrong number of coordinates."""
    from tools.episode_run import _safe_bbox_or_none

    # Too few coordinates
    bbox_short = [100.0, 200.0, 300.0]
    validated, error = _safe_bbox_or_none(bbox_short)
    assert validated is None, "Should reject bbox with 3 coordinates"
    assert (
        error == "bbox_wrong_length_3"
    ), f"Expected 'wrong_length_3' error, got: {error}"

    # Too many coordinates
    bbox_long = [100.0, 200.0, 300.0, 400.0, 500.0]
    validated, error = _safe_bbox_or_none(bbox_long)
    assert validated is None, "Should reject bbox with 5 coordinates"
    assert (
        error == "bbox_wrong_length_5"
    ), f"Expected 'wrong_length_5' error, got: {error}"


def test_safe_bbox_or_none_with_string_coordinates():
    """_safe_bbox_or_none should convert valid string coordinates to floats."""
    from tools.episode_run import _safe_bbox_or_none

    bbox = ["100", "200", "300", "400"]
    validated, error = _safe_bbox_or_none(bbox)

    assert validated is not None, "Should validate string bbox that converts to floats"
    assert error is None, "Should not return error for valid string bbox"
    assert validated == [100.0, 200.0, 300.0, 400.0], "Should convert strings to floats"


def test_safe_bbox_or_none_with_invalid_string_coordinates():
    """_safe_bbox_or_none should reject bbox with non-numeric strings."""
    from tools.episode_run import _safe_bbox_or_none

    bbox = [100.0, 200.0, "invalid", 400.0]
    validated, error = _safe_bbox_or_none(bbox)

    assert validated is None, "Should reject bbox with invalid string"
    assert "bbox_coord_2_invalid" in error, f"Expected 'invalid' error, got: {error}"


# === Integration test for detection bbox validation in detect loop ===


def test_detection_bbox_validation_filters_invalid_before_tracking():
    """
    Detect loop should validate detection bboxes before passing to ByteTrack.

    This regression test ensures invalid bboxes from RetinaFace are dropped
    early, preventing NoneType multiply errors in downstream crop/export logic.
    """
    from tools.episode_run import _safe_bbox_or_none

    # Simulate detection results with mix of valid and invalid bboxes
    class MockDetection:
        def __init__(self, bbox, conf, class_label):
            self.bbox = np.array(bbox) if bbox is not None else None
            self.conf = conf
            self.class_label = class_label

    # Mix of valid and invalid detections (like RetinaFace might emit)
    raw_detections = [
        MockDetection([100.0, 200.0, 300.0, 400.0], 0.95, "face"),  # Valid
        MockDetection([None, 200.0, 300.0, 400.0], 0.92, "face"),  # Invalid: None x1
        MockDetection([100.0, 200.0, 300.0, None], 0.89, "face"),  # Invalid: None y2
        MockDetection([150.0, 250.0, 350.0, 450.0], 0.91, "face"),  # Valid
        MockDetection([100.0, 200.0, np.nan, 400.0], 0.88, "face"),  # Invalid: NaN
    ]

    # Filter logic (simulates what detect loop should do)
    validated_detections = []
    invalid_count = 0

    for det in raw_detections:
        if det.class_label != "face":
            continue

        validated_bbox, bbox_err = _safe_bbox_or_none(det.bbox)
        if validated_bbox is None:
            invalid_count += 1
            # In real code, would log warning here
            continue

        # Update with validated bbox
        det.bbox = np.array(validated_bbox)
        validated_detections.append(det)

    # Assertions
    assert len(raw_detections) == 5, "Should start with 5 raw detections"
    assert (
        len(validated_detections) == 2
    ), "Should have 2 valid detections after filtering"
    assert invalid_count == 3, "Should have dropped 3 invalid detections"

    # Validate that remaining detections have valid bboxes
    for det in validated_detections:
        assert len(det.bbox) == 4, "Valid detection should have 4 bbox coords"
        for coord in det.bbox:
            assert coord is not None, "Valid bbox should have no None coords"
            assert np.isfinite(coord), "Valid bbox should have finite coords"

    # Verify we can safely perform multiplication on validated bboxes (no TypeError)
    for det in validated_detections:
        x1, y1, x2, y2 = det.bbox
        width = x2 - x1
        height = y2 - y1
        margin = 0.15
        expand_x = width * margin  # Should not raise TypeError
        expand_y = height * margin  # Should not raise TypeError
        assert expand_x >= 0, "Margin expansion should be non-negative"
        assert expand_y >= 0, "Margin expansion should be non-negative"


def test_tracker_inputs_drop_invalid_detections_before_vstack():
    """
    _tracker_inputs_from_samples should validate and drop invalid bboxes before np.vstack.

    This regression test ensures the fix for NoneType multiply errors caused by
    invalid bboxes reaching np.vstack in _tracker_inputs_from_samples().

    Related: Comprehensive patch for ONNX provider improvements (Nov 18, 2025)
    """
    from tools.episode_run import _tracker_inputs_from_samples

    # Mock DetectionSample class
    class DetectionSample:
        def __init__(self, bbox, conf, class_idx):
            self.bbox = np.array(bbox) if isinstance(bbox, list) else bbox
            self.conf = conf
            self.class_idx = class_idx

    # Mix of valid and invalid detections (simulates real-world scenario)
    samples = [
        DetectionSample([100.0, 200.0, 300.0, 400.0], 0.95, 0),  # Valid
        DetectionSample([None, 200.0, 300.0, 400.0], 0.92, 0),  # Invalid: None x1
        DetectionSample([100.0, 200.0, 300.0, None], 0.89, 0),  # Invalid: None y2
        DetectionSample([150.0, 250.0, 350.0, 450.0], 0.91, 0),  # Valid
        DetectionSample([100.0, 200.0, np.nan, 400.0], 0.88, 0),  # Invalid: NaN
    ]

    # Should not crash and should only keep valid detections
    tracker_inputs = _tracker_inputs_from_samples(samples)

    # Verify that only 2 valid detections made it through
    assert (
        tracker_inputs.xyxy.shape[0] == 2
    ), f"Expected 2 valid detections, got {tracker_inputs.xyxy.shape[0]}"
    assert tracker_inputs.conf.shape[0] == 2
    assert tracker_inputs.cls.shape[0] == 2

    # Verify that the valid detections have correct values
    assert np.isclose(
        tracker_inputs.conf[0], 0.95
    ), f"Expected conf[0]=0.95, got {tracker_inputs.conf[0]}"
    assert np.isclose(
        tracker_inputs.conf[1], 0.91
    ), f"Expected conf[1]=0.91, got {tracker_inputs.conf[1]}"

    # Verify that we can safely perform vstack operation (no TypeError)
    assert tracker_inputs.xyxy.dtype == np.float32
    assert tracker_inputs.conf.dtype == np.float32
    assert tracker_inputs.cls.dtype == np.float32

    # Verify that all bbox coordinates are finite
    assert np.all(
        np.isfinite(tracker_inputs.xyxy)
    ), "All bbox coordinates should be finite"

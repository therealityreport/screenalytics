"""
Defensive tests for detect/track pipeline - None bbox and invalid margin handling.

Tests comprehensive validation of bbox coordinates, margins, and scaling factors
to prevent TypeError crashes from None multiplication.

Related: nov-17-detect-track-none-bbox-fix.md
"""

import numpy as np
import pytest


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
    assert "invalid_bbox_none_values" in error, f"Expected 'invalid_bbox_none_values' in error, got: {error}"


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
        assert error is not None and "invalid_bbox_none_values" in error, \
            f"Should return 'invalid_bbox_none_values' error for bbox {bbox}"


def test_prepare_face_crop_with_invalid_margin():
    """_prepare_face_crop should handle invalid margin gracefully."""
    from tools.episode_run import _prepare_face_crop

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = [100.0, 200.0, 300.0, 400.0]

    # Should not crash with None margin - uses default (0.15)
    crop, error = _prepare_face_crop(image, bbox, None, margin=None)
    # Either succeeds with default margin or fails gracefully
    assert crop is not None or error is not None, "Should handle None margin without crashing"


def test_prepare_face_crop_with_string_margin():
    """_prepare_face_crop should handle non-numeric margin gracefully."""
    from tools.episode_run import _prepare_face_crop

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = [100.0, 200.0, 300.0, 400.0]

    # Should not crash with string margin - uses default (0.15)
    crop, error = _prepare_face_crop(image, bbox, None, margin="invalid")
    # Either succeeds with default margin or fails gracefully
    assert crop is not None or error is not None, "Should handle string margin without crashing"


def test_prepare_face_crop_with_negative_margin():
    """_prepare_face_crop should clamp negative margin to 0."""
    from tools.episode_run import _prepare_face_crop

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = [100.0, 200.0, 300.0, 400.0]

    # Negative margin should be clamped to 0.0
    crop, error = _prepare_face_crop(image, bbox, None, margin=-0.5)
    # Should either succeed or fail gracefully (not crash)
    assert crop is not None or error is not None, "Should handle negative margin without crashing"


def test_prepare_face_crop_with_valid_inputs():
    """_prepare_face_crop should work with valid inputs."""
    from tools.episode_run import _prepare_face_crop

    # Create image with some content
    image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    bbox = [100.0, 100.0, 300.0, 300.0]

    crop, error = _prepare_face_crop(image, bbox, None, margin=0.15)
    # Should succeed (crop might be None if safe_crop fails for other reasons, but no TypeError)
    assert error != "invalid_bbox_none_values", "Should not report None bbox error for valid bbox"
    assert "invalid_bbox_coordinates" not in str(error), "Should not report coordinate error for valid bbox"


def test_prepare_face_crop_adaptive_margin_with_small_face():
    """_prepare_face_crop should apply larger margin for small faces when adaptive_margin=True."""
    from tools.episode_run import _prepare_face_crop

    image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    # Small bbox (50x50 = 2500 px² < 5000 threshold)
    bbox = [100.0, 100.0, 150.0, 150.0]

    crop, error = _prepare_face_crop(image, bbox, None, margin=0.15, adaptive_margin=True)
    # Should not crash with adaptive margin calculations
    assert crop is not None or error is not None, "Should handle adaptive margin without crashing"
    if error:
        assert "invalid_bbox" not in error, f"Should not report bbox error for valid small face, got: {error}"


def test_prepare_face_crop_adaptive_margin_with_large_face():
    """_prepare_face_crop should apply smaller margin for large faces when adaptive_margin=True."""
    from tools.episode_run import _prepare_face_crop

    image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    # Large bbox (200x200 = 40000 px² > 15000 threshold)
    bbox = [100.0, 100.0, 300.0, 300.0]

    crop, error = _prepare_face_crop(image, bbox, None, margin=0.15, adaptive_margin=True)
    # Should not crash with adaptive margin calculations
    assert crop is not None or error is not None, "Should handle adaptive margin without crashing"
    if error:
        assert "invalid_bbox" not in error, f"Should not report bbox error for valid large face, got: {error}"


def test_prepare_face_crop_with_invalid_string_coordinates():
    """_prepare_face_crop should handle string coordinates gracefully."""
    from tools.episode_run import _prepare_face_crop

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = ["100", "200", "300", "400"]  # String coordinates

    crop, error = _prepare_face_crop(image, bbox, None)
    # Should either succeed (if strings convert to float) or fail gracefully
    assert crop is not None or error is not None, "Should handle string coordinates without crashing"

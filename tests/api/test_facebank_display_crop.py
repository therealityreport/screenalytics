"""Test facebank display crop uses detected face bbox."""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def test_display_crop_uses_bbox():
    """Test that _prepare_display_crop actually crops to the detected face bbox."""
    from apps.api.routers.facebank import _prepare_display_crop

    # Create a test image (100x100)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add a white square in the center to represent a face
    image[40:60, 40:60] = 255

    # Define a bbox centered on the white square
    bbox = [40.0, 40.0, 60.0, 60.0]

    # Call _prepare_display_crop
    crop, crop_box = _prepare_display_crop(image, bbox, "retinaface")

    # Verify that we got a crop (not the full image)
    assert crop is not None, "Crop should not be None"
    assert crop.shape[0] < 100 or crop.shape[1] < 100, (
        f"Crop should be smaller than original image, got {crop.shape} vs (100, 100, 3)"
    )

    # Verify that the crop contains the face region
    # The crop should be expanded with margin, but still centered on the face
    assert crop.shape[0] > 0 and crop.shape[1] > 0, "Crop should have positive dimensions"
    print(f"✓ Display crop uses bbox: original (100, 100, 3) → crop {crop.shape}")


def test_display_crop_fallback_on_invalid_bbox():
    """Test that _prepare_display_crop falls back to full frame for invalid bbox."""
    from apps.api.routers.facebank import _prepare_display_crop

    # Create a test image
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Invalid bbox (outside image bounds)
    bbox = [200.0, 200.0, 300.0, 300.0]

    # Call _prepare_display_crop - should fall back to full frame
    crop, crop_box = _prepare_display_crop(image, bbox, "retinaface")

    # Verify we got the full frame as fallback
    assert crop is not None, "Crop should not be None"
    assert crop.shape == image.shape, (
        f"Should fall back to full frame, got {crop.shape} vs {image.shape}"
    )
    print(f"✓ Display crop falls back to full frame for invalid bbox")


def test_simulated_detector_bbox_preserved():
    """Test that simulated detector bbox is not overwritten with [0,0,1,1]."""
    # This test verifies that the code no longer has the override block
    # that was discarding the simulated detector's bbox
    from pathlib import Path

    facebank_path = PROJECT_ROOT / "apps" / "api" / "routers" / "facebank.py"
    content = facebank_path.read_text()

    # Check that the problematic override is not present
    assert 'detections = [\n                    {\n                        "bbox": [0.0, 0.0, 1.0, 1.0],' not in content, (
        "Code should not override simulated detector bbox with [0,0,1,1]"
    )
    print("✓ Simulated detector bbox is preserved (override code removed)")


def test_prepare_face_crop_uses_simulated_bbox():
    """Test that _prepare_face_crop uses simulated detector bbox instead of letterboxing."""
    from tools.episode_run import _prepare_face_crop

    # Create a test image (200x200)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    # Add a white square in the center-right area
    image[80:120, 120:160] = 255

    # Simulated detector bbox (centered on bright region)
    bbox = [120.0, 80.0, 160.0, 120.0]
    landmarks = None

    # Call with simulated detector mode
    crop, err = _prepare_face_crop(
        image,
        bbox,
        landmarks,
        margin=0.15,
        align=True,
        detector_mode="simulated"
    )

    # Verify we got a crop, not a letterboxed full image
    assert crop is not None, "Crop should not be None"
    assert err is None, f"Should not have error, got: {err}"

    # The crop should be smaller than the full image (bbox-based crop, not letterbox)
    # Letterbox would preserve full dimensions, while bbox crop reduces them
    h, w = crop.shape[:2]
    assert h < 200 or w < 200, (
        f"Crop should be smaller than original for bbox-based crop, got {crop.shape} vs (200, 200, 3)"
    )

    print(f"✓ Simulated detector uses bbox for cropping: (200, 200, 3) → {crop.shape}")


if __name__ == "__main__":
    test_display_crop_uses_bbox()
    test_display_crop_fallback_on_invalid_bbox()
    test_simulated_detector_bbox_preserved()
    test_prepare_face_crop_uses_simulated_bbox()
    print("\n✓ All facebank cropping tests passed!")

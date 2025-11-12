import numpy as np

from tools import episode_run


def test_prepare_face_crop_letterboxes_full_image_when_simulated():
    """Simulated detector should letterbox the entire upload to preserve the face."""
    height, width = 600, 400
    image = np.full((height, width, 3), 10, dtype=np.uint8)
    image[150:450, 100:300] = [5, 15, 240]
    bbox = [0.0, 0.0, float(width), float(height)]

    crop, err = episode_run._prepare_face_crop(
        image,
        bbox,
        landmarks=None,
        detector_mode="simulated",
    )

    assert err is None
    assert crop is not None
    assert crop.shape == (112, 112, 3)
    # Portrait image ⇒ pillarbox padding on the sides.
    assert np.all(crop[:, 0] == 127)
    assert np.all(crop[:, -1] == 127)
    # The bright region from the center of the upload should survive the resize.
    assert crop[..., 2].max() >= 200

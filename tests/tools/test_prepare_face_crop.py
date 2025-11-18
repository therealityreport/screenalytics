import sys
import types

import numpy as np

from tools import episode_run


def _install_fake_face_align(monkeypatch, recorder):
    module = types.ModuleType("insightface.utils")

    class _FaceAlign:
        def norm_crop(self, image, landmark):
            recorder.append(np.asarray(landmark).copy())
            return np.full((16, 16, 3), 255, dtype=np.uint8)

    module.face_align = _FaceAlign()
    pkg = types.ModuleType("insightface")
    pkg.utils = module
    monkeypatch.setitem(sys.modules, "insightface", pkg)
    monkeypatch.setitem(sys.modules, "insightface.utils", module)


def test_prepare_face_crop_skips_degenerate_landmarks(monkeypatch):
    calls: list[np.ndarray] = []
    _install_fake_face_align(monkeypatch, calls)

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[50:150, 50:150] = [0, 0, 255]
    bbox = [50.0, 50.0, 150.0, 150.0]
    degenerate_landmarks = [100.0, 100.0] * 5

    crop, err = episode_run._prepare_face_crop(image, bbox, degenerate_landmarks, margin=0.0)

    assert err is None
    assert crop is not None
    # Should fall back to bbox crop, so fake face_align is never invoked
    assert calls == []
    # Crop still contains the colored square (not a uniform fill)
    assert crop.std() > 0


def test_prepare_face_crop_uses_alignment_when_points_valid(monkeypatch):
    calls: list[np.ndarray] = []
    _install_fake_face_align(monkeypatch, calls)

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[60:140, 80:160] = [0, 255, 0]
    bbox = [60.0, 60.0, 160.0, 160.0]
    valid_landmarks = [
        80.0,
        80.0,
        140.0,
        80.0,
        110.0,
        110.0,
        90.0,
        150.0,
        130.0,
        150.0,
    ]

    crop, err = episode_run._prepare_face_crop(image, bbox, valid_landmarks, margin=0.0)

    assert err is None
    assert crop is not None
    # With valid landmarks the fake aligner should be called once
    assert len(calls) == 1
    # The fake aligner returns a solid white image
    assert np.all(crop == 255)

import sys
import types

import numpy as np

from tools import episode_run


def _install_fake_face_align(monkeypatch, recorder):
    module = types.ModuleType("insightface.utils")

    class _FaceAlign:
        def norm_crop(self, image, landmark):
            recorder.append(np.asarray(landmark).copy())
            return np.full((24, 24, 3), 255, dtype=np.uint8)

    module.face_align = _FaceAlign()
    pkg = types.ModuleType("insightface")
    pkg.utils = module
    monkeypatch.setitem(sys.modules, "insightface", pkg)
    monkeypatch.setitem(sys.modules, "insightface.utils", module)


def test_prepare_face_crop_aligns_when_retinaface_mode(monkeypatch):
    calls: list[np.ndarray] = []
    _install_fake_face_align(monkeypatch, calls)

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[70:130, 90:150] = [0, 200, 0]
    bbox = [60.0, 60.0, 160.0, 160.0]
    landmarks = [
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

    crop, err = episode_run._prepare_face_crop(
        image,
        bbox,
        landmarks,
        margin=0.0,
        detector_mode="retinaface",
    )

    assert err is None
    assert crop is not None
    assert len(calls) == 1
    assert np.all(crop == 255)

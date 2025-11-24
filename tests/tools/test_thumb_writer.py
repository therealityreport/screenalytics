from __future__ import annotations

from pathlib import Path

import numpy as np

from py_screenalytics.artifacts import ensure_dirs
from tools.episode_run import ThumbWriter, _prepare_face_crop, safe_crop


def test_thumb_writer_prefers_prepared_crop(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    ep_id = "demo-s02e01"
    ensure_dirs(ep_id)

    def _fake_imwrite(path, image, jpg_q=85):
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"x" * 2048)
        return True, None

    monkeypatch.setattr("tools.episode_run.safe_imwrite", _fake_imwrite)

    writer = ThumbWriter(ep_id, size=256, jpeg_quality=90)
    rng = np.random.default_rng(42)
    image = (rng.random((200, 200, 3)) * 255).astype(np.uint8)
    image[60:150, 40:160] = 180  # bright face region
    bbox = [50.0, 70.0, 90.0, 120.0]  # tight crop that misses the chin
    raw_crop, _, _ = safe_crop(image, bbox)

    prepared_crop, err = _prepare_face_crop(
        image,
        bbox,
        landmarks=None,
        margin=0.35,
        align=False,
        detector_mode="retinaface",
        adaptive_margin=False,
    )
    assert err is None
    rel_path, abs_path = writer.write(
        image,
        bbox,
        track_id=7,
        frame_idx=15,
        prepared_crop=prepared_crop,
    )
    assert rel_path
    assert abs_path and abs_path.exists()
    meta = writer._last_thumb_meta
    assert meta.get("source_kind") == "prepared"
    source_shape = meta.get("source_shape")
    assert source_shape is not None
    assert source_shape[0] > raw_crop.shape[0]
    assert source_shape[1] > raw_crop.shape[1]

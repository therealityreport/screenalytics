from __future__ import annotations

import pytest

try:
    from tools import episode_run
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency chain
    episode_run = None
    _EPISODE_RUN_IMPORT_ERROR = exc
else:
    _EPISODE_RUN_IMPORT_ERROR = None


def _require_episode_run():
    if episode_run is None:  # pragma: no cover - skip when deps missing
        pytest.skip(f"episode_run import failed: {_EPISODE_RUN_IMPORT_ERROR}")


def test_sanitize_xyxy_clamps_and_rounds():
    _require_episode_run()
    # Negative + overflow coordinates should clamp into the frame bounds.
    box = episode_run.sanitize_xyxy(-5.4, -1.2, 205.7, 99.9, 200, 100)
    assert box == (0, 0, 200, 100)


def test_sanitize_xyxy_rejects_zero_area():
    _require_episode_run()
    assert episode_run.sanitize_xyxy(10.1, 10.4, 10.2, 10.6, 50, 50) is None


def test_save_jpeg_roundtrip_not_constant(tmp_path):
    _require_episode_run()
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    img = numpy.linspace(0.0, 1.0, 32 * 32 * 3, dtype=numpy.float32).reshape(32, 32, 3)
    out_path = tmp_path / "sample.jpg"
    episode_run.save_jpeg(out_path, img, color="rgb", quality=90)
    assert out_path.exists()
    loaded = cv2.imread(str(out_path))
    assert loaded is not None
    assert loaded.max() - loaded.min() > 0

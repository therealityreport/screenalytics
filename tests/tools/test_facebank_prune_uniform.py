import numpy as np
from PIL import Image

from apps.api.services.facebank import FacebankService
from tools import facebank_prune_uniform as prune


def _write_image(path, *, value: int, gradient: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gradient:
        arr = np.linspace(value, value + 40, num=32, dtype=np.float32)
        grid = np.tile(arr[:, None], (1, 32))
        img = np.stack([grid, grid[::-1], grid], axis=2).astype(np.uint8)
    else:
        img = np.full((32, 32, 3), value, dtype=np.uint8)
    Image.fromarray(img).save(path)


def test_prune_facebank_seeds_flags_and_removes(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    fs = FacebankService(data_root)
    show_id = "RHOBH"
    cast_id = "test-cast"

    uniform_path = data_root / "facebank" / show_id / cast_id / "uniform.jpg"
    varied_path = data_root / "facebank" / show_id / cast_id / "varied.jpg"
    _write_image(uniform_path, value=180, gradient=False)
    _write_image(varied_path, value=100, gradient=True)

    fs.add_seed(show_id, cast_id, str(uniform_path), np.zeros(512, dtype=np.float32))
    fs.add_seed(show_id, cast_id, str(varied_path), np.zeros(512, dtype=np.float32))

    stats = prune.prune_facebank_seeds(
        show_id, cast_id=cast_id, threshold=5.0, delete=False
    )
    assert stats["inspected"] == 2
    assert stats["flagged"] == 1
    assert stats["removed"] == 0

    stats = prune.prune_facebank_seeds(
        show_id, cast_id=cast_id, threshold=5.0, delete=True
    )
    assert stats["removed"] == 1

    refreshed = fs._load_facebank(show_id, cast_id)
    assert len(refreshed["seeds"]) == 1
    remaining_uri = refreshed["seeds"][0]["image_uri"]
    assert remaining_uri.endswith("varied.jpg")

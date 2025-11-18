import numpy as np

from apps.api.services.facebank import FacebankService
from tools.facebank_sync_seeds import backfill_facebank_seeds


class _DummyStorage:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str, str]] = []

    def upload_facebank_seed(self, show_id, cast_id, seed_id, local_path):
        key = f"artifacts/facebank/{show_id}/{cast_id}/{seed_id}.jpg"
        self.uploads.append((show_id, cast_id, seed_id, str(local_path)))
        return key


def test_backfill_updates_missing_s3(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    fs = FacebankService(data_root)
    show_id = "demo"
    cast_id = "cast-1"
    seeds_dir = fs._seeds_dir(show_id, cast_id)
    seeds_dir.mkdir(parents=True, exist_ok=True)
    image_path = seeds_dir / "seed.jpg"
    image_path.write_bytes(b"seed")
    fs.add_seed(show_id, cast_id, str(image_path), np.zeros(512, dtype=np.float32))

    dummy_storage = _DummyStorage()
    monkeypatch.setattr("tools.facebank_sync_seeds.StorageService", lambda: dummy_storage)

    stats = backfill_facebank_seeds(show_id)
    assert stats["updated"] == 1
    updated = fs._load_facebank(show_id, cast_id)
    assert updated["seeds"][0]["image_s3_key"].startswith("artifacts/facebank")
    assert updated["seeds"][0]["display_s3_key"].startswith("artifacts/facebank")


def test_backfill_dry_run(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    fs = FacebankService(data_root)
    show_id = "demo"
    cast_id = "cast-2"
    seeds_dir = fs._seeds_dir(show_id, cast_id)
    seeds_dir.mkdir(parents=True, exist_ok=True)
    image_path = seeds_dir / "seed2.jpg"
    image_path.write_bytes(b"seed")
    fs.add_seed(show_id, cast_id, str(image_path), np.zeros(512, dtype=np.float32))

    monkeypatch.setattr("tools.facebank_sync_seeds.StorageService", lambda: None)

    stats = backfill_facebank_seeds(show_id, dry_run=True)
    assert stats["updated"] == 1
    data = fs._load_facebank(show_id, cast_id)
    assert not data["seeds"][0].get("image_s3_key")

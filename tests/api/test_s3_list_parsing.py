from __future__ import annotations

from apps.api.services.storage import StorageService


def test_parse_v2_key_metadata():
    storage = StorageService.__new__(StorageService)
    storage.prefix = "raw/"
    metadata = storage._parse_s3_key_metadata("raw/videos/showA/s03/e04/episode.mp4")
    assert metadata["ep_id"] == "showA-s03e04"
    assert metadata["show"] == "showA"
    assert metadata["season"] == 3
    assert metadata["episode"] == 4
    assert metadata["key_version"] == "v2"

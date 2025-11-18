from pathlib import Path

from apps.api.services.storage import EpisodeContext, StorageService


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def upload_file(self, filename, bucket, key, ExtraArgs=None):  # noqa: ANN001
        self.calls.append(
            {
                "filename": filename,
                "bucket": bucket,
                "key": key,
                "extra": ExtraArgs or {},
            }
        )


def _make_storage(fake_client) -> StorageService:
    svc: StorageService = StorageService.__new__(StorageService)  # type: ignore[call-arg]
    svc.backend = "s3"
    svc._client = fake_client
    svc.bucket = "bucket"
    svc.write_enabled = True
    return svc


def test_put_artifact_sets_headers(tmp_path):
    client = _FakeClient()
    storage = _make_storage(client)
    ctx = EpisodeContext(
        ep_id="demo-s01e01", show_slug="demo", season_number=1, episode_number=1
    )
    local_path = tmp_path / "identities.json"
    local_path.write_text("{}", encoding="utf-8")

    uploaded = storage.put_artifact(ctx, "manifests", local_path, local_path.name)
    assert uploaded is True
    assert client.calls, "expected upload_file to be invoked"
    extra = client.calls[0]["extra"]
    assert extra["ContentType"] == "application/json"
    assert extra["CacheControl"].startswith("max-age=")


def test_upload_dir_applies_cache_headers(tmp_path):
    client = _FakeClient()
    storage = _make_storage(client)
    storage._client = client
    root = tmp_path / "frames"
    (root / "nested").mkdir(parents=True)
    sample = root / "nested" / "frame_000001.jpg"
    sample.write_bytes(b"fake")

    uploaded = storage.upload_dir(root, "artifacts/frames/demo", guess_mime=True)
    assert uploaded == 1
    extra = client.calls[0]["extra"]
    assert extra["ContentType"] == "image/jpeg"
    assert extra["CacheControl"].startswith("max-age=")

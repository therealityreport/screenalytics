
import pytest

from apps.api.services import storage as storage_module
from apps.api.services.storage import StorageService


class _DummyS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict] = {}
        self.presigns: list[tuple[dict, int]] = []

    def put_object(self, **kwargs) -> None:  # noqa: N802
        bucket = kwargs.get("Bucket")
        key = kwargs.get("Key")
        body = kwargs.get("Body", b"")
        if isinstance(body, bytes):
            data = body
        else:
            data = bytes(body)
        self.objects[(bucket, key)] = {
            "body": data,
            "cache_control": kwargs.get("CacheControl"),
            "content_type": kwargs.get("ContentType"),
        }

    def head_object(self, Bucket, Key):  # noqa: N802
        entry = self.objects[(Bucket, Key)]
        return {
            "ContentLength": len(entry["body"]),
            "ContentType": entry["content_type"],
        }

    def generate_presigned_url(
        self, *_args, Params=None, ExpiresIn=None, **_kwargs
    ):  # noqa: N802
        self.presigns.append((Params or {}, ExpiresIn or 0))
        bucket = (Params or {}).get("Bucket", "bucket")
        key = (Params or {}).get("Key", "key")
        return f"https://dummy/{bucket}/{key}"


class _DummyBoto3:
    def __init__(self, client: _DummyS3Client) -> None:
        self._client = client

    def client(self, *_args, **_kwargs):  # noqa: D401, N802
        return self._client


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    monkeypatch.delenv("FACEBANK_S3_BUCKET", raising=False)
    monkeypatch.delenv("FACEBANK_S3_ENDPOINT", raising=False)
    monkeypatch.delenv("FACEBANK_S3_REGION", raising=False)
    monkeypatch.delenv("FACEBANK_S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("FACEBANK_S3_SECRET_KEY", raising=False)
    monkeypatch.delenv("FACEBANK_S3_SIGNATURE", raising=False)
    monkeypatch.delenv("FACEBANK_S3_WRITE", raising=False)


def test_facebank_uploads_use_dedicated_s3_when_backend_local(monkeypatch, tmp_path):
    dummy_client = _DummyS3Client()
    monkeypatch.setattr(storage_module, "_boto3", lambda: _DummyBoto3(dummy_client))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("FACEBANK_S3_BUCKET", "facebank-dedicated")
    monkeypatch.setenv("FACEBANK_S3_SIGNATURE", "")

    storage = StorageService()
    file_path = tmp_path / "seed.jpg"
    file_path.write_bytes(b"face")

    key = storage.upload_facebank_seed("RHOBH", "cast-123", "seed-456", file_path)
    assert key == "artifacts/facebank/RHOBH/cast-123/seed-456.jpg"
    stored = dummy_client.objects[("facebank-dedicated", key)]
    assert stored["content_type"] == "image/jpeg"
    assert stored["cache_control"] == "max-age=31536000,public"

    presigned = storage.presign_get(key)
    assert presigned == f"https://dummy/facebank-dedicated/{key}"
    assert dummy_client.presigns

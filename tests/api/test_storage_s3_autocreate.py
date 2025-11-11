from botocore.exceptions import ClientError

from apps.api.services.storage import StorageService


class FakeSTS:
    def get_caller_identity(self):
        return {"Account": "123456789012"}


class FakeS3Client:
    def __init__(self, exists: bool = True):
        self.exists = exists
        self.created = False

    def head_bucket(self, Bucket):
        if not self.exists:
            raise ClientError({"Error": {"Code": "404"}}, "HeadBucket")

    def create_bucket(self, **kwargs):
        self.created = True


class FakeBoto3:
    def __init__(self, s3_client: FakeS3Client):
        self._s3 = s3_client

    def client(self, name, **_):
        if name == "s3":
            return self._s3
        if name == "sts":
            return FakeSTS()
        raise ValueError(name)


def _setup_env(monkeypatch, auto_create: str):
    monkeypatch.setenv("STORAGE_BACKEND", "s3")
    monkeypatch.setenv("SCREENALYTICS_ENV", "dev")
    monkeypatch.setenv("AWS_S3_PREFIX", "raw/")
    monkeypatch.setenv("S3_AUTO_CREATE", auto_create)
    monkeypatch.delenv("AWS_S3_BUCKET", raising=False)


def test_s3_auto_creates_bucket(monkeypatch):
    _setup_env(monkeypatch, "1")
    fake_client = FakeS3Client(exists=False)
    monkeypatch.setattr("apps.api.services.storage._boto3", lambda: FakeBoto3(fake_client))
    StorageService()
    assert fake_client.created is True


def test_s3_missing_bucket_raises(monkeypatch):
    _setup_env(monkeypatch, "0")
    fake_client = FakeS3Client(exists=False)
    monkeypatch.setattr("apps.api.services.storage._boto3", lambda: FakeBoto3(fake_client))
    try:
        StorageService()
    except RuntimeError as exc:
        assert "Bucket" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when auto create disabled")

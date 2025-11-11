from pathlib import Path

from apps.api.services.storage import StorageService
from py_screenalytics.artifacts import get_path


class _FakeClient:
    def __init__(self) -> None:
        self.download_calls: list[tuple[str, str, str]] = []

    def head_object(self, Bucket, Key):  # noqa: N802 (boto shape)
        return {"ETag": '"fake-etag"', "ContentLength": 7}

    def download_file(self, Bucket, Key, Filename):  # noqa: N802
        self.download_calls.append((Bucket, Key, Filename))
        path = Path(Filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mirror!")


def test_ensure_local_mirror_downloads_from_s3(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    service = StorageService.__new__(StorageService)
    service.backend = "s3"
    service.region = "us-east-1"
    service.prefix = "raw/"
    service.auto_create = False
    service.bucket = "screenalytics"
    service._client = _FakeClient()
    service._client_error_cls = Exception

    ep_id = "demo-episode"
    local_path = get_path(ep_id, "video")
    assert not local_path.exists()

    result = service.ensure_local_mirror(ep_id)
    assert result["local_video_path"].endswith("episode.mp4")
    assert result["bytes"] == len(Path(result["local_video_path"]).read_bytes())
    assert Path(result["local_video_path"]).exists()
    assert service._client.download_calls  # type: ignore[attr-defined]

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import files as files_router


class _FakeStorage:
    def __init__(self, url: str | None = "https://cdn.local") -> None:
        self.calls: list[tuple[str, int, str | None]] = []
        self.url = url

    def presign_get(self, key: str, *, expires_in: int = 3600, content_type: str | None = None) -> str | None:
        self.calls.append((key, expires_in, content_type))
        if not self.url:
            return None
        return f"{self.url}/{key}"


def test_presign_infers_png_content_type(monkeypatch):
    fake_storage = _FakeStorage()
    monkeypatch.setattr(files_router, "storage_service", fake_storage)
    client = TestClient(app)

    key = "artifacts/facebank/rhobh/cast123/seed_d.png"
    resp = client.get("/files/presign", params={"key": key})
    assert resp.status_code == 200
    data = resp.json()
    assert data["content_type"] == "image/png"
    assert fake_storage.calls == [(key, 3600, "image/png")]


def test_presign_infers_jpeg_when_mime_missing(monkeypatch):
    fake_storage = _FakeStorage()
    monkeypatch.setattr(files_router, "storage_service", fake_storage)
    client = TestClient(app)

    key = "artifacts/facebank/rhobh/cast123/seed_d.jpg"
    resp = client.get("/files/presign", params={"key": key})
    assert resp.status_code == 200
    data = resp.json()
    assert data["content_type"] == "image/jpeg"
    assert fake_storage.calls == [(key, 3600, "image/jpeg")]


def test_presign_respects_explicit_mime_override(monkeypatch):
    fake_storage = _FakeStorage()
    monkeypatch.setattr(files_router, "storage_service", fake_storage)
    client = TestClient(app)

    key = "artifacts/facebank/rhobh/cast123/seed_d.png"
    resp = client.get("/files/presign", params={"key": key, "mime": "image/webp"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["content_type"] == "image/webp"
    assert fake_storage.calls == [(key, 3600, "image/webp")]


def test_presign_reports_unavailable_when_storage_missing(monkeypatch):
    fake_storage = _FakeStorage(url=None)
    monkeypatch.setattr(files_router, "storage_service", fake_storage)
    client = TestClient(app)

    resp = client.get("/files/presign", params={"key": "artifacts/foo/bar.jpg"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["error"] == "presign_unavailable"
    assert payload["url"] is None

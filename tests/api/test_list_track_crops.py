import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import episodes as episodes_router
from apps.api.services.storage import StorageService
from py_screenalytics import run_layout


def _setup_env(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "s3")
    monkeypatch.delenv("SCREENALYTICS_CROPS_FALLBACK_ROOT", raising=False)


class _FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class _FakeClient:
    def __init__(self, payload: list[dict]) -> None:
        self._payload = payload

    def get_object(self, Bucket, Key):  # noqa: ANN001
        return {"Body": _FakeBody(json.dumps(self._payload).encode("utf-8"))}


def _make_storage(entries: list[dict]) -> StorageService:
    storage: StorageService = StorageService.__new__(StorageService)  # type: ignore[call-arg]
    storage.backend = "s3"
    storage.bucket = "demo-bucket"
    storage._client = _FakeClient(entries)
    storage._client_error_cls = Exception
    storage.init_error = None
    storage.presign_get = lambda key, expires_in=3600: f"https://example.com/{key}"
    return storage


def _write_faces(ep_id: str, run_id: str, count: int = 3, track_id: int = 1) -> None:
    faces_path = run_layout.run_root(ep_id, run_id) / "faces.jsonl"
    entries = []
    for idx in range(count):
        crop_key = f"artifacts/crops/{ep_id}/track_{track_id:04d}/frame_{idx:06d}.jpg"
        entries.append(
            {
                "ep_id": ep_id,
                "face_id": f"face_{track_id:04d}_{idx:06d}",
                "track_id": track_id,
                "frame_idx": idx,
                "ts": idx * 0.1,
                "crop_s3_key": crop_key,
                "thumb_s3_key": crop_key.replace("frame_", "thumb_"),
            }
        )
    faces_path.parent.mkdir(parents=True, exist_ok=True)
    with faces_path.open("w", encoding="utf-8") as handle:
        for row in entries:
            handle.write(json.dumps(row) + "\n")


@pytest.mark.parametrize("sample", [1, 2])
def test_list_track_crops_remote_pagination(tmp_path, monkeypatch, sample):
    _setup_env(tmp_path, monkeypatch)
    ep_id = "demo-s01e01"
    run_id = "run-crops-1"
    entries = [
        {"key": f"track_0001/frame_{idx:06d}.jpg", "frame_idx": idx, "ts": idx * 0.5}
        for idx in range(4)
    ]
    monkeypatch.setattr(episodes_router, "STORAGE", _make_storage(entries))
    client = TestClient(app)

    resp = client.get(
        f"/episodes/{ep_id}/tracks/1/crops",
        params={"sample": sample, "limit": 1, "run_id": run_id},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert "items" in payload
    assert len(payload["items"]) == 1
    first = payload["items"][0]
    assert first["key"].startswith("track_0001/frame_")
    assert first["url"].endswith(".jpg")
    cursor = payload.get("next_start_after")
    assert cursor

    resp2 = client.get(
        f"/episodes/{ep_id}/tracks/1/crops",
        params={"sample": sample, "limit": 2, "start_after": cursor, "run_id": run_id},
    )
    assert resp2.status_code == 200
    payload2 = resp2.json()
    assert len(payload2["items"]) >= 1
    # ensure returned cursor encodes sampling state when sample > 1
    next_cursor = payload2.get("next_start_after")
    if sample > 1 and next_cursor:
        assert "|" in next_cursor


def test_list_track_crops_sampling_respects_cursor(tmp_path, monkeypatch):
    _setup_env(tmp_path, monkeypatch)
    ep_id = "demo-s01e01"
    run_id = "run-crops-2"
    entries = [
        {"key": f"track_0001/frame_{idx:06d}.jpg", "frame_idx": idx, "ts": idx * 0.5}
        for idx in range(5)
    ]
    monkeypatch.setattr(episodes_router, "STORAGE", _make_storage(entries))
    client = TestClient(app)

    resp = client.get(
        f"/episodes/{ep_id}/tracks/1/crops",
        params={"sample": 2, "limit": 1, "run_id": run_id},
    )
    payload = resp.json()
    cursor = payload["next_start_after"]
    assert cursor.endswith("|1")

    resp2 = client.get(
        f"/episodes/{ep_id}/tracks/1/crops",
        params={"sample": 2, "limit": 1, "start_after": cursor, "run_id": run_id},
    )
    payload2 = resp2.json()
    assert payload2["items"], "expected another sampled crop"
    second_frame = payload2["items"][0]["frame_idx"]
    assert second_frame == payload["items"][0]["frame_idx"] + 2


def test_list_track_crops_remote_uses_presigned(monkeypatch, tmp_path):
    _setup_env(tmp_path, monkeypatch)
    ep_id = "demo-s01e01"
    run_id = "run-remote-1"
    entries = [
        {"key": "track_0001/frame_000000.jpg", "frame_idx": 0, "ts": 0.0},
        {"key": "track_0001/frame_000010.jpg", "frame_idx": 10, "ts": 0.5},
    ]
    monkeypatch.setattr(episodes_router, "STORAGE", _make_storage(entries))

    client = TestClient(app)
    resp = client.get(
        f"/episodes/{ep_id}/tracks/1/crops",
        params={"sample": 1, "limit": 2, "run_id": run_id},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload["items"]) == 2
    assert payload["items"][0]["url"].startswith("https://example.com/")


def test_list_track_crops_rejects_local_fallback_root(tmp_path, monkeypatch):
    data_root = tmp_path / "primary"
    legacy_root = tmp_path / "legacy"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("SCREENALYTICS_CROPS_FALLBACK_ROOT", str(legacy_root))
    monkeypatch.setattr(episodes_router, "STORAGE", StorageService())
    ep_id = "legacy-s01e01"
    run_id = "run-legacy-1"
    client = TestClient(app)
    resp = client.get(
        f"/episodes/{ep_id}/tracks/1/crops",
        params={"sample": 1, "limit": 5, "run_id": run_id},
    )
    assert resp.status_code == 503
    assert "S3 misconfigured" in resp.json().get("message", "")


def test_track_frames_endpoint_returns_media(tmp_path, monkeypatch):
    _setup_env(tmp_path, monkeypatch)
    ep_id = "demo-s01e01"
    run_id = "run-crops-frames"
    run_layout.run_root(ep_id, run_id).mkdir(parents=True, exist_ok=True)
    _write_faces(ep_id, run_id, count=4)
    monkeypatch.setattr(episodes_router, "STORAGE", _make_storage([]))
    client = TestClient(app)
    resp = client.get(
        f"/episodes/{ep_id}/tracks/1/frames",
        params={"sample": 1, "page": 1, "page_size": 2, "run_id": run_id},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["total"] == 4
    assert len(payload["items"]) == 2
    assert payload["items"][0]["frame_idx"] == 0
    first_media = payload["items"][0]["media_url"]
    assert first_media and first_media.endswith(".jpg")


def test_track_frames_endpoint_falls_back_when_no_crops(tmp_path, monkeypatch):
    _setup_env(tmp_path, monkeypatch)
    ep_id = "nogrid-s01e01"
    run_id = "run-frames-1"
    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    faces_path = run_root / "faces.jsonl"
    faces = [
        {"ep_id": ep_id, "face_id": "face_0001_000000", "track_id": 1, "frame_idx": 0, "ts": 0.0},
        {"ep_id": ep_id, "face_id": "face_0001_000001", "track_id": 1, "frame_idx": 1, "ts": 0.1},
    ]
    faces_path.write_text("\n".join(json.dumps(row) for row in faces) + "\n", encoding="utf-8")
    monkeypatch.setattr(episodes_router, "STORAGE", _make_storage([]))
    client = TestClient(app)
    resp = client.get(
        f"/episodes/{ep_id}/tracks/1/frames",
        params={"sample": 1, "page": 1, "page_size": 5, "run_id": run_id},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["total"] == 2
    assert len(payload["items"]) == 2


def test_track_integrity_counts_faces_and_crops(tmp_path, monkeypatch):
    _setup_env(tmp_path, monkeypatch)
    ep_id = "demo-s01e01"
    run_id = "run-integrity-1"
    run_layout.run_root(ep_id, run_id).mkdir(parents=True, exist_ok=True)
    _write_faces(ep_id, run_id, count=3)
    entries = [
        {"key": f"track_0001/frame_{idx:06d}.jpg", "frame_idx": idx, "ts": idx * 0.5}
        for idx in range(3)
    ]
    monkeypatch.setattr(episodes_router, "STORAGE", _make_storage(entries))
    client = TestClient(app)
    resp = client.get(f"/episodes/{ep_id}/tracks/1/integrity", params={"run_id": run_id})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["faces_manifest"] == 3
    assert payload["crops_files"] == 3
    assert payload["ok"]

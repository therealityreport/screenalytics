import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.services.storage import StorageService
from py_screenalytics import run_layout
from py_screenalytics.artifacts import ensure_dirs, get_path


def _setup_track(tmp_path: Path, count: int = 3) -> tuple[str, str]:
    data_root = tmp_path / "data"
    os.environ["SCREENALYTICS_DATA_ROOT"] = str(data_root)
    os.environ["STORAGE_BACKEND"] = "local"
    os.environ.pop("SCREENALYTICS_CROPS_FALLBACK_ROOT", None)
    ep_id = "demo-s01e01"
    run_id = "run-crops-1"
    ensure_dirs(ep_id)
    run_layout.run_root(ep_id, run_id).mkdir(parents=True, exist_ok=True)
    track_dir = get_path(ep_id, "frames_root") / "crops" / "track_0001"
    track_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for idx in range(count):
        frame_path = track_dir / f"frame_{idx:06d}.jpg"
        frame_path.write_bytes(b"test")
        entries.append(
            {
                "key": f"track_0001/frame_{idx:06d}.jpg",
                "frame_idx": idx,
                "ts": idx * 0.5,
            }
        )
    (track_dir / "index.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return ep_id, run_id


def _setup_legacy_track(tmp_path: Path, count: int = 2) -> tuple[str, str]:
    data_root = tmp_path / "primary"
    legacy_root = tmp_path / "legacy"
    os.environ["SCREENALYTICS_DATA_ROOT"] = str(data_root)
    os.environ["STORAGE_BACKEND"] = "local"
    os.environ["SCREENALYTICS_CROPS_FALLBACK_ROOT"] = str(legacy_root)
    ep_id = "legacy-s01e01"
    run_id = "run-legacy-1"
    ensure_dirs(ep_id)
    run_layout.run_root(ep_id, run_id).mkdir(parents=True, exist_ok=True)
    track_dir = legacy_root / ep_id / "tracks" / "track_0001"
    track_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for idx in range(count):
        frame_path = track_dir / f"frame_{idx:06d}.jpg"
        frame_path.write_bytes(b"legacy")
        entries.append(
            {
                "key": f"track_0001/frame_{idx:06d}.jpg",
                "frame_idx": idx,
                "ts": idx * 0.25,
            }
        )
    (track_dir / "index.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return ep_id, run_id


def _write_faces(ep_id: str, run_id: str, count: int = 3, track_id: int = 1) -> None:
    faces_path = run_layout.run_root(ep_id, run_id) / "faces.jsonl"
    entries = []
    for idx in range(count):
        entries.append(
            {
                "ep_id": ep_id,
                "face_id": f"face_{track_id:04d}_{idx:06d}",
                "track_id": track_id,
                "frame_idx": idx,
                "ts": idx * 0.1,
                "crop_rel_path": f"crops/track_{track_id:04d}/frame_{idx:06d}.jpg",
                "thumb_rel_path": f"track_{track_id:04d}/thumb_{idx:06d}.jpg",
            }
        )
    faces_path.parent.mkdir(parents=True, exist_ok=True)
    with faces_path.open("w", encoding="utf-8") as handle:
        for row in entries:
            handle.write(json.dumps(row) + "\n")


@pytest.mark.parametrize("sample", [1, 2])
def test_list_track_crops_local_pagination(tmp_path, sample):
    ep_id, run_id = _setup_track(tmp_path, count=4)
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


def test_list_track_crops_sampling_respects_cursor(tmp_path):
    ep_id, run_id = _setup_track(tmp_path, count=5)
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
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    ep_id = "demo-s01e01"
    run_id = "run-remote-1"
    entries = [
        {"key": "track_0001/frame_000000.jpg", "frame_idx": 0, "ts": 0.0},
        {"key": "track_0001/frame_000010.jpg", "frame_idx": 10, "ts": 0.5},
    ]

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

    from apps.api.routers import episodes as episodes_router

    storage = StorageService()
    storage.backend = "s3"
    storage.bucket = "demo-bucket"
    storage._client = _FakeClient(entries)
    storage._client_error_cls = Exception
    monkeypatch.setattr(
        storage,
        "presign_get",
        lambda key, expires_in=3600: f"https://example.com/{key}",
    )
    monkeypatch.setattr(episodes_router, "STORAGE", storage)

    client = TestClient(app)
    resp = client.get(
        f"/episodes/{ep_id}/tracks/1/crops",
        params={"sample": 1, "limit": 2, "run_id": run_id},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload["items"]) == 2
    assert payload["items"][0]["url"].startswith("https://example.com/")


def test_list_track_crops_uses_fallback_root(tmp_path):
    ep_id, run_id = _setup_legacy_track(tmp_path, count=2)
    client = TestClient(app)
    resp = client.get(
        f"/episodes/{ep_id}/tracks/1/crops",
        params={"sample": 1, "limit": 5, "run_id": run_id},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["items"], "expected crops from fallback root"
    for item in payload["items"]:
        assert item["url"].endswith(".jpg")


def test_track_frames_endpoint_returns_media(tmp_path):
    ep_id, run_id = _setup_track(tmp_path, count=4)
    _write_faces(ep_id, run_id, count=4)
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


def test_track_frames_endpoint_falls_back_when_no_crops(tmp_path):
    data_root = tmp_path / "data"
    os.environ["SCREENALYTICS_DATA_ROOT"] = str(data_root)
    ep_id = "nogrid-s01e01"
    run_id = "run-frames-1"
    ensure_dirs(ep_id)
    run_layout.run_root(ep_id, run_id).mkdir(parents=True, exist_ok=True)
    _write_faces(ep_id, run_id, count=2)
    client = TestClient(app)
    resp = client.get(
        f"/episodes/{ep_id}/tracks/1/frames",
        params={"sample": 1, "page": 1, "page_size": 5, "run_id": run_id},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["total"] == 2
    assert len(payload["items"]) == 2


def test_track_integrity_counts_faces_and_crops(tmp_path):
    ep_id, run_id = _setup_track(tmp_path, count=3)
    _write_faces(ep_id, run_id, count=3)
    client = TestClient(app)
    resp = client.get(f"/episodes/{ep_id}/tracks/1/integrity", params={"run_id": run_id})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["faces_manifest"] == 3
    assert payload["crops_files"] == 3
    assert payload["ok"]

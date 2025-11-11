import os
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import episodes as episodes_router
from apps.api.services.episodes import EpisodeStore
from apps.api.services.storage import StorageService
from py_screenalytics.artifacts import ensure_dirs, get_path


def _create_episode_dirs(ep_id: str) -> None:
    ensure_dirs(ep_id)
    video_dir = get_path(ep_id, "video").parent
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "episode.mp4").write_bytes(b"vid")
    frames_dir = get_path(ep_id, "frames_root")
    (frames_dir / "crops").mkdir(parents=True, exist_ok=True)
    manifests_dir = get_path(ep_id, "detections").parent
    manifests_dir.mkdir(parents=True, exist_ok=True)
    analytics_dir = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")) / "analytics" / ep_id
    analytics_dir.mkdir(parents=True, exist_ok=True)


def _prepare_store(tmp_path: Path, monkeypatch, count: int = 2) -> list[str]:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    episodes_router.EPISODE_STORE = EpisodeStore()
    episodes_router.STORAGE = StorageService()
    store = episodes_router.EPISODE_STORE
    ep_ids: list[str] = []
    for idx in range(count):
        record = store.upsert(show_ref=f"demo{idx}", season_number=1, episode_number=idx + 1)
        ep_ids.append(record.ep_id)
        _create_episode_dirs(record.ep_id)
    return ep_ids


def test_delete_episode_removes_local_data(tmp_path, monkeypatch):
    ep_ids = _prepare_store(tmp_path, monkeypatch, count=1)
    ep_id = ep_ids[0]
    client = TestClient(app)

    resp = client.request(
        "DELETE",
        f"/episodes/{ep_id}",
        json={"delete_local": True, "delete_artifacts": False, "delete_raw": False},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ep_id"] == ep_id
    assert payload["deleted"]["local_dirs"] >= 1

    assert not get_path(ep_id, "video").parent.exists()
    assert not get_path(ep_id, "frames_root").exists()
    assert not get_path(ep_id, "detections").parent.exists()

    store = EpisodeStore()
    assert not store.exists(ep_id)


def test_purge_all_invokes_s3_prefix_cleanup(tmp_path, monkeypatch):
    ep_ids = _prepare_store(tmp_path, monkeypatch, count=2)
    deleted_prefixes: list[str] = []

    def fake_delete(bucket: str, prefix: str, storage=None) -> int:  # noqa: ANN001
        deleted_prefixes.append(prefix)
        return 5

    monkeypatch.setattr("apps.api.routers.episodes.delete_s3_prefix", fake_delete)
    client = TestClient(app)

    resp = client.post(
        "/episodes/purge_all",
        json={
            "confirm": "DELETE ALL",
            "delete_local": True,
            "delete_artifacts": True,
            "delete_raw": True,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] == len(ep_ids)
    assert payload["deleted"]["s3_objects"] == 5 * len(deleted_prefixes)
    assert deleted_prefixes, "expected prefixes to be scheduled for deletion"
    assert len(deleted_prefixes) >= len(ep_ids) * 4

    for ep_id in ep_ids:
        assert not get_path(ep_id, "frames_root").exists()

    store = EpisodeStore()
    assert store.list() == []

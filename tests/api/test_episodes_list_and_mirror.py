from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import episodes as episodes_router
from apps.api.services.episodes import EpisodeStore
from apps.api.services.storage import StorageService
from py_screenalytics.artifacts import get_path


def _reset_services(monkeypatch) -> None:
    monkeypatch.setattr(episodes_router, "EPISODE_STORE", EpisodeStore())
    monkeypatch.setattr(episodes_router, "STORAGE", StorageService())


def _create_episode(
    client: TestClient, show_ref: str, season: int, episode: int
) -> str:
    resp = client.post(
        "/episodes",
        json={
            "show_slug_or_id": show_ref,
            "season_number": season,
            "episode_number": episode,
        },
    )
    assert resp.status_code == 200
    return resp.json()["ep_id"]


def test_episode_list_detail_and_mirror_local(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    _reset_services(monkeypatch)

    client = TestClient(app)
    ep_id = _create_episode(client, "rhoslc", 5, 7)

    listing = client.get("/episodes")
    assert listing.status_code == 200
    episodes = listing.json()["episodes"]
    assert any(item["ep_id"] == ep_id for item in episodes)

    detail = client.get(f"/episodes/{ep_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["ep_id"] == ep_id
    assert payload["s3"]["bucket"] == "local"
    assert payload["s3"]["v2_key"].endswith("episode.mp4")
    assert payload["s3"]["v2_exists"] is False
    assert payload["s3"]["v1_key"].endswith("episode.mp4")
    assert payload["local"]["exists"] is False

    local_video = get_path(ep_id, "video")
    local_video.parent.mkdir(parents=True, exist_ok=True)
    sample_bytes = b"episode-bytes"
    local_video.write_bytes(sample_bytes)

    mirror_resp = client.post(f"/episodes/{ep_id}/mirror")
    assert mirror_resp.status_code == 200
    mirror_data = mirror_resp.json()
    assert mirror_data["local_video_path"].endswith("episode.mp4")
    assert mirror_data["bytes"] == len(sample_bytes)
    assert Path(mirror_data["local_video_path"]).exists()
    assert (
        mirror_data["used_key_version"] is None
        or mirror_data["used_key_version"] == "v2"
    )

    hydrate_resp = client.post(f"/episodes/{ep_id}/hydrate")
    assert hydrate_resp.status_code == 200
    assert hydrate_resp.json()["local_video_path"].endswith("episode.mp4")

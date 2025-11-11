import subprocess

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import jobs as jobs_router
from apps.api.services.episodes import EpisodeStore
from py_screenalytics.artifacts import ensure_dirs, get_path


def test_detect_track_stub(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "upload-show-s01e01"
    ensure_dirs(ep_id)
    video_path = get_path(ep_id, "video")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"fake mp4 bytes")
    jobs_router.EPISODE_STORE = EpisodeStore()
    jobs_router.EPISODE_STORE.upsert_ep_id(ep_id=ep_id, show_slug="upload-show", season=1, episode=1)

    detections_path = get_path(ep_id, "detections")
    tracks_path = get_path(ep_id, "tracks")

    def _fake_run(command, **kwargs):  # noqa: ANN001 - signature matches subprocess.run
        detections_path.write_text('{"ep_id": "%s"}\n' % ep_id, encoding="utf-8")
        tracks_path.write_text('{"track_id": 1}\n{"track_id": 2}\n', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("apps.api.routers.jobs.subprocess.run", _fake_run)

    client = TestClient(app)
    resp = client.post("/jobs/detect_track", json={"ep_id": ep_id, "stub": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["job"] == "detect_track"
    assert body["detections_count"] == 1
    assert body["tracks_count"] == 2
    assert body["tracker"] == "bytetrack"
    assert "--stub" in body["command"]

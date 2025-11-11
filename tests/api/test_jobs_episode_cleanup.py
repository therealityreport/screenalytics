from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import jobs as jobs_router
from apps.api.services.episodes import EpisodeStore
from apps.api.services.jobs import JobService
from py_screenalytics.artifacts import ensure_dirs, get_path


def test_episode_cleanup_async_enqueues_job(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    ep_id = "demo-s01e02"
    ensure_dirs(ep_id)
    video_path = get_path(ep_id, "video")
    video_path.write_bytes(b"vid")

    jobs_router.EPISODE_STORE = EpisodeStore()
    jobs_router.EPISODE_STORE.upsert_ep_id(ep_id=ep_id, show_slug="demo", season=1, episode=2)

    service = JobService(data_root=data_root)
    captured: dict = {}

    def _fake_launch(**kwargs):
        captured.update(kwargs)
        return {
            "job_id": "cleanup-job",
            "ep_id": ep_id,
            "state": "running",
            "started_at": "now",
            "progress_file": str(kwargs["progress_path"]),
            "requested": kwargs["requested"],
        }

    monkeypatch.setattr(service, "_launch_job", _fake_launch)
    monkeypatch.setattr(jobs_router, "JOB_SERVICE", service)

    client = TestClient(app)
    resp = client.post(
        "/jobs/episode_cleanup_async",
        json={"ep_id": ep_id, "stride": 3, "actions": ["split_tracks", "recluster"]},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["job_id"] == "cleanup-job"
    assert captured["job_type"] == "episode_cleanup"
    command = captured["command"]
    assert str(video_path) in command
    assert "--actions" in command
    assert captured["requested"]["actions"] == ["split_tracks", "recluster"]

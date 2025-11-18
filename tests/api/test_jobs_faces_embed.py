import types

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import jobs as jobs_router
from apps.api.services import jobs as jobs_service
from apps.api.services.episodes import EpisodeStore
from apps.api.services.jobs import JobService
from py_screenalytics.artifacts import ensure_dirs, get_path


def test_faces_embed_async_fails_fast_when_arcface_missing(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "demo-s01e04"
    ensure_dirs(ep_id)
    track_path = get_path(ep_id, "tracks")
    track_path.parent.mkdir(parents=True, exist_ok=True)
    track_path.write_text("{}", encoding="utf-8")

    jobs_router.EPISODE_STORE = EpisodeStore()
    jobs_router.EPISODE_STORE.upsert_ep_id(
        ep_id=ep_id, show_slug="demo", season=1, episode=4
    )

    service = JobService(data_root=data_root)
    monkeypatch.setattr(jobs_router, "JOB_SERVICE", service)

    fake_episode_run = types.SimpleNamespace(
        ensure_retinaface_ready=lambda device, det_thresh=None: (True, None, "cpu"),
        ensure_arcface_ready=lambda device: (False, "ArcFace weights missing", None),
        RETINAFACE_HELP="retina help",
        ARC_FACE_HELP="ArcFace weights missing or could not initialize. See README 'Models' or run scripts/fetch_models.py.",
    )
    monkeypatch.setattr(jobs_service, "episode_run", fake_episode_run, raising=False)

    client = TestClient(app)
    resp = client.post("/jobs/faces_embed_async", json={"ep_id": ep_id})
    assert resp.status_code == 400
    assert "ArcFace" in resp.json()["detail"]

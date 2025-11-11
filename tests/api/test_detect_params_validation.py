import types

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.services.episodes import EpisodeStore
from apps.api.services.jobs import JobService
from apps.api.routers import jobs as jobs_router
from apps.api.services import jobs as jobs_service
from py_screenalytics.artifacts import ensure_dirs, get_path


def test_detect_track_invalid_detector(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "demo-s01e01"
    ensure_dirs(ep_id)
    video_path = get_path(ep_id, "video")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"fake-bytes")
    jobs_router.EPISODE_STORE = EpisodeStore()
    jobs_router.EPISODE_STORE.upsert_ep_id(ep_id=ep_id, show_slug="demo", season=1, episode=1)

    client = TestClient(app)
    response = client.post(
        "/jobs/detect_track",
        json={"ep_id": ep_id, "detector": "not_a_real_detector"},
    )
    assert response.status_code == 400
    assert "Unsupported detector" in response.json().get("detail", "")


def test_detect_track_invalid_tracker(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "demo-s01e02"
    ensure_dirs(ep_id)
    video_path = get_path(ep_id, "video")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"fake-bytes-2")
    jobs_router.EPISODE_STORE = EpisodeStore()
    jobs_router.EPISODE_STORE.upsert_ep_id(ep_id=ep_id, show_slug="demo", season=1, episode=2)

    client = TestClient(app)
    response = client.post(
        "/jobs/detect_track",
        json={"ep_id": ep_id, "tracker": "nope"},
    )
    assert response.status_code == 400
    assert "Unsupported tracker" in response.json().get("detail", "")


def test_detect_track_retinaface_missing_models_returns_400(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "demo-s01e03"
    ensure_dirs(ep_id)
    video_path = get_path(ep_id, "video")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"fake-video")

    jobs_router.EPISODE_STORE = EpisodeStore()
    jobs_router.EPISODE_STORE.upsert_ep_id(ep_id=ep_id, show_slug="demo", season=1, episode=3)
    monkeypatch.setattr(jobs_router, "JOB_SERVICE", JobService(data_root=data_root))
    fake_episode_run = types.SimpleNamespace(
        ensure_retinaface_ready=lambda device, det_thresh=None: (False, "weights missing", None),
        RETINAFACE_HELP="RetinaFace weights missing or could not initialize. See README 'Models' or run scripts/fetch_models.py.",
    )
    monkeypatch.setattr(jobs_service, "episode_run", fake_episode_run)

    client = TestClient(app)
    response = client.post(
        "/jobs/detect_track",
        json={"ep_id": ep_id, "device": "cpu", "detector": "retinaface"},
    )
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "RetinaFace weights missing or could not initialize" in detail

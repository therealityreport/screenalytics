import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient

if "psycopg2" not in sys.modules:
    psycopg2_stub = types.ModuleType("psycopg2")
    psycopg2_stub.connect = lambda *args, **kwargs: None
    extras = types.ModuleType("psycopg2.extras")
    extras.RealDictCursor = object
    sys.modules["psycopg2"] = psycopg2_stub
    sys.modules["psycopg2.extras"] = extras

from apps.api.main import app
from apps.api.routers import jobs as jobs_router
from apps.api.services import jobs as jobs_service
from apps.api.services.episodes import EpisodeStore
from apps.api.services.jobs import JobService
from py_screenalytics.artifacts import ensure_dirs, get_path


def test_faces_embed_async_fails_fast_when_arcface_missing(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = EpisodeStore.make_ep_id("demo", 1, 4)
    ensure_dirs(ep_id)
    track_path = get_path(ep_id, "tracks")
    track_path.parent.mkdir(parents=True, exist_ok=True)
    track_path.write_text("{}", encoding="utf-8")
    video_path = get_path(ep_id, "video")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"\x00\x01video")

    jobs_router.EPISODE_STORE = EpisodeStore()
    jobs_router.EPISODE_STORE.upsert_ep_id(ep_id=ep_id, show_slug="demo", season=1, episode=4)

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


def test_faces_embed_sync_fails_when_video_missing(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = EpisodeStore.make_ep_id("demo", 1, 4)
    ensure_dirs(ep_id)
    track_path = get_path(ep_id, "tracks")
    track_path.parent.mkdir(parents=True, exist_ok=True)
    track_path.write_text("{}", encoding="utf-8")

    jobs_router.EPISODE_STORE = EpisodeStore()
    jobs_router.EPISODE_STORE.upsert_ep_id(ep_id=ep_id, show_slug="demo", season=1, episode=4)
    jobs_router.JOB_SERVICE = JobService(data_root=data_root)

    called = {"run_invoked": False}

    def _fail_if_called(*args, **kwargs):
        called["run_invoked"] = True
        raise AssertionError("_run_job_with_optional_sse should not be called when video is missing")

    monkeypatch.setattr(jobs_router, "_run_job_with_optional_sse", _fail_if_called)

    client = TestClient(app)
    resp = client.post("/jobs/faces_embed", json={"ep_id": ep_id})
    assert resp.status_code == 400
    assert "video" in resp.text.lower()
    assert called["run_invoked"] is False


def test_faces_embed_sync_succeeds_when_video_present(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = EpisodeStore.make_ep_id("demo", 1, 1)
    ensure_dirs(ep_id)
    track_path = get_path(ep_id, "tracks")
    track_path.parent.mkdir(parents=True, exist_ok=True)
    track_path.write_text("{}", encoding="utf-8")

    # Make a small readable "video" file; validation only checks readability
    video_path = get_path(ep_id, "video")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"\x00\x01fakevideo")

    jobs_router.EPISODE_STORE = EpisodeStore()
    jobs_router.EPISODE_STORE.upsert_ep_id(ep_id=ep_id, show_slug="demo", season=1, episode=1)
    jobs_router.JOB_SERVICE = JobService(data_root=data_root)

    faces_path = get_path(ep_id, "detections").parent / "faces.jsonl"
    embeds_path = data_root / "embeds" / ep_id / "faces.npy"

    def _fake_run_job(command, request, progress_file=None, cpu_threads=None):
        faces_path.parent.mkdir(parents=True, exist_ok=True)
        faces_path.write_text("{}", encoding="utf-8")
        embeds_path.parent.mkdir(parents=True, exist_ok=True)
        embeds_path.write_bytes(b"\x93NUMPY")  # minimal header prefix
        return {"job": "faces_embed", "ep_id": ep_id, "command": command}

    monkeypatch.setattr(jobs_router, "_run_job_with_optional_sse", _fake_run_job)
    monkeypatch.setattr(jobs_router.JOB_SERVICE, "ensure_arcface_ready", lambda device: "cpu")

    client = TestClient(app)
    resp = client.post("/jobs/faces_embed", json={"ep_id": ep_id})
    assert resp.status_code != 400
    assert faces_path.exists(), "faces.jsonl should be created on success"
    assert embeds_path.exists(), "faces.npy should be created on success"

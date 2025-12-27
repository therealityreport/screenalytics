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
from py_screenalytics import run_layout
from py_screenalytics.artifacts import ensure_dirs, get_path


def test_faces_embed_async_fails_fast_when_arcface_missing(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = EpisodeStore.make_ep_id("demo", 1, 4)
    run_id = "run-faces-embed-async"
    ensure_dirs(ep_id)
    track_path = get_path(ep_id, "tracks")
    track_path.parent.mkdir(parents=True, exist_ok=True)
    track_path.write_text("{}", encoding="utf-8")
    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "tracks.jsonl").write_text("{}", encoding="utf-8")
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
    resp = client.post("/jobs/faces_embed_async", json={"ep_id": ep_id, "run_id": run_id})
    assert resp.status_code == 400
    detail = resp.json().get("detail") or resp.json().get("message", "")
    assert "ArcFace" in detail


def test_faces_embed_sync_fails_when_video_missing(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = EpisodeStore.make_ep_id("demo", 1, 4)
    run_id = "run-faces-embed-video-missing"
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
    resp = client.post("/jobs/faces_embed", json={"ep_id": ep_id, "run_id": run_id})
    assert resp.status_code == 400
    assert "video" in resp.text.lower()
    assert called["run_invoked"] is False


def test_faces_embed_sync_succeeds_when_video_present(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = EpisodeStore.make_ep_id("demo", 1, 1)
    run_id = "run-faces-embed-success"
    ensure_dirs(ep_id)
    track_path = get_path(ep_id, "tracks")
    track_path.parent.mkdir(parents=True, exist_ok=True)
    track_path.write_text("{}", encoding="utf-8")

    # Make a small readable "video" file; validation only checks readability
    video_path = get_path(ep_id, "video")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"\x00\x01fakevideo")
    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "tracks.jsonl").write_text("{}", encoding="utf-8")

    jobs_router.EPISODE_STORE = EpisodeStore()
    jobs_router.EPISODE_STORE.upsert_ep_id(ep_id=ep_id, show_slug="demo", season=1, episode=1)
    jobs_router.JOB_SERVICE = JobService(data_root=data_root)

    def _fake_enqueue(*, ep_id: str, run_id: str, stage: str, params: dict | None, source: str = "test") -> dict:
        return {
            "status": "queued",
            "ep_id": ep_id,
            "run_id": run_id,
            "stage": stage,
            "job_id": "job-faces-embed",
            "params_hash": "hash",
        }

    monkeypatch.setattr(jobs_router, "_enqueue_run_stage_job", _fake_enqueue)
    monkeypatch.setattr(jobs_router.JOB_SERVICE, "ensure_arcface_ready", lambda device: "cpu")

    client = TestClient(app)
    resp = client.post("/jobs/faces_embed", json={"ep_id": ep_id, "run_id": run_id})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["state"] == "queued"
    assert payload["job_id"] == "job-faces-embed"

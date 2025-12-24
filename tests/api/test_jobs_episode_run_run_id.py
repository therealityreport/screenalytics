from __future__ import annotations

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
from apps.api.services.episodes import EpisodeStore
from apps.api.services.jobs import JobService
from py_screenalytics import run_layout
from py_screenalytics.artifacts import ensure_dirs, get_path


def test_episode_run_api_generates_run_id(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = EpisodeStore.make_ep_id("demo", 1, 1)
    ensure_dirs(ep_id)
    video_path = get_path(ep_id, "video")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"\x00\x01fakevideo")

    jobs_router.EPISODE_STORE = EpisodeStore()
    jobs_router.EPISODE_STORE.upsert_ep_id(ep_id=ep_id, show_slug="demo", season=1, episode=1)

    service = JobService(data_root=data_root)
    monkeypatch.setattr(jobs_router, "JOB_SERVICE", service)

    class _FakeResult:
        def __init__(self, run_id: str) -> None:
            self.run_id = run_id
            self.success = True
            self.error = None

        def to_dict(self) -> dict:
            return {"run_id": self.run_id, "success": True}

    import py_screenalytics.pipeline as pipeline

    def _fake_run_episode(ep_id_value: str, video_path_value: Path, config) -> _FakeResult:
        return _FakeResult(config.run_id)

    monkeypatch.setattr(pipeline, "run_episode", _fake_run_episode)

    client = TestClient(app)
    resp = client.post("/jobs/jobs/episode-run", json={"ep_id": ep_id})
    assert resp.status_code == 200
    payload = resp.json()

    run_id = payload.get("run_id")
    assert isinstance(run_id, str) and run_id.strip()
    assert run_layout.run_root(ep_id, run_id).exists()

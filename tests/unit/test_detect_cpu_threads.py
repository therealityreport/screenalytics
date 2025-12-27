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
from apps.api.services import jobs as jobs_service
from apps.api.services.jobs import JobService
from py_screenalytics.artifacts import ensure_dirs, get_path


def test_launch_job_sets_cpu_thread_env(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    service = JobService(data_root=data_root)
    monkeypatch.setattr(service, "ensure_retinaface_ready", lambda *args, **kwargs: "cpu")
    monkeypatch.setattr(JobService, "_monitor_process", lambda *args, **kwargs: None)

    env_seen = {}

    class FakeProc:
        def __init__(self, *args, **kwargs):
            env_seen.update(kwargs.get("env", {}))
            self.pid = 999
            self.returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(jobs_service.subprocess, "Popen", FakeProc)

    ep_id = "cpu-threads"
    ensure_dirs(ep_id)
    video_path = get_path(ep_id, "video")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"\x00\x01video")

    service.start_detect_track_job(
        ep_id=ep_id,
        stride=4,
        fps=None,
        device="cpu",
        video_path=video_path,
        save_frames=False,
        save_crops=False,
        jpeg_quality=85,
        detector="retinaface",
        tracker="bytetrack",
        max_gap=30,
        det_thresh=0.5,
        scene_detector="pyscenedetect",
        scene_threshold=27.0,
        scene_min_len=12,
        scene_warmup_dets=3,
        track_high_thresh=None,
        new_track_thresh=None,
        track_buffer=None,
        min_box_area=None,
        profile=None,
        cpu_threads=2,
    )

    assert env_seen["OMP_NUM_THREADS"] == "2"
    assert env_seen["MKL_NUM_THREADS"] == "2"
    assert env_seen["OPENBLAS_NUM_THREADS"] == "2"
    assert env_seen["VECLIB_MAXIMUM_THREADS"] == "2"
    assert env_seen["NUMEXPR_NUM_THREADS"] == "2"
    assert env_seen["OPENCV_NUM_THREADS"] == "2"
    assert env_seen["ORT_INTRA_OP_NUM_THREADS"] == "2"
    assert env_seen["ORT_INTER_OP_NUM_THREADS"] == "1"
    assert env_seen["SCREENALYTICS_MAX_CPU_THREADS"] == "2"


def test_detect_track_async_propagates_cpu_threads(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    jobs_router.JOB_SERVICE = JobService(data_root=data_root)
    jobs_router.EPISODE_STORE = EpisodeStore()
    ep_id = EpisodeStore.make_ep_id("demo", 1, 3)
    run_id = "run-cpu-threads"
    jobs_router.EPISODE_STORE.upsert_ep_id(ep_id=ep_id, show_slug="demo", season=1, episode=3)

    video_path = get_path(ep_id, "video")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"\x00\x01video")

    captured_requested = {}

    def _fake_delay(ep_id_value, options):  # noqa: ANN001
        captured_requested.update(options)
        return types.SimpleNamespace(id="async-cpu-job")

    fake_tasks = types.SimpleNamespace(run_detect_track_task=types.SimpleNamespace(delay=_fake_delay))
    monkeypatch.setitem(sys.modules, "apps.api.tasks", fake_tasks)
    monkeypatch.setattr(JobService, "ensure_retinaface_ready", lambda *args, **kwargs: "cpu")

    client = TestClient(app)
    resp = client.post(
        "/jobs/detect_track_async",
        json={
            "ep_id": ep_id,
            "run_id": run_id,
            "stride": 4,
            "device": "cpu",
            "save_frames": False,
            "save_crops": False,
            "cpu_threads": 2,
        },
    )
    assert resp.status_code == 200
    assert captured_requested.get("cpu_threads") == 2


def test_faces_embed_async_propagates_cpu_threads(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    jobs_router.JOB_SERVICE = JobService(data_root=data_root)
    jobs_router.EPISODE_STORE = EpisodeStore()
    ep_id = EpisodeStore.make_ep_id("demo", 1, 5)
    run_id = "run-cpu-embed"
    jobs_router.EPISODE_STORE.upsert_ep_id(ep_id=ep_id, show_slug="demo", season=1, episode=5)

    from py_screenalytics import run_layout
    track_path = run_layout.run_root(ep_id, run_id) / "tracks.jsonl"
    track_path.parent.mkdir(parents=True, exist_ok=True)
    track_path.write_text("{}", encoding="utf-8")
    video_path = get_path(ep_id, "video")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"\x00\x01video")

    captured_requested = {}

    def _fake_delay(ep_id_value, options):  # noqa: ANN001
        captured_requested.update(options)
        return types.SimpleNamespace(id="faces-cpu-job")

    fake_tasks = types.SimpleNamespace(run_faces_embed_task=types.SimpleNamespace(delay=_fake_delay))
    monkeypatch.setitem(sys.modules, "apps.api.tasks", fake_tasks)
    monkeypatch.setattr(JobService, "ensure_arcface_ready", lambda *args, **kwargs: "cpu")

    client = TestClient(app)
    resp = client.post(
        "/jobs/faces_embed_async",
        json={
            "ep_id": ep_id,
            "run_id": run_id,
            "device": "cpu",
            "save_frames": False,
            "save_crops": False,
            "cpu_threads": 2,
        },
    )
    assert resp.status_code == 200
    assert captured_requested.get("cpu_threads") == 2

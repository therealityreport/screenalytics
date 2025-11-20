from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta
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
from apps.api.routers import episodes as episodes_router
from apps.api.routers import jobs as jobs_router
from apps.api.services.episodes import EpisodeStore
from apps.api.services.jobs import JobService
from py_screenalytics.artifacts import ensure_dirs, get_path
from tools.episode_run import _write_run_marker


def _reset_services(data_root: Path) -> JobService:
    jobs_router.JOB_SERVICE = JobService(data_root=data_root)
    store = EpisodeStore()
    jobs_router.EPISODE_STORE = store
    episodes_router.EPISODE_STORE = store
    return jobs_router.JOB_SERVICE


def test_detect_rerun_invalidation_removes_downstream(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    service = _reset_services(data_root)

    ep_id = EpisodeStore.make_ep_id("demo", 1, 1)
    ensure_dirs(ep_id)
    store = jobs_router.EPISODE_STORE
    store.upsert_ep_id(ep_id=ep_id, show_slug="demo", season=1, episode=1)

    video_path = get_path(ep_id, "video")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"\x00\x01video")

    manifests_dir = get_path(ep_id, "detections").parent
    embeds_dir = manifests_dir.parent / "embeds" / ep_id
    detections_path = manifests_dir / "detections.jsonl"
    tracks_path = manifests_dir / "tracks.jsonl"
    detections_path.parent.mkdir(parents=True, exist_ok=True)
    detections_path.write_text('{"d":1}\n', encoding="utf-8")
    tracks_path.write_text('{"t":1}\n', encoding="utf-8")

    faces_path = manifests_dir / "faces.jsonl"
    faces_path.write_text('{"f":1}\n', encoding="utf-8")
    identities_path = manifests_dir / "identities.json"
    identities_path.write_text('{"identities": []}', encoding="utf-8")
    embeds_dir.mkdir(parents=True, exist_ok=True)
    (embeds_dir / "faces.npy").write_bytes(b"\x93NUMPY")
    (embeds_dir / "tracks.npy").write_bytes(b"\x93NUMPY")
    (embeds_dir / "track_ids.json").write_text("[0,1]", encoding="utf-8")

    monkeypatch.setattr(service, "ensure_retinaface_ready", lambda *args, **kwargs: "cpu")

    launch_called = {}

    def _fake_launch_job(self, **kwargs):
        launch_called["requested"] = kwargs.get("requested")
        return {
            "job_id": "fake",
            "state": "running",
            "progress_file": str(kwargs.get("progress_path")),
            "requested": kwargs.get("requested"),
        }

    monkeypatch.setattr(JobService, "_launch_job", _fake_launch_job)

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
        cpu_threads=None,
    )

    assert detections_path.exists()
    assert tracks_path.exists()
    assert detections_path.read_text(encoding="utf-8").startswith('{"d":1}')
    assert tracks_path.read_text(encoding="utf-8").startswith('{"t":1}')
    assert not faces_path.exists()
    assert not identities_path.exists()
    assert not (embeds_dir / "faces.npy").exists()
    assert not (embeds_dir / "tracks.npy").exists()
    assert not (embeds_dir / "track_ids.json").exists()
    assert launch_called, "detect/track rerun should enqueue a job"


def test_status_marks_faces_and_cluster_stale_after_redetect(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    service = _reset_services(data_root)

    ep_id = EpisodeStore.make_ep_id("demo", 1, 2)
    ensure_dirs(ep_id)
    jobs_router.EPISODE_STORE.upsert_ep_id(ep_id=ep_id, show_slug="demo", season=1, episode=2)

    manifests_dir = get_path(ep_id, "detections").parent
    detections_path = manifests_dir / "detections.jsonl"
    tracks_path = manifests_dir / "tracks.jsonl"
    detections_path.parent.mkdir(parents=True, exist_ok=True)
    detections_path.write_text('{"d":1}\n', encoding="utf-8")
    tracks_path.write_text('{"t":1}\n', encoding="utf-8")
    faces_path = manifests_dir / "faces.jsonl"
    faces_path.write_text('{"f":1}\n', encoding="utf-8")
    identities_path = manifests_dir / "identities.json"
    identities_path.write_text('{"identities": [1]}', encoding="utf-8")

    base_time = datetime.utcnow().replace(microsecond=0)
    detect_finished = base_time.isoformat() + "Z"
    faces_finished = (base_time + timedelta(seconds=10)).isoformat() + "Z"
    cluster_finished = (base_time + timedelta(seconds=20)).isoformat() + "Z"

    _write_run_marker(
        ep_id,
        "detect_track",
        {"phase": "detect_track", "status": "success", "tracks": 2, "detections": 2, "finished_at": detect_finished},
    )
    _write_run_marker(
        ep_id,
        "faces_embed",
        {"phase": "faces_embed", "status": "success", "faces": 2, "finished_at": faces_finished},
    )
    _write_run_marker(
        ep_id,
        "cluster",
        {"phase": "cluster", "status": "success", "identities": 1, "faces": 2, "finished_at": cluster_finished},
    )

    client = TestClient(app)
    initial = client.get(f"/episodes/{ep_id}/status")
    assert initial.status_code == 200
    payload = initial.json()
    assert payload["faces_embed"]["status"] == "success"
    assert payload["cluster"]["status"] == "success"

    def _fake_launch_job(self, **kwargs):
        new_finished = (base_time + timedelta(hours=1)).isoformat() + "Z"
        _write_run_marker(
            ep_id,
            "detect_track",
            {
                "phase": "detect_track",
                "status": "success",
                "tracks": 2,
                "detections": 2,
                "finished_at": new_finished,
            },
        )
        return {"job_id": "redetect", "state": "succeeded", "progress_file": str(kwargs.get("progress_path"))}

    monkeypatch.setattr(service, "ensure_retinaface_ready", lambda *args, **kwargs: "cpu")
    monkeypatch.setattr(JobService, "_launch_job", _fake_launch_job)

    video_path = get_path(ep_id, "video")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"\x00\x01video")

    service.start_detect_track_job(
        ep_id=ep_id,
        stride=6,
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
        cpu_threads=None,
    )

    updated = client.get(f"/episodes/{ep_id}/status")
    assert updated.status_code == 200
    updated_payload = updated.json()
    assert updated_payload["faces_embed"]["status"] == "stale"
    assert updated_payload["faces_embed"]["source"] == "outdated_after_redetect"
    assert updated_payload["cluster"]["status"] == "stale"
    assert updated_payload["cluster"]["source"] == "outdated_after_redetect"
    assert updated_payload["detect_track"]["finished_at"] > updated_payload["faces_embed"]["finished_at"]

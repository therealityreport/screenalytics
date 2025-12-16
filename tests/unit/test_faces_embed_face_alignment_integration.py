from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from py_screenalytics.artifacts import ensure_dirs, get_path
from tools import episode_run


class _FakeFrameDecoder:
    def __init__(self, _path: Path) -> None:
        self._closed = False

    def read(self, frame_idx: int):
        # Deterministic gradient frame with enough variance to pass quality checks.
        h, w = 120, 160
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(w, dtype=np.uint8)[None, :]
        frame[:, :, 1] = np.arange(h, dtype=np.uint8)[:, None]
        frame[:, :, 2] = np.uint8(frame_idx % 255)
        return frame

    def get_cache_stats(self) -> dict:
        return {"hits": 0, "misses": 0, "hit_rate": 0.0, "size": 0}

    def close(self) -> None:
        self._closed = True


class _FakeEmbedder:
    def __init__(self) -> None:
        self.resolved_device = "cpu"
        self.seen_max_values: list[int] = []

    def ensure_ready(self) -> None:
        return None

    def encode(self, crops):
        for crop in crops:
            self.seen_max_values.append(int(np.max(np.asarray(crop))))
        base = np.linspace(-1.0, 1.0, 512, dtype=np.float32)
        base /= np.linalg.norm(base)
        return np.vstack([base for _ in crops])


def _write_single_track(ep_id: str, *, run_id: str | None = None) -> None:
    track_path = episode_run._tracks_path_for_run(ep_id, run_id)
    track_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "track_id": 1,
        "class": "face",
        "bboxes_sampled": [
            {"frame_idx": 0, "ts": 0.0, "bbox_xyxy": [10, 10, 80, 90]},
        ],
    }
    with track_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _patch_face_alignment_enabled(monkeypatch) -> None:
    monkeypatch.setattr(
        episode_run,
        "_load_alignment_config",
        lambda: {
            "face_alignment": {
                "enabled": True,
                "model": {"landmarks_type": "2D", "flip_input": False},
                "processing": {"device": "cpu"},
                "quality": {"min_face_size": 20},
                "output": {"crop_size": 112, "crop_margin": 0.0},
            }
        },
    )


def _patch_fake_embedding_runtime(monkeypatch, fake_embedder: _FakeEmbedder) -> None:
    # Avoid cv2 blur scoring in unit tests (and keep thresholds permissive).
    monkeypatch.setattr(episode_run, "_estimate_blur_score", lambda _img: 999.0)

    def _fake_get_backend(*, backend_type: str, device: str, tensorrt_config: str, allow_cpu_fallback: bool):
        return fake_embedder

    monkeypatch.setattr(episode_run, "get_embedding_backend", _fake_get_backend)
    monkeypatch.setattr(episode_run, "FrameDecoder", _FakeFrameDecoder)

    # Patch FAN alignment to avoid requiring the face-alignment dependency in unit tests.
    from py_screenalytics.face_alignment import fan2d as fan2d_mod

    class _DummyAligner:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def align_face(self, image_bgr, bbox_xyxy):
            return [[float(i), float(i)] for i in range(68)]

    def _fake_aligned_crop(image_bgr, landmarks_68, *, crop_size: int = 112, margin: float = 0.0):
        crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        crop[:, :, 0] = np.arange(crop_size, dtype=np.uint8)[:, None]
        crop[:, :, 1] = np.arange(crop_size, dtype=np.uint8)[None, :]
        crop[:, :, 2] = 200
        return crop

    monkeypatch.setattr(fan2d_mod, "Fan2dAligner", _DummyAligner)
    monkeypatch.setattr(fan2d_mod, "align_face_crop", _fake_aligned_crop)
    monkeypatch.setattr(fan2d_mod, "compute_alignment_quality", lambda *_a, **_k: 1.0)


def test_faces_embed_uses_aligned_crop_when_face_alignment_enabled(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "test-face-align-integration"
    ensure_dirs(ep_id)

    video_path = get_path(ep_id, "video")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.touch()

    _write_single_track(ep_id)

    fake_embedder = _FakeEmbedder()
    _patch_face_alignment_enabled(monkeypatch)
    _patch_fake_embedding_runtime(monkeypatch, fake_embedder)

    manifests_dir = episode_run._manifests_dir_for_run(ep_id, run_id=None)
    progress_path = manifests_dir / "progress.json"

    args = SimpleNamespace(
        ep_id=ep_id,
        device="cpu",
        save_frames=False,
        save_crops=False,
        jpeg_quality=85,
        thumb_size=64,
        progress_file=str(progress_path),
        max_samples_per_track=1,
        min_samples_per_track=1,
        sample_every_n_frames=1,
    )

    summary = episode_run._run_faces_embed_stage(args, storage=None, ep_ctx=None, s3_prefixes=None)
    assert summary.get("stage") == "faces_embed"

    # Ensure the embedding backend saw our "aligned" crop (blue channel=200).
    assert fake_embedder.seen_max_values, "expected embedder.encode to be called"
    assert max(fake_embedder.seen_max_values) >= 200

    faces_path = manifests_dir / "faces.jsonl"
    assert faces_path.exists()
    faces = [json.loads(line) for line in faces_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert faces and faces[0].get("alignment_used") is True
    assert faces[0].get("alignment_source") in {"fan2d", "aligned_faces"}

    aligned_faces_path = manifests_dir / "face_alignment" / "aligned_faces.jsonl"
    assert aligned_faces_path.exists()
    aligned_rows = [
        json.loads(line)
        for line in aligned_faces_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert aligned_rows and aligned_rows[0].get("track_id") == 1
    assert "landmarks_68" in aligned_rows[0]


def test_faces_embed_writes_run_scoped_alignment_artifacts_when_run_id_set(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "test-face-align-run-scoped"
    run_id = "runABC"
    ensure_dirs(ep_id)

    video_path = get_path(ep_id, "video")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.touch()

    _write_single_track(ep_id, run_id=run_id)

    fake_embedder = _FakeEmbedder()
    _patch_face_alignment_enabled(monkeypatch)
    _patch_fake_embedding_runtime(monkeypatch, fake_embedder)

    manifests_dir = episode_run._manifests_dir_for_run(ep_id, run_id=run_id)
    progress_path = manifests_dir / "progress.json"

    args = SimpleNamespace(
        ep_id=ep_id,
        run_id=run_id,
        device="cpu",
        save_frames=False,
        save_crops=False,
        jpeg_quality=85,
        thumb_size=64,
        progress_file=str(progress_path),
        max_samples_per_track=1,
        min_samples_per_track=1,
        sample_every_n_frames=1,
    )

    summary = episode_run._run_faces_embed_stage(args, storage=None, ep_ctx=None, s3_prefixes=None)
    assert summary.get("stage") == "faces_embed"

    aligned_faces_path = manifests_dir / "face_alignment" / "aligned_faces.jsonl"
    assert aligned_faces_path.exists()

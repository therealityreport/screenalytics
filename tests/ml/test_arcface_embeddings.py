from __future__ import annotations

import json
import os
from types import SimpleNamespace
from pathlib import Path

import pytest

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional ML dependency
    pytest.skip("numpy is required for ML embedding tests", allow_module_level=True)

from py_screenalytics.artifacts import ensure_dirs, get_path
from tools import episode_run

RUN_ML_TESTS = os.environ.get("RUN_ML_TESTS") == "1"
pytestmark = pytest.mark.skipif(not RUN_ML_TESTS, reason="set RUN_ML_TESTS=1 to run ML integration tests")


def _make_sample_video(target: Path, frame_count: int = 5, size: tuple[int, int] = (80, 80)) -> Path:
    import cv2  # type: ignore

    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(target), fourcc, 10, (width, height))
    if not writer.isOpened():  # pragma: no cover
        raise RuntimeError("Unable to create synthetic video for test")
    for idx in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        offset = 10 + idx * 2
        cv2.rectangle(frame, (offset, offset), (offset + 20, offset + 26), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return target


class _FakeArcFaceEmbedder:
    def __init__(self, device: str, **_: object) -> None:  # noqa: D401
        self.device = device
        self.resolved_device = device

    def ensure_ready(self) -> None:
        return None

    def encode(self, crops):
        base = np.linspace(-1.0, 1.0, 512, dtype=np.float32)
        base /= np.linalg.norm(base)
        return np.vstack([base for _ in crops])


def _write_sample_tracks(ep_id: str, frame_count: int = 3) -> None:
    track_path = get_path(ep_id, "tracks")
    rows = []
    for track_id in range(1, 3):
        samples = []
        for frame_idx in range(frame_count):
            samples.append(
                {
                    "frame_idx": frame_idx,
                    "ts": frame_idx * 0.5,
                    "bbox_xyxy": [20 + track_id * 5, 15, 40 + track_id * 5, 35],
                }
            )
        rows.append(
            {
                "track_id": track_id,
                "class": "face",
                "first_ts": 0.0,
                "last_ts": (frame_count - 1) * 0.5,
                "frame_count": frame_count,
                "bboxes_sampled": samples,
            }
        )
    track_path.parent.mkdir(parents=True, exist_ok=True)
    with track_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_arcface_embeddings_are_unit_norm(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "arcface-test-ep"
    ensure_dirs(ep_id)
    video_path = get_path(ep_id, "video")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    _make_sample_video(video_path)
    _write_sample_tracks(ep_id)

    manifests_dir = get_path(ep_id, "detections").parent
    progress_path = manifests_dir / "progress.json"

    args = SimpleNamespace(
        ep_id=ep_id,
        device="cpu",
        save_frames=False,
        save_crops=False,
        jpeg_quality=85,
        thumb_size=128,
        progress_file=str(progress_path),
    )

    monkeypatch.setattr(episode_run, "ArcFaceEmbedder", _FakeArcFaceEmbedder)

    summary = episode_run._run_faces_embed_stage(args, storage=None, ep_ctx=None, s3_prefixes=None)
    faces_path = manifests_dir / "faces.jsonl"
    assert faces_path.exists()

    embeddings = []
    with faces_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            emb = np.asarray(payload["embedding"], dtype=np.float32)
            embeddings.append(emb)
            assert emb.shape[0] == 512
            assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)
            assert payload.get("embedding_model") == episode_run.ARC_FACE_MODEL_NAME
            assert payload.get("detector") == "retinaface"
    assert embeddings, "expected faces to be written"
    assert summary["stage"] == "faces_embed"
    assert summary["detector"] == "retinaface"
    assert summary.get("embedding_model") == episode_run.ARC_FACE_MODEL_NAME
    assert summary.get("tracker") == "bytetrack"

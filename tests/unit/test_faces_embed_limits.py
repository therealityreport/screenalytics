from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from tools import episode_run


def _write_tracks(tmp_path: Path, track_specs: dict[int, int]) -> Path:
    track_path = tmp_path / "tracks.jsonl"
    with track_path.open("w", encoding="utf-8") as handle:
        for track_id, frame_count in track_specs.items():
            samples = [
                {"frame_idx": idx, "ts": float(idx) / 10.0, "bbox_xyxy": [0, 0, 10, 10]}
                for idx in range(frame_count)
            ]
            row = {"track_id": track_id, "bboxes_sampled": samples}
            handle.write(json.dumps(row) + "\n")
    return track_path


def test_per_track_sampling_cap_uniform(tmp_path) -> None:
    track_path = _write_tracks(tmp_path, {1: 10})
    samples = episode_run._load_track_samples(
        track_path,
        sort_by_frame=True,
        max_samples_per_track=3,
        min_samples_per_track=1,
        sample_every_n_frames=1,
    )
    assert len(samples) == 3
    frames = [s["frame_idx"] for s in samples]
    assert min(frames) == 0
    assert max(frames) == 9
    assert len(set(frames)) == 3


def test_episode_cap_keeps_tracks_and_downsamples(tmp_path) -> None:
    track_path = _write_tracks(tmp_path, {1: 8, 2: 8, 3: 8})
    samples = episode_run._load_track_samples(
        track_path,
        sort_by_frame=True,
        max_samples_per_track=6,
        min_samples_per_track=1,
        sample_every_n_frames=1,
        max_faces_total=10,
    )
    assert len(samples) == 10
    counts = Counter(s["track_id"] for s in samples)
    assert sum(counts.values()) == 10
    assert all(count >= 1 for count in counts.values())


def test_frame_exporter_direct_s3(monkeypatch, tmp_path) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    class FakeStorage:
        def __init__(self) -> None:
            self.uploads: list[tuple[str, bytes, str | None]] = []
            self.write_enabled = True

        def s3_enabled(self) -> bool:
            return True

        def upload_bytes(self, data: bytes, key: str, *, content_type: str | None = None) -> bool:
            self.uploads.append((key, data, content_type))
            return True

    storage = FakeStorage()
    prefixes = {
        "frames": "artifacts/frames/demo/",
        "crops": "artifacts/crops/demo/",
    }
    exporter = episode_run.FrameExporter(
        "demo-s01e01",
        save_frames=True,
        save_crops=True,
        jpeg_quality=80,
        storage=storage,  # type: ignore[arg-type]
        s3_prefixes=prefixes,
    )
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    exporter.export(1, image, [(7, [10, 10, 50, 50])], ts=0.0)
    exporter.write_indexes()

    keys = [entry[0] for entry in storage.uploads]
    assert f"{prefixes['frames']}frame_000001.jpg" in keys
    assert f"{prefixes['crops']}track_0007/frame_000001.jpg" in keys
    assert f"{prefixes['crops']}track_0007/index.json" in keys
    assert not exporter.frames_dir.exists(), "frames directory should not be created for direct S3 uploads"

from __future__ import annotations

import json

from apps.api.services import track_reps


def test_compute_track_representative_finds_run_scoped_crops(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setattr(track_reps, "REP_DET_MIN", 0.0)
    monkeypatch.setattr(track_reps, "REP_STD_MIN", 0.0)

    ep_id = "ep123"
    run_id = "runABC"
    manifests_dir = data_root / "manifests" / ep_id
    manifests_dir.mkdir(parents=True, exist_ok=True)

    (manifests_dir / "tracks.jsonl").write_text(json.dumps({"track_id": 1}) + "\n", encoding="utf-8")

    face = {
        "track_id": 1,
        "frame_idx": 1,
        "conf": 0.9,
        "crop_std": 10.0,
        "bbox": [0, 0, 10, 10],
        "embedding": [0.01] * 512,
    }
    (manifests_dir / "faces.jsonl").write_text(json.dumps(face) + "\n", encoding="utf-8")

    crop_path = (
        data_root
        / "frames"
        / ep_id
        / "runs"
        / run_id
        / "crops"
        / "track_0001"
        / "frame_000001.png"
    )
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.write_bytes(b"fake")

    rep = track_reps.compute_track_representative(ep_id, 1, run_id=run_id)
    assert rep is not None
    assert rep.get("crop_key") == "crops/track_0001/frame_000001.png"

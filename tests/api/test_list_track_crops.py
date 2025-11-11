import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from py_screenalytics.artifacts import ensure_dirs, get_path


def _setup_track(tmp_path: Path, count: int = 3) -> str:
    data_root = tmp_path / "data"
    os.environ["SCREENALYTICS_DATA_ROOT"] = str(data_root)
    os.environ["STORAGE_BACKEND"] = "local"
    ep_id = "demo-s01e01"
    ensure_dirs(ep_id)
    track_dir = get_path(ep_id, "frames_root") / "crops" / "track_0001"
    track_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for idx in range(count):
        frame_path = track_dir / f"frame_{idx:06d}.jpg"
        frame_path.write_bytes(b"test")
        entries.append({"key": f"track_0001/frame_{idx:06d}.jpg", "frame_idx": idx, "ts": idx * 0.5})
    (track_dir / "index.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return ep_id


@pytest.mark.parametrize("sample", [1, 2])
def test_list_track_crops_local_pagination(tmp_path, sample):
    ep_id = _setup_track(tmp_path, count=4)
    client = TestClient(app)

    resp = client.get(
        f"/episodes/{ep_id}/tracks/1/crops",
        params={"sample": sample, "limit": 1},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert "items" in payload
    assert len(payload["items"]) == 1
    first = payload["items"][0]
    assert first["key"].startswith("track_0001/frame_")
    assert first["url"].endswith(".jpg")
    cursor = payload.get("next_start_after")
    assert cursor

    resp2 = client.get(
        f"/episodes/{ep_id}/tracks/1/crops",
        params={"sample": sample, "limit": 2, "start_after": cursor},
    )
    assert resp2.status_code == 200
    payload2 = resp2.json()
    assert len(payload2["items"]) >= 1
    # ensure returned cursor encodes sampling state when sample > 1
    next_cursor = payload2.get("next_start_after")
    if sample > 1 and next_cursor:
        assert "|" in next_cursor


def test_list_track_crops_sampling_respects_cursor(tmp_path):
    ep_id = _setup_track(tmp_path, count=5)
    client = TestClient(app)

    resp = client.get(
        f"/episodes/{ep_id}/tracks/1/crops",
        params={"sample": 2, "limit": 1},
    )
    payload = resp.json()
    cursor = payload["next_start_after"]
    assert cursor.endswith("|1")

    resp2 = client.get(
        f"/episodes/{ep_id}/tracks/1/crops",
        params={"sample": 2, "limit": 1, "start_after": cursor},
    )
    payload2 = resp2.json()
    assert payload2["items"], "expected another sampled crop"
    second_frame = payload2["items"][0]["frame_idx"]
    assert second_frame == payload["items"][0]["frame_idx"] + 2

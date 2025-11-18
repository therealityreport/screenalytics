from __future__ import annotations

import pytest
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from apps.api.main import app
from tests.api._sse_utils import collect_sse_events, write_sample_faces, write_sample_tracks

pytest.importorskip("numpy")


def test_sse_cluster_done_and_close(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    ep_id = "cluster-sse"
    write_sample_tracks(ep_id, sample_count=8)
    write_sample_faces(ep_id, face_count=8)

    events_to_emit = [
        {"phase": "cluster", "frames_done": 0, "frames_total": 8},
        {"phase": "cluster", "frames_done": 8, "frames_total": 8, "step": "done", "summary": {"stage": "cluster"}},
        {"phase": "done", "step": "cluster"},
    ]

    async def _fake_event_stream():
        import json

        for payload in events_to_emit:
            yield f"data: {json.dumps(payload)}\n\n"

    def _fake_run(command, request, progress_file=None):
        return StreamingResponse(_fake_event_stream(), media_type="text/event-stream")

    monkeypatch.setattr("apps.api.routers.jobs._run_job_with_optional_sse", _fake_run)

    client = TestClient(app)
    headers = {"accept": "text/event-stream"}
    with client.stream("POST", "/jobs/cluster", headers=headers, json={"ep_id": ep_id}) as response:
        assert response.status_code == 200
        events = collect_sse_events(response)
        stream_closed = response.is_closed

    cluster_events = [body for _, body in events if body.get("phase") == "cluster"]
    assert cluster_events, "expected cluster progress events"
    assert cluster_events[0]["frames_done"] == 0
    final_cluster = cluster_events[-1]
    assert final_cluster.get("step") == "done"
    assert final_cluster["frames_done"] == final_cluster["frames_total"]
    assert final_cluster.get("summary", {}).get("stage") == "cluster"

    done_events = [body for _, body in events if body.get("phase") == "done"]
    assert done_events[-1].get("step") == "cluster"
    assert stream_closed is True

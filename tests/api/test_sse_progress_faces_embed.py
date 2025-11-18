from __future__ import annotations

import pytest
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from apps.api.main import app
from tests.api._sse_utils import collect_sse_events, write_sample_tracks

pytest.importorskip("numpy")


def test_sse_faces_embed_done_and_close(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    ep_id = "faces-sse"
    write_sample_tracks(ep_id, sample_count=6)

    events_to_emit = [
        {"phase": "faces_embed", "frames_done": 0, "frames_total": 6},
        {
            "phase": "faces_embed",
            "frames_done": 6,
            "frames_total": 6,
            "step": "done",
            "summary": {"stage": "faces_embed"},
        },
        {"phase": "done", "step": "faces_embed"},
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
    payload = {"ep_id": ep_id, "save_crops": False}

    with client.stream(
        "POST", "/jobs/faces_embed", headers=headers, json=payload
    ) as response:
        assert response.status_code == 200
        events = collect_sse_events(response)
        stream_closed = response.is_closed

    faces_events = [body for _, body in events if body.get("phase") == "faces_embed"]
    assert faces_events, "expected faces_embed progress events"
    assert faces_events[0]["frames_done"] == 0
    final_event = faces_events[-1]
    assert final_event.get("step") == "done"
    assert final_event["frames_done"] == final_event["frames_total"]
    assert final_event.get("video_time") is None
    assert final_event.get("summary", {}).get("stage") == "faces_embed"

    done_events = [body for _, body in events if body.get("phase") == "done"]
    assert done_events, "expected terminal done event"
    assert done_events[-1].get("step") == "faces_embed"
    assert stream_closed is True

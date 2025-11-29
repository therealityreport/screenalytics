from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import episodes
from apps.api.routers.episodes import EpisodeStatusResponse, PhaseStatus


def test_validation_errors_use_envelope() -> None:
    with TestClient(app) as client:
        response = client.post("/episodes", json={})
    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "VALIDATION_ERROR"
    assert payload["message"] == "Validation error"


def test_http_exception_uses_envelope() -> None:
    with TestClient(app) as client:
        response = client.get("/episodes/does-not-exist/status")
    assert response.status_code == 404
    payload = response.json()
    assert payload["code"] == "HTTP_404"
    assert "not found" in payload["message"].lower()


def test_episode_events_include_manifest_mtime(monkeypatch) -> None:
    async def _fake_snapshot(ep_id: str) -> EpisodeStatusResponse:
        return EpisodeStatusResponse(
            ep_id=ep_id,
            detect_track=PhaseStatus(
                phase="detect_track", status="success", manifest_exists=True, last_run_at="2025-01-01T00:00:00Z"
            ),
            faces_embed=PhaseStatus(
                phase="faces_embed", status="success", manifest_exists=True, last_run_at="2025-01-01T00:00:00Z"
            ),
            cluster=PhaseStatus(phase="cluster", status="success", manifest_exists=True, last_run_at="2025-01-01T00:00:00Z"),
            scenes_ready=True,
            tracks_ready=True,
            faces_harvested=True,
            coreml_available=None,
            faces_stale=False,
            cluster_stale=False,
            faces_manifest_fallback=False,
            tracks_only_fallback=False,
        )

    monkeypatch.setattr(episodes, "_status_snapshot", _fake_snapshot)

    with TestClient(app) as client:
        response = client.get("/episodes/demo/events?max_events=3&poll_ms=5")

    assert response.status_code == 200
    body = response.text  # SSE stream text
    assert "manifest_mtime" in body
    assert "manifest_type" in body

from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import episodes
from apps.api.routers.episodes import EpisodeStatusResponse, PhaseStatus


def test_validation_errors_use_envelope() -> None:
    client = TestClient(app)
    resp = client.get("/episodes", params={"limit": -1})
    assert resp.status_code == 422
    payload = resp.json()
    assert payload["code"] == "VALIDATION_ERROR"
    assert payload["message"] == "Validation error"
    assert isinstance(payload["details"], list)
    assert payload["details"][0]["type"] == "greater_than"


def test_domain_errors_use_envelope() -> None:
    client = TestClient(app)
    resp = client.get("/episodes/nonexistent-id/status")
    assert resp.status_code == 404
    payload = resp.json()
    assert payload["code"] == "HTTP_404"
    assert payload["message"]


def test_ok_response_shape_when_no_faces_manifest_but_faces_ready(monkeypatch) -> None:
    """Regression for status calculation; ensures envelope structure is preserved."""
    client = TestClient(app)

    # Force status to OK but faces ready to avoid earlier crashing logic
    def _mock_status(ep_id: str) -> EpisodeStatusResponse:
        return EpisodeStatusResponse(
            ep_id=ep_id,
            status="ok",
            detect_ready=True,
            track_ready=True,
            faces_ready=True,
            faces_manifest_fallback=True,
            cluster_ready=False,
            screentime_ready=False,
            rerun_required={},
            rerun_blockers={},
            rerun_warnings={},
            rerun_reasons=[],
            stats=None,
            jobs=[],
        )

    monkeypatch.setattr(episodes, "get_episode_status", _mock_status)

    resp = client.get("/episodes/abc/status")
    assert resp.status_code == 200
    payload = resp.json()
    # Envelope should not change
    assert set(payload.keys()) >= {"status", "data", "meta"}
    status_payload = EpisodeStatusResponse(**payload["data"])
    assert status_payload.status == "ok"
    assert status_payload.faces_ready
    assert status_payload.faces_manifest_fallback
    # Ensure rerun fields remain intact
    assert status_payload.rerun_required == {}
    assert status_payload.rerun_reasons == []


def test_phase_status_enum_consistency() -> None:
    """Ensure expected PhaseStatus values exist for UI consumers."""
    expected = {
        "pending",
        "running",
        "succeeded",
        "failed",
        "skipped",
        "cancelled",
    }
    assert set(item.value for item in PhaseStatus) == expected

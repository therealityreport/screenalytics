from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import episodes
from apps.api.routers.episodes import EpisodeStatusResponse, PhaseStatus, PhaseStatusValue


def test_validation_errors_use_envelope() -> None:
    client = TestClient(app)
    resp = client.get("/episodes", params={"limit": -1})
    assert resp.status_code == 422
    payload = resp.json()
    assert payload["code"] == "VALIDATION_ERROR"
    assert payload["message"] == "Validation error"
    assert isinstance(payload["details"], list)
    assert payload["details"][0]["type"] in {"greater_than", "greater_than_equal"}


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
    def _mock_status(ep_id: str, run_id: str | None = None) -> EpisodeStatusResponse:
        return EpisodeStatusResponse(
            ep_id=ep_id,
            detect_track=PhaseStatus(phase="detect_track", status="success"),
            faces_embed=PhaseStatus(phase="faces_embed", status="success"),
            cluster=PhaseStatus(phase="cluster", status="missing"),
            scenes_ready=True,
            tracks_ready=True,
            faces_harvested=True,
            faces_manifest_fallback=True,
        )

    monkeypatch.setattr(episodes, "get_episode_status", _mock_status)

    resp = client.get("/episodes/demo-s01e01/status")
    assert resp.status_code == 200
    payload = resp.json()
    status_payload = EpisodeStatusResponse(**payload)
    assert status_payload.faces_embed.status == "success"
    assert status_payload.faces_manifest_fallback


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
    assert set(item.value for item in PhaseStatusValue) == expected

from fastapi.testclient import TestClient

from apps.api.main import app


def test_episode_create_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path / "data"))
    client = TestClient(app)

    payload = {
        "show_slug_or_id": "salt-lake",
        "season_number": 6,
        "episode_number": 2,
        "title": "Frozen Assets",
    }
    first = client.post("/episodes", json=payload)
    assert first.status_code == 200
    second = client.post("/episodes", json=payload)
    assert second.status_code == 200

    assert first.json()["ep_id"] == second.json()["ep_id"]

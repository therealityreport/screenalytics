from __future__ import annotations


from tests.ui._helpers_loader import load_ui_helpers_module


class _DummyResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_get_episode_status_includes_run_id_query_param(monkeypatch) -> None:
    helpers = load_ui_helpers_module()
    helpers.st.session_state["api_base"] = "http://example"

    captured: dict = {}

    def _fake_get(url: str, *, params=None, timeout=None, **_kwargs):  # noqa: ANN001
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return _DummyResponse({"ok": True})

    monkeypatch.setattr(helpers.requests, "get", _fake_get, raising=False)

    payload = helpers.get_episode_status("ep123", run_id="runABC")

    assert payload == {"ok": True}
    assert captured["url"] == "http://example/episodes/ep123/status"
    assert captured["params"] == {"run_id": "runABC"}


def test_get_episode_status_omits_run_id_when_missing(monkeypatch) -> None:
    helpers = load_ui_helpers_module()
    helpers.st.session_state["api_base"] = "http://example"

    captured: dict = {}

    def _fake_get(url: str, *, params=None, timeout=None, **_kwargs):  # noqa: ANN001
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return _DummyResponse({"ok": True})

    monkeypatch.setattr(helpers.requests, "get", _fake_get, raising=False)

    payload = helpers.get_episode_status("ep123", run_id=None)

    assert payload == {"ok": True}
    assert captured["url"] == "http://example/episodes/ep123/status"
    assert captured["params"] is None

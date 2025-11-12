from __future__ import annotations

from tests.ui._helpers_loader import load_ui_helpers_module


def test_resolve_thumb_handles_png_presign(monkeypatch):
    helpers = load_ui_helpers_module()
    helpers.st.session_state.clear()
    helpers.st.session_state["api_base"] = "http://api.test"

    calls: list[tuple[str, dict | None]] = []

    class _Response:
        def __init__(self, ok: bool = True, json_data: dict | None = None):
            self.ok = ok
            self._json = json_data or {}
            self.content = b""
            self.headers = {"content-type": "application/json"}

        def json(self):
            return self._json

    def _fake_get(url, params=None, timeout=None):
        calls.append((url, params))
        if url == "http://api.test/files/presign":
            return _Response(json_data={"url": "https://cdn.test/seed.png", "content_type": "image/png"})
        raise AssertionError(f"Unexpected request: {url}")

    monkeypatch.setattr(helpers.requests, "get", _fake_get, raising=False)

    src = "artifacts/facebank/rhobh/cast123/seed_d.png"
    resolved = helpers.resolve_thumb(src)
    assert resolved == "https://cdn.test/seed.png"
    assert calls and "mime" not in (calls[0][1] or {})

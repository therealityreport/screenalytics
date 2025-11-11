from __future__ import annotations

from tests.ui._helpers_loader import load_ui_helpers_module


class _FakeResponse:
    def __init__(self, lines):
        self.headers = {"Content-Type": "text/event-stream"}
        self._lines = lines
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode: bool = False):
        for line in self._lines:
            yield line if decode_unicode else line.encode()

    def close(self):
        self.closed = True


def _install_fake_post(helpers_module, fake_response):  # noqa: ANN001
    def _fake_post(url, json=None, headers=None, stream=False, timeout=None):  # noqa: ANN001
        return fake_response

    helpers_module.requests.post = _fake_post  # type: ignore[attr-defined]


def _install_fake_get(helpers_module, payload):  # noqa: ANN001
    class _FakeGetResponse:
        def __init__(self, body):
            self._body = body
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):  # noqa: D401
            return self._body

    def _fake_get(url, timeout=None):  # noqa: ANN001
        return _FakeGetResponse(payload)

    helpers_module.requests.get = _fake_get  # type: ignore[attr-defined]


def test_attempt_sse_run_returns_on_faces_step_done():
    helpers = load_ui_helpers_module()
    helpers.st.session_state["api_base"] = "http://testserver"
    fake_response = _FakeResponse(
        [
            "event: progress",
            'data: {"phase": "faces_embed", "frames_done": 5, "frames_total": 5, "step": "done", "summary": {"stage": "faces_embed"}}',
            "",
        ]
    )
    _install_fake_post(helpers, fake_response)

    updates = []
    summary, error_message, completed = helpers.attempt_sse_run(
        "/jobs/faces_embed",
        {"ep_id": "demo"},
        update_cb=lambda payload: updates.append(payload),
    )

    assert completed is True
    assert error_message is None
    assert summary and summary.get("stage") == "faces_embed"
    assert updates and updates[-1].get("step") == "done"


def test_attempt_sse_run_returns_on_cluster_step_done():
    helpers = load_ui_helpers_module()
    helpers.st.session_state["api_base"] = "http://testserver"
    fake_response = _FakeResponse(
        [
            "event: progress",
            'data: {"phase": "cluster", "frames_done": 10, "frames_total": 10, "step": "done", "summary": {"stage": "cluster"}}',
            "",
        ]
    )
    _install_fake_post(helpers, fake_response)

    summary, error_message, completed = helpers.attempt_sse_run(
        "/jobs/cluster",
        {"ep_id": "demo"},
        update_cb=lambda payload: None,
    )

    assert completed is True
    assert error_message is None
    assert summary and summary.get("stage") == "cluster"


def test_progress_ends_on_done_or_status_success():
    helpers = load_ui_helpers_module()
    helpers.st.session_state["api_base"] = "http://testserver"
    fake_response = _FakeResponse(
        [
            "event: progress",
            'data: {"phase": "faces_embed", "frames_done": 5, "frames_total": 10}',
            "",
        ]
    )
    _install_fake_post(helpers, fake_response)
    status_payload = {
        "faces_embed": {"status": "success", "faces": 10},
        "cluster": {"status": "missing"},
    }
    _install_fake_get(helpers, status_payload)

    summary, error_message, completed = helpers.attempt_sse_run(
        "/jobs/faces_embed",
        {"ep_id": "demo"},
        update_cb=lambda payload: None,
    )

    assert completed is True
    assert error_message is None
    assert summary and summary.get("stage") == "faces_embed"
    assert summary.get("faces") == 10

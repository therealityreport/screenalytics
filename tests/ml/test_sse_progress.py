from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Iterator, List

if "requests" not in sys.modules:
    mock_requests = types.SimpleNamespace(RequestException=Exception, HTTPError=Exception)
    sys.modules["requests"] = mock_requests
if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = types.SimpleNamespace()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELPERS_PATH = PROJECT_ROOT / "apps" / "workspace-ui" / "ui_helpers.py"
spec = importlib.util.spec_from_file_location("workspace_ui_helpers", HELPERS_PATH)
helpers = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(helpers)  # type: ignore[assignment]


class _FakeResponse:
    def __init__(self, lines: List[str]) -> None:
        self._lines = lines
        self.closed = False

    def iter_lines(self, decode_unicode: bool = True) -> Iterator[str]:
        for line in self._lines:
            yield line

    def close(self) -> None:
        self.closed = True


def test_sse_progress_eta_monotonic() -> None:
    lines = [
        "event: progress",
        'data: {"phase":"detect","frames_done":25,"frames_total":100,"secs_done":12,"secs_total":60}',
        "",
        "event: progress",
        'data: {"phase":"detect","frames_done":50,"frames_total":100,"secs_done":25,"secs_total":60}',
        "",
        "event: progress",
        'data: {"phase":"track","frames_done":100,"frames_total":100,"secs_done":55,"secs_total":60}',
        "",
        "event: done",
        'data: {"phase":"done","frames_done":100,"frames_total":100,"secs_done":60,"secs_total":60}',
        "",
    ]
    response = _FakeResponse(lines)
    prev_eta = None
    for event_name, payload in helpers.iter_sse_events(response):
        if event_name not in {"progress", "done"}:
            continue
        ratio = helpers.progress_ratio(payload)
        assert 0.0 <= ratio <= 1.0
        eta = helpers.eta_seconds(payload)
        if eta is not None and prev_eta is not None:
            assert eta <= prev_eta + 1e-6
        if eta is not None:
            prev_eta = eta
    assert response.closed

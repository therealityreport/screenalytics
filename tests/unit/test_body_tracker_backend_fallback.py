"""Unit tests for body tracker backend fallback diagnostics."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_body_tracker_records_iou_fallback_when_supervision_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.path.insert(0, str(PROJECT_ROOT))

    real_import = builtins.__import__

    def _fake_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
        if name == "supervision":
            raise ImportError("No module named 'supervision'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    from FEATURES.body_tracking.src.track_bodies import BodyTracker

    tracker = BodyTracker(tracker_type="bytetrack")
    tracker._init_tracker()

    assert tracker.tracker_backend_configured == "bytetrack"
    assert tracker.tracker_backend_actual == "iou_fallback"
    assert tracker.tracker_fallback_reason == "supervision_missing"


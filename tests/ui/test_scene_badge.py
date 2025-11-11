from __future__ import annotations

from tests.ui._helpers_loader import load_ui_helpers_module


def test_scene_badge_formats_label() -> None:
    helpers = load_ui_helpers_module()
    summary = {"scene_cuts": {"count": 42}}
    badge = helpers.scene_cuts_badge_text(summary)
    assert badge == "Scene cuts: 42"


def test_scene_badge_missing_returns_none() -> None:
    helpers = load_ui_helpers_module()
    assert helpers.scene_cuts_badge_text({}) is None

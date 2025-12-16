from __future__ import annotations

from tests.ui._helpers_loader import load_ui_helpers_module


def test_feature_status_registry_loads() -> None:
    helpers = load_ui_helpers_module()
    registry, error = helpers.load_feature_status_registry()
    assert error is None
    assert isinstance(registry, dict)
    assert "features" in registry
    assert "face_alignment" in registry["features"]

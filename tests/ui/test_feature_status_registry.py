from __future__ import annotations

from tests.ui._helpers_loader import load_ui_helpers_module


def test_feature_status_registry_loads() -> None:
    helpers = load_ui_helpers_module()
    registry, error = helpers.load_feature_status_registry()
    assert error is None
    assert isinstance(registry, dict)
    assert registry.get("schema_version") == 2
    assert "features" in registry
    assert "face_alignment" in registry["features"]

    face_alignment = registry["features"]["face_alignment"]
    assert "integrated_in_jobs" in face_alignment
    assert "integrated_in_autorun" in face_alignment
    assert "enabled_by_default" in face_alignment
    assert "how_to_enable" in face_alignment
    assert "evidence_paths" in face_alignment

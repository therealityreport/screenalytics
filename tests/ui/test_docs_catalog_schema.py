from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = PROJECT_ROOT / "docs" / "_meta" / "docs_catalog.json"


def test_docs_catalog_json_loads_and_has_required_keys() -> None:
    if not CATALOG_PATH.exists():
        pytest.skip("docs/_meta/docs_catalog.json not present on this branch yet.")

    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert isinstance(data.get("docs"), list)
    assert isinstance(data.get("features"), dict)

    for feature_id in ("face_alignment", "arcface_tensorrt", "vision_analytics"):
        assert feature_id in data["features"]


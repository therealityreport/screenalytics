"""Unit tests for screentime gain % formatting in the PDF run debug report."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_gain_pct_face_only_zero_returns_na() -> None:
    """If face_only==0 and gain>0, the report must not render a misleading 0.0%."""
    pytest.importorskip("reportlab")
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    from apps.api.services.run_export import _format_percent

    rendered = _format_percent(144.75, 0.0, na="N/A")
    assert rendered == "N/A"
    assert "0.0%" not in rendered


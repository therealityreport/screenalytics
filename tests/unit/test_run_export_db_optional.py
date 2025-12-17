"""Unit tests for optional DB configuration in the PDF run debug report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ascii_strings(data: bytes, *, min_len: int = 4) -> list[str]:
    """Extract printable ASCII runs (similar to the `strings` utility)."""
    out: list[str] = []
    buf: list[str] = []
    for b in data:
        if 32 <= b <= 126:
            buf.append(chr(b))
        else:
            if len(buf) >= min_len:
                out.append("".join(buf))
            buf = []
    if len(buf) >= min_len:
        out.append("".join(buf))
    return out


def test_pdf_db_not_configured_renders_na_instead_of_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("reportlab")

    sys.path.insert(0, str(PROJECT_ROOT))

    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SCREENALYTICS_PDF_NO_COMPRESSION", "1")
    monkeypatch.delenv("SCREENALYTICS_FAKE_DB", raising=False)
    monkeypatch.delenv("DB_URL", raising=False)

    from py_screenalytics import run_layout
    from apps.api.services.run_export import build_screentime_run_debug_pdf

    ep_id = "ep-test"
    run_id = "run123"
    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    (run_root / "tracks.jsonl").write_text(
        json.dumps({"track_id": 1, "first_ts": 0.0, "last_ts": 10.0}) + "\n",
        encoding="utf-8",
    )

    pdf_bytes, _name = build_screentime_run_debug_pdf(ep_id=ep_id, run_id=run_id)
    combined = "\n".join(_ascii_strings(pdf_bytes))

    assert "Not configured" in combined
    assert "DB_URL is not set" in combined
    assert "No (DB_URL is not set)" not in combined
    assert "No \\(DB_URL is not set\\)" not in combined
    assert "unavailable (DB not configured)" not in combined
    assert "unavailable \\(DB not configured\\)" in combined
    assert "unavailable (DB error)" not in combined
    assert "unavailable \\(DB error\\)" not in combined

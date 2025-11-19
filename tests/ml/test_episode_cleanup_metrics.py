"""Integration test for episode_cleanup with before/after validation.

Tests cleanup workflow and validates:
- cleanup_report.json exists and contains before/after stats
- Metrics improve in the right direction
- No dangling track_id or identity_id references
- Profile support works correctly

Note: Full cleanup test requires real episode with artifacts.
These tests use synthetic data to validate the workflow.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ML_TESTS = os.environ.get("RUN_ML_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_ML_TESTS,
    reason="set RUN_ML_TESTS=1 to run ML integration tests"
)


def _read_jsonl(path: Path) -> list[dict]:
    """Read JSONL file into list of dicts."""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@pytest.mark.timeout(600)
@pytest.mark.skip(reason="Cleanup requires full pipeline artifacts - implement after pipeline integration tests pass")
def test_episode_cleanup_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test cleanup workflow on a full episode (requires detect→track→faces→cluster).

    This test is currently skipped pending full fixture creation.
    TODO: Create fixture with deliberately messy tracks for cleanup testing.
    """

    try:
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    except ImportError:
        pass

    # TODO: Create full pipeline fixture
    # 1. Run detect/track with settings that create fragmented tracks
    # 2. Run faces_embed
    # 3. Run cluster
    # 4. Run cleanup with all actions
    # 5. Validate before/after metrics

    pytest.skip("Full cleanup test not yet implemented")


@pytest.mark.timeout(300)
def test_cleanup_report_schema_placeholder(tmp_path: Path) -> None:
    """Placeholder test for cleanup_report.json schema validation.

    TODO: Validate cleanup_report.json contains:
    - tracks_before / tracks_after
    - faces_before / faces_after
    - clusters_before / clusters_after
    - short_track_fraction_before/after
    - singleton_fraction_before/after
    """
    pytest.skip("Cleanup report schema test pending full implementation")


# Note: Episode cleanup tests require substantial setup.
# Priority is to get the other integration tests passing first,
# then we can build full cleanup test fixtures.

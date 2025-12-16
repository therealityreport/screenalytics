from __future__ import annotations

import json
from pathlib import Path

from tests.ui._helpers_loader import load_ui_helpers_module


def test_feature_status_bucket_rules() -> None:
    helpers = load_ui_helpers_module()

    # implemented_sandbox can never be COMPLETE, even if "integrated" flags are set.
    assert (
        helpers.classify_feature_status_bucket(
            "implemented_sandbox",
            integrated_in_jobs=["harvest"],
            integrated_in_autorun=True,
        )
        == "in_progress"
    )

    # implemented_production without integration flags must show as IN PROGRESS.
    assert (
        helpers.classify_feature_status_bucket(
            "implemented_production",
            integrated_in_jobs=[],
            integrated_in_autorun=False,
        )
        == "in_progress"
    )

    # implemented_production with integration can be COMPLETE.
    assert (
        helpers.classify_feature_status_bucket(
            "implemented_production",
            integrated_in_jobs=["harvest"],
            integrated_in_autorun=False,
        )
        == "complete"
    )

    # Feature-level "partial" remains IN PROGRESS even if a phase is complete.
    assert (
        helpers.classify_feature_status_bucket(
            "partial",
            integrated_in_jobs=["harvest"],
            integrated_in_autorun=True,
        )
        == "in_progress"
    )

    # Future work stays in TODO.
    assert (
        helpers.classify_feature_status_bucket(
            "not_started",
            integrated_in_jobs=[],
            integrated_in_autorun=False,
        )
        == "todo"
    )

    # heuristic_stub is IN PROGRESS.
    assert (
        helpers.classify_feature_status_bucket(
            "heuristic_stub",
            integrated_in_jobs=[],
            integrated_in_autorun=False,
        )
        == "in_progress"
    )

    # scaffold_only is IN PROGRESS.
    assert (
        helpers.classify_feature_status_bucket(
            "scaffold_only",
            integrated_in_jobs=[],
            integrated_in_autorun=False,
        )
        == "in_progress"
    )

    # draft is TODO.
    assert (
        helpers.classify_feature_status_bucket(
            "draft",
            integrated_in_jobs=[],
            integrated_in_autorun=False,
        )
        == "todo"
    )

    # outdated is TODO.
    assert (
        helpers.classify_feature_status_bucket(
            "outdated",
            integrated_in_jobs=[],
            integrated_in_autorun=False,
        )
        == "todo"
    )

    # future is TODO.
    assert (
        helpers.classify_feature_status_bucket(
            "future",
            integrated_in_jobs=[],
            integrated_in_autorun=False,
        )
        == "todo"
    )

    # complete without integration flags is IN PROGRESS (not COMPLETE).
    assert (
        helpers.classify_feature_status_bucket(
            "complete",
            integrated_in_jobs=[],
            integrated_in_autorun=False,
        )
        == "in_progress"
    )

    # complete with autorun but no jobs is COMPLETE.
    assert (
        helpers.classify_feature_status_bucket(
            "complete",
            integrated_in_jobs=[],
            integrated_in_autorun=True,
        )
        == "complete"
    )


def test_registry_sandbox_phases_never_complete() -> None:
    """Ensure no implemented_sandbox phase in feature_status.json will classify as COMPLETE."""
    helpers = load_ui_helpers_module()

    # Load the actual registry
    repo_root = Path(__file__).resolve().parents[2]
    registry_path = repo_root / "docs" / "_meta" / "feature_status.json"
    if not registry_path.exists():
        # Skip if registry doesn't exist in test environment
        return

    with open(registry_path) as f:
        registry = json.load(f)

    features = registry.get("features", {})
    sandbox_complete_violations = []

    for feature_id, feature_meta in features.items():
        if not isinstance(feature_meta, dict):
            continue

        phases = feature_meta.get("phases", {})
        for phase_id, phase_info in phases.items():
            if not isinstance(phase_info, dict):
                continue

            status = phase_info.get("status", "")
            jobs = phase_info.get("integrated_in_jobs", [])
            autorun = phase_info.get("integrated_in_autorun", False)

            bucket = helpers.classify_feature_status_bucket(
                status, integrated_in_jobs=jobs, integrated_in_autorun=autorun
            )

            # implemented_sandbox should NEVER be in complete bucket
            if status == "implemented_sandbox" and bucket == "complete":
                sandbox_complete_violations.append(f"{feature_id}.{phase_id}")

    assert not sandbox_complete_violations, (
        f"implemented_sandbox phases must not be COMPLETE: {sandbox_complete_violations}"
    )


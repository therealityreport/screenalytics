from __future__ import annotations

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


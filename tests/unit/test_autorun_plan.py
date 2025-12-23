"""Unit tests for the Episode Details Auto-Run stage plan helpers."""

from __future__ import annotations

from py_screenalytics.autorun_plan import autorun_complete, build_autorun_stage_plan


def test_stage_plan_requires_setup_stages_only() -> None:
    assert build_autorun_stage_plan() == [
        "detect",
        "faces",
        "cluster",
        "body_tracking",
        "track_fusion",
        "pdf",
    ]


def test_autorun_complete_requires_all_enabled_stages() -> None:
    stage_plan = build_autorun_stage_plan()
    assert autorun_complete(stage_plan, {}) is False
    assert (
        autorun_complete(
            stage_plan,
            {
                "detect": True,
                "faces": True,
                "cluster": True,
                "body_tracking": True,
                "track_fusion": True,
                "pdf": False,
            },
        )
        is False
    )
    assert (
        autorun_complete(
            stage_plan,
            {
                "detect": True,
                "faces": True,
                "cluster": True,
                "body_tracking": True,
                "track_fusion": True,
                "pdf": True,
            },
        )
        is True
    )

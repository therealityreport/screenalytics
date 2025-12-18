"""Unit tests for the Episode Details Auto-Run stage plan helpers."""

from __future__ import annotations

from py_screenalytics.autorun_plan import autorun_complete, build_autorun_stage_plan


def test_stage_plan_excludes_downstream_when_disabled() -> None:
    assert build_autorun_stage_plan(body_tracking_enabled=False, track_fusion_enabled=False) == [
        "detect",
        "faces",
        "cluster",
        "screentime",
        "pdf",
    ]


def test_stage_plan_includes_body_tracking_when_enabled() -> None:
    assert build_autorun_stage_plan(body_tracking_enabled=True, track_fusion_enabled=False) == [
        "detect",
        "faces",
        "cluster",
        "body_tracking",
        "screentime",
        "pdf",
    ]


def test_stage_plan_includes_track_fusion_when_enabled() -> None:
    assert build_autorun_stage_plan(body_tracking_enabled=True, track_fusion_enabled=True) == [
        "detect",
        "faces",
        "cluster",
        "body_tracking",
        "track_fusion",
        "screentime",
        "pdf",
    ]


def test_autorun_complete_requires_all_enabled_stages() -> None:
    stage_plan = build_autorun_stage_plan(body_tracking_enabled=True, track_fusion_enabled=True)
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
                "screentime": True,
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
                "screentime": True,
                "pdf": True,
            },
        )
        is True
    )


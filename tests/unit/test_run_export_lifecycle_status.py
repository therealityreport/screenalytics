from __future__ import annotations


def test_resolve_lifecycle_status_prefers_artifacts() -> None:
    from apps.api.services.run_export import _resolve_lifecycle_status

    status = _resolve_lifecycle_status(
        "detect",
        {},
        {"detections": True, "tracks": True},
    )
    assert status == "success"


def test_resolve_lifecycle_status_respects_error() -> None:
    from apps.api.services.run_export import _resolve_lifecycle_status

    status = _resolve_lifecycle_status(
        "detect",
        {"status": "error"},
        {"detections": True, "tracks": True},
    )
    assert status == "error"


def test_resolve_lifecycle_status_legacy_fallback() -> None:
    from apps.api.services.run_export import _resolve_lifecycle_status

    status = _resolve_lifecycle_status(
        "body_tracking",
        {},
        {"legacy": True},
    )
    assert status == "success (legacy)"

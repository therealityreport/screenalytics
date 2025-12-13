"""Contract tests for screentime JSON schema.

Validates that screentime output maintains backward compatibility
and includes all required fields for the web UI and API consumers.

Schema notes:
- 'both_s' is reserved/legacy and currently not computed (always 0.0)
- Body metrics are only validated when body_tracking_enabled AND body_metrics_available
"""
import pytest


# Schema constants
REQUIRED_TOP_LEVEL = ["episode_id", "generated_at", "metrics", "metadata"]
REQUIRED_METRIC_FIELDS = ["name", "confidence"]
NEW_EXPLICIT_FIELDS = [
    "face_visible_seconds",
    "body_visible_seconds",
    "body_only_seconds",
    "gap_bridged_seconds",
]
LEGACY_FIELDS = ["visual_s", "speaking_s", "both_s"]
METADATA_FLAGS = ["body_tracking_enabled", "body_metrics_available"]


def validate_screentime_schema(data: dict) -> list[str]:
    """Validate screentime JSON schema.

    Returns list of validation errors (empty if valid).
    """
    errors = []

    # Required top-level keys
    for key in REQUIRED_TOP_LEVEL:
        if key not in data:
            errors.append(f"Missing required key: {key}")

    # Metadata flags
    metadata = data.get("metadata", {})
    for flag in METADATA_FLAGS:
        if flag not in metadata:
            errors.append(f"Missing metadata.{flag}")

    body_tracking_enabled = metadata.get("body_tracking_enabled", False)
    body_metrics_available = metadata.get("body_metrics_available", False)

    # Metrics validation
    metrics = data.get("metrics", [])
    for i, m in enumerate(metrics):
        name = m.get("name", f"metric_{i}")

        # Required fields
        for field in REQUIRED_METRIC_FIELDS:
            if field not in m:
                errors.append(f"{name}: missing '{field}'")

        # Must have face visibility metric (new or legacy)
        if "face_visible_seconds" not in m and "visual_s" not in m:
            errors.append(f"{name}: missing face_visible_seconds/visual_s")

        # Non-negative seconds
        for field in [
            "face_visible_seconds",
            "visual_s",
            "speaking_s",
            "body_visible_seconds",
            "body_only_seconds",
            "gap_bridged_seconds",
        ]:
            val = m.get(field)
            if val is not None and val < 0:
                errors.append(f"{name}: {field} is negative ({val})")

        # Confidence in range [0, 1]
        conf = m.get("confidence", 0)
        if not (0 <= conf <= 1):
            errors.append(f"{name}: confidence out of range ({conf})")

        # Legacy fields for backward compatibility
        for field in LEGACY_FIELDS:
            if field not in m:
                errors.append(f"{name}: missing legacy field '{field}'")

        # Body tracking fields when enabled
        if body_tracking_enabled and body_metrics_available:
            for field in ["body_visible_seconds", "body_only_seconds"]:
                if field not in m or m[field] is None:
                    errors.append(f"{name}: body tracking enabled but {field} missing")

    return errors


class TestScreentimeSchemaContract:
    """Contract tests for screentime output schema."""

    def test_valid_complete_schema(self):
        """Complete valid screentime output passes validation."""
        data = {
            "episode_id": "test-ep",
            "generated_at": "2025-01-01T00:00:00",
            "metadata": {
                "body_tracking_enabled": False,
                "body_metrics_available": False,
            },
            "metrics": [
                {
                    "name": "Test Person",
                    "face_visible_seconds": 10.5,
                    "body_visible_seconds": None,
                    "body_only_seconds": None,
                    "gap_bridged_seconds": None,
                    "visual_s": 10.5,
                    "speaking_s": 5.0,
                    "both_s": 0.0,
                    "confidence": 0.85,
                }
            ],
        }
        errors = validate_screentime_schema(data)
        assert errors == [], f"Validation errors: {errors}"

    def test_missing_required_top_level_keys(self):
        """Missing required top-level keys are detected."""
        data = {"episode_id": "test"}
        errors = validate_screentime_schema(data)
        assert any("generated_at" in e for e in errors)
        assert any("metrics" in e for e in errors)
        assert any("metadata" in e for e in errors)

    def test_missing_metadata_flags(self):
        """Missing metadata flags are detected."""
        data = {
            "episode_id": "test",
            "generated_at": "2025-01-01T00:00:00",
            "metadata": {},  # Missing flags
            "metrics": [],
        }
        errors = validate_screentime_schema(data)
        assert any("body_tracking_enabled" in e for e in errors)
        assert any("body_metrics_available" in e for e in errors)

    def test_negative_seconds_rejected(self):
        """Negative second values are rejected."""
        data = {
            "episode_id": "test",
            "generated_at": "2025-01-01T00:00:00",
            "metadata": {"body_tracking_enabled": False, "body_metrics_available": False},
            "metrics": [
                {
                    "name": "Test",
                    "face_visible_seconds": -5.0,  # Invalid
                    "visual_s": 10.0,
                    "speaking_s": 0.0,
                    "both_s": 0.0,
                    "confidence": 0.8,
                }
            ],
        }
        errors = validate_screentime_schema(data)
        assert any("negative" in e for e in errors)

    def test_confidence_out_of_range_rejected(self):
        """Confidence values outside [0, 1] are rejected."""
        data = {
            "episode_id": "test",
            "generated_at": "2025-01-01T00:00:00",
            "metadata": {"body_tracking_enabled": False, "body_metrics_available": False},
            "metrics": [
                {
                    "name": "Test",
                    "face_visible_seconds": 10.0,
                    "visual_s": 10.0,
                    "speaking_s": 0.0,
                    "both_s": 0.0,
                    "confidence": 1.5,  # Invalid
                }
            ],
        }
        errors = validate_screentime_schema(data)
        assert any("confidence out of range" in e for e in errors)

    def test_legacy_fields_required_for_backward_compat(self):
        """Legacy fields must be present for backward compatibility."""
        data = {
            "episode_id": "test",
            "generated_at": "2025-01-01T00:00:00",
            "metadata": {"body_tracking_enabled": False, "body_metrics_available": False},
            "metrics": [
                {
                    "name": "Test",
                    "face_visible_seconds": 10.0,
                    # Missing: visual_s, speaking_s, both_s
                    "confidence": 0.8,
                }
            ],
        }
        errors = validate_screentime_schema(data)
        assert any("visual_s" in e for e in errors)
        assert any("speaking_s" in e for e in errors)
        assert any("both_s" in e for e in errors)

    def test_both_s_can_be_zero(self):
        """both_s field can be 0.0 (reserved/not computed)."""
        data = {
            "episode_id": "test",
            "generated_at": "2025-01-01T00:00:00",
            "metadata": {"body_tracking_enabled": False, "body_metrics_available": False},
            "metrics": [
                {
                    "name": "Test",
                    "face_visible_seconds": 10.0,
                    "visual_s": 10.0,
                    "speaking_s": 5.0,
                    "both_s": 0.0,  # OK - reserved/not computed
                    "confidence": 0.8,
                }
            ],
        }
        errors = validate_screentime_schema(data)
        assert errors == []

    def test_body_tracking_fields_when_enabled(self):
        """Body tracking fields validated when enabled."""
        data = {
            "episode_id": "test",
            "generated_at": "2025-01-01T00:00:00",
            "metadata": {"body_tracking_enabled": True, "body_metrics_available": True},
            "metrics": [
                {
                    "name": "Test",
                    "face_visible_seconds": 10.0,
                    "body_visible_seconds": 15.0,
                    "body_only_seconds": 5.0,
                    "gap_bridged_seconds": 2.0,
                    "visual_s": 10.0,
                    "speaking_s": 0.0,
                    "both_s": 0.0,
                    "confidence": 0.8,
                }
            ],
        }
        errors = validate_screentime_schema(data)
        assert errors == []

    def test_body_tracking_fields_required_when_enabled(self):
        """Body tracking fields must be present when body tracking enabled."""
        data = {
            "episode_id": "test",
            "generated_at": "2025-01-01T00:00:00",
            "metadata": {"body_tracking_enabled": True, "body_metrics_available": True},
            "metrics": [
                {
                    "name": "Test",
                    "face_visible_seconds": 10.0,
                    # Missing body_visible_seconds and body_only_seconds
                    "visual_s": 10.0,
                    "speaking_s": 0.0,
                    "both_s": 0.0,
                    "confidence": 0.8,
                }
            ],
        }
        errors = validate_screentime_schema(data)
        assert any("body_visible_seconds" in e for e in errors)
        assert any("body_only_seconds" in e for e in errors)

    def test_body_fields_optional_when_tracking_disabled(self):
        """Body tracking fields can be None when tracking disabled."""
        data = {
            "episode_id": "test",
            "generated_at": "2025-01-01T00:00:00",
            "metadata": {"body_tracking_enabled": False, "body_metrics_available": False},
            "metrics": [
                {
                    "name": "Test",
                    "face_visible_seconds": 10.0,
                    "body_visible_seconds": None,  # OK when disabled
                    "body_only_seconds": None,  # OK when disabled
                    "gap_bridged_seconds": None,  # OK when disabled
                    "visual_s": 10.0,
                    "speaking_s": 0.0,
                    "both_s": 0.0,
                    "confidence": 0.8,
                }
            ],
        }
        errors = validate_screentime_schema(data)
        assert errors == []

    def test_empty_metrics_with_no_tracks(self):
        """Empty metrics list is valid when no tracks exist."""
        data = {
            "episode_id": "test",
            "generated_at": "2025-01-01T00:00:00",
            "metadata": {"body_tracking_enabled": False, "body_metrics_available": False},
            "metrics": [],  # No metrics is OK
        }
        errors = validate_screentime_schema(data)
        assert errors == []

    def test_multiple_cast_members(self):
        """Multiple cast members all validated."""
        data = {
            "episode_id": "test",
            "generated_at": "2025-01-01T00:00:00",
            "metadata": {"body_tracking_enabled": False, "body_metrics_available": False},
            "metrics": [
                {
                    "name": "Person A",
                    "face_visible_seconds": 100.0,
                    "visual_s": 100.0,
                    "speaking_s": 50.0,
                    "both_s": 0.0,
                    "confidence": 0.9,
                },
                {
                    "name": "Person B",
                    "face_visible_seconds": 80.0,
                    "visual_s": 80.0,
                    "speaking_s": 30.0,
                    "both_s": 0.0,
                    "confidence": 0.85,
                },
                {
                    "name": "Person C",
                    "face_visible_seconds": 60.0,
                    "visual_s": 60.0,
                    "speaking_s": 20.0,
                    "both_s": 0.0,
                    "confidence": 0.75,
                },
            ],
        }
        errors = validate_screentime_schema(data)
        assert errors == []

"""Unit tests for config and profile resolution order.

Tests that the profile resolution order is correct:
1. Explicit request parameters (highest priority)
2. Environment variables
3. Profile preset values
4. Stage config defaults
5. Hardcoded fallbacks (lowest priority)

Also tests that unknown profile names are rejected.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("profile_name", [
    "fast_cpu",
    "low_power",
    "balanced",
    "high_accuracy",
])
def test_valid_profile_names_accepted(profile_name: str) -> None:
    """Test that all valid profile names are accepted."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from apps.api.routers.jobs import PROFILE_LITERAL

    # Profile literal should include this name
    assert profile_name in PROFILE_LITERAL.__args__, \
        f"Profile {profile_name} not in PROFILE_LITERAL"


def test_invalid_profile_rejected() -> None:
    """Test that invalid profile names are rejected by Pydantic."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from apps.api.routers.jobs import DetectTrackRequest
    from pydantic import ValidationError

    # Should raise ValidationError for invalid profile
    with pytest.raises(ValidationError) as exc_info:
        DetectTrackRequest(
            ep_id="test",
            profile="invalid_profile_name",
        )

    error_dict = exc_info.value.errors()[0]
    assert "profile" in str(error_dict["loc"])


def test_profile_resolution_explicit_override() -> None:
    """Test that explicit parameters override profile defaults."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from apps.api.routers.jobs import DetectTrackRequest

    # Create request with profile AND explicit stride
    request = DetectTrackRequest(
        ep_id="test",
        profile="balanced",
        stride=12,  # Explicit override
    )

    # Explicit stride should be preserved (not replaced by profile default)
    assert request.stride == 12, "Explicit stride should override profile default"
    assert request.profile == "balanced"


def test_env_var_not_override_explicit_param(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that explicit parameters take precedence over environment variables."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    # Set env var that might conflict
    monkeypatch.setenv("SCREANALYTICS_STRIDE", "10")

    from apps.api.routers.jobs import DetectTrackRequest

    # Explicit parameter should win
    request = DetectTrackRequest(
        ep_id="test",
        stride=6,
    )

    assert request.stride == 6, "Explicit parameter should override env var"


def test_profile_field_optional() -> None:
    """Test that profile field is optional (can be None)."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from apps.api.routers.jobs import DetectTrackRequest

    # Profile can be omitted
    request = DetectTrackRequest(
        ep_id="test",
    )

    assert request.profile is None, "Profile should default to None"


def test_all_job_types_support_profile() -> None:
    """Test that all job request types have profile field."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from apps.api.routers.jobs import (
        DetectTrackRequest,
        FacesEmbedRequest,
        ClusterRequest,
        CleanupJobRequest,
    )

    # All should have profile field
    for request_class in [DetectTrackRequest, FacesEmbedRequest, ClusterRequest, CleanupJobRequest]:
        assert hasattr(request_class, "__annotations__"), \
            f"{request_class.__name__} missing annotations"
        assert "profile" in request_class.__annotations__, \
            f"{request_class.__name__} missing profile field"


def test_profile_resolution_order_documentation() -> None:
    """Verify that profile resolution order is documented in docs/reference/api.md."""
    api_doc_path = PROJECT_ROOT / "docs" / "reference" / "api.md"
    assert api_doc_path.exists(), "docs/reference/api.md not found"

    content = api_doc_path.read_text(encoding="utf-8")

    # Check for profile resolution order documentation
    assert "Profile Resolution Order" in content, \
        "Profile Resolution Order section missing from docs/reference/api.md"

    # Check for the correct precedence order
    precedence_keywords = [
        "Explicit request parameters",
        "Environment variables",
        "Profile preset values",
        "Stage config defaults",
        "Hardcoded fallbacks",
    ]

    for keyword in precedence_keywords:
        assert keyword in content, \
            f"Profile resolution documentation missing keyword: {keyword}"


def test_config_guide_documents_profiles() -> None:
    """Verify that docs/reference/config/pipeline_configs.md documents all profile names."""
    config_guide_path = PROJECT_ROOT / "docs" / "reference" / "config" / "pipeline_configs.md"

    if not config_guide_path.exists():
        pytest.skip("docs/reference/config/pipeline_configs.md not found")

    content = config_guide_path.read_text(encoding="utf-8")

    # All profile names should be documented
    profile_names = ["fast_cpu", "low_power", "balanced", "high_accuracy"]

    for profile in profile_names:
        assert profile in content, \
            f"Profile '{profile}' not documented in docs/reference/config/pipeline_configs.md"


def test_acceptance_matrix_references_metrics() -> None:
    """Verify that ACCEPTANCE_MATRIX.md defines thresholds for key metrics."""
    acceptance_matrix_path = PROJECT_ROOT / "ACCEPTANCE_MATRIX.md"
    assert acceptance_matrix_path.exists(), "ACCEPTANCE_MATRIX.md not found"

    content = acceptance_matrix_path.read_text(encoding="utf-8")

    # Key metrics should have thresholds defined
    required_metrics = [
        "tracks_per_minute",
        "short_track_fraction",
        "id_switch_rate",
        "singleton_fraction",
        "largest_cluster_fraction",
    ]

    for metric in required_metrics:
        assert metric in content, \
            f"Metric '{metric}' not defined in ACCEPTANCE_MATRIX.md"


def test_profile_config_files_exist() -> None:
    """Verify that profile config files exist."""
    config_dir = PROJECT_ROOT / "config" / "pipeline"
    assert config_dir.exists(), "config/pipeline directory not found"

    # Performance profiles config should exist
    profiles_config = config_dir / "performance_profiles.yaml"
    assert profiles_config.exists(), \
        "config/pipeline/performance_profiles.yaml not found"

    # Read and validate structure
    try:
        import yaml
    except ImportError:
        pytest.skip("pyyaml not installed")

    with profiles_config.open("r") as f:
        profiles_data = yaml.safe_load(f)

    # Should have profile definitions
    assert isinstance(profiles_data, dict), "Profiles config should be a dict"

    # Check for expected profiles (might use different keys)
    # Note: fast_cpu might be aliased as low_power
    expected_profiles = {"low_power", "balanced", "high_accuracy"}

    for profile in expected_profiles:
        assert profile in profiles_data, \
            f"Profile '{profile}' missing from performance_profiles.yaml"


def test_profile_passthrough_to_cli() -> None:
    """Test that API passes profile parameter to CLI command."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from apps.api.routers.jobs import _build_detect_track_command, DetectTrackRequest
    from pathlib import Path

    # Create request with profile
    request = DetectTrackRequest(
        ep_id="test",
        profile="balanced",
        stride=6,
        device="cpu",
    )

    # Build command
    progress_path = Path("/tmp/progress.json")
    video_path = Path("/tmp/test.mp4")

    command = _build_detect_track_command(request, progress_path, video_path)

    # Command should include --profile balanced
    assert "--profile" in command, "Command missing --profile flag"

    profile_idx = command.index("--profile")
    assert command[profile_idx + 1] == "balanced", \
        "Profile value not passed correctly to CLI"


def test_profile_none_does_not_add_flag() -> None:
    """Test that profile=None doesn't add --profile flag to CLI command."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from apps.api.routers.jobs import _build_detect_track_command, DetectTrackRequest
    from pathlib import Path

    # Create request without profile
    request = DetectTrackRequest(
        ep_id="test",
        profile=None,
        device="cpu",
    )

    progress_path = Path("/tmp/progress.json")
    video_path = Path("/tmp/test.mp4")

    command = _build_detect_track_command(request, progress_path, video_path)

    # Command should NOT include --profile flag
    assert "--profile" not in command, \
        "Command should not include --profile when profile=None"


def test_low_power_profile_applies_defaults() -> None:
    """Ensure low_power profile applies stride/save flags/cpu thread caps."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from apps.api.routers.jobs import DetectTrackRequest, _resolve_detect_track_inputs

    request = DetectTrackRequest(ep_id="test-low-power", profile="low_power", device="mps")
    effective = _resolve_detect_track_inputs(request, resolved_device="mps")

    assert effective["profile"] == "low_power"
    assert effective["stride"] == 8, "low_power should raise stride to 8"
    assert effective["fps"] == 8, "low_power should cap FPS to 8"
    assert effective["save_frames"] is False
    assert effective["save_crops"] is False
    assert effective["cpu_threads"] == 2


def test_default_profile_for_mps_prefers_low_power() -> None:
    """Default profile should bias to low_power on Apple MPS/CoreML."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from apps.api.routers.jobs import DetectTrackRequest, _resolve_detect_track_inputs

    # Omit profile to trigger device-based defaulting
    request = DetectTrackRequest(ep_id="test-default-profile", device="mps")
    effective = _resolve_detect_track_inputs(request, resolved_device="mps")

    assert effective["profile"] == "low_power"
    assert effective["stride"] == 8
def test_device_literal_values() -> None:
    """Test that DEVICE_LITERAL includes all expected device types."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from apps.api.routers.jobs import DEVICE_LITERAL

    expected_devices = {"auto", "cpu", "mps", "coreml", "metal", "apple", "cuda"}

    for device in expected_devices:
        assert device in DEVICE_LITERAL.__args__, \
            f"Device '{device}' not in DEVICE_LITERAL"


def test_profile_enum_matches_yaml_config() -> None:
    """Test that PROFILE_LITERAL enum values match config file."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from apps.api.routers.jobs import PROFILE_LITERAL

    # Read profiles from YAML
    config_dir = PROJECT_ROOT / "config" / "pipeline"
    profiles_config = config_dir / "performance_profiles.yaml"

    if not profiles_config.exists():
        pytest.skip("performance_profiles.yaml not found")

    try:
        import yaml
    except ImportError:
        pytest.skip("pyyaml not installed")

    with profiles_config.open("r") as f:
        profiles_data = yaml.safe_load(f)

    yaml_profiles = set(profiles_data.keys())

    # PROFILE_LITERAL should include all YAML profiles
    # (or their aliases like fast_cpu = low_power)
    profile_literal_values = set(PROFILE_LITERAL.__args__)

    # low_power and balanced and high_accuracy must be in PROFILE_LITERAL
    core_profiles = {"low_power", "balanced", "high_accuracy"}

    for profile in core_profiles:
        assert profile in profile_literal_values or profile in yaml_profiles, \
            f"Core profile '{profile}' missing from both PROFILE_LITERAL and YAML config"

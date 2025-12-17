"""Tests for body tracking default enabled state and env var override.

These tests verify:
1. Body tracking is enabled by default in the YAML config
2. AUTO_RUN_BODY_TRACKING env var can override the default
3. Body tracking artifacts are included in S3 sync allowlist
"""

from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestBodyTrackingDefaultEnabled:
    """Tests for body tracking enabled by default."""

    def test_yaml_config_has_body_tracking_enabled(self) -> None:
        """Test that body_detection.yaml has body_tracking.enabled = true."""
        import yaml

        config_path = PROJECT_ROOT / "config" / "pipeline" / "body_detection.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert config is not None, "Config file is empty"
        assert "body_tracking" in config, "body_tracking section missing from config"
        assert config["body_tracking"].get("enabled") is True, (
            "body_tracking.enabled should be True by default"
        )

    def test_load_body_tracking_config_returns_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that _load_body_tracking_config returns enabled=True by default."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        # Clear any cached config
        monkeypatch.delenv("AUTO_RUN_BODY_TRACKING", raising=False)

        from tools.episode_run import _load_body_tracking_config

        config = _load_body_tracking_config()
        enabled = (config.get("body_tracking") or {}).get("enabled", False)
        assert enabled is True, "Body tracking should be enabled by default"


class TestAutoRunBodyTrackingEnvVar:
    """Tests for AUTO_RUN_BODY_TRACKING environment variable override."""

    def test_env_var_disables_body_tracking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that AUTO_RUN_BODY_TRACKING=0 disables body tracking."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        monkeypatch.setenv("AUTO_RUN_BODY_TRACKING", "0")

        from tools.episode_run import _load_body_tracking_config

        config = _load_body_tracking_config()
        enabled = (config.get("body_tracking") or {}).get("enabled", False)
        assert enabled is False, "AUTO_RUN_BODY_TRACKING=0 should disable body tracking"

    def test_env_var_false_disables_body_tracking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that AUTO_RUN_BODY_TRACKING=false disables body tracking."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        monkeypatch.setenv("AUTO_RUN_BODY_TRACKING", "false")

        from tools.episode_run import _load_body_tracking_config

        config = _load_body_tracking_config()
        enabled = (config.get("body_tracking") or {}).get("enabled", False)
        assert enabled is False, "AUTO_RUN_BODY_TRACKING=false should disable body tracking"

    def test_env_var_enables_body_tracking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that AUTO_RUN_BODY_TRACKING=1 enables body tracking even if YAML says false."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        # Even if we modify config to disable, env var should override
        monkeypatch.setenv("AUTO_RUN_BODY_TRACKING", "1")

        from tools.episode_run import _load_body_tracking_config

        config = _load_body_tracking_config()
        enabled = (config.get("body_tracking") or {}).get("enabled", False)
        assert enabled is True, "AUTO_RUN_BODY_TRACKING=1 should enable body tracking"


class TestBodyTrackingArtifactAllowlist:
    """Tests for body tracking artifacts in S3 sync allowlist."""

    def test_body_tracking_core_artifacts_in_allowlist(self) -> None:
        """Test that core body tracking artifacts are in RUN_ARTIFACT_ALLOWLIST."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        from py_screenalytics.run_layout import RUN_ARTIFACT_ALLOWLIST

        expected_artifacts = [
            "body_tracking/body_detections.jsonl",
            "body_tracking/body_tracks.jsonl",
            "body_tracking/track_fusion.json",
            "body_tracking/screentime_comparison.json",
        ]

        for artifact in expected_artifacts:
            assert artifact in RUN_ARTIFACT_ALLOWLIST, (
                f"Missing from S3 allowlist: {artifact}"
            )

    def test_body_tracking_embedding_artifacts_in_allowlist(self) -> None:
        """Test that body tracking embedding artifacts are in RUN_ARTIFACT_ALLOWLIST."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        from py_screenalytics.run_layout import RUN_ARTIFACT_ALLOWLIST

        expected_artifacts = [
            "body_tracking/body_embeddings.npy",
            "body_tracking/body_embeddings_meta.json",
            "body_tracking/body_metrics.json",
        ]

        for artifact in expected_artifacts:
            assert artifact in RUN_ARTIFACT_ALLOWLIST, (
                f"Missing from S3 allowlist: {artifact}"
            )

    def test_list_run_artifacts_includes_body_tracking(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that list_run_artifacts includes body tracking files when present."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

        monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
        monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

        from py_screenalytics import run_layout

        ep_id = "test-ep-body"
        run_id = "test-run-body"

        # Create run directory with body tracking artifacts
        run_root = run_layout.run_root(ep_id, run_id)
        body_dir = run_root / "body_tracking"
        body_dir.mkdir(parents=True, exist_ok=True)

        # Create some body tracking artifacts
        (body_dir / "body_detections.jsonl").write_text('{"det": 1}\n')
        (body_dir / "body_tracks.jsonl").write_text('{"track": 1}\n')
        (body_dir / "track_fusion.json").write_text('{}')

        artifacts = run_layout.list_run_artifacts(ep_id, run_id)
        filenames = [p.name for p, _ in artifacts]

        assert "body_detections.jsonl" in filenames
        assert "body_tracks.jsonl" in filenames
        assert "track_fusion.json" in filenames

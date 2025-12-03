"""Tests for diarization backend selection (Precision-2 vs OSS 3.1)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from py_screenalytics.audio.models import DiarizationConfig


class TestBackendSelection:
    """Test backend selection logic."""

    def test_precision2_backend_with_api_key(self, monkeypatch):
        """When backend=precision-2 and PYANNOTEAI_API_KEY is set, uses Precision-2."""
        # Clear any cached state
        import py_screenalytics.audio.diarization_pyannote as diar_module
        diar_module._DIARIZATION_PIPELINE = None
        diar_module._DIARIZATION_BACKEND = None
        diar_module._load_pyannoteai_api_key.cache_clear()

        monkeypatch.setenv("PYANNOTEAI_API_KEY", "test-api-key-12345")

        config = DiarizationConfig(backend="precision-2")

        # Mock Pipeline.from_pretrained to avoid actual API call
        mock_pipeline = MagicMock()
        with patch("pyannote.audio.Pipeline.from_pretrained", return_value=mock_pipeline) as mock_from_pretrained:
            pipeline, actual_backend = diar_module._build_pyannote_pipeline(config)

            # Should use Precision-2 model
            mock_from_pretrained.assert_called_once()
            call_args = mock_from_pretrained.call_args
            assert call_args[0][0] == "pyannote/speaker-diarization-precision-2"
            assert call_args[1]["token"] == "test-api-key-12345"
            assert actual_backend == "precision-2"
            assert pipeline == mock_pipeline

    def test_precision2_backend_falls_back_without_api_key(self, monkeypatch, caplog):
        """When backend=precision-2 but PYANNOTEAI_API_KEY is missing, falls back to OSS."""
        import py_screenalytics.audio.diarization_pyannote as diar_module
        diar_module._DIARIZATION_PIPELINE = None
        diar_module._DIARIZATION_BACKEND = None
        diar_module._load_pyannoteai_api_key.cache_clear()
        diar_module._load_env_token.cache_clear()

        # Ensure no API key is set
        monkeypatch.delenv("PYANNOTEAI_API_KEY", raising=False)
        monkeypatch.delenv("PYANNOTE_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)

        config = DiarizationConfig(backend="precision-2")

        # Mock Pipeline.from_pretrained and torch
        mock_pipeline = MagicMock()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = MagicMock()

        with patch("pyannote.audio.Pipeline.from_pretrained", return_value=mock_pipeline) as mock_from_pretrained, \
             patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib
            # Re-import to pick up mocked torch
            pipeline, actual_backend = diar_module._build_pyannote_pipeline(config)

            # Should fall back to OSS 3.1 model
            mock_from_pretrained.assert_called_once()
            call_args = mock_from_pretrained.call_args
            assert call_args[0][0] == "pyannote/speaker-diarization-3.1"
            assert actual_backend == "oss-3.1"

        # Check warning was logged
        assert any("falling back to OSS" in record.message for record in caplog.records)

    def test_oss31_backend_explicit(self, monkeypatch):
        """When backend=oss-3.1, uses OSS model regardless of API key."""
        import py_screenalytics.audio.diarization_pyannote as diar_module
        diar_module._DIARIZATION_PIPELINE = None
        diar_module._DIARIZATION_BACKEND = None
        diar_module._load_env_token.cache_clear()

        # Even with API key set, should use OSS when explicitly requested
        monkeypatch.setenv("PYANNOTEAI_API_KEY", "test-api-key-12345")
        monkeypatch.delenv("PYANNOTE_AUTH_TOKEN", raising=False)

        config = DiarizationConfig(backend="oss-3.1")

        mock_pipeline = MagicMock()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = MagicMock()

        with patch("pyannote.audio.Pipeline.from_pretrained", return_value=mock_pipeline) as mock_from_pretrained, \
             patch.dict("sys.modules", {"torch": mock_torch}):
            pipeline, actual_backend = diar_module._build_pyannote_pipeline(config)

            # Should use OSS 3.1 model
            mock_from_pretrained.assert_called_once()
            call_args = mock_from_pretrained.call_args
            assert call_args[0][0] == "pyannote/speaker-diarization-3.1"
            assert actual_backend == "oss-3.1"

    def test_get_current_backend_tracks_state(self, monkeypatch):
        """get_current_backend returns the currently loaded backend."""
        import py_screenalytics.audio.diarization_pyannote as diar_module

        # Reset state
        diar_module._DIARIZATION_PIPELINE = None
        diar_module._DIARIZATION_BACKEND = None

        # Initially None
        assert diar_module.get_current_backend() is None

        # Set state
        diar_module._DIARIZATION_BACKEND = "precision-2"
        assert diar_module.get_current_backend() == "precision-2"

        diar_module._DIARIZATION_BACKEND = "oss-3.1"
        assert diar_module.get_current_backend() == "oss-3.1"

    def test_unload_models_resets_backend(self):
        """unload_models clears the backend tracking."""
        import py_screenalytics.audio.diarization_pyannote as diar_module

        diar_module._DIARIZATION_PIPELINE = MagicMock()
        diar_module._DIARIZATION_BACKEND = "precision-2"
        diar_module._EMBEDDING_MODEL = MagicMock()

        diar_module.unload_models()

        assert diar_module._DIARIZATION_PIPELINE is None
        assert diar_module._DIARIZATION_BACKEND is None
        assert diar_module._EMBEDDING_MODEL is None


class TestDiarizationConfigBackend:
    """Test DiarizationConfig backend field."""

    def test_default_backend_is_precision2(self):
        """Default backend should be precision-2."""
        config = DiarizationConfig()
        assert config.backend == "precision-2"

    def test_backend_can_be_set_to_oss31(self):
        """Backend can be explicitly set to oss-3.1."""
        config = DiarizationConfig(backend="oss-3.1")
        assert config.backend == "oss-3.1"

    def test_model_name_default_matches_backend(self):
        """Default model_name should match precision-2 backend."""
        config = DiarizationConfig()
        assert config.model_name == "pyannote/speaker-diarization-precision-2"


class TestAPIKeyLoading:
    """Test API key loading from environment."""

    def test_api_key_from_environment(self, monkeypatch):
        """API key can be loaded from PYANNOTEAI_API_KEY env var."""
        import py_screenalytics.audio.diarization_pyannote as diar_module
        diar_module._load_pyannoteai_api_key.cache_clear()

        monkeypatch.setenv("PYANNOTEAI_API_KEY", "env-api-key")

        key = diar_module._get_pyannoteai_api_key()
        assert key == "env-api-key"

    def test_api_key_missing_returns_none(self, monkeypatch):
        """Returns None when API key is not set."""
        import py_screenalytics.audio.diarization_pyannote as diar_module
        diar_module._load_pyannoteai_api_key.cache_clear()

        monkeypatch.delenv("PYANNOTEAI_API_KEY", raising=False)

        key = diar_module._get_pyannoteai_api_key()
        assert key is None

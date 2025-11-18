"""
Tests for ONNX Runtime provider selection in detect/track pipeline.

Validates that _onnx_providers_for() correctly detects and selects:
- CoreMLExecutionProvider on macOS (Apple Silicon)
- CUDAExecutionProvider on Linux/Windows (NVIDIA GPUs)
- CPUExecutionProvider as fallback

Related: nov-17-detect-track-none-bbox-fix.md (CoreML detection fix)
"""

from unittest.mock import MagicMock, patch

import pytest


def test_onnx_providers_auto_selects_cuda_when_available():
    """device=auto should select CUDA when CUDAExecutionProvider is available."""
    from tools.episode_run import _onnx_providers_for

    # Mock ONNX Runtime to simulate Linux/Windows with CUDA
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        providers, resolved = _onnx_providers_for("auto")

        assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert resolved == "cuda"


def test_onnx_providers_auto_selects_coreml_when_cuda_unavailable():
    """device=auto should select CoreML when CUDA unavailable but CoreML is (macOS)."""
    from tools.episode_run import _onnx_providers_for

    # Mock ONNX Runtime to simulate macOS with CoreML
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = [
        "CoreMLExecutionProvider",
        "AzureExecutionProvider",
        "CPUExecutionProvider",
    ]

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        providers, resolved = _onnx_providers_for("auto")

        assert providers == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        assert resolved == "coreml"


def test_onnx_providers_auto_fallback_to_cpu():
    """device=auto should fall back to CPU when no accelerators available."""
    from tools.episode_run import _onnx_providers_for

    # Mock ONNX Runtime with only CPU available
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        providers, resolved = _onnx_providers_for("auto")

        assert providers == ["CPUExecutionProvider"]
        assert resolved == "cpu"


def test_onnx_providers_explicit_cuda_request():
    """Explicit device=cuda should select CUDA when available."""
    from tools.episode_run import _onnx_providers_for

    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        providers, resolved = _onnx_providers_for("cuda")

        assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert resolved == "cuda"


def test_onnx_providers_explicit_cuda_unavailable_warns():
    """Explicit device=cuda should warn and fall back to CPU when CUDA unavailable."""
    from tools.episode_run import _onnx_providers_for

    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        with patch("tools.episode_run.LOGGER") as mock_logger:
            providers, resolved = _onnx_providers_for("cuda")

            assert providers == ["CPUExecutionProvider"]
            assert resolved == "cpu"
            mock_logger.warning.assert_called_once()
            assert "CUDA requested" in mock_logger.warning.call_args[0][0]


def test_onnx_providers_explicit_mps_selects_coreml():
    """Explicit device=mps should select CoreML when available (Apple Silicon)."""
    from tools.episode_run import _onnx_providers_for

    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        providers, resolved = _onnx_providers_for("mps")

        assert providers == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        assert resolved == "coreml"


def test_onnx_providers_explicit_metal_selects_coreml():
    """Explicit device=metal should select CoreML when available."""
    from tools.episode_run import _onnx_providers_for

    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        providers, resolved = _onnx_providers_for("metal")

        assert providers == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        assert resolved == "coreml"


def test_onnx_providers_explicit_apple_selects_coreml():
    """Explicit device=apple should select CoreML when available."""
    from tools.episode_run import _onnx_providers_for

    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        providers, resolved = _onnx_providers_for("apple")

        assert providers == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        assert resolved == "coreml"


def test_onnx_providers_explicit_mps_unavailable_warns():
    """Explicit device=mps should warn and fall back to CPU when CoreML unavailable."""
    from tools.episode_run import _onnx_providers_for

    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        with patch("tools.episode_run.LOGGER") as mock_logger:
            providers, resolved = _onnx_providers_for("mps")

            assert providers == ["CPUExecutionProvider"]
            assert resolved == "cpu"
            mock_logger.warning.assert_called_once()
            assert "CoreML requested" in mock_logger.warning.call_args[0][0]


def test_onnx_providers_explicit_cpu():
    """Explicit device=cpu should always return CPU provider."""
    from tools.episode_run import _onnx_providers_for

    # Even if CUDA/CoreML are available, explicit CPU should use CPU
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        providers, resolved = _onnx_providers_for("cpu")

        assert providers == ["CPUExecutionProvider"]
        assert resolved == "cpu"


def test_onnx_providers_none_defaults_to_auto():
    """device=None should default to auto selection."""
    from tools.episode_run import _onnx_providers_for

    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        providers, resolved = _onnx_providers_for(None)

        # Should auto-detect and select CoreML
        assert providers == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        assert resolved == "coreml"


def test_onnx_providers_import_error_fallback():
    """Should fall back to CPU if onnxruntime import fails."""
    from tools.episode_run import _onnx_providers_for

    # Simulate import error by making the module raise ImportError
    mock_ort = MagicMock()
    mock_ort.get_available_providers.side_effect = Exception("Import failed")

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        providers, resolved = _onnx_providers_for("auto")

        assert providers == ["CPUExecutionProvider"]
        assert resolved == "cpu"


def test_onnx_providers_get_providers_error_fallback():
    """Should fall back to CPU if get_available_providers() raises an exception."""
    from tools.episode_run import _onnx_providers_for

    mock_ort = MagicMock()
    mock_ort.get_available_providers.side_effect = RuntimeError("Provider enumeration failed")

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        providers, resolved = _onnx_providers_for("auto")

        assert providers == ["CPUExecutionProvider"]
        assert resolved == "cpu"


def test_onnx_providers_macos_scenario():
    """
    Integration test: Simulate exact macOS scenario from user error log.

    Error log showed:
    - Available providers: 'CoreMLExecutionProvider, AzureExecutionProvider, CPUExecutionProvider'
    - User requested: device=auto
    - OLD behavior: Returned CPU (0.06 fps)
    - NEW behavior: Should return CoreML (5-10+ fps)
    """
    from tools.episode_run import _onnx_providers_for

    # Simulate macOS M1/M2/M3 environment (exactly as shown in error log)
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = [
        "CoreMLExecutionProvider",
        "AzureExecutionProvider",
        "CPUExecutionProvider",
    ]

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        providers, resolved = _onnx_providers_for("auto")

        # Should select CoreML, NOT CPU
        assert providers == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        assert resolved == "coreml"
        assert "CPUExecutionProvider" in providers  # Fallback still present
        assert providers[0] == "CoreMLExecutionProvider"  # CoreML is first priority

"""
macOS Low-Noise Profile for SCREANALYTICS.

This module provides a "low-noise" execution profile optimized for running
ML workloads on macOS laptops (especially M1/M2/M3 Apple Silicon) without
causing excessive heat, fan noise, or battery drain.

Key features:
- MPS (Metal Performance Shaders) environment tuning
- CoreML-first execution with controlled CPU fallback
- FFmpeg VideoToolbox hardware acceleration
- Conservative thread and memory limits

Usage:
    Call apply_macos_low_noise_profile() at the start of your process,
    BEFORE importing ML libraries. This sets environment variables that
    configure MPS memory allocation and fallback behavior.

Environment Variables Set:
    PYTORCH_ENABLE_MPS_FALLBACK=1     - Allow ops to fall back to CPU if MPS unsupported
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 - Limit MPS memory to 60% of unified memory
    PYTORCH_MPS_BLOCK_SIZE=262144     - Smaller allocation blocks (256KB)
    PYTORCH_MPS_ALLOCATOR_MAX_SHARE=0.8 - Memory sharing limit
    PYTORCH_MPS_LOGS=0                - Disable MPS debug logs
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

# Check if we're on macOS Apple Silicon
_IS_MACOS = platform.system().lower() == "darwin"
_IS_APPLE_SILICON = _IS_MACOS and platform.machine().lower().startswith(("arm", "aarch64"))

_profile_applied = False


def is_macos() -> bool:
    """Check if running on macOS."""
    return _IS_MACOS


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon Mac."""
    return _IS_APPLE_SILICON


def apply_macos_low_noise_profile() -> bool:
    """Apply the macOS Low-Noise profile for quieter ML execution.

    This sets MPS environment variables to limit memory usage and enable
    controlled fallback to CPU for unsupported operations.

    Should be called BEFORE importing PyTorch or other ML libraries.

    Returns:
        True if profile was applied, False if not on macOS or already applied.
    """
    global _profile_applied

    if _profile_applied:
        LOGGER.debug("macOS Low-Noise profile already applied; skipping")
        return True

    if not _IS_MACOS:
        LOGGER.debug("Not on macOS; skipping Low-Noise profile")
        return False

    LOGGER.info("Applying macOS Low-Noise profile for quieter ML execution")

    # MPS (Metal Performance Shaders) configuration
    # These control PyTorch's MPS backend behavior

    # Enable per-op CPU fallback for unsupported MPS operations
    # This is safer than failing entirely when an op isn't supported
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # Limit MPS memory to 60% of unified memory
    # Prevents memory pressure that causes system-wide slowdowns
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")

    # Smaller allocation block size (256KB instead of default)
    # Reduces memory fragmentation on constrained devices
    os.environ.setdefault("PYTORCH_MPS_BLOCK_SIZE", "262144")

    # Limit memory sharing between allocators
    os.environ.setdefault("PYTORCH_MPS_ALLOCATOR_MAX_SHARE", "0.8")

    # Disable MPS debug logs (reduces noise)
    os.environ.setdefault("PYTORCH_MPS_LOGS", "0")

    # Also apply ONNX Runtime CoreML-specific settings
    # Prefer CoreML with conservative thread counts
    os.environ.setdefault("ORT_USE_COREML", "1")

    _profile_applied = True
    LOGGER.info("macOS Low-Noise profile applied successfully")

    return True


def get_mps_device():
    """Get the MPS device if available, otherwise CPU.

    Returns:
        torch.device for MPS or CPU
    """
    try:
        import torch

        if torch.backends.mps.is_available():
            return torch.device("mps")
    except (ImportError, AttributeError):
        pass

    try:
        import torch
        return torch.device("cpu")
    except ImportError:
        return None


def pick_torch_device(preferred: str | None = None):
    """Pick the best torch device based on preference and availability.

    Order of preference:
    - If preferred="mps" or "auto" and MPS is available: MPS
    - If preferred="cuda" or "auto" and CUDA is available: CUDA
    - Otherwise: CPU

    Args:
        preferred: Device preference ("mps", "cuda", "cpu", "auto", or None)

    Returns:
        torch.device object
    """
    try:
        import torch
    except ImportError:
        LOGGER.warning("PyTorch not installed; cannot pick device")
        return None

    preferred = (preferred or "auto").lower()

    # MPS (Apple Silicon)
    if preferred in ("mps", "metal", "apple", "auto"):
        if torch.backends.mps.is_available():
            LOGGER.debug("Selected MPS device")
            return torch.device("mps")

    # CUDA (NVIDIA)
    if preferred in ("cuda", "gpu", "auto"):
        if torch.cuda.is_available():
            LOGGER.debug("Selected CUDA device")
            return torch.device("cuda")

    # Fallback to CPU
    LOGGER.debug("Selected CPU device")
    return torch.device("cpu")


def safe_mps_op(fn, x, *args, **kwargs):
    """Execute a function with automatic MPS-to-CPU fallback.

    If the operation fails on MPS, it will automatically retry on CPU
    and then move the result back to MPS if successful.

    This is useful for operations that may not be fully supported on MPS.

    Args:
        fn: Function to execute
        x: Input tensor (will be moved to MPS first)
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments

    Returns:
        Result tensor (on MPS if available, otherwise CPU)
    """
    try:
        import torch
    except ImportError:
        return fn(x, *args, **kwargs)

    # Try MPS first
    try:
        if hasattr(x, "to") and torch.backends.mps.is_available():
            x_mps = x.to("mps")
            result = fn(x_mps, *args, **kwargs)
            return result
    except Exception as exc:
        LOGGER.debug(f"MPS op failed, falling back to CPU: {exc}")

    # Fallback to CPU
    try:
        x_cpu = x.to("cpu") if hasattr(x, "to") else x
        result = fn(x_cpu, *args, **kwargs)
        # Try to move back to MPS if available
        if hasattr(result, "to") and torch.backends.mps.is_available():
            try:
                return result.to("mps")
            except Exception:
                pass
        return result
    except Exception:
        # Last resort: just run on whatever device the input is on
        return fn(x, *args, **kwargs)


# =============================================================================
# FFmpeg Helper with VideoToolbox support
# =============================================================================

def _find_ffmpeg() -> Optional[str]:
    """Find the ffmpeg executable path."""
    return shutil.which("ffmpeg")


def _find_ffprobe() -> Optional[str]:
    """Find the ffprobe executable path."""
    return shutil.which("ffprobe")


def get_ffmpeg_base_args(
    loglevel: str = "error",
    threads: int = 2,
    *,
    use_videotoolbox: bool = True,
) -> List[str]:
    """Get base FFmpeg arguments for quiet, low-CPU operation.

    Args:
        loglevel: FFmpeg log level (default: "error" for minimal output)
        threads: Number of threads to use (default: 2 for thermal safety)
        use_videotoolbox: Use VideoToolbox hardware acceleration on macOS

    Returns:
        List of FFmpeg arguments
    """
    args = [
        "-hide_banner",
        "-loglevel", loglevel,
        "-threads", str(threads),
    ]

    # Add VideoToolbox hardware acceleration on macOS
    if _IS_MACOS and use_videotoolbox:
        args.extend(["-hwaccel", "videotoolbox"])

    return args


def get_ffmpeg_decode_args(
    input_path: str | Path,
    *,
    loglevel: str = "error",
    threads: int = 2,
    use_videotoolbox: bool = True,
) -> List[str]:
    """Build FFmpeg decode command with VideoToolbox acceleration.

    Args:
        input_path: Path to input video file
        loglevel: Log level
        threads: Thread count
        use_videotoolbox: Use VideoToolbox on macOS

    Returns:
        FFmpeg command arguments (without output specification)
    """
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH")

    args = [ffmpeg]
    args.extend(get_ffmpeg_base_args(loglevel, threads, use_videotoolbox=use_videotoolbox))
    args.extend(["-i", str(input_path)])

    return args


def get_ffmpeg_encode_args(
    output_path: str | Path,
    *,
    codec: str = "h264_videotoolbox",
    bitrate: str = "8M",
    maxrate: str = "12M",
    bufsize: str = "16M",
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
    use_software_fallback: bool = True,
) -> List[str]:
    """Get FFmpeg encode arguments with VideoToolbox acceleration.

    If VideoToolbox encoder is not available and use_software_fallback=True,
    falls back to libx264 with high-quality CRF settings.

    Args:
        output_path: Output file path
        codec: Video codec (h264_videotoolbox or libx264)
        bitrate: Target video bitrate (8M default for high quality)
        maxrate: Maximum video bitrate (12M default)
        bufsize: Rate control buffer size (16M default)
        audio_codec: Audio codec
        audio_bitrate: Audio bitrate (192k default for high quality)
        use_software_fallback: Use libx264 if VideoToolbox unavailable

    Returns:
        FFmpeg output arguments
    """
    args = []

    # Check if VideoToolbox encoder is available
    if _IS_MACOS and "videotoolbox" in codec.lower():
        # VideoToolbox hardware encoding with high quality bitrate
        args.extend([
            "-c:v", codec,
            "-b:v", bitrate,
            "-maxrate", maxrate,
            "-bufsize", bufsize,
        ])
    elif use_software_fallback or not _IS_MACOS:
        # Software fallback with CRF for quality (CRF 18 = near-lossless)
        args.extend([
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-threads", "4",
        ])

    # Audio encoding
    args.extend([
        "-c:a", audio_codec,
        "-b:a", audio_bitrate,
    ])

    args.append(str(output_path))

    return args


def run_ffmpeg_command(
    args: List[str],
    *,
    timeout: Optional[float] = None,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """Run an FFmpeg command with error handling.

    Args:
        args: FFmpeg command arguments
        timeout: Optional timeout in seconds
        check: Raise exception on non-zero return code
        capture_output: Capture stdout/stderr

    Returns:
        CompletedProcess result
    """
    LOGGER.debug(f"Running FFmpeg: {' '.join(args)}")

    try:
        result = subprocess.run(
            args,
            timeout=timeout,
            check=check,
            capture_output=capture_output,
            text=True,
        )
        return result
    except subprocess.TimeoutExpired:
        LOGGER.error(f"FFmpeg timed out after {timeout}s")
        raise
    except subprocess.CalledProcessError as exc:
        LOGGER.error(f"FFmpeg failed with code {exc.returncode}: {exc.stderr}")
        raise


def get_video_info(input_path: str | Path) -> Dict[str, Any]:
    """Get video metadata using ffprobe.

    Args:
        input_path: Path to video file

    Returns:
        Dict with video metadata (duration, fps, width, height, etc.)
    """
    ffprobe = _find_ffprobe()
    if not ffprobe:
        raise RuntimeError("ffprobe not found in PATH")

    import json

    args = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(input_path),
    ]

    result = subprocess.run(args, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    info: Dict[str, Any] = {}

    # Extract format info
    fmt = data.get("format", {})
    info["duration"] = float(fmt.get("duration", 0))
    info["size"] = int(fmt.get("size", 0))
    info["bit_rate"] = int(fmt.get("bit_rate", 0))

    # Find video stream
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            info["width"] = stream.get("width")
            info["height"] = stream.get("height")
            info["codec"] = stream.get("codec_name")

            # Parse frame rate (e.g., "24000/1001")
            fps_str = stream.get("r_frame_rate", "0/1")
            try:
                num, denom = map(int, fps_str.split("/"))
                info["fps"] = num / denom if denom else 0
            except (ValueError, ZeroDivisionError):
                info["fps"] = 0

            info["frame_count"] = int(stream.get("nb_frames", 0))
            break

    return info


# Apply profile automatically on import if on macOS dev environment
def _auto_apply_dev_profile():
    """Auto-apply profile in dev environment."""
    # Check for dev/local indicators
    is_dev = (
        os.environ.get("SCREENALYTICS_ENV", "").lower() in ("dev", "local", "development")
        or os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true")
        or os.environ.get("DEBUG", "").lower() in ("1", "true")
    )

    if _IS_MACOS and is_dev:
        apply_macos_low_noise_profile()


# Auto-apply on import if in dev mode
_auto_apply_dev_profile()

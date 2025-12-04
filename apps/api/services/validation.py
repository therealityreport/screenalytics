"""Validation and error handling utilities for storage, jobs, and pipeline configuration.

This module provides:
- A16: STORAGE_BACKEND validation at startup
- A17: S3 credentials pre-flight check
- A18: JPEG quality validation
- A19: Stride/FPS combination validation
- A20: Device selection validation
- B22: Error taxonomy (recoverable vs fatal errors)
- B25: Early video corruption detection
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

# =============================================================================
# A16: STORAGE_BACKEND Validation
# =============================================================================

VALID_STORAGE_BACKENDS = frozenset({"local", "s3", "minio", "hybrid"})

# Cached validation result to avoid repeated checks
_storage_config_cache: Optional["StorageConfigResult"] = None
_storage_config_cache_time: float = 0
_STORAGE_CONFIG_CACHE_TTL = 300  # 5 minutes


@dataclass
class StorageConfigResult:
    """Result of storage backend configuration validation."""

    backend: str
    is_valid: bool
    is_fallback: bool = False
    fallback_reason: Optional[str] = None
    original_backend: Optional[str] = None
    bucket: Optional[str] = None
    region: Optional[str] = None
    has_credentials: bool = False
    s3_endpoint: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "is_valid": self.is_valid,
            "is_fallback": self.is_fallback,
            "fallback_reason": self.fallback_reason,
            "original_backend": self.original_backend,
            "bucket": self.bucket,
            "region": self.region,
            "has_credentials": self.has_credentials,
            "s3_endpoint": self.s3_endpoint,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_storage_backend_config(force_refresh: bool = False) -> StorageConfigResult:
    """Validate STORAGE_BACKEND environment configuration.

    Returns a StorageConfigResult with details about the current storage configuration,
    including whether it's valid, if fallback was applied, and any errors/warnings.

    This implements requirement A16: STORAGE_BACKEND check is silent fix.
    """
    global _storage_config_cache, _storage_config_cache_time

    # Use cache if available and not expired
    if not force_refresh and _storage_config_cache is not None:
        if time.time() - _storage_config_cache_time < _STORAGE_CONFIG_CACHE_TTL:
            return _storage_config_cache

    raw_backend = os.environ.get("STORAGE_BACKEND", "").strip().lower()
    errors: List[str] = []
    warnings: List[str] = []

    # Check if backend is specified
    if not raw_backend:
        warnings.append(
            "STORAGE_BACKEND not set; defaulting to 'local'. "
            "Set STORAGE_BACKEND=local|s3|minio|hybrid explicitly to avoid this warning."
        )
        raw_backend = "local"

    # Validate backend value
    is_fallback = False
    fallback_reason = None
    original_backend = None

    if raw_backend not in VALID_STORAGE_BACKENDS:
        original_backend = raw_backend
        fallback_reason = f"Invalid STORAGE_BACKEND '{raw_backend}'. Valid options: {', '.join(sorted(VALID_STORAGE_BACKENDS))}"
        errors.append(fallback_reason)
        is_fallback = True
        raw_backend = "local"

    # For S3/MinIO backends, validate required configuration
    bucket = None
    region = None
    has_credentials = False
    s3_endpoint = None

    if raw_backend in ("s3", "minio", "hybrid"):
        bucket = (
            os.environ.get("SCREENALYTICS_S3_BUCKET")
            or os.environ.get("AWS_S3_BUCKET")
            or os.environ.get("SCREENALYTICS_OBJECT_STORE_BUCKET")
            or os.environ.get("S3_BUCKET")
            or os.environ.get("BUCKET")
        )
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        s3_endpoint = (
            os.environ.get("SCREENALYTICS_OBJECT_STORE_ENDPOINT")
            or os.environ.get("AWS_ENDPOINT_URL")
        )

        # Check credentials
        has_access_key = bool(
            os.environ.get("AWS_ACCESS_KEY_ID")
            or os.environ.get("SCREENALYTICS_OBJECT_STORE_ACCESS_KEY")
        )
        has_secret_key = bool(
            os.environ.get("AWS_SECRET_ACCESS_KEY")
            or os.environ.get("SCREENALYTICS_OBJECT_STORE_SECRET_KEY")
        )
        # Also check for AWS profile or instance credentials
        has_profile = bool(os.environ.get("AWS_PROFILE"))
        has_credentials = (has_access_key and has_secret_key) or has_profile

        if raw_backend == "s3" and not has_credentials:
            # For S3, missing credentials might use instance profile, so just warn
            warnings.append(
                "No AWS credentials found (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE). "
                "If running on AWS with instance profiles, this may be fine."
            )

        if raw_backend == "minio":
            if not s3_endpoint:
                errors.append(
                    "STORAGE_BACKEND=minio requires SCREENALYTICS_OBJECT_STORE_ENDPOINT to be set."
                )
                is_fallback = True
                original_backend = original_backend or "minio"
                fallback_reason = "MinIO endpoint not configured"
                raw_backend = "local"
            elif not has_credentials:
                errors.append(
                    "STORAGE_BACKEND=minio requires access credentials. "
                    "Set SCREENALYTICS_OBJECT_STORE_ACCESS_KEY and SCREENALYTICS_OBJECT_STORE_SECRET_KEY."
                )
                is_fallback = True
                original_backend = original_backend or "minio"
                fallback_reason = "MinIO credentials not configured"
                raw_backend = "local"

    # Log configuration status
    if errors:
        for error in errors:
            LOGGER.error("[storage-config] %s", error)
    if warnings:
        for warning in warnings:
            LOGGER.warning("[storage-config] %s", warning)

    if is_fallback:
        LOGGER.warning(
            "[storage-config] Falling back to local storage due to configuration errors. "
            "Original backend: %s, Reason: %s",
            original_backend,
            fallback_reason,
        )

    result = StorageConfigResult(
        backend=raw_backend,
        is_valid=not errors,
        is_fallback=is_fallback,
        fallback_reason=fallback_reason,
        original_backend=original_backend,
        bucket=bucket,
        region=region,
        has_credentials=has_credentials,
        s3_endpoint=s3_endpoint,
        errors=errors,
        warnings=warnings,
    )

    _storage_config_cache = result
    _storage_config_cache_time = time.time()

    return result


def get_storage_backend_config() -> StorageConfigResult:
    """Get validated storage backend configuration.

    This is the recommended way to access storage configuration,
    ensuring validation has been performed.
    """
    return validate_storage_backend_config()


# =============================================================================
# A17: S3 Credentials Pre-flight Check
# =============================================================================

# Cache pre-flight results for a short period
_s3_preflight_cache: Dict[str, Tuple[bool, str, float]] = {}
_S3_PREFLIGHT_CACHE_TTL = 60  # 1 minute


@dataclass
class S3PreflightResult:
    """Result of S3 credentials pre-flight check."""

    success: bool
    error: Optional[str] = None
    bucket: Optional[str] = None
    region: Optional[str] = None
    latency_ms: float = 0
    checked_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "error": self.error,
            "bucket": self.bucket,
            "region": self.region,
            "latency_ms": self.latency_ms,
            "checked_at": self.checked_at.isoformat() if self.checked_at else None,
        }


def check_s3_credentials_preflight(
    force_check: bool = False,
    timeout_seconds: float = 10.0,
) -> S3PreflightResult:
    """Perform a pre-flight check of S3 credentials before job start.

    This implements requirement A17: No validation of S3 credentials before job start.

    Performs a minimal S3 operation (head_bucket) to verify:
    - Credentials are valid and not expired
    - Bucket exists and is accessible
    - Network connectivity to S3

    Results are cached for 1 minute to avoid hammering S3.
    """
    config = get_storage_backend_config()

    # Not applicable for local storage
    if config.backend == "local":
        return S3PreflightResult(
            success=True,
            error=None,
            bucket=None,
            checked_at=datetime.utcnow(),
        )

    # Check cache
    cache_key = f"{config.backend}:{config.bucket}:{config.s3_endpoint}"
    if not force_check and cache_key in _s3_preflight_cache:
        cached_success, cached_error, cached_time = _s3_preflight_cache[cache_key]
        if time.time() - cached_time < _S3_PREFLIGHT_CACHE_TTL:
            return S3PreflightResult(
                success=cached_success,
                error=cached_error,
                bucket=config.bucket,
                region=config.region,
                checked_at=datetime.fromtimestamp(cached_time),
            )

    # Check if boto3 is available
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
    except ImportError:
        error = "boto3 not installed; cannot verify S3 credentials"
        _s3_preflight_cache[cache_key] = (False, error, time.time())
        return S3PreflightResult(
            success=False,
            error=error,
            bucket=config.bucket,
            checked_at=datetime.utcnow(),
        )

    start_time = time.time()

    try:
        # Build client kwargs
        client_kwargs: Dict[str, Any] = {"region_name": config.region}

        if config.s3_endpoint:
            client_kwargs["endpoint_url"] = config.s3_endpoint

        if config.backend == "minio":
            from botocore.client import Config

            access_key = os.environ.get("SCREENALYTICS_OBJECT_STORE_ACCESS_KEY", "minio")
            secret_key = os.environ.get("SCREENALYTICS_OBJECT_STORE_SECRET_KEY", "miniosecret")
            signature = os.environ.get("SCREENALYTICS_OBJECT_STORE_SIGNATURE", "s3v4")
            client_kwargs["aws_access_key_id"] = access_key
            client_kwargs["aws_secret_access_key"] = secret_key
            client_kwargs["config"] = Config(signature_version=signature)

        client = boto3.client("s3", **client_kwargs)

        # Perform head_bucket as a minimal credential check
        bucket = config.bucket or "screenalytics"
        client.head_bucket(Bucket=bucket)

        latency_ms = (time.time() - start_time) * 1000
        _s3_preflight_cache[cache_key] = (True, None, time.time())

        LOGGER.info(
            "[s3-preflight] Credentials valid for bucket '%s' (latency: %.1fms)",
            bucket,
            latency_ms,
        )

        return S3PreflightResult(
            success=True,
            bucket=bucket,
            region=config.region,
            latency_ms=latency_ms,
            checked_at=datetime.utcnow(),
        )

    except NoCredentialsError:
        error = "S3 credentials not found or invalid. Check AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE."
        _s3_preflight_cache[cache_key] = (False, error, time.time())
        LOGGER.error("[s3-preflight] %s", error)
        return S3PreflightResult(
            success=False,
            error=error,
            bucket=config.bucket,
            checked_at=datetime.utcnow(),
        )

    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "Unknown")
        error_msg = exc.response.get("Error", {}).get("Message", str(exc))

        if error_code == "403":
            error = f"S3 credentials valid but access denied to bucket '{config.bucket}'. Check IAM permissions."
        elif error_code == "404":
            error = f"S3 bucket '{config.bucket}' does not exist."
        elif error_code == "ExpiredToken":
            error = "S3 credentials have expired. Refresh your AWS credentials."
        else:
            error = f"S3 error ({error_code}): {error_msg}"

        _s3_preflight_cache[cache_key] = (False, error, time.time())
        LOGGER.error("[s3-preflight] %s", error)
        return S3PreflightResult(
            success=False,
            error=error,
            bucket=config.bucket,
            checked_at=datetime.utcnow(),
        )

    except BotoCoreError as exc:
        error = f"S3 connection error: {exc}"
        _s3_preflight_cache[cache_key] = (False, error, time.time())
        LOGGER.error("[s3-preflight] %s", error)
        return S3PreflightResult(
            success=False,
            error=error,
            bucket=config.bucket,
            checked_at=datetime.utcnow(),
        )

    except Exception as exc:
        error = f"Unexpected error during S3 pre-flight check: {exc}"
        _s3_preflight_cache[cache_key] = (False, error, time.time())
        LOGGER.exception("[s3-preflight] %s", error)
        return S3PreflightResult(
            success=False,
            error=error,
            bucket=config.bucket,
            checked_at=datetime.utcnow(),
        )


def require_s3_credentials_for_job() -> None:
    """Ensure S3 credentials are valid before starting a job.

    Raises:
        StorageConfigurationError: If S3 credentials are invalid/expired.
    """
    config = get_storage_backend_config()

    if config.backend not in ("s3", "minio", "hybrid"):
        return  # Not using S3, nothing to check

    result = check_s3_credentials_preflight()

    if not result.success:
        raise StorageConfigurationError(
            f"S3 credentials invalid/expired – fix before running jobs. {result.error}"
        )


# =============================================================================
# A18: JPEG Quality Validation
# =============================================================================

JPEG_QUALITY_MIN = 10
JPEG_QUALITY_MAX = 100
JPEG_QUALITY_DEFAULT = 72


@dataclass
class JpegQualityResult:
    """Result of JPEG quality validation."""

    value: int
    is_valid: bool
    was_clamped: bool = False
    original_value: Optional[int] = None
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "is_valid": self.is_valid,
            "was_clamped": self.was_clamped,
            "original_value": self.original_value,
            "warning": self.warning,
        }


def validate_jpeg_quality(
    value: Any,
    clamp: bool = True,
    raise_on_invalid: bool = False,
) -> JpegQualityResult:
    """Validate JPEG quality value.

    This implements requirement A18: JPEG quality not validated.

    Args:
        value: The JPEG quality value to validate.
        clamp: If True, clamp invalid values to nearest valid value instead of rejecting.
        raise_on_invalid: If True, raise ValueError for invalid values.

    Returns:
        JpegQualityResult with validated value and any warnings.
    """
    # Handle None or empty
    if value is None or value == "":
        return JpegQualityResult(
            value=JPEG_QUALITY_DEFAULT,
            is_valid=True,
            was_clamped=False,
        )

    # Convert to int
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        if raise_on_invalid:
            raise ValueError(f"JPEG quality must be an integer, got: {value}")
        LOGGER.warning("[jpeg-quality] Invalid value '%s', using default %d", value, JPEG_QUALITY_DEFAULT)
        return JpegQualityResult(
            value=JPEG_QUALITY_DEFAULT,
            is_valid=False,
            was_clamped=True,
            original_value=None,
            warning=f"Invalid JPEG quality '{value}', using default {JPEG_QUALITY_DEFAULT}",
        )

    # Validate range
    if JPEG_QUALITY_MIN <= int_value <= JPEG_QUALITY_MAX:
        return JpegQualityResult(
            value=int_value,
            is_valid=True,
        )

    # Out of range
    if raise_on_invalid and not clamp:
        raise ValueError(
            f"JPEG quality {int_value} out of range [{JPEG_QUALITY_MIN}, {JPEG_QUALITY_MAX}]"
        )

    if clamp:
        clamped = max(JPEG_QUALITY_MIN, min(JPEG_QUALITY_MAX, int_value))
        warning = (
            f"JPEG quality {int_value} out of range [{JPEG_QUALITY_MIN}, {JPEG_QUALITY_MAX}], "
            f"clamped to {clamped}"
        )
        LOGGER.warning("[jpeg-quality] %s", warning)
        return JpegQualityResult(
            value=clamped,
            is_valid=False,
            was_clamped=True,
            original_value=int_value,
            warning=warning,
        )

    # Reject
    return JpegQualityResult(
        value=JPEG_QUALITY_DEFAULT,
        is_valid=False,
        original_value=int_value,
        warning=f"JPEG quality {int_value} out of range [{JPEG_QUALITY_MIN}, {JPEG_QUALITY_MAX}]",
    )


# =============================================================================
# A19: Stride/FPS Combination Validation
# =============================================================================

# Thresholds for stride/FPS validation
EFFECTIVE_FPS_MIN_WARNING = 0.5  # frames/sec - warn if below
EFFECTIVE_FPS_MIN_ERROR = 0.1  # frames/sec - error if below
EFFECTIVE_FPS_MAX_WARNING = 30.0  # frames/sec - warn if above


@dataclass
class StrideFpsResult:
    """Result of stride/FPS validation."""

    stride: int
    fps: float
    effective_fps: float
    is_valid: bool
    severity: str = "ok"  # "ok", "warning", "error"
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stride": self.stride,
            "fps": self.fps,
            "effective_fps": round(self.effective_fps, 3),
            "is_valid": self.is_valid,
            "severity": self.severity,
            "message": self.message,
        }


def validate_stride_fps(
    stride: int,
    video_fps: float,
    video_duration_sec: Optional[float] = None,
) -> StrideFpsResult:
    """Validate stride/FPS combination for reasonable sampling rate.

    This implements requirement A19: Stride/FPS combination not validated.

    Args:
        stride: Frame stride (sample every N frames)
        video_fps: Video frames per second
        video_duration_sec: Optional video duration for additional context

    Returns:
        StrideFpsResult with validation details and any warnings.
    """
    # Validate inputs
    if stride <= 0:
        return StrideFpsResult(
            stride=stride,
            fps=video_fps,
            effective_fps=0,
            is_valid=False,
            severity="error",
            message="Frame stride must be positive",
        )

    if video_fps <= 0:
        return StrideFpsResult(
            stride=stride,
            fps=video_fps,
            effective_fps=0,
            is_valid=False,
            severity="error",
            message="Video FPS must be positive",
        )

    # Calculate effective sampling rate
    effective_fps = video_fps / stride

    # Check for very sparse sampling (might miss faces)
    if effective_fps < EFFECTIVE_FPS_MIN_ERROR:
        total_frames = int(video_duration_sec * effective_fps) if video_duration_sec else None
        frame_info = f" ({total_frames} total frames)" if total_frames else ""
        return StrideFpsResult(
            stride=stride,
            fps=video_fps,
            effective_fps=effective_fps,
            is_valid=False,
            severity="error",
            message=(
                f"Extremely sparse sampling: {effective_fps:.3f} frames/sec{frame_info}. "
                f"This will likely miss most faces. Consider reducing stride from {stride}."
            ),
        )

    if effective_fps < EFFECTIVE_FPS_MIN_WARNING:
        seconds_between = 1.0 / effective_fps
        return StrideFpsResult(
            stride=stride,
            fps=video_fps,
            effective_fps=effective_fps,
            is_valid=True,
            severity="warning",
            message=(
                f"Sparse sampling: {effective_fps:.2f} frames/sec (~{seconds_between:.1f}s between samples). "
                f"Faces in brief scenes may be missed."
            ),
        )

    # Check for very dense sampling (may be excessive)
    if effective_fps > EFFECTIVE_FPS_MAX_WARNING:
        return StrideFpsResult(
            stride=stride,
            fps=video_fps,
            effective_fps=effective_fps,
            is_valid=True,
            severity="warning",
            message=(
                f"Dense sampling: {effective_fps:.1f} frames/sec. "
                f"This may increase processing time significantly with diminishing returns. "
                f"Consider increasing stride from {stride}."
            ),
        )

    return StrideFpsResult(
        stride=stride,
        fps=video_fps,
        effective_fps=effective_fps,
        is_valid=True,
        severity="ok",
    )


# =============================================================================
# A20: Device Selection Validation
# =============================================================================

DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
DEVICE_COREML = "coreml"
DEVICE_MPS = "mps"
DEVICE_AUTO = "auto"

VALID_DEVICES = frozenset({DEVICE_CPU, DEVICE_CUDA, DEVICE_COREML, DEVICE_MPS, DEVICE_AUTO})


@dataclass
class DeviceCapabilities:
    """Information about available compute devices."""

    has_cuda: bool = False
    cuda_device_count: int = 0
    cuda_device_names: List[str] = field(default_factory=list)
    has_coreml: bool = False
    has_mps: bool = False
    is_apple_silicon: bool = False
    cpu_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_cuda": self.has_cuda,
            "cuda_device_count": self.cuda_device_count,
            "cuda_device_names": self.cuda_device_names,
            "has_coreml": self.has_coreml,
            "has_mps": self.has_mps,
            "is_apple_silicon": self.is_apple_silicon,
            "cpu_count": self.cpu_count,
        }


# Cache device capabilities
_device_capabilities_cache: Optional[DeviceCapabilities] = None


def get_device_capabilities(force_refresh: bool = False) -> DeviceCapabilities:
    """Detect available compute devices on this system.

    This implements requirement A20: Device selection validated too late.
    """
    global _device_capabilities_cache

    if not force_refresh and _device_capabilities_cache is not None:
        return _device_capabilities_cache

    caps = DeviceCapabilities()

    # CPU count
    try:
        caps.cpu_count = os.cpu_count() or 1
    except Exception:
        caps.cpu_count = 1

    # Check Apple Silicon
    caps.is_apple_silicon = (
        sys.platform == "darwin"
        and platform.machine().lower() in ("arm64", "aarch64")
    )

    # Check CUDA
    try:
        import torch

        caps.has_cuda = torch.cuda.is_available()
        if caps.has_cuda:
            caps.cuda_device_count = torch.cuda.device_count()
            caps.cuda_device_names = [
                torch.cuda.get_device_name(i) for i in range(caps.cuda_device_count)
            ]
    except ImportError:
        # Try nvidia-smi as fallback
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                names = [n.strip() for n in result.stdout.strip().split("\n") if n.strip()]
                caps.has_cuda = len(names) > 0
                caps.cuda_device_count = len(names)
                caps.cuda_device_names = names
        except Exception:
            pass

    # Check CoreML
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        caps.has_coreml = any("coreml" in p.lower() for p in providers)
    except ImportError:
        pass

    # Check MPS (Apple Metal Performance Shaders)
    if caps.is_apple_silicon:
        try:
            import torch

            caps.has_mps = torch.backends.mps.is_available()
        except (ImportError, AttributeError):
            pass

    _device_capabilities_cache = caps
    return caps


def list_supported_devices() -> List[str]:
    """Return list of devices actually supported on this system."""
    caps = get_device_capabilities()
    devices = [DEVICE_CPU]

    if caps.has_cuda:
        devices.append(DEVICE_CUDA)
    if caps.has_coreml:
        devices.append(DEVICE_COREML)
    if caps.has_mps:
        devices.append(DEVICE_MPS)

    return devices


@dataclass
class DeviceValidationResult:
    """Result of device validation."""

    requested: str
    resolved: str
    is_valid: bool
    was_fallback: bool = False
    fallback_reason: Optional[str] = None
    warning: Optional[str] = None
    capabilities: Optional[DeviceCapabilities] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "is_valid": self.is_valid,
            "was_fallback": self.was_fallback,
            "fallback_reason": self.fallback_reason,
            "warning": self.warning,
            "capabilities": self.capabilities.to_dict() if self.capabilities else None,
        }


def normalize_device(
    device_str: Optional[str],
    allow_fallback: bool = True,
) -> DeviceValidationResult:
    """Normalize and validate a device selection string.

    This implements requirement A20: Device selection validated too late.

    Args:
        device_str: The device string to validate ("cpu", "cuda", "coreml", "mps", "auto")
        allow_fallback: If True, fallback to CPU when requested device unavailable.
                       If False, raise an error for unavailable devices.

    Returns:
        DeviceValidationResult with resolved device and validation info.

    Raises:
        DeviceUnavailableError: If allow_fallback=False and device is unavailable.
    """
    caps = get_device_capabilities()
    supported = list_supported_devices()

    # Normalize input
    raw = (device_str or "auto").strip().lower()

    # Handle aliases
    if raw in ("0", "gpu"):
        raw = DEVICE_CUDA
    elif raw == "":
        raw = DEVICE_AUTO

    # Check if valid device name
    if raw not in VALID_DEVICES:
        if allow_fallback:
            LOGGER.warning("[device] Unknown device '%s', falling back to CPU", device_str)
            return DeviceValidationResult(
                requested=device_str or "auto",
                resolved=DEVICE_CPU,
                is_valid=False,
                was_fallback=True,
                fallback_reason=f"Unknown device '{device_str}'",
                capabilities=caps,
            )
        raise DeviceUnavailableError(f"Unknown device: {device_str}")

    # Handle auto-detection
    if raw == DEVICE_AUTO:
        if DEVICE_CUDA in supported:
            resolved = DEVICE_CUDA
        elif DEVICE_COREML in supported:
            resolved = DEVICE_COREML
        elif DEVICE_MPS in supported:
            resolved = DEVICE_MPS
        else:
            resolved = DEVICE_CPU

        return DeviceValidationResult(
            requested="auto",
            resolved=resolved,
            is_valid=True,
            capabilities=caps,
        )

    # Check if requested device is available
    if raw == DEVICE_CUDA and not caps.has_cuda:
        if allow_fallback:
            LOGGER.warning(
                "[device] CUDA requested but not available (no NVIDIA GPU detected), "
                "falling back to CPU"
            )
            return DeviceValidationResult(
                requested=raw,
                resolved=DEVICE_CPU,
                is_valid=False,
                was_fallback=True,
                fallback_reason="CUDA not available (no NVIDIA GPU detected)",
                warning="CUDA requested but not available; using CPU instead",
                capabilities=caps,
            )
        raise DeviceUnavailableError(
            "CUDA requested but not available. No NVIDIA GPU detected on this system."
        )

    if raw == DEVICE_COREML and not caps.has_coreml:
        if allow_fallback:
            LOGGER.warning(
                "[device] CoreML requested but not available, falling back to CPU"
            )
            return DeviceValidationResult(
                requested=raw,
                resolved=DEVICE_CPU,
                is_valid=False,
                was_fallback=True,
                fallback_reason="CoreML not available (requires onnxruntime-coreml on macOS)",
                warning="CoreML requested but not available; using CPU instead",
                capabilities=caps,
            )
        raise DeviceUnavailableError(
            "CoreML requested but not available. Install onnxruntime-coreml on macOS."
        )

    if raw == DEVICE_MPS and not caps.has_mps:
        if allow_fallback:
            LOGGER.warning(
                "[device] MPS requested but not available, falling back to CPU"
            )
            return DeviceValidationResult(
                requested=raw,
                resolved=DEVICE_CPU,
                is_valid=False,
                was_fallback=True,
                fallback_reason="MPS not available (requires Apple Silicon with PyTorch)",
                warning="MPS requested but not available; using CPU instead",
                capabilities=caps,
            )
        raise DeviceUnavailableError(
            "MPS requested but not available. Requires Apple Silicon with PyTorch MPS support."
        )

    # Device is available
    return DeviceValidationResult(
        requested=raw,
        resolved=raw,
        is_valid=True,
        capabilities=caps,
    )


# =============================================================================
# B22: Error Taxonomy
# =============================================================================


class ErrorCategory(Enum):
    """Categories of errors for distinct handling."""

    TRANSIENT = auto()  # Network/S3 timeouts, 5xx errors - should retry
    CONFIG = auto()  # Bad credentials, invalid env, unsupported device - fail fast
    DATA_INTEGRITY = auto()  # Corrupted video, invalid manifests - fail with clear message
    UNKNOWN = auto()  # Uncategorized errors


@dataclass
class CategorizedError:
    """An error with its category for appropriate handling."""

    category: ErrorCategory
    message: str
    original_exception: Optional[Exception] = None
    retry_suggested: bool = False
    max_retries: int = 0
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.name,
            "message": self.message,
            "retry_suggested": self.retry_suggested,
            "max_retries": self.max_retries,
            "context": self.context,
        }


def categorize_error(exc: Exception, context: Optional[Dict[str, Any]] = None) -> CategorizedError:
    """Categorize an exception for appropriate handling.

    This implements requirement B22: No distinction between recoverable and fatal errors.
    """
    ctx = context or {}
    exc_str = str(exc).lower()
    exc_type = type(exc).__name__

    # Check for transient network/S3 errors
    transient_indicators = [
        "timeout",
        "timed out",
        "connection reset",
        "connection refused",
        "connection aborted",
        "temporary failure",
        "service unavailable",
        "503",
        "502",
        "504",
        "throttl",
        "rate limit",
        "too many requests",
        "429",
        "network",
        "socket",
        "ssl",
    ]

    for indicator in transient_indicators:
        if indicator in exc_str or indicator in exc_type.lower():
            return CategorizedError(
                category=ErrorCategory.TRANSIENT,
                message=f"Transient error (may resolve with retry): {exc}",
                original_exception=exc,
                retry_suggested=True,
                max_retries=3,
                context=ctx,
            )

    # Check for configuration errors
    config_indicators = [
        "credential",
        "authentication",
        "authorization",
        "access denied",
        "403",
        "401",
        "invalid key",
        "expired",
        "not configured",
        "missing config",
        "invalid config",
        "environment",
        "no such device",
        "device not found",
        "cuda not available",
        "coreml not available",
    ]

    for indicator in config_indicators:
        if indicator in exc_str:
            return CategorizedError(
                category=ErrorCategory.CONFIG,
                message=f"Configuration error (fix before retrying): {exc}",
                original_exception=exc,
                retry_suggested=False,
                context=ctx,
            )

    # Check for data integrity errors
    data_indicators = [
        "corrupt",
        "invalid format",
        "decode error",
        "codec",
        "malformed",
        "truncated",
        "checksum",
        "hash mismatch",
        "invalid json",
        "json decode",
        "manifest",
        "not a valid",
    ]

    for indicator in data_indicators:
        if indicator in exc_str:
            return CategorizedError(
                category=ErrorCategory.DATA_INTEGRITY,
                message=f"Data integrity error: {exc}",
                original_exception=exc,
                retry_suggested=False,
                context=ctx,
            )

    # Unknown category
    return CategorizedError(
        category=ErrorCategory.UNKNOWN,
        message=str(exc),
        original_exception=exc,
        retry_suggested=False,
        context=ctx,
    )


# =============================================================================
# B25: Video Corruption Detection
# =============================================================================


@dataclass
class VideoValidationResult:
    """Result of video file validation."""

    is_valid: bool
    path: str
    duration_sec: Optional[float] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    codec: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    frames_tested: int = 0
    frames_decoded_ok: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "path": self.path,
            "duration_sec": self.duration_sec,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            "errors": self.errors,
            "warnings": self.warnings,
            "frames_tested": self.frames_tested,
            "frames_decoded_ok": self.frames_decoded_ok,
        }


def validate_video_file(
    video_path: Path | str,
    test_frame_positions: Optional[List[float]] = None,
) -> VideoValidationResult:
    """Validate a video file for corruption before processing.

    This implements requirement B25: Video corruption not detected early.

    Tests:
    1. File exists and is readable
    2. Video metadata can be extracted (duration, fps, dimensions)
    3. Sample frames can be decoded (start, middle, end)

    Args:
        video_path: Path to the video file
        test_frame_positions: Optional list of positions (0.0-1.0) to test.
                             Defaults to [0.0, 0.5, 1.0] (start, middle, end)

    Returns:
        VideoValidationResult with validation details
    """
    path = Path(video_path)
    result = VideoValidationResult(is_valid=False, path=str(path))

    # Check file exists
    if not path.exists():
        result.errors.append(f"Video file not found: {path}")
        return result

    if not path.is_file():
        result.errors.append(f"Path is not a file: {path}")
        return result

    # Check file size
    try:
        file_size = path.stat().st_size
        if file_size == 0:
            result.errors.append("Video file is empty (0 bytes)")
            return result
        if file_size < 1000:
            result.warnings.append(f"Video file is suspiciously small ({file_size} bytes)")
    except OSError as exc:
        result.errors.append(f"Cannot read file metadata: {exc}")
        return result

    # Try to open video with OpenCV
    try:
        import cv2
    except ImportError:
        result.errors.append("OpenCV (cv2) not installed; cannot validate video")
        return result

    cap = None
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            result.errors.append(f"Cannot open video file (corrupted or unsupported format): {path}")
            return result

        # Extract metadata
        result.fps = cap.get(cv2.CAP_PROP_FPS)
        result.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        result.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        result.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        result.codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        # Calculate duration
        if result.fps and result.fps > 0 and result.frame_count:
            result.duration_sec = result.frame_count / result.fps

        # Validate metadata
        if not result.fps or result.fps <= 0:
            result.errors.append("Invalid or missing FPS in video metadata")

        if not result.frame_count or result.frame_count <= 0:
            result.errors.append("Invalid or missing frame count in video metadata")

        if not result.width or not result.height or result.width <= 0 or result.height <= 0:
            result.errors.append("Invalid or missing video dimensions")

        if result.errors:
            return result

        # Test frame decoding at key positions
        positions = test_frame_positions or [0.0, 0.5, 1.0]
        result.frames_tested = len(positions)

        for pos in positions:
            # Calculate frame index (0.0 = start, 1.0 = end)
            frame_idx = int(pos * (result.frame_count - 1))
            frame_idx = max(0, min(frame_idx, result.frame_count - 1))

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret or frame is None:
                pos_label = {0.0: "start", 0.5: "middle", 1.0: "end"}.get(pos, f"position {pos:.0%}")
                result.errors.append(
                    f"Failed to decode frame at {pos_label} (frame {frame_idx}/{result.frame_count})"
                )
            else:
                result.frames_decoded_ok += 1

        if result.frames_decoded_ok < result.frames_tested:
            if result.frames_decoded_ok == 0:
                result.errors.append(
                    "All test frames failed to decode. Video may be severely corrupted."
                )
            else:
                result.warnings.append(
                    f"Only {result.frames_decoded_ok}/{result.frames_tested} test frames decoded. "
                    "Video may have partial corruption."
                )

        # Set valid if no errors
        result.is_valid = len(result.errors) == 0

    except Exception as exc:
        result.errors.append(f"Error validating video: {exc}")

    finally:
        if cap is not None:
            cap.release()

    return result


# =============================================================================
# Custom Exceptions
# =============================================================================


class StorageConfigurationError(Exception):
    """Raised when storage backend is misconfigured."""

    pass


class DeviceUnavailableError(Exception):
    """Raised when a requested compute device is not available."""

    pass


class VideoCorruptionError(Exception):
    """Raised when a video file is corrupted or cannot be processed."""

    def __init__(self, message: str, validation_result: Optional[VideoValidationResult] = None):
        super().__init__(message)
        self.validation_result = validation_result


class JobConfigurationError(Exception):
    """Raised when job configuration is invalid."""

    pass


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Storage validation (A16)
    "VALID_STORAGE_BACKENDS",
    "StorageConfigResult",
    "validate_storage_backend_config",
    "get_storage_backend_config",
    # S3 pre-flight (A17)
    "S3PreflightResult",
    "check_s3_credentials_preflight",
    "require_s3_credentials_for_job",
    # JPEG quality (A18)
    "JPEG_QUALITY_MIN",
    "JPEG_QUALITY_MAX",
    "JPEG_QUALITY_DEFAULT",
    "JpegQualityResult",
    "validate_jpeg_quality",
    # Stride/FPS (A19)
    "StrideFpsResult",
    "validate_stride_fps",
    # Device validation (A20)
    "VALID_DEVICES",
    "DeviceCapabilities",
    "DeviceValidationResult",
    "get_device_capabilities",
    "list_supported_devices",
    "normalize_device",
    # Error taxonomy (B22)
    "ErrorCategory",
    "CategorizedError",
    "categorize_error",
    # Video validation (B25)
    "VideoValidationResult",
    "validate_video_file",
    # Exceptions
    "StorageConfigurationError",
    "DeviceUnavailableError",
    "VideoCorruptionError",
    "JobConfigurationError",
]

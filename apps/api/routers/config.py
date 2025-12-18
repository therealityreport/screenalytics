"""Configuration API endpoints.

Exposes configuration values like thresholds for UI components,
storage backend status, device capabilities, and validation results.
"""

import logging
import os
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from pydantic import BaseModel, Field

from apps.api.config.suggestions import (
    get_all_thresholds,
    get_effective_thresholds,
    update_suggestion_thresholds,
    reset_suggestion_thresholds,
    SUGGESTION_THRESHOLDS,
    TIMEOUTS,
)

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["config"])


# =============================================================================
# Pydantic Models
# =============================================================================


class SuggestionThresholdUpdate(BaseModel):
    """Request model for updating suggestion thresholds."""

    cast_high: Optional[float] = Field(
        None,
        ge=0.5,
        le=1.0,
        description="High confidence threshold (0.5-1.0). Suggestions at or above this are HIGH confidence.",
    )
    cast_medium: Optional[float] = Field(
        None,
        ge=0.3,
        le=1.0,
        description="Medium confidence threshold (0.3-cast_high). Suggestions at or above this are MEDIUM confidence.",
    )
    cast_auto_assign: Optional[float] = Field(
        None,
        ge=0.5,
        le=1.0,
        description="Auto-assign threshold (cast_high-1.0). Only auto-assign when above this.",
    )


# =============================================================================
# Threshold Endpoints
# =============================================================================


@router.get("/thresholds")
def get_thresholds() -> dict:
    """Get all Smart Suggestions thresholds and configuration values.

    Returns effective configuration values (with any user overrides applied)
    that UI components can use for:
    - Confidence labels (high/medium/low based on thresholds)
    - Threshold display in UI (e.g., "≥68%")
    - Timeout values

    Response:
        {
            "suggestion": {
                "cast_high": 0.68,
                "cast_high_label": "≥68%",
                "cast_medium": 0.50,
                ...
            },
            "grouping": {...},
            "timeouts": {...},
            "api_base_url": "http://...",
            "has_overrides": false
        }
    """
    return {
        "status": "success",
        **get_effective_thresholds(),
    }


@router.get("/thresholds/cast")
def get_cast_thresholds() -> dict:
    """Get cast/facebank similarity thresholds only.

    Returns effective thresholds (with any user overrides applied).

    Returns:
        {
            "high": 0.68,
            "high_label": "≥68%",
            "medium": 0.50,
            "medium_label": "≥50%",
            "auto_assign": 0.85,
            "auto_assign_label": "≥85%",
            "has_overrides": false
        }
    """
    effective = get_effective_thresholds()
    suggestion = effective.get("suggestion", {})
    return {
        "status": "success",
        "high": suggestion.get("cast_high", SUGGESTION_THRESHOLDS["cast_high"]),
        "high_label": f"≥{int(suggestion.get('cast_high', SUGGESTION_THRESHOLDS['cast_high']) * 100)}%",
        "medium": suggestion.get("cast_medium", SUGGESTION_THRESHOLDS["cast_medium"]),
        "medium_label": f"≥{int(suggestion.get('cast_medium', SUGGESTION_THRESHOLDS['cast_medium']) * 100)}%",
        "auto_assign": suggestion.get("cast_auto_assign", SUGGESTION_THRESHOLDS["cast_auto_assign"]),
        "auto_assign_label": f"≥{int(suggestion.get('cast_auto_assign', SUGGESTION_THRESHOLDS['cast_auto_assign']) * 100)}%",
        "has_overrides": effective.get("has_overrides", False),
    }


@router.put("/thresholds/suggestion")
def update_thresholds(body: SuggestionThresholdUpdate) -> dict:
    """Update suggestion confidence thresholds.

    Allows customizing the HIGH/MEDIUM/LOW confidence cutoffs for cast suggestions.
    Changes are persisted and will survive server restarts.

    Validation rules:
    - cast_high: Must be between 0.5 and 1.0
    - cast_medium: Must be between 0.3 and cast_high
    - cast_auto_assign: Must be between cast_high and 1.0

    Request body:
        {
            "cast_high": 0.68,      // Optional
            "cast_medium": 0.50,   // Optional
            "cast_auto_assign": 0.85  // Optional
        }

    Response:
        Updated effective thresholds (same format as GET /config/thresholds)
    """
    try:
        result = update_suggestion_thresholds(
            cast_high=body.cast_high,
            cast_medium=body.cast_medium,
            cast_auto_assign=body.cast_auto_assign,
        )
        LOGGER.info(
            "[thresholds] Updated suggestion thresholds: high=%s, medium=%s, auto=%s",
            body.cast_high,
            body.cast_medium,
            body.cast_auto_assign,
        )
        return {"status": "success", **result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/thresholds/suggestion")
def reset_thresholds() -> dict:
    """Reset suggestion thresholds to default values.

    Removes any user-configured overrides and restores defaults:
    - cast_high: 0.68 (68%)
    - cast_medium: 0.50 (50%)
    - cast_auto_assign: 0.85 (85%)

    Response:
        Default thresholds (same format as GET /config/thresholds)
    """
    result = reset_suggestion_thresholds()
    LOGGER.info("[thresholds] Reset suggestion thresholds to defaults")
    return {"status": "success", **result}


@router.get("/timeouts")
def get_timeouts() -> dict:
    """Get API timeout configuration.

    Returns:
        {
            "default": 30,
            "heavy": 60,
            "fast": 15
        }
    """
    return {
        "status": "success",
        "default": TIMEOUTS["api_default"],
        "heavy": TIMEOUTS["api_heavy"],
        "fast": TIMEOUTS["api_fast"],
    }


# =============================================================================
# Storage Backend Status (A16-A17)
# =============================================================================


@router.get("/storage")
def get_storage_status() -> Dict[str, Any]:
    """Get current storage backend status and validation results.

    Returns storage backend type, S3 credentials status (if applicable),
    and any warnings about configuration.

    Response:
        {
            "backend_type": "local" | "s3" | "minio" | "hybrid",
            "supports_presigned_urls": bool,
            "validation": {
                "backend": "local",
                "is_fallback": false,
                "original_backend": null,
                "bucket": "screenalytics",
                "warnings": [],
                "config_source": "STORAGE_BACKEND env var"
            },
            "s3_preflight": {  // Only if S3/MinIO/hybrid
                "success": true,
                "bucket": "screenalytics",
                "region": "us-east-1",
                "latency_ms": 123.4,
                "checked_at": "2024-..."
            }
        }
    """
    try:
        from apps.api.services.storage_backend import get_storage_backend_status

        status = get_storage_backend_status()

        # Add S3 preflight check if using S3-based backend
        backend_type = status.get("backend_type", "local")
        if backend_type in ("s3", "minio", "hybrid"):
            try:
                from apps.api.services.validation import check_s3_credentials_preflight

                preflight = check_s3_credentials_preflight()
                status["s3_preflight"] = preflight.to_dict()
            except Exception as exc:
                LOGGER.debug("[storage-status] Failed to run S3 preflight: %s", exc)
                status["s3_preflight"] = {"success": False, "error": str(exc)}

        return {"status": "success", **status}

    except ImportError as exc:
        LOGGER.warning("[storage-status] Validation module not available: %s", exc)
        return {
            "status": "error",
            "error": "Storage validation module not available",
            "backend_type": "unknown",
        }


# =============================================================================
# DB Health (optional DB_URL)
# =============================================================================


@router.get("/db_health")
def get_db_health() -> Dict[str, Any]:
    """Lightweight DB connectivity + schema preflight (optional).

    DB-backed features (identity locks, suggestion batches, audit trails) require DB_URL.
    This endpoint never raises; it returns a structured error instead.
    """
    db_url = (os.getenv("DB_URL") or "").strip()
    fake_db = (os.getenv("SCREENALYTICS_FAKE_DB") or "").strip() == "1"
    result: Dict[str, Any] = {
        "configured": bool(db_url) or fake_db,
        "fake_db": fake_db,
        "psycopg2_available": None,
        "ok": None,
        "latency_ms": None,
        "tables": {},
        "migrations_ok": None,
        "error": None,
    }

    if fake_db:
        result["psycopg2_available"] = False
        result["ok"] = True
        result["migrations_ok"] = True
        result["tables"] = {"mode": "fake_db"}
        return result

    if not db_url:
        result["psycopg2_available"] = False
        result["ok"] = False
        result["migrations_ok"] = False
        result["error"] = "DB_URL is not set"
        return result

    try:
        import psycopg2  # type: ignore
    except Exception as exc:
        result["psycopg2_available"] = False
        result["ok"] = False
        result["migrations_ok"] = False
        result["error"] = f"psycopg2_import_error: {type(exc).__name__}: {exc}"
        return result

    result["psycopg2_available"] = True
    started = time.perf_counter()
    try:
        conn = psycopg2.connect(db_url, connect_timeout=3)  # type: ignore[arg-type]
    except Exception as exc:
        result["ok"] = False
        result["migrations_ok"] = False
        result["error"] = f"connect_error: {type(exc).__name__}: {exc}"
        return result

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()

                tables = [
                    "runs",
                    "job_runs",
                    "identity_locks",
                    "suggestion_batches",
                    "suggestions",
                    "suggestion_applies",
                ]
                table_status: Dict[str, bool] = {}
                for table in tables:
                    cur.execute("SELECT to_regclass(%s);", (f"public.{table}",))
                    row = cur.fetchone()
                    table_status[table] = bool(row and row[0])
                result["tables"] = table_status
                result["migrations_ok"] = all(table_status.values())
        result["ok"] = True
        result["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    except Exception as exc:
        result["ok"] = False
        result["migrations_ok"] = False
        result["error"] = f"query_error: {type(exc).__name__}: {exc}"
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return result


# =============================================================================
# Device Capabilities (A20)
# =============================================================================


@router.get("/device")
def get_device_capabilities() -> Dict[str, Any]:
    """Get available device capabilities for ML inference.

    Returns detected GPU/accelerator availability and recommended device selection.

    Response:
        {
            "has_cuda": bool,
            "has_coreml": bool,
            "has_mps": bool,
            "cuda_device_name": "NVIDIA RTX 4090" | null,
            "recommended_device": "cuda" | "coreml" | "mps" | "cpu",
            "recommended_profile": "balanced" | "low_power" | "performance"
        }
    """
    try:
        from apps.api.services.validation import detect_device_capabilities

        caps = detect_device_capabilities()
        return {"status": "success", **caps.to_dict()}

    except ImportError as exc:
        LOGGER.warning("[device-caps] Validation module not available: %s", exc)
        return {
            "status": "error",
            "error": "Device validation module not available",
            "recommended_device": "cpu",
            "recommended_profile": "low_power",
        }


@router.get("/device/validate")
def validate_device(
    device: str = Query(..., description="Device to validate (cpu, cuda, coreml, mps)"),
) -> Dict[str, Any]:
    """Validate a specific device selection.

    Returns whether the device is available and any warnings.

    Response:
        {
            "requested_device": "cuda",
            "normalized_device": "cuda",
            "is_available": true,
            "fallback_device": null,
            "warning": null
        }
    """
    try:
        from apps.api.services.validation import validate_device_selection

        result = validate_device_selection(device)
        return {"status": "success", **result.to_dict()}

    except ImportError as exc:
        LOGGER.warning("[device-validate] Validation module not available: %s", exc)
        return {
            "status": "error",
            "error": "Device validation module not available",
            "requested_device": device,
            "is_available": device.lower() == "cpu",
        }


# =============================================================================
# JPEG Quality Validation (A18)
# =============================================================================


@router.get("/validate/jpeg-quality")
def validate_jpeg_quality(
    value: int = Query(..., description="JPEG quality value to validate (10-100)"),
) -> Dict[str, Any]:
    """Validate a JPEG quality value.

    Response:
        {
            "value": 72,
            "is_valid": true,
            "was_clamped": false,
            "original_value": null,
            "warning": null
        }
    """
    try:
        from apps.api.services.validation import validate_jpeg_quality as do_validate

        result = do_validate(value)
        return {"status": "success", **result.to_dict()}

    except ImportError as exc:
        # Fallback validation
        clamped = max(10, min(100, int(value)))
        return {
            "status": "success",
            "value": clamped,
            "is_valid": 10 <= value <= 100,
            "was_clamped": clamped != value,
            "original_value": value if clamped != value else None,
        }


# =============================================================================
# Stride/FPS Validation (A19)
# =============================================================================


@router.get("/validate/stride-fps")
def validate_stride_fps(
    stride: int = Query(..., description="Frame stride (sample every N frames)"),
    fps: float = Query(..., description="Video frames per second"),
    duration: Optional[float] = Query(None, description="Video duration in seconds"),
) -> Dict[str, Any]:
    """Validate stride/FPS combination for reasonable sampling rate.

    Response:
        {
            "stride": 6,
            "fps": 24.0,
            "effective_fps": 4.0,
            "is_valid": true,
            "severity": "ok",
            "message": null
        }
    """
    try:
        from apps.api.services.validation import validate_stride_fps as do_validate

        result = do_validate(stride, fps, duration)
        return {"status": "success", **result.to_dict()}

    except ImportError as exc:
        # Fallback validation
        if stride <= 0 or fps <= 0:
            return {
                "status": "error",
                "error": "Stride and FPS must be positive",
            }
        effective_fps = fps / stride
        return {
            "status": "success",
            "stride": stride,
            "fps": fps,
            "effective_fps": round(effective_fps, 3),
            "is_valid": effective_fps >= 0.1,
            "severity": "warning" if effective_fps < 0.5 else "ok",
        }


# =============================================================================
# Video Validation (B25)
# =============================================================================


@router.post("/validate/video")
def validate_video(
    video_path: str = Query(..., description="Path to video file to validate"),
) -> Dict[str, Any]:
    """Validate a video file for corruption.

    Performs sample frame decoding to detect corrupt or incomplete videos.

    Response:
        {
            "is_valid": true,
            "path": "/path/to/video.mp4",
            "frame_count": 3600,
            "fps": 24.0,
            "duration_sec": 150.0,
            "sample_frames_decoded": 10,
            "error": null,
            "error_category": null
        }
    """
    try:
        from apps.api.services.validation import validate_video_file

        result = validate_video_file(video_path)
        return {"status": "success" if result.is_valid else "error", **result.to_dict()}

    except ImportError as exc:
        LOGGER.warning("[video-validate] Validation module not available: %s", exc)
        return {
            "status": "error",
            "error": "Video validation module not available",
            "is_valid": False,
        }


# =============================================================================
# Dry Run Estimation (E35)
# =============================================================================


@router.get("/dry-run/{ep_id}")
def dry_run_estimate(
    ep_id: str,
    stride: int = Query(6, description="Frame stride"),
    save_frames: bool = Query(False, description="Save extracted frames"),
    save_crops: bool = Query(True, description="Save face crops"),
) -> Dict[str, Any]:
    """Estimate resource requirements for detect/track job without running it.

    Response:
        {
            "ep_id": "SHOW-s01e01",
            "video_path": "/path/to/video.mp4",
            "video_duration_sec": 2700.0,
            "video_fps": 24.0,
            "stride": 6,
            "estimated_frames": 10800,
            "estimated_faces": 54000,
            "estimated_disk_mb": 512.0,
            "estimated_runtime_sec": 180.0,
            "warnings": []
        }
    """
    try:
        from apps.api.services.job_manager import dry_run_detect_track

        result = dry_run_detect_track(
            ep_id=ep_id,
            stride=stride,
            save_frames=save_frames,
            save_crops=save_crops,
        )
        return {"status": "success", **result.to_dict()}

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ImportError as exc:
        LOGGER.warning("[dry-run] Job manager module not available: %s", exc)
        return {
            "status": "error",
            "error": "Dry run estimation not available",
            "ep_id": ep_id,
        }


# =============================================================================
# Resource Limits (E38)
# =============================================================================


@router.get("/resources")
def get_resource_status() -> Dict[str, Any]:
    """Get current resource limits and usage.

    Response:
        {
            "limits": {
                "max_cpu_threads": 4,
                "max_memory_mb": null,
                "max_gpu_memory_mb": null
            },
            "usage": {
                "cpu_percent": 45.0,
                "memory_mb": 2048.0,
                "gpu_memory_mb": null,
                "available_cpu_threads": 8
            }
        }
    """
    try:
        from apps.api.services.job_manager import get_resource_manager

        rm = get_resource_manager()
        limits = rm.limits
        usage = rm.get_current_usage()

        return {
            "status": "success",
            "limits": limits.to_dict(),
            "usage": usage.to_dict(),
        }

    except ImportError as exc:
        LOGGER.warning("[resources] Job manager module not available: %s", exc)
        return {
            "status": "error",
            "error": "Resource manager not available",
        }

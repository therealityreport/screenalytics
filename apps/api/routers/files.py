"""File serving and presign endpoints."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Query
from mimetypes import guess_type

from apps.api.services.storage import StorageService

LOGGER = logging.getLogger(__name__)
router = APIRouter()
storage_service = StorageService()


def _infer_mime_for_key(key: str) -> str:
    mime, _ = guess_type(key)
    if mime:
        return mime
    lowered = key.lower()
    if lowered.endswith(".png"):
        return "image/png"
    if lowered.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


@router.get("/files/presign")
def presign_image(
    key: str = Query(..., description="S3 object key"),
    mime: str | None = Query(None, description="Content type override"),
    ttl: int = Query(3600, ge=60, le=86400, description="URL expiration in seconds"),
) -> dict:
    """Generate a presigned URL for an S3 object with proper content-type headers."""
    resolved_mime = mime or _infer_mime_for_key(key)
    url = storage_service.presign_get(key, expires_in=ttl, content_type=resolved_mime)
    if not url:
        return {"error": "presign_unavailable", "url": None, "key": key}

    return {"url": url, "key": key, "expires_in": ttl, "content_type": resolved_mime}


@router.get("/files/health")
def check_image_health(path_or_key: str = Query(..., description="Local path or S3 key")) -> dict:
    """Check if an image exists and is accessible, with optional image diagnostics."""
    import hashlib

    mime_guess = _infer_mime_for_key(path_or_key)

    # Check if it's a local path
    if path_or_key.startswith("/") or path_or_key.startswith("data/"):
        try:
            p = Path(path_or_key)
            if p.exists() and p.is_file():
                size = p.stat().st_size
                data = p.read_bytes()
                sha1 = hashlib.sha1(data).hexdigest()

                result = {
                    "exists": True,
                    "size": size,
                    "sha1": sha1,
                    "source": "local",
                    "presign_ok": False,
                    "mime_guess": mime_guess,
                    "mime": mime_guess,
                }

                # Try to extract image dimensions and variance
                if path_or_key.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    try:
                        from PIL import Image
                        import numpy as np

                        with Image.open(p) as img:
                            result["dimensions"] = {
                                "width": img.width,
                                "height": img.height,
                                "mode": img.mode,
                            }
                            # Calculate std dev for diagnostic
                            arr = np.asarray(img)
                            result["std_dev"] = float(arr.std())
                            pil_mime = Image.MIME.get(getattr(img, "format", "") or "")
                            result["mime_image"] = pil_mime
                            if pil_mime:
                                result["mime"] = pil_mime
                    except Exception as img_err:
                        result["image_error"] = str(img_err)

                return result
        except Exception as e:
            return {
                "exists": False,
                "error": str(e),
                "source": "local",
                "mime_guess": mime_guess,
            }

    # Assume it's an S3 key
    if storage_service.s3_enabled():
        url = storage_service.presign_get(path_or_key, content_type=mime_guess)
        return {
            "exists": bool(url),
            "source": "s3",
            "presign_ok": bool(url),
            "key": path_or_key,
            "mime_guess": mime_guess,
            "mime": mime_guess,
        }

    return {
        "exists": False,
        "error": "Not a local file and S3 not enabled",
        "mime_guess": mime_guess,
    }


@router.get("/health/detector")
def check_detector_health() -> dict:
    """Check RetinaFace detector availability and status."""
    import os
    from pathlib import Path

    result = {
        "retinaface_ready": False,
        "resolved_provider": None,
        "models_present": [],
        "reason_if_false": None,
    }

    # Check for simulated mode environment variable
    if os.getenv("SCREANALYTICS_VISION_SIM") == "1":
        result["reason_if_false"] = "SCREANALYTICS_VISION_SIM=1 (forced simulated mode)"
        return result

    # Try to initialize detector to check availability
    try:
        from tools.episode_run import RetinaFaceDetectorBackend

        detector = RetinaFaceDetectorBackend(device="auto")
        # Try to load model to verify it's available
        detector._get_or_create_model()

        result["retinaface_ready"] = True

        # Try to get ONNX runtime provider info
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            result["resolved_provider"] = providers[0] if providers else "Unknown"
        except Exception:
            result["resolved_provider"] = "ONNX Runtime not available"

    except ImportError as e:
        result["reason_if_false"] = f"Import error: {str(e)}"
    except Exception as e:
        result["reason_if_false"] = f"Initialization error: {str(e)}"

    # Check for model files in expected locations
    model_paths = [
        Path("~/.insightface/models/buffalo_l").expanduser(),
        Path("~/.insightface/models/antelopev2").expanduser(),
    ]

    for model_dir in model_paths:
        if model_dir.exists():
            onnx_files = list(model_dir.glob("*.onnx"))
            result["models_present"].extend([f.name for f in onnx_files])

    return result


__all__ = ["router"]

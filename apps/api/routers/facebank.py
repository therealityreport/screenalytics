"""Facebank management API endpoints."""

from __future__ import annotations

import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import List

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from apps.api.services.cast import CastService
from apps.api.services.facebank import FacebankService
from apps.api.services.jobs import JobService

router = APIRouter()
cast_service = CastService()
facebank_service = FacebankService()
job_service = JobService()
LOGGER = logging.getLogger(__name__)

class FacebankResponse(BaseModel):
    show_id: str
    cast_id: str
    seeds: List[dict]
    exemplars: List[dict]
    stats: dict


class SeedUploadResponse(BaseModel):
    fb_id: str
    cast_id: str
    image_uri: str
    embedding_dim: int
    created_at: str


class DeleteSeedsRequest(BaseModel):
    seed_ids: List[str] = Field(..., description="List of seed IDs to delete")


def _emit_facebank_refresh(show_id: str, cast_id: str, action: str, seed_ids: List[str]) -> str | None:
    if not seed_ids:
        return None
    try:
        record = job_service.emit_facebank_refresh(
            show_id,
            cast_id,
            action=action,
            seed_ids=seed_ids,
        )
    except Exception as exc:  # pragma: no cover - best-effort notification
        LOGGER.warning("Facebank refresh emit failed for %s/%s: %s", show_id, cast_id, exc)
        return None
    return record.get("job_id")


@router.get("/cast/{cast_id}/facebank")
def get_facebank(cast_id: str, show_id: str) -> FacebankResponse:
    """Get facebank data for a cast member."""
    # Verify cast member exists
    member = cast_service.get_cast_member(show_id, cast_id)
    if not member:
        raise HTTPException(status_code=404, detail=f"Cast member {cast_id} not found")

    facebank = facebank_service.get_facebank(show_id, cast_id)
    return FacebankResponse(**facebank)


@router.post("/cast/{cast_id}/seeds/upload")
async def upload_seeds(
    cast_id: str,
    show_id: str,
    files: List[UploadFile] = File(...),
) -> dict:
    """Upload seed images for a cast member.

    Images are validated (must contain exactly 1 face), embedded with ArcFace,
    and stored in the facebank.
    """
    # Verify cast member exists
    member = cast_service.get_cast_member(show_id, cast_id)
    if not member:
        raise HTTPException(status_code=404, detail=f"Cast member {cast_id} not found")

    # Import face detection and embedding here to avoid loading models at startup
    try:
        from FEATURES.detection.src.retinaface_detector import RetinaFaceDetector
        from FEATURES.recognition.src.arcface_embedder import ArcFaceEmbedder
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Face detection/embedding modules not available: {exc}",
        ) from exc

    # Initialize detector and embedder
    detector = RetinaFaceDetector(device="auto")
    embedder = ArcFaceEmbedder(device="auto")
    detector.ensure_ready()
    embedder.ensure_ready()

    uploaded_seeds = []
    errors = []

    for file in files:
        try:
            # Read image
            contents = await file.read()
            image_bytes = BytesIO(contents)

            # Convert to numpy array
            import cv2
            file_bytes = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                errors.append({"file": file.filename, "error": "Failed to decode image"})
                continue

            # Detect faces
            detections = detector.detect(image)

            # Validate: must have exactly 1 face
            if len(detections) == 0:
                errors.append({"file": file.filename, "error": "No face detected"})
                continue
            if len(detections) > 1:
                errors.append({"file": file.filename, "error": f"{len(detections)} faces detected, expected 1"})
                continue

            detection = detections[0]
            bbox = detection.bbox
            landmarks = detection.landmarks

            # Compute quality metrics
            h, w = image.shape[:2]
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            bbox_ratio = bbox_area / (w * h)

            quality = {
                "sharpness": 1.0,  # TODO: compute Laplacian variance
                "occlusion": 0.0,  # TODO: detect occlusion
                "bbox_ratio": float(bbox_ratio),
                "bbox": [float(x) for x in bbox.tolist()],
            }

            # Compute embedding
            embedding = embedder.embed_face(image, bbox, landmarks)

            # Save image
            import uuid
            seed_id = str(uuid.uuid4())
            seeds_dir = facebank_service._seeds_dir(show_id, cast_id)
            image_path = seeds_dir / f"{seed_id}.jpg"

            # Convert to uint8 if needed
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            # Ensure BGR format and contiguous
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = np.ascontiguousarray(image)

            # Save as JPEG
            cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Add to facebank
            seed_entry = facebank_service.add_seed(
                show_id,
                cast_id,
                str(image_path),
                embedding,
                quality=quality,
            )

            uploaded_seeds.append({
                "fb_id": seed_entry["fb_id"],
                "image_uri": seed_entry["image_uri"],
                "filename": file.filename,
            })

        except Exception as exc:
            errors.append({"file": file.filename, "error": str(exc)})

    refresh_job_id = None
    if uploaded_seeds:
        refresh_job_id = _emit_facebank_refresh(
            show_id,
            cast_id,
            "upload",
            [seed["fb_id"] for seed in uploaded_seeds],
        )

    return {
        "cast_id": cast_id,
        "uploaded": len(uploaded_seeds),
        "failed": len(errors),
        "seeds": uploaded_seeds,
        "errors": errors,
        "refresh_job_id": refresh_job_id,
    }


@router.delete("/cast/{cast_id}/seeds")
def delete_seeds(cast_id: str, show_id: str, body: DeleteSeedsRequest) -> dict:
    """Delete seed images from a cast member's facebank."""
    # Verify cast member exists
    member = cast_service.get_cast_member(show_id, cast_id)
    if not member:
        raise HTTPException(status_code=404, detail=f"Cast member {cast_id} not found")

    deleted_count = facebank_service.delete_seeds(show_id, cast_id, body.seed_ids)
    refresh_job_id = None
    if deleted_count:
        refresh_job_id = _emit_facebank_refresh(show_id, cast_id, "delete", body.seed_ids)

    return {
        "cast_id": cast_id,
        "deleted": deleted_count,
        "requested": len(body.seed_ids),
        "refresh_job_id": refresh_job_id,
    }


__all__ = ["router"]

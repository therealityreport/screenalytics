"""Facebank management API endpoints."""

from __future__ import annotations

import json
import logging
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from apps.api.services.cast import CastService
from apps.api.services.facebank import FacebankService
from apps.api.services.jobs import JobService
from apps.api.services.people import PeopleService
from apps.api.services.storage import StorageService
from tools import episode_run

router = APIRouter()
cast_service = CastService()
facebank_service = FacebankService()
job_service = JobService()
people_service = PeopleService()
storage_service = StorageService()
LOGGER = logging.getLogger(__name__)

DIAG = os.getenv("DIAG_LOG", "0") == "1"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(int(raw), 1)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _normalize_format(value: str | None, default: str) -> str:
    normalized = (value or default).strip().lower()
    if normalized in {"jpg", "jpeg"}:
        return "jpg"
    if normalized == "png":
        return "png"
    return default


SEED_DISPLAY_SIZE = _env_int("SEED_DISPLAY_SIZE", 512)
SEED_EMBED_SIZE = _env_int("SEED_EMBED_SIZE", 112)
SEED_DISPLAY_MIN = max(_env_int("SEED_DISPLAY_MIN", 512), 64)
SEED_DISPLAY_MAX = max(SEED_DISPLAY_MIN, _env_int("SEED_DISPLAY_MAX", 1024))
SEED_DISPLAY_FORMAT = _normalize_format(os.getenv("SEED_DISPLAY_FORMAT"), "png")
SEED_EMBED_FORMAT = _normalize_format(os.getenv("SEED_EMBED_FORMAT"), "png")
SEED_FACE_MARGIN = _env_float("SEED_FACE_MARGIN", 0.35)
SEED_MIN_FACE_FRAC = max(_env_float("SEED_MIN_FACE_FRAC", 0.10), 0.0)
JPEG_QUALITY = int(os.getenv("SEED_JPEG_QUALITY", "92"))
_FACEBANK_KEEP_ORIG_RAW = os.getenv("FACEBANK_KEEP_ORIG")
FACEBANK_KEEP_ORIG = (_FACEBANK_KEEP_ORIG_RAW or "1").strip().lower() in {"1", "true", "yes", "on"}
SEED_ORIG_FORMAT = "png"

DISPLAY_MIME = "image/png" if SEED_DISPLAY_FORMAT == "png" else "image/jpeg"
EMBED_MIME = "image/png" if SEED_EMBED_FORMAT == "png" else "image/jpeg"


def _diag(tag: str, **kw) -> None:
    """Diagnostic logger enabled via DIAG_LOG=1."""
    if DIAG:
        LOGGER.info("[DIAG:%s] %s", tag, json.dumps(kw, ensure_ascii=False))


SIMULATED_DETECTOR_MESSAGE = (
    "Using simulated detector. Crops use full image without alignment. Install RetinaFace to improve results."
)


def _letterbox_square(image: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        arr = np.expand_dims(arr, axis=-1)
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return np.full((size, size, 3), 0, dtype=np.uint8)
    scale = min(size / h, size / w)
    new_w = max(int(round(w * scale)), 1)
    new_h = max(int(round(h * scale)), 1)

    resample_attr = getattr(Image, "Resampling", Image)
    resample_filter = getattr(resample_attr, "LANCZOS", getattr(Image, "BICUBIC", Image.NEAREST))

    # Convert BGR to RGB for PIL, resize, then convert back
    pil_img = Image.fromarray(arr[..., ::-1])
    resized = pil_img.resize((new_w, new_h), resample=resample_filter)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas.paste(resized, (left, top))
    return np.ascontiguousarray(np.asarray(canvas)[..., ::-1])


def _resize_display_image(image_bgr: np.ndarray) -> tuple[np.ndarray, list[int], bool]:
    """Resize display crop within configured bounds while preserving aspect ratio."""
    arr = np.asarray(image_bgr)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("display image must be HxWx3")
    h, w = arr.shape[:2]
    long_side = max(h, w)
    if long_side <= 0:
        blank = np.zeros((SEED_DISPLAY_MIN, SEED_DISPLAY_MIN, 3), dtype=np.uint8)
        return blank, [SEED_DISPLAY_MIN, SEED_DISPLAY_MIN], True
    scale = 1.0
    if long_side > SEED_DISPLAY_MAX:
        scale = SEED_DISPLAY_MAX / long_side
    elif long_side >= SEED_DISPLAY_MIN:
        scale = 1.0
    else:
        scale = 1.0
    new_w = max(int(round(w * scale)), 1)
    new_h = max(int(round(h * scale)), 1)
    result = arr
    if scale != 1.0:
        resample_attr = getattr(Image, "Resampling", Image)
        resample_filter = getattr(resample_attr, "LANCZOS", getattr(Image, "BICUBIC", Image.NEAREST))
        pil_img = Image.fromarray(arr[..., ::-1])
        resized = pil_img.resize((new_w, new_h), resample=resample_filter)
        pil_img.close()
        result = np.asarray(resized)[..., ::-1]
        resized.close()
    return np.ascontiguousarray(result), [new_w, new_h], long_side < SEED_DISPLAY_MIN


def _expand_square_bbox(bbox: list[float], margin: float, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    side = max(box_w, box_h)
    side = side * (1.0 + max(margin, 0.0) * 2.0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half = side / 2.0
    new_x1 = cx - half
    new_y1 = cy - half
    new_x2 = cx + half
    new_y2 = cy + half

    if new_x1 < 0:
        new_x2 -= new_x1
        new_x1 = 0.0
    if new_y1 < 0:
        new_y2 -= new_y1
        new_y1 = 0.0
    if new_x2 > width:
        shift = new_x2 - width
        new_x1 -= shift
        new_x2 = float(width)
    if new_y2 > height:
        shift = new_y2 - height
        new_y1 -= shift
        new_y2 = float(height)

    new_x1 = max(new_x1, 0.0)
    new_y1 = max(new_y1, 0.0)
    new_x2 = max(new_x2, new_x1 + 1.0)
    new_y2 = max(new_y2, new_y1 + 1.0)
    return [int(round(new_x1)), int(round(new_y1)), int(round(new_x2)), int(round(new_y2))]


def _prepare_display_crop(
    image_bgr: np.ndarray,
    bbox: list[float],
    detector_mode: str,
) -> tuple[np.ndarray, list[int]]:
    h, w = image_bgr.shape[:2]
    full_box = [0, 0, w, h]
    return np.ascontiguousarray(image_bgr.copy()), full_box


def _save_derivative(image_bgr: np.ndarray, path: Path, fmt: str) -> None:
    target_fmt = "PNG" if fmt == "png" else "JPEG"
    img_rgb = np.ascontiguousarray(image_bgr[..., ::-1])
    image = Image.fromarray(img_rgb)
    save_kwargs: dict = {"optimize": True} if target_fmt == "PNG" else {"quality": JPEG_QUALITY}
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format=target_fmt, **save_kwargs)
    image.close()


class FacebankResponse(BaseModel):
    show_id: str
    cast_id: str
    seeds: List[dict]
    exemplars: List[dict]
    stats: dict
    featured_seed_id: str | None = None


class SeedUploadResponse(BaseModel):
    fb_id: str
    cast_id: str
    image_uri: str
    embedding_dim: int
    created_at: str


class FeatureSeedResponse(BaseModel):
    cast_id: str
    seed_id: str
    image_uri: str


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


def _hydrate_seed_urls(facebank_payload: dict) -> dict:
    """Replace image_uri entries with presigned URLs when S3 keys exist, or API URLs for local storage."""
    show_id = facebank_payload.get("show_id")
    cast_id = facebank_payload.get("cast_id")
    seeds = facebank_payload.get("seeds", [])
    for seed in seeds:
        key = seed.get("display_s3_key") or seed.get("image_s3_key")
        if key:
            url = storage_service.presign_get(key)
            if url:
                seed["display_url"] = url
                seed["image_uri"] = url
                seed.setdefault("display_key", key)
                continue
        seed_id = seed.get("fb_id")
        if seed_id and show_id and cast_id:
            api_path = f"/cast/{cast_id}/seeds/{seed_id}/image?show_id={show_id}"
            seed["image_uri"] = api_path
            seed["display_url"] = api_path
    return facebank_payload


@router.get("/cast/{cast_id}/facebank")
def get_facebank(cast_id: str, show_id: str) -> FacebankResponse:
    """Get facebank data for a cast member."""
    member = cast_service.get_cast_member(show_id, cast_id)
    if not member:
        raise HTTPException(status_code=404, detail=f"Cast member {cast_id} not found")

    facebank = facebank_service.get_facebank(show_id, cast_id)
    facebank = _hydrate_seed_urls(facebank)
    return FacebankResponse(**facebank)


@router.post("/cast/{cast_id}/seeds/upload")
async def upload_seeds(
    cast_id: str,
    show_id: str,
    files: List[UploadFile] = File(...),
) -> JSONResponse:
    """Upload seed images for a cast member."""
    member = cast_service.get_cast_member(show_id, cast_id)
    if not member:
        raise HTTPException(status_code=404, detail=f"Cast member {cast_id} not found")

    try:
        from FEATURES.detection.src.run_retinaface import RetinaFaceDetector
        from tools.episode_run import ArcFaceEmbedder, _prepare_face_crop
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Face detection/embedding modules not available: {exc}",
        ) from exc

    detector_ready, detector_error, detector_provider = episode_run.ensure_retinaface_ready("cpu")
    detector_cfg = {"ctx_id": -1, "force_simulated": not detector_ready}
    detector = RetinaFaceDetector(detector_cfg)
    detector_simulated = bool(getattr(detector, "simulated", False))
    if detector_simulated:
        detector_ready = False
    detector_mode = "simulated" if detector_simulated else "retinaface"
    detector_message = None
    if not detector_ready or detector_simulated:
        detector_message = SIMULATED_DETECTOR_MESSAGE
        if detector_error:
            detector_message = f"{detector_message} ({detector_error})"

    embedder = ArcFaceEmbedder(device="cpu")
    embedder.ensure_ready()

    _diag(
        "SEED_INIT",
        detector_ready=detector_ready,
        detector_simulated=detector_simulated,
        detector_mode=detector_mode,
        detector_error=detector_error,
        detector_provider=detector_provider,
    )

    uploaded_seeds = []
    errors = []

    for file in files:
        try:
            contents = await file.read()
            try:
                pil_image = Image.open(BytesIO(contents))
                pil_image = ImageOps.exif_transpose(pil_image)
                pil_image = pil_image.convert("RGB")
            except (UnidentifiedImageError, OSError):
                errors.append({"file": file.filename, "error": "Unsupported or corrupt image"})
                continue

            image_rgb = np.asarray(pil_image)
            pil_image.close()
            if image_rgb.size == 0:
                errors.append({"file": file.filename, "error": "Unsupported or corrupt image"})
                continue
            image = np.ascontiguousarray(image_rgb[..., ::-1])

            detections = detector(image)
            if detector_simulated:
                detections = [
                    {
                        "bbox": [0.0, 0.0, 1.0, 1.0],
                        "landmarks": [],
                        "conf": 0.0,
                    }
                ]

            if len(detections) == 0:
                errors.append({"file": file.filename, "error": "No face detected"})
                continue
            if len(detections) > 1:
                errors.append({"file": file.filename, "error": f"{len(detections)} faces detected, expected 1"})
                continue

            detection = detections[0]
            bbox_rel = detection["bbox"]
            landmarks_rel = detection["landmarks"]
            conf = detection["conf"]

            h, w = image.shape[:2]
            bbox = [
                bbox_rel[0] * w,
                bbox_rel[1] * h,
                bbox_rel[2] * w,
                bbox_rel[3] * h,
            ]

            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            bbox_ratio = bbox_area / (w * h)

            if detector_mode != "simulated" and bbox_ratio < SEED_MIN_FACE_FRAC:
                pct = bbox_ratio * 100.0
                min_pct = SEED_MIN_FACE_FRAC * 100.0
                raise HTTPException(
                    status_code=422,
                    detail=f"Face too small ({pct:.1f}%). Minimum is {min_pct:.0f}% of the image",
                )

            quality = {
                "sharpness": 1.0,
                "occlusion": 0.0,
                "bbox_ratio": float(bbox_ratio),
                "bbox": bbox,
                "conf": float(conf),
                "detector": detector_mode,
                "detector_model": getattr(detector, "model_id", "retinaface"),
            }
            if detector_provider:
                quality["detector_provider"] = detector_provider

            landmarks_abs = [coord * (w if i % 2 == 0 else h) for i, coord in enumerate(landmarks_rel or [])]

            _diag(
                "SEED_INPUT",
                filename=file.filename,
                image_shape=(h, w),
                bbox_abs=bbox,
                bbox_area=float(bbox_area),
                bbox_ratio=float(bbox_ratio),
                conf=float(conf),
                detector_mode=detector_mode,
            )

            display_crop, display_bbox = _prepare_display_crop(image, bbox, detector_mode)
            display_bgr, display_dims, display_low_res = _resize_display_image(display_crop)

            crop, crop_err = _prepare_face_crop(
                image,
                bbox,
                landmarks_abs,
                margin=SEED_FACE_MARGIN,
                detector_mode=detector_mode,
            )

            crop_shape = crop.shape if crop is not None else None
            crop_std = float(crop.std()) if crop is not None else 0.0
            display_std = float(display_bgr.std()) if display_bgr is not None else 0.0
            _diag(
                "SEED_CROP",
                filename=file.filename,
                crop_shape=crop_shape,
                crop_std=crop_std,
                display_std=display_std,
                crop_err=crop_err,
            )

            if crop is None:
                errors.append({"file": file.filename, "error": f"Failed to crop face: {crop_err}"})
                continue

            embed_crop = np.ascontiguousarray(crop)
            if embed_crop.dtype != np.uint8:
                embed_crop = np.clip(embed_crop, 0, 255).astype(np.uint8, copy=False)

            embeddings = embedder.encode([embed_crop])
            embedding = embeddings[0]

            embed_for_save = embed_crop
            if SEED_EMBED_SIZE != embed_crop.shape[0]:
                embed_for_save = _letterbox_square(embed_crop, SEED_EMBED_SIZE)

            import uuid

            seed_id = str(uuid.uuid4())
            seeds_dir = facebank_service._seeds_dir(show_id, cast_id)
            orig_filename = f"{seed_id}_o.{SEED_ORIG_FORMAT}"
            display_filename = f"{seed_id}_d.{SEED_DISPLAY_FORMAT}"
            embed_filename = f"{seed_id}_e.{SEED_EMBED_FORMAT}"
            orig_path = seeds_dir / orig_filename
            display_path = seeds_dir / display_filename
            embed_path = seeds_dir / embed_filename

            if FACEBANK_KEEP_ORIG:
                _save_derivative(image.astype(np.uint8, copy=False), orig_path, SEED_ORIG_FORMAT)

            _save_derivative(display_bgr.astype(np.uint8, copy=False), display_path, SEED_DISPLAY_FORMAT)
            _save_derivative(embed_for_save, embed_path, SEED_EMBED_FORMAT)
            embed_dims = [int(embed_for_save.shape[1]), int(embed_for_save.shape[0])]

            quality.update({
                "display_bbox": display_bbox,
                "display_std": display_std,
                "embed_std": crop_std,
                "display_dims": display_dims,
                "embed_size": SEED_EMBED_SIZE,
                "display_low_res": display_low_res,
                "source_dims": [w, h],
            })

            display_s3_key = None
            embed_s3_key = None
            orig_s3_key = None
            try:
                display_s3_key = storage_service.upload_facebank_seed(
                    show_id,
                    cast_id,
                    seed_id,
                    display_path,
                    object_name=f"seeds/{display_filename}",
                    content_type_hint=DISPLAY_MIME,
                )
                embed_s3_key = storage_service.upload_facebank_seed(
                    show_id,
                    cast_id,
                    seed_id,
                    embed_path,
                    object_name=f"seeds/{embed_filename}",
                    content_type_hint=EMBED_MIME,
                )
                if FACEBANK_KEEP_ORIG:
                    orig_s3_key = storage_service.upload_facebank_seed(
                        show_id,
                        cast_id,
                        seed_id,
                        orig_path,
                        object_name=f"seeds/{orig_filename}",
                        content_type_hint="image/png",
                    )
            except Exception as exc:  # pragma: no cover - best effort mirror
                LOGGER.error(
                    "Failed to upload facebank derivatives %s/%s/%s: %s",
                    show_id,
                    cast_id,
                    seed_id,
                    exc,
                )

            _diag(
                "SEED_S3",
                filename=file.filename,
                seed_id=seed_id,
                display_key=display_s3_key,
                embed_key=embed_s3_key,
                s3_enabled=storage_service.s3_enabled(),
            )

            seed_entry = facebank_service.add_seed(
                show_id,
                cast_id,
                str(display_path),
                embedding,
                quality=quality,
                image_s3_key=display_s3_key,
                embed_image_path=str(embed_path),
                embed_s3_key=embed_s3_key,
                seed_id=seed_id,
                seed_storage_id=seed_id,
                display_uri=str(display_path),
                embed_uri=str(embed_path),
                display_dims=display_dims,
                embed_dims=embed_dims,
                display_low_res=display_low_res,
                detector_mode=detector_mode,
                orig_image_path=str(orig_path) if FACEBANK_KEEP_ORIG else None,
                orig_s3_key=orig_s3_key,
            )

            presigned = storage_service.presign_get(display_s3_key, content_type=DISPLAY_MIME) if display_s3_key else None

            uploaded_seeds.append({
                "fb_id": seed_entry["fb_id"],
                "image_uri": presigned or seed_entry["image_uri"],
                "filename": file.filename,
                "display_key": display_s3_key,
                "display_dims": display_dims,
                "orig_key": orig_s3_key,
            })

            try:
                display_bytes = display_path.stat().st_size
            except OSError:
                display_bytes = None
            LOGGER.info(
                "Facebank seed upload show=%s cast=%s seed=%s display=%sx%s bytes=%s low_res=%s key=%s",
                show_id,
                cast_id,
                seed_entry["fb_id"],
                display_dims[0],
                display_dims[1],
                display_bytes,
                display_low_res,
                display_s3_key,
            )

            _diag(
                "SEED_RETURN",
                filename=file.filename,
                seed_id=seed_entry["fb_id"],
                presigned_url=presigned,
                fallback_uri=seed_entry["image_uri"],
                embedding_dim=len(embedding),
            )

        except HTTPException:
            raise
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

    response_payload = {
        "cast_id": cast_id,
        "uploaded": len(uploaded_seeds),
        "failed": len(errors),
        "seeds": uploaded_seeds,
        "errors": errors,
        "refresh_job_id": refresh_job_id,
        "detector": detector_mode,
        "detector_ready": detector_mode != "simulated",
        "detector_message": detector_message,
        "detector_provider": detector_provider,
    }

    _diag(
        "SEED_BATCH",
        cast_id=cast_id,
        uploaded_count=len(uploaded_seeds),
        failed_count=len(errors),
        detector_mode=detector_mode,
        refresh_job_id=refresh_job_id,
    )

    status_code = status.HTTP_202_ACCEPTED if detector_mode == "simulated" else status.HTTP_200_OK
    return JSONResponse(response_payload, status_code=status_code)


@router.delete("/cast/{cast_id}/seeds")
def delete_seeds(cast_id: str, show_id: str, body: DeleteSeedsRequest) -> dict:
    """Delete seed images from a cast member's facebank."""
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


@router.post("/cast/{cast_id}/seeds/{seed_id}/feature")
def feature_seed(cast_id: str, seed_id: str, show_id: str) -> FeatureSeedResponse:
    """Mark a seed as the featured facebank image."""
    member = cast_service.get_cast_member(show_id, cast_id)
    if not member:
        raise HTTPException(status_code=404, detail=f"Cast member {cast_id} not found")
    try:
        seed_entry = facebank_service.set_featured_seed(show_id, cast_id, seed_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    image_uri = seed_entry.get("image_uri")
    image_s3_key = seed_entry.get("display_s3_key") or seed_entry.get("image_s3_key")
    display_uri = None
    if image_s3_key:
        display_uri = storage_service.presign_get(image_s3_key)
    elif image_uri:
        display_uri = image_uri

    person = people_service.find_person_by_cast_id(show_id, cast_id)
    if not person:
        people_service.create_person(
            show_id,
            name=member.get("name") or cast_id,
            rep_crop=image_uri,
            rep_crop_s3_key=image_s3_key,
            cast_id=cast_id,
        )
    else:
        people_service.update_person(
            show_id,
            person["person_id"],
            name=member.get("name"),
            rep_crop=image_uri,
            rep_crop_s3_key=image_s3_key,
            cast_id=cast_id,
        )

    return FeatureSeedResponse(cast_id=cast_id, seed_id=seed_id, image_uri=display_uri or "")


@router.get("/cast/{cast_id}/seeds/{seed_id}/image")
def get_seed_image(cast_id: str, seed_id: str, show_id: str) -> FileResponse:
    """Serve a seed image file (fallback for local storage without S3)."""
    facebank = facebank_service.get_facebank(show_id, cast_id)
    seed = next((s for s in facebank.get("seeds", []) if s.get("fb_id") == seed_id), None)
    if not seed:
        raise HTTPException(status_code=404, detail=f"Seed {seed_id} not found")

    image_uri = seed.get("image_uri")
    if not image_uri:
        raise HTTPException(status_code=404, detail="Seed has no image URI")
    path = Path(image_uri)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Seed image not found on disk")
    return FileResponse(path)


__all__ = ["router"]

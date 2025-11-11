from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import cv2  # type: ignore
import numpy as np

LOGGER = logging.getLogger(__name__)


def clip_bbox(x1: float, y1: float, x2: float, y2: float, *, W: int, H: int) -> tuple[int, int, int, int] | None:
    """Clamp an XYXY box to integer pixel coordinates."""
    if W <= 1 or H <= 1:
        return None
    try:
        x1 = max(0, min(int(x1), W - 1))
        x2 = max(0, min(int(x2), W))
        y1 = max(0, min(int(y1), H - 1))
        y2 = max(0, min(int(y2), H))
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def to_u8_bgr(image: np.ndarray) -> np.ndarray:
    """Return a contiguous uint8 BGR image."""
    if image is None:
        return image
    arr = np.asarray(image)
    if arr.size == 0:
        return arr
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255)
        max_val = float(arr.max()) if arr.size else 0.0
        if max_val <= 1.0:
            arr = arr * 255.0
        arr = arr.astype(np.uint8, copy=False)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    else:
        arr = np.broadcast_to(arr[..., None], arr.shape + (3,))
    return np.ascontiguousarray(arr)


def safe_crop(frame_bgr, bbox: Iterable[float]) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, str | None]:
    """Crop using clip_bbox + dtype normalization."""
    if frame_bgr is None:
        return None, None, "frame_missing"
    arr = np.asarray(frame_bgr)
    if arr.ndim < 2:
        return None, None, "invalid_frame"
    H, W = arr.shape[:2]
    try:
        x1, y1, x2, y2 = bbox
    except Exception:
        return None, None, "invalid_bbox"
    clipped = clip_bbox(x1, y1, x2, y2, W=W, H=H)
    if clipped is None:
        return None, None, "degenerate_bbox"
    rx1, ry1, rx2, ry2 = clipped
    crop = arr[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return None, clipped, "empty_slice"
    return to_u8_bgr(crop), clipped, None


def safe_imwrite(path: str | Path, image, jpg_q: int = 85) -> tuple[bool, str | None]:
    """Write JPEGs with variance + size guards."""
    if image is None:
        return False, "image_missing"
    img = to_u8_bgr(np.asarray(image))
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jpeg_q = max(1, min(int(jpg_q or 85), 100))
    variance = float(np.std(img)) if img.size else 0.0
    range_val = (
        float(np.nanmax(img)) - float(np.nanmin(img))
        if img.size
        else 0.0
    )
    ok = cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    if not ok:
        return False, "imwrite_failed"
    try:
        size_bytes = out_path.stat().st_size
    except OSError:
        size_bytes = 0
    if size_bytes < 1024:
        try:
            out_path.unlink()
        except OSError:
            pass
        return False, "tiny_file"
    if variance <= 0.05 and range_val <= 1.0:
        try:
            out_path.unlink()
        except OSError:
            pass
        LOGGER.warning(
            "Removed near-uniform image %s (std=%.5f range=%.3f)", out_path, variance, range_val
        )
        return False, "near_uniform_gray"
    return True, None

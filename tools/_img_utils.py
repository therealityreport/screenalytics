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


def safe_crop(
    frame_bgr, bbox: Iterable[float]
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, str | None]:
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


def safe_imwrite(
    path: str | Path,
    image,
    jpg_q: int = 85,
    *,
    use_png: bool = False,
    png_compression: int = 3,
) -> tuple[bool, str | None]:
    """Write images with variance + size guards.

    Args:
        path: Output file path (extension determines format if use_png not set)
        image: Image array to write
        jpg_q: JPEG quality (1-100, default 85)
        use_png: Force PNG format for maximum quality (lossless)
        png_compression: PNG compression level (0-9, default 3 for balance)

    Returns:
        (success, error_reason) tuple
    """
    if image is None:
        return False, "image_missing"
    img = to_u8_bgr(np.asarray(image))
    out_path = Path(path)

    # Determine format from extension or use_png flag
    suffix = out_path.suffix.lower()
    is_png = use_png or suffix == ".png"

    # If use_png is True but path has .jpg, change extension
    if use_png and suffix in (".jpg", ".jpeg"):
        out_path = out_path.with_suffix(".png")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    variance = float(np.std(img)) if img.size else 0.0
    range_val = float(np.nanmax(img)) - float(np.nanmin(img)) if img.size else 0.0

    if is_png:
        # PNG: lossless compression for maximum quality
        compression = max(0, min(int(png_compression), 9))
        ok = cv2.imwrite(str(out_path), img, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    else:
        # JPEG: lossy compression
        jpeg_q = max(1, min(int(jpg_q or 85), 100))
        ok = cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])

    if not ok:
        return False, "imwrite_failed"
    try:
        size_bytes = out_path.stat().st_size
    except OSError:
        size_bytes = 0
    # Lower threshold to 256 bytes - small face crops (20-30px) can be under 1KB
    # but still valid. Only catch truly degenerate/corrupted writes.
    if size_bytes < 256:
        try:
            out_path.unlink()
        except OSError:
            # File may already be removed by another cleanup step.
            pass
        return False, "tiny_file"
    if variance <= 0.05 and range_val <= 1.0:
        try:
            out_path.unlink()
        except OSError:
            # Ignore if concurrent deletion already removed the file.
            pass
        LOGGER.warning(
            "Removed near-uniform image %s (std=%.5f range=%.3f)",
            out_path,
            variance,
            range_val,
        )
        return False, "near_uniform_gray"
    return True, None


def encode_png_bytes(image, *, color: str = "bgr", compression: int = 3) -> bytes | None:
    """Encode image to PNG bytes for S3 upload (lossless).

    Args:
        image: Image array
        color: Color space ("bgr" or "rgb")
        compression: PNG compression level (0-9)

    Returns:
        PNG bytes or None if encoding fails
    """
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.size == 0:
        return None
    arr = to_u8_bgr(arr)
    if color == "rgb":
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    compression = max(0, min(int(compression), 9))
    success, encoded = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    if not success:
        return None
    return encoded.tobytes()

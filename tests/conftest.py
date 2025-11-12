import os
import sys
import types
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("STORAGE_BACKEND", "local")

# Provide a lightweight cv2 stub when OpenCV is unavailable (CI/unit tests).
try:  # pragma: no cover - exercised only when cv2 is missing
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - dependency guard
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.COLOR_GRAY2BGR = 0
    cv2_stub.COLOR_BGR2GRAY = 1
    cv2_stub.COLOR_BGR2RGB = 2
    cv2_stub.IMWRITE_JPEG_QUALITY = 0

    def _cvt_color(image, code):
        arr = np.asarray(image)
        if code == cv2_stub.COLOR_GRAY2BGR:
            if arr.ndim == 2:
                return np.repeat(arr[..., None], 3, axis=2)
            return arr
        if code == cv2_stub.COLOR_BGR2GRAY:
            if arr.ndim == 2:
                return arr
            return arr[..., 0]
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[..., ::-1]
        return arr

    def _resize(image, dsize):
        width, height = dsize
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = arr[..., None]
        y_idx = np.linspace(0, arr.shape[0] - 1, max(height, 1)).astype(int)
        x_idx = np.linspace(0, arr.shape[1] - 1, max(width, 1)).astype(int)
        resized = arr[y_idx][:, x_idx]
        if resized.ndim == 2:
            return np.repeat(resized[..., None], 3, axis=2)
        return resized

    def _imwrite(path, img, params=None):
        return True

    def _min_max_loc(image):
        arr = np.asarray(image)
        if arr.size == 0:
            return 0.0, 0.0, (0, 0), (0, 0)
        min_val = float(arr.min())
        max_val = float(arr.max())
        min_idx = int(arr.argmin())
        max_idx = int(arr.argmax())
        width = arr.shape[1] if arr.ndim >= 2 else 1
        min_loc = (min_idx % width, min_idx // width)
        max_loc = (max_idx % width, max_idx // width)
        return min_val, max_val, min_loc, max_loc

    cv2_stub.cvtColor = _cvt_color
    cv2_stub.resize = _resize
    cv2_stub.imwrite = _imwrite
    cv2_stub.minMaxLoc = _min_max_loc
    sys.modules["cv2"] = cv2_stub

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class HeadPoseEstimate:
    """Lightweight head pose estimate in degrees."""

    yaw: float
    pitch: float
    roll: float
    reprojection_error_px: float | None = None
    source: str = "pnp_2d_landmarks"


def _rotation_matrix_to_euler_xyz(rot: np.ndarray) -> tuple[float, float, float]:
    """Return (pitch, yaw, roll) in radians for rotation matrix using XYZ convention."""
    if rot.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation matrix, got {rot.shape}")

    sy = math.sqrt(float(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0]))
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        yaw = math.atan2(float(-rot[2, 0]), sy)
        roll = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        pitch = math.atan2(float(-rot[1, 2]), float(rot[1, 1]))
        yaw = math.atan2(float(-rot[2, 0]), sy)
        roll = 0.0

    return pitch, yaw, roll


def estimate_head_pose_pnp(
    landmarks_68: Sequence[Sequence[float]],
    *,
    image_shape: Sequence[int],
) -> HeadPoseEstimate | None:
    """Estimate head pose (yaw/pitch/roll) from 68-point 2D landmarks via solvePnP.

    This is a deterministic MVP pose estimator that does not require a 3DDFA model.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        return None

    lm = np.asarray(landmarks_68, dtype=np.float32)
    if lm.ndim != 2 or lm.shape[0] < 68 or lm.shape[1] < 2:
        return None

    try:
        height = int(image_shape[0])
        width = int(image_shape[1])
    except Exception:
        return None
    if height <= 0 or width <= 0:
        return None

    # Standard 3D face model reference points (approximate, in arbitrary units).
    model_points = np.asarray(
        [
            (0.0, 0.0, 0.0),  # nose tip
            (0.0, -330.0, -65.0),  # chin
            (-225.0, 170.0, -135.0),  # left eye left corner
            (225.0, 170.0, -135.0),  # right eye right corner
            (-150.0, -150.0, -125.0),  # left mouth corner
            (150.0, -150.0, -125.0),  # right mouth corner
        ],
        dtype=np.float32,
    )

    # 2D landmark indices for the corresponding facial features.
    image_points = np.asarray(
        [
            lm[30],  # nose tip
            lm[8],  # chin
            lm[36],  # left eye left corner
            lm[45],  # right eye right corner
            lm[48],  # left mouth corner
            lm[54],  # right mouth corner
        ],
        dtype=np.float32,
    )

    focal_length = float(width)
    center = (width / 2.0, height / 2.0)
    camera_matrix = np.asarray(
        [
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    rot, _ = cv2.Rodrigues(rvec)
    pitch_rad, yaw_rad, roll_rad = _rotation_matrix_to_euler_xyz(rot)

    yaw = float(math.degrees(yaw_rad))
    pitch = float(math.degrees(pitch_rad))
    roll = float(math.degrees(roll_rad))

    reprojection_error_px: float | None = None
    try:
        projected, _ = cv2.projectPoints(model_points, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        err = np.linalg.norm(projected - image_points, axis=1).mean()
        reprojection_error_px = float(err)
    except Exception:
        reprojection_error_px = None

    return HeadPoseEstimate(
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        reprojection_error_px=reprojection_error_px,
        source="pnp_2d_landmarks",
    )

"""
Export aligned faces to JSONL and optional crop images.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import cv2

from .run_fan_alignment import AlignedFace


logger = logging.getLogger(__name__)


def export_aligned_faces(
    aligned_faces: List[AlignedFace],
    output_path: Path,
    crops_dir: Optional[Path] = None,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
) -> int:
    """
    Export aligned faces to JSONL and optionally save crop images.

    Args:
        aligned_faces: List of AlignedFace results
        output_path: Path to output JSONL file
        crops_dir: Directory for crop images (optional)
        image_format: Image format for crops ("jpg" or "png")
        jpeg_quality: JPEG quality (1-100)

    Returns:
        Number of faces exported
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if crops_dir:
        crops_dir = Path(crops_dir)
        crops_dir.mkdir(parents=True, exist_ok=True)

    exported = 0

    with open(output_path, "w") as f:
        for aligned in aligned_faces:
            # Save crop if requested
            if crops_dir and aligned.crop is not None:
                crop_filename = f"frame_{aligned.frame_idx:06d}"
                if aligned.track_id is not None:
                    crop_filename += f"_track_{aligned.track_id}"
                crop_filename += f".{image_format}"

                crop_path = crops_dir / crop_filename

                if image_format == "jpg":
                    cv2.imwrite(
                        str(crop_path),
                        aligned.crop,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                    )
                else:
                    cv2.imwrite(str(crop_path), aligned.crop)

                aligned.crop_path = str(crop_path.relative_to(output_path.parent))

            # Write JSONL entry
            f.write(json.dumps(aligned.to_dict()) + "\n")
            exported += 1

    logger.info(f"Exported {exported} aligned faces to {output_path}")

    if crops_dir:
        # Write marker file
        (crops_dir / ".done").touch()
        logger.info(f"Saved {exported} crop images to {crops_dir}")

    return exported


def load_aligned_faces(input_path: Path) -> List[AlignedFace]:
    """
    Load aligned faces from JSONL file.

    Args:
        input_path: Path to aligned_faces.jsonl

    Returns:
        List of AlignedFace objects
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Aligned faces not found: {input_path}")

    faces = []
    with open(input_path) as f:
        for line in f:
            data = json.loads(line)
            faces.append(AlignedFace.from_dict(data))

    logger.info(f"Loaded {len(faces)} aligned faces from {input_path}")
    return faces


def compute_alignment_stats(
    aligned_faces: List,
    quality_threshold: float = 0.6,
) -> dict:
    """
    Compute statistics over aligned faces.

    Args:
        aligned_faces: List of AlignedFace objects or dicts
        quality_threshold: Threshold for counting high-quality faces

    Returns:
        Dict with statistics
    """
    import numpy as np

    if not aligned_faces:
        return {"total_faces": 0, "count": 0}

    def get_attr(obj, key, default=None):
        """Get attribute from object or dict."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    confidences = [get_attr(f, "confidence", 0.9) for f in aligned_faces]
    track_ids = set(
        get_attr(f, "track_id")
        for f in aligned_faces
        if get_attr(f, "track_id") is not None
    )
    frame_indices = [get_attr(f, "frame_idx", 0) for f in aligned_faces]

    stats = {
        "total_faces": len(aligned_faces),
        "count": len(aligned_faces),
        "unique_tracks": len(track_ids),
        "frames_covered": len(set(frame_indices)),
        "frame_range": [min(frame_indices), max(frame_indices)],
        "confidence_mean": float(np.mean(confidences)),
        "confidence_std": float(np.std(confidences)),
        "confidence_min": float(np.min(confidences)),
        "confidence_max": float(np.max(confidences)),
    }

    # Quality stats if available
    qualities = [
        get_attr(f, "alignment_quality")
        for f in aligned_faces
        if get_attr(f, "alignment_quality") is not None
    ]
    if qualities:
        stats["mean_quality"] = float(np.mean(qualities))
        stats["alignment_quality_mean"] = float(np.mean(qualities))
        stats["alignment_quality_std"] = float(np.std(qualities))
        stats["quality_above_threshold"] = sum(1 for q in qualities if q >= quality_threshold)

    return stats

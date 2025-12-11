"""
Pipeline integration for face alignment.

Provides a clean interface for integrating FAN-based alignment
into the main episode_run.py pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


logger = logging.getLogger(__name__)


def load_face_alignment_config(config_path: Optional[Path] = None) -> Dict:
    """
    Load face alignment configuration.

    Args:
        config_path: Optional override path

    Returns:
        Config dict with defaults for missing keys
    """
    config_path = config_path or Path("config/pipeline/face_alignment.yaml")

    defaults = {
        "face_alignment": {
            "enabled": False,  # Disabled by default until integrated
            "model": {
                "type": "2d",
                "landmarks_type": "2D",
                "flip_input": False,
            },
            "processing": {
                "batch_size": 16,
                "stride": 1,
                "device": "auto",
            },
            "quality": {
                "min_face_size": 20,
                "min_confidence": 0.5,
            },
            "output": {
                "export_crops": False,
                "crop_size": 112,
                "crop_margin": 0.0,
            },
        },
        "use_aligned_crops_for_embeddings": False,
    }

    if not config_path.exists():
        logger.warning(f"Face alignment config not found at {config_path}, using defaults")
        return defaults

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Merge with defaults
        if config:
            for key, value in config.items():
                if key in defaults and isinstance(value, dict):
                    defaults[key].update(value)
                else:
                    defaults[key] = value

        return defaults
    except Exception as e:
        logger.warning(f"Failed to load face alignment config: {e}")
        return defaults


def is_face_alignment_enabled(config: Dict) -> bool:
    """Check if face alignment is enabled in config."""
    return config.get("face_alignment", {}).get("enabled", False)


def run_face_alignment_stage(
    episode_id: str,
    manifest_dir: Path,
    video_path: Path,
    config: Dict,
    device: Optional[str] = None,
    skip_existing: bool = False,
) -> Tuple[bool, Optional[Path]]:
    """
    Run face alignment stage for an episode.

    Args:
        episode_id: Episode identifier
        manifest_dir: Path to manifest directory
        video_path: Path to video file
        config: Face alignment configuration
        device: Optional device override
        skip_existing: Skip if output exists

    Returns:
        Tuple of (success, output_path)
    """
    if not is_face_alignment_enabled(config):
        logger.info("[FACE_ALIGNMENT] Skipped - disabled in config")
        return True, None

    output_dir = manifest_dir / "face_alignment"
    output_path = output_dir / "aligned_faces.jsonl"

    if skip_existing and output_path.exists():
        logger.info(f"[FACE_ALIGNMENT] Skipped - output exists: {output_path}")
        return True, output_path

    try:
        from .face_alignment_runner import FaceAlignmentConfig, FaceAlignmentRunner

        # Build config from pipeline config
        fa_config = config.get("face_alignment", {})
        model_config = fa_config.get("model", {})
        processing_config = fa_config.get("processing", {})
        quality_config = fa_config.get("quality", {})
        output_config = fa_config.get("output", {})

        runner_config = FaceAlignmentConfig(
            model_type=model_config.get("type", "2d"),
            landmarks_type=model_config.get("landmarks_type", "2D"),
            flip_input=model_config.get("flip_input", False),
            batch_size=processing_config.get("batch_size", 16),
            stride=processing_config.get("stride", 1),
            device=device or processing_config.get("device", "auto"),
            min_face_size=quality_config.get("min_face_size", 20),
            min_confidence=quality_config.get("min_confidence", 0.5),
            export_crops=output_config.get("export_crops", False),
            crop_size=output_config.get("crop_size", 112),
            crop_margin=output_config.get("crop_margin", 0.0),
        )

        runner = FaceAlignmentRunner(
            episode_id=episode_id,
            video_path=video_path,
            output_dir=output_dir,
            skip_existing=skip_existing,
        )
        runner.config = runner_config

        logger.info(f"[FACE_ALIGNMENT] Running for {episode_id}")
        runner.run_alignment()

        logger.info(f"[FACE_ALIGNMENT] Complete: {output_path}")
        return True, output_path

    except ImportError as e:
        logger.error(f"[FACE_ALIGNMENT] Import error: {e}")
        logger.error("Install face-alignment package: pip install face-alignment")
        return False, None
    except FileNotFoundError as e:
        logger.error(f"[FACE_ALIGNMENT] File not found: {e}")
        return False, None
    except Exception as e:
        logger.exception(f"[FACE_ALIGNMENT] Failed: {e}")
        return False, None


def load_aligned_crops_for_embedding(
    manifest_dir: Path,
    track_ids: Optional[List[int]] = None,
) -> Dict[int, List[Dict]]:
    """
    Load aligned faces for use in embedding stage.

    Args:
        manifest_dir: Path to manifest directory
        track_ids: Optional filter for specific tracks

    Returns:
        Dict mapping track_id to list of aligned face dicts
    """
    aligned_path = manifest_dir / "face_alignment" / "aligned_faces.jsonl"

    if not aligned_path.exists():
        logger.warning(f"Aligned faces not found: {aligned_path}")
        return {}

    by_track = {}
    with open(aligned_path) as f:
        for line in f:
            face = json.loads(line)
            track_id = face.get("track_id")

            if track_id is None:
                continue

            if track_ids is not None and track_id not in track_ids:
                continue

            if track_id not in by_track:
                by_track[track_id] = []
            by_track[track_id].append(face)

    logger.info(f"Loaded aligned faces for {len(by_track)} tracks")
    return by_track


def get_aligned_crop_path(
    manifest_dir: Path,
    frame_idx: int,
    track_id: Optional[int] = None,
) -> Optional[Path]:
    """
    Get path to aligned crop image if it exists.

    Args:
        manifest_dir: Path to manifest directory
        frame_idx: Frame index
        track_id: Optional track ID

    Returns:
        Path to crop image or None
    """
    crops_dir = manifest_dir / "face_alignment" / "aligned_crops"

    if not crops_dir.exists():
        return None

    # Try different naming patterns
    patterns = [
        f"frame_{frame_idx:06d}_track_{track_id}.jpg" if track_id else None,
        f"frame_{frame_idx:06d}.jpg",
        f"frame_{frame_idx:06d}_track_{track_id}.png" if track_id else None,
        f"frame_{frame_idx:06d}.png",
    ]

    for pattern in patterns:
        if pattern:
            path = crops_dir / pattern
            if path.exists():
                return path

    return None

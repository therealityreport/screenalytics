"""Facebank seed selection and export for permanent identity storage.

This module selects high-quality face crops from episode identities and writes them
to the permanent facebank for similarity matching across episodes.

Quality Criteria:
- Detection confidence >= 0.75
- Crop sharpness (std) >= 15.0
- Similarity to identity centroid >= 0.70
- No skip flag
- Crop file must exist

Selection Strategy:
- Sort by quality score (weighted combination of det_score, sharpness, similarity)
- Take top N frames (default 20, max 50)
- Ensure diversity (avoid selecting consecutive frames)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

# Quality thresholds for seed selection
SEED_DET_MIN = float(os.getenv("SEED_DET_MIN", "0.75"))
SEED_STD_MIN = float(os.getenv("SEED_STD_MIN", "15.0"))
SEED_SIM_MIN = float(os.getenv("SEED_SIM_MIN", "0.70"))
SEED_MAX_COUNT = int(os.getenv("SEED_MAX_COUNT", "20"))
SEED_DIVERSITY_GAP = int(os.getenv("SEED_DIVERSITY_GAP", "10"))  # Min frame gap between seeds


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    norm = np.linalg.norm(vector) + 1e-12
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors (0-1)."""
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def _extract_quality_metrics(face: Dict[str, Any]) -> Tuple[float, float, float]:
    """Extract quality metrics from a face record.

    Returns:
        (det_score, crop_std, box_area)
    """
    # Detection confidence
    det_score = face.get("det_score") or face.get("conf") or face.get("confidence") or 0.0
    if not det_score:
        quality = face.get("quality")
        if isinstance(quality, dict):
            det_score = quality.get("det") or 0.0

    # Crop standard deviation (sharpness)
    std = face.get("crop_std") or 0.0
    if not std:
        quality = face.get("quality")
        if isinstance(quality, dict):
            std = quality.get("std") or 0.0

    # Face box area (if available)
    box = face.get("box") or face.get("bbox")
    box_area = 0.0
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            width = abs(float(box[2]) - float(box[0]))
            height = abs(float(box[3]) - float(box[1]))
            box_area = width * height
        except (TypeError, ValueError, IndexError):
            box_area = 0.0

    return float(det_score), float(std), float(box_area)


def _compute_quality_score(
    det_score: float,
    crop_std: float,
    box_area: float,
    similarity: float,
) -> float:
    """Compute a weighted quality score for seed selection.

    Higher scores = better quality.

    Weights:
    - Detection confidence: 30%
    - Sharpness (crop_std): 30%
    - Similarity to identity: 30%
    - Face box area: 10%
    """
    # Normalize scores to 0-1 range
    det_norm = min(max(det_score, 0.0), 1.0)

    # Normalize std (higher is sharper, cap at 100 for normalization)
    std_norm = min(crop_std / 100.0, 1.0)

    # Normalize similarity (already 0-1)
    sim_norm = min(max(similarity, 0.0), 1.0)

    # Normalize box area (cap at 100,000 pixels squared)
    area_norm = min(box_area / 100000.0, 1.0)

    # Weighted combination
    score = (0.30 * det_norm) + (0.30 * std_norm) + (0.30 * sim_norm) + (0.10 * area_norm)

    return float(score)


def _compute_identity_centroid(faces: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    """Compute the centroid embedding for an identity from its faces."""
    embeddings: List[np.ndarray] = []

    for face in faces:
        if face.get("skip"):
            continue

        embedding = face.get("embedding")
        if not embedding:
            continue

        embeddings.append(np.array(embedding, dtype=np.float32))

    if not embeddings:
        LOGGER.warning("No valid embeddings found for identity centroid")
        return None

    # Compute centroid as L2-normalized mean
    mean_embed = np.mean(embeddings, axis=0)
    return l2_normalize(mean_embed)


def _resolve_crop_path(ep_id: str, face: Dict[str, Any]) -> Optional[Path]:
    """Resolve the crop file path for a face."""
    from py_screenalytics.artifacts import get_path

    # Try crop_rel_path first
    crop_rel = face.get("crop_rel_path")
    if crop_rel:
        frames_root = get_path(ep_id, "frames_root")
        crop_path = frames_root / crop_rel
        if crop_path.exists():
            return crop_path

        # Try fallback root
        fallback_root = Path(os.environ.get("SCREENALYTICS_CROPS_FALLBACK_ROOT", "data/crops")).expanduser()
        crop_path = fallback_root / ep_id / crop_rel.replace("crops/", "")
        if crop_path.exists():
            return crop_path

    # Try thumb_rel_path as fallback
    thumb_rel = face.get("thumb_rel_path")
    if thumb_rel:
        frames_root = get_path(ep_id, "frames_root")
        thumb_path = frames_root / thumb_rel
        if thumb_path.exists():
            return thumb_path

    return None


def select_facebank_seeds(
    ep_id: str,
    identity_id: str,
    identity_faces: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Select high-quality seed frames for facebank export.

    Args:
        ep_id: Episode ID
        identity_id: Identity/cluster ID
        identity_faces: List of face records for this identity

    Returns:
        List of seed records with metadata:
        [
            {
                "face_id": "...",
                "frame_idx": 123,
                "track_id": 5,
                "crop_path": Path(...),
                "embedding": [...],
                "quality": {"det": 0.85, "std": 25.3, "sim": 0.92, "score": 0.87},
                "ts": "2025-11-18T10:30:00Z"
            },
            ...
        ]
    """
    if not identity_faces:
        LOGGER.warning(f"No faces provided for identity {identity_id}")
        return []

    # Compute identity centroid for similarity scoring
    centroid = _compute_identity_centroid(identity_faces)
    if centroid is None:
        LOGGER.warning(f"Cannot compute centroid for identity {identity_id}, using fallback")
        # Fallback: select based on quality only (no similarity scoring)
        centroid = None

    # Score all candidates
    candidates: List[Tuple[Dict[str, Any], Path, float, float]] = []  # (face, crop_path, quality_score, similarity)

    for face in identity_faces:
        # Skip flagged faces
        if face.get("skip"):
            continue

        # Resolve crop file
        crop_path = _resolve_crop_path(ep_id, face)
        if not crop_path:
            continue

        # Extract quality metrics
        det_score, crop_std, box_area = _extract_quality_metrics(face)

        # Compute similarity to centroid
        similarity = 0.0
        if centroid is not None:
            embedding = face.get("embedding")
            if embedding:
                face_embed = np.array(embedding, dtype=np.float32)
                similarity = cosine_similarity(face_embed, centroid)

        # Apply quality gates
        if det_score < SEED_DET_MIN:
            continue
        if crop_std < SEED_STD_MIN:
            continue
        if centroid is not None and similarity < SEED_SIM_MIN:
            continue

        # Compute overall quality score
        quality_score = _compute_quality_score(det_score, crop_std, box_area, similarity)

        candidates.append((face, crop_path, quality_score, similarity))

    if not candidates:
        LOGGER.warning(
            f"No candidates passed quality gates for {identity_id}. "
            f"Thresholds: det>={SEED_DET_MIN}, std>={SEED_STD_MIN}, sim>={SEED_SIM_MIN}"
        )
        return []

    # Sort by quality score (descending)
    candidates.sort(key=lambda x: x[2], reverse=True)

    # Select diverse seeds (avoid consecutive frames)
    selected_seeds: List[Dict[str, Any]] = []
    last_frame_idx: Optional[int] = None

    for face, crop_path, quality_score, similarity in candidates:
        if len(selected_seeds) >= SEED_MAX_COUNT:
            break

        # Ensure diversity: skip frames too close to the last selected frame
        frame_idx = face.get("frame_idx")
        if last_frame_idx is not None and frame_idx is not None:
            if abs(frame_idx - last_frame_idx) < SEED_DIVERSITY_GAP:
                continue

        # Extract final quality metrics
        det_score, crop_std, box_area = _extract_quality_metrics(face)

        seed = {
            "face_id": face.get("face_id"),
            "frame_idx": frame_idx,
            "track_id": face.get("track_id"),
            "crop_path": crop_path,
            "embedding": face.get("embedding"),
            "quality": {
                "det": round(float(det_score), 3),
                "std": round(float(crop_std), 1),
                "sim": round(float(similarity), 3),
                "score": round(float(quality_score), 4),
            },
            "ts": face.get("ts"),
        }

        selected_seeds.append(seed)
        last_frame_idx = frame_idx

    LOGGER.info(f"Selected {len(selected_seeds)} seeds for {identity_id} from {len(candidates)} candidates")

    return selected_seeds


def write_facebank_seeds(
    person_id: str,
    seeds: List[Dict[str, Any]],
    facebank_root: Path,
) -> Path:
    """Write seed frames to permanent facebank storage.

    Atomically writes seeds and index to {facebank_root}/{person_id}/

    Args:
        person_id: Person/cast ID
        seeds: List of seed records from select_facebank_seeds()
        facebank_root: Root directory for facebank storage

    Returns:
        Path to the person's facebank directory

    Raises:
        OSError: If file operations fail
        ValueError: If inputs are invalid
    """
    if not person_id or not person_id.strip():
        raise ValueError("person_id must be a non-empty string")

    if not seeds:
        raise ValueError("seeds list cannot be empty")

    # Sanitize person_id (prevent path traversal)
    safe_person_id = person_id.strip().replace("/", "_").replace("\\", "_")
    if safe_person_id != person_id.strip():
        LOGGER.warning(f"Sanitized person_id from '{person_id}' to '{safe_person_id}'")

    person_dir = facebank_root / safe_person_id

    # Atomic write: use temp directory, then move
    temp_dir = Path(tempfile.mkdtemp(prefix=f"facebank_{safe_person_id}_", dir=facebank_root.parent))

    try:
        # Copy seed images
        seed_index: List[Dict[str, Any]] = []

        for idx, seed in enumerate(seeds):
            crop_path = seed.get("crop_path")
            if not crop_path or not isinstance(crop_path, Path):
                LOGGER.warning(f"Skipping seed {idx} with invalid crop_path")
                continue

            if not crop_path.exists():
                LOGGER.warning(f"Skipping seed {idx} with missing crop file: {crop_path}")
                continue

            # Generate facebank filename
            frame_idx = seed.get("frame_idx", idx)
            track_id = seed.get("track_id", 0)
            suffix = crop_path.suffix or ".jpg"
            seed_filename = f"seed_{idx:03d}_track{track_id:04d}_frame{frame_idx:06d}{suffix}"

            dest_path = temp_dir / seed_filename

            # Copy file
            try:
                shutil.copy2(crop_path, dest_path)
            except OSError as e:
                LOGGER.error(f"Failed to copy {crop_path} to {dest_path}: {e}")
                raise

            # Build index entry
            index_entry = {
                "seed_id": f"seed_{idx:03d}",
                "filename": seed_filename,
                "face_id": seed.get("face_id"),
                "frame_idx": frame_idx,
                "track_id": track_id,
                "quality": seed.get("quality"),
                "ts": seed.get("ts"),
                "exported_at": _now_iso(),
            }

            # Include embedding if present
            embedding = seed.get("embedding")
            if embedding:
                index_entry["embedding"] = embedding

            seed_index.append(index_entry)

        if not seed_index:
            raise ValueError("No valid seeds could be written (all crop files missing)")

        # Write index.json
        index_path = temp_dir / "index.json"
        index_data = {
            "person_id": safe_person_id,
            "seeds_count": len(seed_index),
            "seeds": seed_index,
            "created_at": _now_iso(),
            "schema_version": "facebank_v1",
        }

        try:
            index_path.write_text(json.dumps(index_data, indent=2), encoding="utf-8")
        except OSError as e:
            LOGGER.error(f"Failed to write index.json: {e}")
            raise

        # Atomic move: replace existing person directory
        person_dir.parent.mkdir(parents=True, exist_ok=True)

        if person_dir.exists():
            # Backup existing directory
            backup_dir = person_dir.parent / f"{safe_person_id}.bak.{int(datetime.utcnow().timestamp())}"
            try:
                shutil.move(str(person_dir), str(backup_dir))
                LOGGER.info(f"Backed up existing facebank to {backup_dir}")
            except OSError as e:
                LOGGER.warning(f"Failed to backup existing facebank: {e}")

        # Move temp to final location
        try:
            shutil.move(str(temp_dir), str(person_dir))
        except OSError as e:
            LOGGER.error(f"Failed to move temp directory to {person_dir}: {e}")
            raise

        LOGGER.info(f"Wrote {len(seed_index)} seeds to facebank: {person_dir}")

        return person_dir

    except Exception:
        # Cleanup temp directory on error
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        raise

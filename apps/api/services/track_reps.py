"""Track representatives and cluster centroids computation service."""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from py_screenalytics.artifacts import get_path
from py_screenalytics import run_layout

LOGGER = logging.getLogger(__name__)

# Quality gates for representative frames
REP_DET_MIN = float(os.getenv("REP_DET_MIN", "0.50"))  # Lowered from 0.60 to accept more frames
REP_STD_MIN = float(os.getenv("REP_STD_MIN", "1.0"))
REP_MAX_FRAMES_PER_TRACK = int(os.getenv("REP_MAX_FRAMES_PER_TRACK", "50"))
REP_MIN_SIM_TO_CENTROID = float(os.getenv("REP_MIN_SIM_TO_CENTROID", "0.50"))
REP_HIGH_SIM_THRESHOLD = float(
    os.getenv("REP_HIGH_SIM_THRESHOLD", "0.85")
)  # Accept lower quality if similarity is excellent


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    norm = np.linalg.norm(vector) + 1e-12
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors (0-1)."""
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def _normalize_run_id(run_id: str | None) -> str | None:
    if run_id is None:
        return None
    candidate = str(run_id).strip()
    if not candidate:
        return None
    try:
        return run_layout.normalize_run_id(candidate)
    except ValueError:
        return None


def _manifests_dir(ep_id: str, *, run_id: str | None = None) -> Path:
    run_id_norm = _normalize_run_id(run_id)
    if run_id_norm:
        return run_layout.run_root(ep_id, run_id_norm)
    return get_path(ep_id, "detections").parent


def _faces_path(ep_id: str, *, run_id: str | None = None) -> Path:
    return _manifests_dir(ep_id, run_id=run_id) / "faces.jsonl"


def _tracks_path(ep_id: str, *, run_id: str | None = None) -> Path:
    run_id_norm = _normalize_run_id(run_id)
    if run_id_norm:
        return _manifests_dir(ep_id, run_id=run_id_norm) / "tracks.jsonl"
    return get_path(ep_id, "tracks")


def _track_reps_path(ep_id: str, *, run_id: str | None = None) -> Path:
    return _manifests_dir(ep_id, run_id=run_id) / "track_reps.jsonl"


def _cluster_centroids_path(ep_id: str, *, run_id: str | None = None) -> Path:
    return _manifests_dir(ep_id, run_id=run_id) / "cluster_centroids.json"


def _identities_path(ep_id: str, *, run_id: str | None = None) -> Path:
    return _manifests_dir(ep_id, run_id=run_id) / "identities.json"


def _crops_root(ep_id: str) -> Path:
    return get_path(ep_id, "frames_root") / "crops"


def _discover_crop_path(
    ep_id: str,
    track_id: int,
    frame_idx: int,
    *,
    run_id: str | None = None,
) -> Optional[str]:
    """Find the crop file for a specific track and frame.

    Primary: data/frames/<ep_id>/crops/track_<ID>/frame_<frame_idx>.jpg
    Fallback 1: data/crops/<ep_id>/tracks/track_<ID>/frame_<frame_idx>.jpg
    Fallback 2: artifacts/crops/<show>/<season>/<episode>/tracks/track_<ID>/frame_<frame_idx>.jpg
    """
    track_component = f"track_{track_id:04d}"
    frame_filename_jpg = f"frame_{frame_idx:06d}.jpg"
    frame_filename_png = f"frame_{frame_idx:06d}.png"

    frames_root = get_path(ep_id, "frames_root")
    candidate_crops_roots: list[Path] = []
    run_candidates: list[str] = []
    if run_id:
        try:
            run_candidates.append(run_layout.normalize_run_id(run_id))
        except ValueError:
            pass
    active_run = run_layout.read_active_run_id(ep_id)
    if active_run and active_run not in run_candidates:
        run_candidates.append(active_run)
    for rid in run_candidates:
        candidate_crops_roots.append(frames_root / "runs" / rid / "crops")
    candidate_crops_roots.append(frames_root / "crops")

    for crops_root in candidate_crops_roots:
        jpg_path = crops_root / track_component / frame_filename_jpg
        if jpg_path.exists():
            return f"crops/{track_component}/{frame_filename_jpg}"
        png_path = crops_root / track_component / frame_filename_png
        if png_path.exists():
            return f"crops/{track_component}/{frame_filename_png}"

    # Fallback locations
    fallback_root = Path(os.environ.get("SCREENALYTICS_CROPS_FALLBACK_ROOT", "data/crops")).expanduser()
    legacy1 = fallback_root / ep_id / "tracks" / track_component / frame_filename_jpg
    if legacy1.exists():
        return f"crops/{track_component}/{frame_filename_jpg}"

    legacy1_png = fallback_root / ep_id / "tracks" / track_component / frame_filename_png
    if legacy1_png.exists():
        return f"crops/{track_component}/{frame_filename_png}"

    return None


def _load_faces_for_track(ep_id: str, track_id: int, *, run_id: str | None = None) -> List[Dict[str, Any]]:
    """Load all faces for a specific track."""
    faces_path = _faces_path(ep_id, run_id=run_id)
    if not faces_path.exists():
        return []

    track_faces: List[Dict[str, Any]] = []
    with faces_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                face = json.loads(line)
            except json.JSONDecodeError:
                continue

            if face.get("track_id") == track_id:
                track_faces.append(face)

    return track_faces


def _extract_quality_metrics(face: Dict[str, Any]) -> Tuple[float, float, float]:
    """Extract quality metrics from a face record.

    Returns:
        (det_score, crop_std, box_area)
    """
    # Detection confidence
    det_score = face.get("det_score") or face.get("conf") or 0.0
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
        # box format: [x1, y1, x2, y2] or [x, y, w, h]
        # Assume [x1, y1, x2, y2] format
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
) -> float:
    """Compute a weighted quality score for representative frame selection.

    Higher scores = better quality.

    Weights:
    - Detection confidence: 40%
    - Sharpness (crop_std): 35%
    - Face box area: 25%
    """
    # Normalize scores to 0-1 range
    det_norm = min(max(det_score, 0.0), 1.0)

    # Normalize std (higher is sharper, cap at 100 for normalization)
    std_norm = min(crop_std / 100.0, 1.0)

    # Normalize box area (cap at 100,000 pixels squared)
    area_norm = min(box_area / 100000.0, 1.0)

    # Weighted combination
    score = (0.40 * det_norm) + (0.35 * std_norm) + (0.25 * area_norm)

    return float(score)


def _passes_quality_gates(face: Dict[str, Any], crop_path: Optional[str]) -> bool:
    """Check if a face passes minimum quality gates for representative selection."""
    det_score, crop_std, _ = _extract_quality_metrics(face)

    if det_score < REP_DET_MIN:
        return False

    if crop_std < REP_STD_MIN:
        return False

    # Crop file must exist
    if not crop_path:
        return False

    # Skip flag
    if face.get("skip"):
        return False

    return True


def compute_track_representative(
    ep_id: str,
    track_id: int,
    *,
    run_id: str | None = None,
) -> Optional[Dict[str, Any]]:
    """Compute representative frame and centroid for a track.

    Selects the HIGHEST-QUALITY frame that passes similarity and quality thresholds,
    not just the first frame that passes gates.

    Returns:
        {
            "track_id": "track_0001",
            "rep_frame": 123,
            "crop_key": "crops/track_0001/frame_000123.jpg",
            "embed": [...512-d L2-norm...],
            "quality": {"det": 0.82, "std": 15.3, "box_area": 12345, "score": 0.87},
            "sim_to_centroid": 0.95,  # optional
            "low_confidence": false,
            "rep_low_quality": false  # true when had to fall back to low-quality frames
        }
    """
    faces = _load_faces_for_track(ep_id, track_id, run_id=run_id)
    if not faces:
        return None

    # Sort by frame index
    faces.sort(key=lambda f: f.get("frame_idx", float("inf")))

    # STEP 1: Collect embeddings for track centroid computation (from quality-passing frames)
    embeddings: List[np.ndarray] = []
    for face in faces[:REP_MAX_FRAMES_PER_TRACK]:
        embedding = face.get("embedding")
        if not embedding:
            continue

        frame_idx = face.get("frame_idx")
        if frame_idx is None:
            continue

        crop_path = _discover_crop_path(ep_id, track_id, frame_idx, run_id=run_id)
        if _passes_quality_gates(face, crop_path):
            embeddings.append(np.array(embedding, dtype=np.float32))

    # Fallback: use all available embeddings if no accepted frames
    if not embeddings:
        for face in faces[:REP_MAX_FRAMES_PER_TRACK]:
            embedding = face.get("embedding")
            if embedding:
                embeddings.append(np.array(embedding, dtype=np.float32))

    # Compute track centroid if we have embeddings
    track_centroid: Optional[np.ndarray] = None
    no_embeddings = False
    if embeddings:
        mean_embed = np.mean(embeddings, axis=0)
        track_centroid = l2_normalize(mean_embed)
    else:
        LOGGER.info(f"No embeddings found for track {track_id} in {ep_id}, selecting by quality only")
        no_embeddings = True

    # STEP 2: Score all candidate frames and select the BEST one
    candidates: List[Tuple[Dict[str, Any], str, float, float]] = []  # (face, crop_path, quality_score, similarity)
    skipped_candidates: List[Tuple[Dict[str, Any], str, float, float]] = []  # Fallback for all-skipped tracks

    for face in faces:
        frame_idx = face.get("frame_idx")
        if frame_idx is None:
            continue

        crop_path = _discover_crop_path(ep_id, track_id, frame_idx, run_id=run_id)
        if not crop_path:
            continue

        # Extract quality metrics
        det_score, crop_std, box_area = _extract_quality_metrics(face)
        quality_score = _compute_quality_score(det_score, crop_std, box_area)

        # Compute similarity to track centroid (if we have one)
        embedding = face.get("embedding")
        similarity = 0.0
        if embedding and track_centroid is not None:
            face_embed = np.array(embedding, dtype=np.float32)
            similarity = cosine_similarity(face_embed, track_centroid)

        # Separate skipped faces into fallback list
        if face.get("skip"):
            skipped_candidates.append((face, crop_path, quality_score, similarity))
            continue

        candidates.append((face, crop_path, quality_score, similarity))

    # Fall back to skipped candidates if no non-skipped candidates exist
    # This ensures tracks with all-skipped faces still get a representative
    all_skipped_fallback = False
    if not candidates:
        if skipped_candidates:
            LOGGER.info(f"Track {track_id}: using skipped faces as fallback (all faces were auto-skipped)")
            candidates = skipped_candidates
            all_skipped_fallback = True
        else:
            LOGGER.warning(f"No valid candidates for track {track_id} in {ep_id}")
            return None

    # First pass: find best frame that passes BOTH quality gates AND similarity threshold
    high_quality_candidates = [
        (face, crop_path, quality_score, similarity)
        for face, crop_path, quality_score, similarity in candidates
        if _passes_quality_gates(face, crop_path) and similarity >= REP_MIN_SIM_TO_CENTROID
    ]

    rep_face = None
    rep_crop_key = None
    rep_similarity = 0.0
    rep_quality_score = 0.0
    low_confidence = False
    rep_low_quality = False

    if high_quality_candidates:
        # Sort by quality score (descending) and pick the best
        high_quality_candidates.sort(key=lambda x: x[2], reverse=True)
        rep_face, rep_crop_key, rep_quality_score, rep_similarity = high_quality_candidates[0]
    else:
        # Second pass: prioritize high-similarity frames even if they fail quality gates
        # If similarity is excellent (≥0.85), accept lower quality frames
        high_sim_candidates = [
            (face, crop_path, quality_score, similarity)
            for face, crop_path, quality_score, similarity in candidates
            if similarity >= REP_HIGH_SIM_THRESHOLD
        ]

        if high_sim_candidates:
            # Sort by similarity first, then quality
            high_sim_candidates.sort(key=lambda x: (x[3], x[2]), reverse=True)
            rep_face, rep_crop_key, rep_quality_score, rep_similarity = high_sim_candidates[0]
            LOGGER.info(
                "Track %d: using high-similarity rep (sim=%.3f, score=%.3f)",
                track_id,
                rep_similarity,
                rep_quality_score,
            )
        else:
            # Third pass: find best frame that passes quality gates (ignore similarity)
            quality_only_candidates = [
                (face, crop_path, quality_score, similarity)
                for face, crop_path, quality_score, similarity in candidates
                if _passes_quality_gates(face, crop_path)
            ]

            if quality_only_candidates:
                quality_only_candidates.sort(key=lambda x: x[2], reverse=True)
                rep_face, rep_crop_key, rep_quality_score, rep_similarity = quality_only_candidates[0]
                low_confidence = True
                LOGGER.info(
                    "Track %d: using quality-only rep with sim %.3f < %.3f",
                    track_id,
                    rep_similarity,
                    REP_MIN_SIM_TO_CENTROID,
                )
            else:
                # Final fallback: pick best available frame by quality score (no gates)
                candidates.sort(key=lambda x: x[2], reverse=True)
                rep_face, rep_crop_key, rep_quality_score, rep_similarity = candidates[0]
                low_confidence = True
                rep_low_quality = True
                LOGGER.warning(
                    "Track %d: using low-quality fallback rep (score=%.3f, sim=%.3f)",
                    track_id,
                    rep_quality_score,
                    rep_similarity,
                )

    if not rep_face:
        LOGGER.warning(f"No representative found for track {track_id} in {ep_id}")
        return None

    # Extract final quality metrics
    det_score, crop_std, box_area = _extract_quality_metrics(rep_face)

    result = {
        "track_id": f"track_{track_id:04d}",
        "rep_frame": rep_face.get("frame_idx"),
        "crop_key": rep_crop_key,
        "embed": track_centroid.tolist() if track_centroid is not None else None,
        "quality": {
            "det": round(float(det_score), 3),
            "std": round(float(crop_std), 1),
            "box_area": round(float(box_area), 1),
            "score": round(float(rep_quality_score), 4),
        },
    }

    if rep_similarity > 0:
        result["sim_to_centroid"] = round(float(rep_similarity), 4)

    if low_confidence:
        result["low_confidence"] = True

    if rep_low_quality:
        result["rep_low_quality"] = True

    if all_skipped_fallback:
        result["all_skipped"] = True

    if no_embeddings:
        result["no_embeddings"] = True

    return result


def compute_all_track_reps(
    ep_id: str,
    *,
    run_id: str | None = None,
    return_stats: bool = False,
) -> List[Dict[str, Any]] | Dict[str, Any]:
    """Compute track representatives for all tracks in an episode.

    Args:
        ep_id: Episode ID
        return_stats: If True, return dict with reps and statistics

    Returns:
        If return_stats is False: List of track rep dicts
        If return_stats is True: Dict with 'reps', 'tracks_processed', 'tracks_with_reps', 'tracks_skipped'
    """
    tracks_path = _tracks_path(ep_id, run_id=run_id)
    if not tracks_path.exists():
        LOGGER.warning(f"No tracks file found for {ep_id}")
        if return_stats:
            return {"reps": [], "tracks_processed": 0, "tracks_with_reps": 0, "tracks_skipped": 0}
        return []

    track_reps: List[Dict[str, Any]] = []
    tracks_processed = 0
    tracks_skipped = 0

    with tracks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                track = json.loads(line)
            except json.JSONDecodeError:
                continue

            track_id = track.get("track_id")
            if track_id is None:
                continue

            tracks_processed += 1
            rep = compute_track_representative(ep_id, track_id, run_id=run_id)
            if rep:
                track_reps.append(rep)
            else:
                tracks_skipped += 1

    if return_stats:
        return {
            "reps": track_reps,
            "tracks_processed": tracks_processed,
            "tracks_with_reps": len(track_reps),
            "tracks_skipped": tracks_skipped,
        }
    return track_reps


def write_track_reps(ep_id: str, track_reps: List[Dict[str, Any]], *, run_id: str | None = None) -> Path:
    """Write track representatives to track_reps.jsonl."""
    run_id_norm = _normalize_run_id(run_id)
    path = _track_reps_path(ep_id, run_id=run_id_norm)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for rep in track_reps:
            f.write(json.dumps(rep) + "\n")

    LOGGER.info(f"Wrote {len(track_reps)} track representatives to {path}")

    # Backwards compatibility: keep legacy root copy in sync for UI/features that
    # still read from data/manifests/{ep_id}/track_reps.jsonl.
    if run_id_norm:
        legacy_path = _track_reps_path(ep_id, run_id=None)
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = legacy_path.with_suffix(legacy_path.suffix + ".tmp")
        try:
            shutil.copy2(path, tmp)
            tmp.replace(legacy_path)
        except OSError as exc:
            LOGGER.warning("Failed to promote track reps to legacy path %s: %s", legacy_path, exc)
    return path


def load_track_reps(ep_id: str, *, run_id: str | None = None) -> Dict[str, Dict[str, Any]]:
    """Load track representatives from track_reps.jsonl.

    Returns: Dict mapping track_id -> rep data
    """
    path = _track_reps_path(ep_id, run_id=run_id)
    if not path.exists():
        return {}

    reps: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rep = json.loads(line)
            except json.JSONDecodeError:
                continue

            track_id = rep.get("track_id")
            if track_id:
                reps[track_id] = rep

    return reps


def compute_cluster_centroids(
    ep_id: str,
    track_reps: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    run_id: str | None = None,
) -> Dict[str, Any]:
    """Compute cluster centroids from track representatives.

    Args:
        ep_id: Episode ID
        track_reps: Optional pre-loaded track representatives. If None, will load from disk.

    Returns:
        {
            "id_0001": {
                "centroid": [...512-d...],
                "tracks": ["track_0001", "track_0022", ...],
                "cohesion": 0.74
            },
            ...
        }
    """
    if track_reps is None:
        track_reps = load_track_reps(ep_id, run_id=run_id)

    if not track_reps:
        LOGGER.warning(f"No track representatives found for {ep_id}")
        return {}

    # Load identities to get cluster assignments
    identities_path = _identities_path(ep_id, run_id=run_id)
    if not identities_path.exists():
        LOGGER.warning(f"No identities file found for {ep_id}")
        return {}

    identities_data = json.loads(identities_path.read_text(encoding="utf-8"))
    identities = identities_data.get("identities", [])

    # Build cluster -> track mapping
    cluster_to_tracks: Dict[str, List[str]] = {}
    for identity in identities:
        cluster_id = identity.get("identity_id")
        if not cluster_id:
            continue

        track_ids = identity.get("track_ids", [])
        # Normalize track IDs to track_XXXX format
        normalized_tracks = []
        for tid in track_ids:
            if isinstance(tid, int):
                normalized_tracks.append(f"track_{tid:04d}")
            elif isinstance(tid, str):
                if tid.startswith("track_"):
                    normalized_tracks.append(tid)
                else:
                    try:
                        normalized_tracks.append(f"track_{int(tid):04d}")
                    except (TypeError, ValueError):
                        continue

        cluster_to_tracks[cluster_id] = normalized_tracks

    # Compute centroids
    centroids: Dict[str, Any] = {}
    for cluster_id, track_ids in cluster_to_tracks.items():
        # Collect track centroids for this cluster
        track_centroids: List[np.ndarray] = []
        valid_tracks: List[str] = []

        for track_id in track_ids:
            rep = track_reps.get(track_id)
            if not rep:
                continue

            embed = rep.get("embed")
            if not embed:
                continue

            track_centroids.append(np.array(embed, dtype=np.float32))
            valid_tracks.append(track_id)

        if not track_centroids:
            continue

        # Compute cluster centroid as mean of track centroids
        mean_centroid = np.mean(track_centroids, axis=0)
        cluster_centroid = l2_normalize(mean_centroid)

        # Compute cohesion as mean similarity of tracks to centroid
        similarities = [cosine_similarity(tc, cluster_centroid) for tc in track_centroids]
        cohesion = float(np.mean(similarities))

        centroids[cluster_id] = {
            "centroid": cluster_centroid.tolist(),
            "tracks": valid_tracks,
            "cohesion": round(cohesion, 3),
        }

    return centroids


def write_cluster_centroids(ep_id: str, centroids: Dict[str, Any], *, run_id: str | None = None) -> Path:
    """Write cluster centroids to cluster_centroids.json."""
    run_id_norm = _normalize_run_id(run_id)
    path = _cluster_centroids_path(ep_id, run_id=run_id_norm)
    path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "ep_id": ep_id,
        "centroids": centroids,
        "computed_at": _now_iso(),
    }

    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    LOGGER.info(f"Wrote {len(centroids)} cluster centroids to {path}")

    # Backwards compatibility: keep legacy root copy in sync for UI/features that
    # still read from data/manifests/{ep_id}/cluster_centroids.json.
    if run_id_norm:
        legacy_path = _cluster_centroids_path(ep_id, run_id=None)
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = legacy_path.with_suffix(legacy_path.suffix + ".tmp")
        try:
            shutil.copy2(path, tmp)
            tmp.replace(legacy_path)
        except OSError as exc:
            LOGGER.warning("Failed to promote cluster centroids to legacy path %s: %s", legacy_path, exc)
    return path


def load_cluster_centroids(ep_id: str, *, run_id: str | None = None) -> Dict[str, Any]:
    """Load cluster centroids from cluster_centroids.json.

    For legacy formats, derives track lists from identities.json.
    """
    path = _cluster_centroids_path(ep_id, run_id=run_id)
    if not path.exists():
        LOGGER.warning(f"cluster_centroids.json does not exist for {ep_id}")
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        centroids = data.get("centroids") if isinstance(data, dict) else None

        # Handle old format (list) vs new format (dict)
        if isinstance(centroids, list):
            LOGGER.warning(f"cluster_centroids.json has old list format for {ep_id}, converting...")

            # Load identities to get track_ids for each cluster
            from apps.api.services.identities import load_identities

            identities_data = load_identities(ep_id, run_id=run_id)
            identity_tracks_map = {}
            for identity in identities_data.get("identities", []):
                identity_id = identity.get("identity_id")
                if identity_id:
                    track_ids = identity.get("track_ids", [])
                    # Convert to track_XXXX format
                    identity_tracks_map[identity_id] = [f"track_{int(tid):04d}" for tid in track_ids]

            # Convert old format to new format
            centroids_dict = {}
            for item in centroids:
                if isinstance(item, dict) and "cluster_id" in item:
                    cluster_id = item["cluster_id"]
                    tracks = identity_tracks_map.get(cluster_id, [])
                    centroids_dict[cluster_id] = {
                        "centroid": item.get("centroid", []),
                        "tracks": tracks,  # Derived from identities.json
                        "cohesion": item.get("cohesion"),  # Preserve if available
                    }

            LOGGER.info(f"Converted {len(centroids_dict)} legacy centroids with tracks from identities.json")
            return centroids_dict
        elif isinstance(centroids, dict):
            return centroids
        else:
            LOGGER.error(f"Unexpected centroids type: {type(centroids)}")
            return {}
    except json.JSONDecodeError as e:
        LOGGER.warning(f"Failed to parse cluster centroids for {ep_id}: {e}")
        return {}
    except Exception as e:
        LOGGER.error(f"Unexpected error loading cluster centroids for {ep_id}: {e}")
        return {}


def build_cluster_track_reps(
    ep_id: str,
    cluster_id: str,
    track_reps: Optional[Dict[str, Dict[str, Any]]] = None,
    cluster_centroids: Optional[Dict[str, Any]] = None,
    *,
    run_id: str | None = None,
) -> Dict[str, Any]:
    """Build track representatives with similarity scores for a cluster.

    Returns:
        {
            "cluster_id": "id_0001",
            "cluster_centroid": [...],
            "cohesion": 0.74,
            "tracks": [
                {
                    "track_id": "track_0001",
                    "rep_frame": 123,
                    "crop_key": "crops/track_0001/frame_000123.jpg",
                    "similarity": 0.82,
                    "quality": {"det": 0.82, "std": 15.3}
                },
                ...
            ],
            "total_tracks": 13
        }
    """
    if track_reps is None:
        track_reps = load_track_reps(ep_id, run_id=run_id)

    if cluster_centroids is None:
        cluster_centroids = load_cluster_centroids(ep_id, run_id=run_id)

    cluster_data = cluster_centroids.get(cluster_id)

    # If cluster not in centroids (e.g., newly created via manual assignment),
    # fall back to loading track_ids directly from identities.json
    if not cluster_data:
        from apps.api.services.identities import load_identities

        identities_data = load_identities(ep_id)
        identities = identities_data.get("identities", [])
        identity = next(
            (ident for ident in identities if ident.get("identity_id") == cluster_id),
            None,
        )
        if not identity:
            return {
                "cluster_id": cluster_id,
                "cluster_centroid": None,
                "cohesion": None,
                "tracks": [],
                "total_tracks": 0,
            }

        # Get track_ids from identity and normalize to track_XXXX format
        raw_track_ids = identity.get("track_ids", [])
        track_ids = []
        for tid in raw_track_ids:
            if isinstance(tid, int):
                track_ids.append(f"track_{tid:04d}")
            elif isinstance(tid, str):
                if tid.startswith("track_"):
                    track_ids.append(tid)
                else:
                    try:
                        track_ids.append(f"track_{int(tid):04d}")
                    except (TypeError, ValueError):
                        continue

        # Build tracks output without centroid similarity (no centroid available)
        tracks_output: List[Dict[str, Any]] = []
        for track_id in track_ids:
            rep = track_reps.get(track_id)
            if not rep:
                # Try to compute on-the-fly for tracks missing from track_reps.jsonl
                # This handles tracks with all-skipped faces that weren't included before
                try:
                    tid_int = int(track_id.replace("track_", "")) if isinstance(track_id, str) else int(track_id)
                    computed_rep = compute_track_representative(ep_id, tid_int)
                    if computed_rep:
                        rep = computed_rep
                        LOGGER.info(f"Computed on-the-fly rep for {track_id}")
                except Exception as e:
                    LOGGER.warning(f"Failed to compute rep for {track_id}: {e}")
            if not rep:
                tracks_output.append(
                    {
                        "track_id": track_id,
                        "rep_frame": None,
                        "crop_key": None,
                        "similarity": None,
                        "quality": None,
                        "embedding": None,
                    }
                )
                continue

            tracks_output.append(
                {
                    "track_id": track_id,
                    "rep_frame": rep.get("rep_frame"),
                    "crop_key": rep.get("crop_key"),
                    "similarity": None,  # No centroid to compare against
                    "quality": rep.get("quality"),
                    "embedding": rep.get("embed"),  # Include embedding for cross-track scoring
                }
            )

        return {
            "cluster_id": cluster_id,
            "cluster_centroid": None,
            "cohesion": None,
            "tracks": tracks_output,
            "total_tracks": len(tracks_output),
        }

    cluster_centroid_vec = np.array(cluster_data["centroid"], dtype=np.float32)
    track_ids = cluster_data.get("tracks", [])
    cohesion = cluster_data.get("cohesion")

    # Build track reps with similarity scores
    tracks_output: List[Dict[str, Any]] = []
    for track_id in track_ids:
        rep = track_reps.get(track_id)
        if not rep:
            # Try to compute on-the-fly for tracks missing from track_reps.jsonl
            # This handles tracks with all-skipped faces that weren't included before
            try:
                tid_int = int(track_id.replace("track_", "")) if isinstance(track_id, str) else int(track_id)
                computed_rep = compute_track_representative(ep_id, tid_int)
                if computed_rep:
                    rep = computed_rep
                    LOGGER.info(f"Computed on-the-fly rep for {track_id} in cluster {cluster_id}")
            except Exception as e:
                LOGGER.warning(f"Failed to compute rep for {track_id}: {e}")
        if not rep:
            # Include missing tracks with placeholder data
            tracks_output.append(
                {
                    "track_id": track_id,
                    "rep_frame": None,
                    "crop_key": None,
                    "similarity": None,
                    "quality": None,
                    "embedding": None,
                }
            )
            continue

        # Compute similarity
        track_centroid_vec = np.array(rep["embed"], dtype=np.float32)
        similarity = cosine_similarity(track_centroid_vec, cluster_centroid_vec)

        tracks_output.append(
            {
                "track_id": track_id,
                "rep_frame": rep.get("rep_frame"),
                "crop_key": rep.get("crop_key"),
                "similarity": round(float(similarity), 3),
                "quality": rep.get("quality"),
                "embedding": rep.get("embed"),  # Include embedding for cross-track scoring
            }
        )

    return {
        "cluster_id": cluster_id,
        "cluster_centroid": cluster_data["centroid"],
        "cohesion": cohesion,
        "tracks": tracks_output,
        "total_tracks": len(tracks_output),
    }


def generate_track_reps_and_centroids(ep_id: str, *, run_id: str | None = None) -> Dict[str, Any]:
    """Complete pipeline: compute track reps and cluster centroids.

    Returns: Summary with paths and counts matching expected keys:
        - tracks_processed: Total tracks attempted
        - tracks_with_reps: Tracks with valid representatives
        - tracks_skipped: Tracks skipped (no embeddings)
        - centroids_computed: Number of cluster centroids computed
    """
    LOGGER.info(f"Generating track representatives for {ep_id}")
    track_reps_result = compute_all_track_reps(ep_id, run_id=run_id, return_stats=True)
    track_reps_list = track_reps_result["reps"]
    track_reps_map = {rep["track_id"]: rep for rep in track_reps_list}

    track_reps_path = write_track_reps(ep_id, track_reps_list, run_id=run_id)

    LOGGER.info(f"Generating cluster centroids for {ep_id}")
    centroids = compute_cluster_centroids(ep_id, track_reps_map, run_id=run_id)

    centroids_path = write_cluster_centroids(ep_id, centroids, run_id=run_id)

    return {
        # Keys expected by run_refresh_similarity_task
        "tracks_processed": track_reps_result["tracks_processed"],
        "tracks_with_reps": track_reps_result["tracks_with_reps"],
        "tracks_skipped": track_reps_result["tracks_skipped"],
        "centroids_computed": len(centroids),
        # Additional paths for debugging/logging
        "track_reps_path": str(track_reps_path),
        "cluster_centroids_path": str(centroids_path),
    }

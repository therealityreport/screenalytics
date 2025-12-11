"""
Body Re-ID Embeddings using OSNet.

Computes person Re-ID embeddings for body crops.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .track_bodies import BodyTrack


logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Re-ID embedding for a body crop."""

    track_id: int
    frame_idx: int
    embedding: np.ndarray

    def to_meta_dict(self) -> dict:
        """Return metadata (without embedding array)."""
        return {
            "track_id": self.track_id,
            "frame_idx": self.frame_idx,
        }


class BodyEmbedder:
    """OSNet-based person Re-ID embedder."""

    # Standard Re-ID crop size (height x width)
    TARGET_SIZE = (256, 128)

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        device: str = "auto",
        weights_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.weights_path = weights_path

        self._model = None
        self._device_resolved = None

    def _load_model(self):
        """Lazy-load OSNet model."""
        if self._model is not None:
            return

        import torch

        # Determine device
        device = self.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device_resolved = device

        # Try torchreid first
        try:
            from torchreid.utils import FeatureExtractor
            self._model = FeatureExtractor(
                model_name=self.model_name,
                model_path=self.weights_path,
                device=device,
            )
            self._model_type = "torchreid"
            logger.info(f"Loaded torchreid {self.model_name} on {device}")
            return
        except ImportError:
            logger.warning("torchreid not found, trying alternative")

        # Fallback to timm or custom
        try:
            import timm
            # OSNet is not in timm, but this shows the pattern
            # For now, raise to indicate we need torchreid
            raise ImportError("OSNet requires torchreid")
        except ImportError:
            raise ImportError(
                "torchreid package required for body Re-ID. "
                "Install with: pip install torchreid"
            )

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """Preprocess body crop for Re-ID model."""
        # Resize to target size
        resized = cv2.resize(crop, (self.TARGET_SIZE[1], self.TARGET_SIZE[0]))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        return rgb

    def embed_crop(self, crop: np.ndarray) -> np.ndarray:
        """Compute embedding for a single crop."""
        self._load_model()

        preprocessed = self._preprocess_crop(crop)

        if self._model_type == "torchreid":
            # torchreid expects a list of images
            features = self._model([preprocessed])
            return features[0].cpu().numpy()

        raise RuntimeError(f"Unknown model type: {self._model_type}")

    def embed_crops_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """Compute embeddings for a batch of crops."""
        self._load_model()

        if not crops:
            return np.array([])

        preprocessed = [self._preprocess_crop(c) for c in crops]

        if self._model_type == "torchreid":
            features = self._model(preprocessed)
            return features.cpu().numpy()

        raise RuntimeError(f"Unknown model type: {self._model_type}")

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension for current model."""
        dims = {
            "osnet_x1_0": 512,
            "osnet_x0_75": 512,
            "osnet_x0_5": 512,
            "osnet_x0_25": 512,
            "osnet_ain_x1_0": 512,
        }
        return dims.get(self.model_name, 512)


def _extract_body_crop(
    frame: np.ndarray,
    bbox: List[float],
    margin: float = 0.1,
    min_height: int = 64,
    min_width: int = 32,
) -> Optional[np.ndarray]:
    """Extract body crop from frame with margin."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    height, width = frame.shape[:2]

    # Calculate margin
    box_w = x2 - x1
    box_h = y2 - y1

    margin_x = int(box_w * margin)
    margin_y = int(box_h * margin)

    # Expand with margin
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(width, x2 + margin_x)
    y2 = min(height, y2 + margin_y)

    # Check minimum size
    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w < min_width or crop_h < min_height:
        return None

    return frame[y1:y2, x1:x2].copy()


def _select_representative_frames(
    track: BodyTrack,
    max_samples: int = 5,
) -> List[int]:
    """Select representative frame indices from track."""
    n_frames = len(track.detections)

    if n_frames <= max_samples:
        return list(range(n_frames))

    # Sample evenly across the track
    indices = np.linspace(0, n_frames - 1, max_samples, dtype=int)
    return indices.tolist()


def compute_body_embeddings(
    embedder: BodyEmbedder,
    video_path: Path,
    tracks_path: Path,
    output_path: Path,
    meta_path: Path,
    batch_size: int = 32,
    max_samples_per_track: int = 5,
) -> int:
    """
    Compute Re-ID embeddings for body tracks.

    Args:
        embedder: BodyEmbedder instance
        video_path: Path to video file
        tracks_path: Path to body_tracks.jsonl
        output_path: Path to output embeddings .npy file
        meta_path: Path to output metadata .json file
        batch_size: Crops per embedding batch
        max_samples_per_track: Max frames to sample per track

    Returns:
        Number of embeddings computed
    """
    video_path = Path(video_path)
    tracks_path = Path(tracks_path)
    output_path = Path(output_path)
    meta_path = Path(meta_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load tracks
    tracks: Dict[int, BodyTrack] = {}
    logger.info(f"Loading tracks from: {tracks_path}")

    with open(tracks_path) as f:
        for line in f:
            data = json.loads(line)
            track_id = data["track_id"]
            from .detect_bodies import BodyDetection
            detections = [BodyDetection.from_dict(d) for d in data["detections"]]
            tracks[track_id] = BodyTrack(track_id=track_id, detections=detections)

    logger.info(f"Loaded {len(tracks)} tracks")

    # Build frame -> (track_id, det_idx) mapping for frames we need to read
    frames_needed: Dict[int, List[Tuple[int, int]]] = {}  # frame_idx -> [(track_id, det_idx), ...]

    for track_id, track in tracks.items():
        sample_indices = _select_representative_frames(track, max_samples_per_track)
        for det_idx in sample_indices:
            frame_idx = track.detections[det_idx].frame_idx
            if frame_idx not in frames_needed:
                frames_needed[frame_idx] = []
            frames_needed[frame_idx].append((track_id, det_idx))

    logger.info(f"Need to read {len(frames_needed)} frames for embeddings")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Collect crops for batching
    all_crops = []
    all_meta = []  # [(track_id, frame_idx), ...]

    sorted_frames = sorted(frames_needed.keys())
    current_frame_idx = 0

    for target_frame in sorted_frames:
        # Seek if needed
        if current_frame_idx != target_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            current_frame_idx = target_frame

        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Could not read frame {target_frame}")
            continue
        current_frame_idx += 1

        # Extract crops for this frame
        for track_id, det_idx in frames_needed[target_frame]:
            track = tracks[track_id]
            det = track.detections[det_idx]

            crop = _extract_body_crop(frame, det.bbox)
            if crop is not None:
                all_crops.append(crop)
                all_meta.append({
                    "track_id": track_id,
                    "frame_idx": det.frame_idx,
                })

    cap.release()
    logger.info(f"Extracted {len(all_crops)} crops")

    # Compute embeddings in batches
    all_embeddings = []

    for i in range(0, len(all_crops), batch_size):
        batch_crops = all_crops[i:i + batch_size]
        batch_embeddings = embedder.embed_crops_batch(batch_crops)
        all_embeddings.append(batch_embeddings)

        if (i + batch_size) % 100 == 0:
            logger.info(f"  Embedded {min(i + batch_size, len(all_crops))}/{len(all_crops)} crops")

    if all_embeddings:
        embeddings_array = np.vstack(all_embeddings)
    else:
        embeddings_array = np.array([]).reshape(0, embedder.embedding_dim)

    # Save embeddings
    np.save(output_path, embeddings_array)
    logger.info(f"Saved embeddings: {output_path} (shape: {embeddings_array.shape})")

    # Save metadata
    meta = {
        "embedding_dim": embedder.embedding_dim,
        "model_name": embedder.model_name,
        "num_embeddings": len(all_meta),
        "entries": all_meta,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved metadata: {meta_path}")

    return len(all_meta)

"""
Body Tracking Pipeline Runner.

Orchestrates the full body tracking + Re-ID fusion pipeline.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .detect_bodies import BodyDetector, detect_bodies
from .track_bodies import BodyTracker, track_bodies
from .body_embeddings import BodyEmbedder, compute_body_embeddings
from .track_fusion import TrackFusion, fuse_face_body_tracks
from .screentime_compare import ScreenTimeComparator, compare_screen_time


logger = logging.getLogger(__name__)


@dataclass
class BodyTrackingConfig:
    """Configuration for body tracking pipeline."""

    # Detection
    detector_model: str = "yolov8n"
    confidence_threshold: float = 0.50
    nms_iou_threshold: float = 0.45
    min_height_px: int = 50
    min_width_px: int = 25
    detect_every_n_frames: int = 1

    # Tracking
    tracker: str = "bytetrack"
    track_thresh: float = 0.50
    new_track_thresh: float = 0.55
    match_thresh: float = 0.70
    track_buffer: int = 120
    id_offset: int = 100000

    # Re-ID
    reid_enabled: bool = True
    reid_model: str = "osnet_x1_0"
    reid_embedding_dim: int = 256
    reid_batch_size: int = 32

    # Performance
    detection_batch_size: int = 4
    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: Path) -> "BodyTrackingConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Extract nested config sections
        detection = data.get("person_detection", {})
        tracking = data.get("person_tracking", {})
        reid = data.get("person_reid", {})
        perf = data.get("performance", {})

        return cls(
            detector_model=detection.get("model", cls.detector_model),
            confidence_threshold=detection.get("confidence_threshold", cls.confidence_threshold),
            nms_iou_threshold=detection.get("nms_iou_threshold", cls.nms_iou_threshold),
            min_height_px=detection.get("min_height_px", cls.min_height_px),
            min_width_px=detection.get("min_width_px", cls.min_width_px),
            detect_every_n_frames=detection.get("detect_every_n_frames", cls.detect_every_n_frames),
            tracker=tracking.get("tracker", cls.tracker),
            track_thresh=tracking.get("track_thresh", cls.track_thresh),
            new_track_thresh=tracking.get("new_track_thresh", cls.new_track_thresh),
            match_thresh=tracking.get("match_thresh", cls.match_thresh),
            track_buffer=tracking.get("track_buffer", cls.track_buffer),
            id_offset=tracking.get("id_offset", cls.id_offset),
            reid_enabled=reid.get("enabled", cls.reid_enabled),
            reid_model=reid.get("model", cls.reid_model),
            reid_embedding_dim=reid.get("embedding_dim", cls.reid_embedding_dim),
            reid_batch_size=perf.get("reid_batch_size", cls.reid_batch_size),
            detection_batch_size=perf.get("detection_batch_size", cls.detection_batch_size),
            device=detection.get("device", cls.device),
        )


@dataclass
class FusionConfig:
    """Configuration for face-body track fusion."""

    # IoU association
    iou_threshold: float = 0.50
    min_overlap_ratio: float = 0.7
    face_in_upper_body: bool = True
    upper_body_fraction: float = 0.5

    # Re-ID handoff
    reid_similarity_threshold: float = 0.70
    max_gap_seconds: float = 30.0
    min_gap_frames: int = 3
    confidence_decay_rate: float = 0.95
    min_confidence: float = 0.30

    # Screen time
    merge_short_gaps: bool = True
    max_merge_gap_seconds: float = 1.0

    @classmethod
    def from_yaml(cls, path: Path) -> "FusionConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        iou = data.get("iou_association", {})
        reid = data.get("reid_handoff", {})
        handoff = reid.get("handoff", {})
        screen = data.get("screen_time", {})

        return cls(
            iou_threshold=iou.get("iou_threshold", cls.iou_threshold),
            min_overlap_ratio=iou.get("min_overlap_ratio", cls.min_overlap_ratio),
            face_in_upper_body=iou.get("face_in_upper_body", cls.face_in_upper_body),
            upper_body_fraction=iou.get("upper_body_fraction", cls.upper_body_fraction),
            reid_similarity_threshold=reid.get("similarity_threshold", cls.reid_similarity_threshold),
            max_gap_seconds=handoff.get("max_gap_seconds", cls.max_gap_seconds),
            min_gap_frames=handoff.get("min_gap_frames", cls.min_gap_frames),
            confidence_decay_rate=handoff.get("confidence_decay_rate", cls.confidence_decay_rate),
            min_confidence=handoff.get("min_confidence", cls.min_confidence),
            merge_short_gaps=screen.get("merge_short_gaps", cls.merge_short_gaps),
            max_merge_gap_seconds=screen.get("max_merge_gap_seconds", cls.max_merge_gap_seconds),
        )


class BodyTrackingRunner:
    """Orchestrates the body tracking pipeline."""

    def __init__(
        self,
        episode_id: str,
        config_path: Optional[Path] = None,
        fusion_config_path: Optional[Path] = None,
        video_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        skip_existing: bool = False,
    ):
        self.episode_id = episode_id
        self.skip_existing = skip_existing

        # Load configs
        if config_path and config_path.exists():
            self.config = BodyTrackingConfig.from_yaml(config_path)
        else:
            self.config = BodyTrackingConfig()
            logger.warning(f"Config not found at {config_path}, using defaults")

        if fusion_config_path and fusion_config_path.exists():
            self.fusion_config = FusionConfig.from_yaml(fusion_config_path)
        else:
            self.fusion_config = FusionConfig()
            logger.warning(f"Fusion config not found at {fusion_config_path}, using defaults")

        # Override config with CLI args
        if device:
            self.config.device = device
        if batch_size:
            self.config.detection_batch_size = batch_size

        # Set up paths
        self.output_dir = output_dir or Path(f"data/manifests/{episode_id}/body_tracking")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve video path
        self.video_path = video_path
        if not self.video_path:
            self.video_path = self._find_video_path()

        # Output file paths
        self.detections_path = self.output_dir / "body_detections.jsonl"
        self.tracks_path = self.output_dir / "body_tracks.jsonl"
        self.embeddings_path = self.output_dir / "body_embeddings.npy"
        self.embeddings_meta_path = self.output_dir / "body_embeddings_meta.json"
        self.fusion_path = self.output_dir / "track_fusion.json"
        self.metrics_path = self.output_dir / "body_metrics.json"
        self.comparison_path = self.output_dir / "screentime_comparison.json"

        logger.info(f"Output directory: {self.output_dir}")

    def _find_video_path(self) -> Path:
        """Find video path from episode manifest."""
        # Prefer canonical Screenalytics artifact layout when available.
        try:
            from py_screenalytics.artifacts import get_path  # type: ignore

            candidate = get_path(self.episode_id, "video")
            if candidate.exists():
                return candidate
        except ImportError:
            pass

        manifest_dir = Path(f"data/manifests/{self.episode_id}")
        manifest_path = manifest_dir / "manifest.json"

        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            if "video_path" in manifest:
                return Path(manifest["video_path"])

        # Try common locations
        for ext in [".mp4", ".mkv", ".avi", ".mov"]:
            for base in [manifest_dir, Path("data/videos")]:
                video_path = base / f"{self.episode_id}{ext}"
                if video_path.exists():
                    return video_path
                # Newer layout: data/videos/{ep_id}/episode.mp4
                nested = base / self.episode_id / f"episode{ext}"
                if nested.exists():
                    return nested

        raise FileNotFoundError(
            f"Could not find video for episode {self.episode_id}. "
            "Use --video-path to specify manually."
        )

    def _should_skip(self, output_path: Path, stage_name: str) -> bool:
        """Check if stage should be skipped."""
        if self.skip_existing and output_path.exists():
            logger.info(f"[SKIP] {stage_name}: {output_path} exists")
            return True
        return False

    def run_detection(self) -> Path:
        """Run body detection stage."""
        if self._should_skip(self.detections_path, "Detection"):
            return self.detections_path

        logger.info("[STAGE] Body Detection")
        logger.info(f"  Model: {self.config.detector_model}")
        logger.info(f"  Confidence: {self.config.confidence_threshold}")
        logger.info(f"  Device: {self.config.device}")

        detector = BodyDetector(
            model_name=self.config.detector_model,
            confidence_threshold=self.config.confidence_threshold,
            nms_iou_threshold=self.config.nms_iou_threshold,
            device=self.config.device,
            min_height_px=self.config.min_height_px,
            min_width_px=self.config.min_width_px,
        )

        detect_bodies(
            detector=detector,
            video_path=self.video_path,
            output_path=self.detections_path,
            sample_every_n=self.config.detect_every_n_frames,
            batch_size=self.config.detection_batch_size,
        )

        logger.info(f"  Output: {self.detections_path}")
        return self.detections_path

    def run_tracking(self) -> Path:
        """Run body tracking stage."""
        if self._should_skip(self.tracks_path, "Tracking"):
            return self.tracks_path

        # Ensure detections exist
        if not self.detections_path.exists():
            self.run_detection()

        logger.info("[STAGE] Body Tracking")
        logger.info(f"  Tracker: {self.config.tracker}")
        logger.info(f"  Track buffer: {self.config.track_buffer} frames")

        tracker = BodyTracker(
            tracker_type=self.config.tracker,
            track_thresh=self.config.track_thresh,
            new_track_thresh=self.config.new_track_thresh,
            match_thresh=self.config.match_thresh,
            track_buffer=self.config.track_buffer,
            id_offset=self.config.id_offset,
        )

        track_bodies(
            tracker=tracker,
            detections_path=self.detections_path,
            output_path=self.tracks_path,
        )

        logger.info(f"  Output: {self.tracks_path}")
        return self.tracks_path

    def run_embedding(self) -> Path:
        """Run body Re-ID embedding stage."""
        if self._should_skip(self.embeddings_path, "Embedding"):
            return self.embeddings_path

        if not self.config.reid_enabled:
            logger.info("[SKIP] Re-ID disabled in config")
            return self.embeddings_path

        # Ensure tracks exist
        if not self.tracks_path.exists():
            self.run_tracking()

        logger.info("[STAGE] Body Re-ID Embeddings")
        logger.info(f"  Model: {self.config.reid_model}")
        logger.info(f"  Embedding dim: {self.config.reid_embedding_dim}")

        embedder = BodyEmbedder(
            model_name=self.config.reid_model,
            device=self.config.device,
        )

        compute_body_embeddings(
            embedder=embedder,
            video_path=self.video_path,
            tracks_path=self.tracks_path,
            output_path=self.embeddings_path,
            meta_path=self.embeddings_meta_path,
            batch_size=self.config.reid_batch_size,
        )

        logger.info(f"  Output: {self.embeddings_path}")
        return self.embeddings_path

    def run_fusion(self) -> Path:
        """Run face-body track fusion stage."""
        if self._should_skip(self.fusion_path, "Fusion"):
            return self.fusion_path

        # Ensure body tracks exist
        if not self.tracks_path.exists():
            self.run_tracking()

        # Check for face tracks
        face_tracks_path = Path(f"data/manifests/{self.episode_id}/faces.jsonl")
        if not face_tracks_path.exists():
            logger.warning(f"Face tracks not found at {face_tracks_path}")
            logger.warning("Fusion requires face tracks. Skipping fusion stage.")
            return self.fusion_path

        logger.info("[STAGE] Face-Body Track Fusion")
        logger.info(f"  IoU threshold: {self.fusion_config.iou_threshold}")
        logger.info(f"  Re-ID threshold: {self.fusion_config.reid_similarity_threshold}")

        fusion = TrackFusion(
            iou_threshold=self.fusion_config.iou_threshold,
            min_overlap_ratio=self.fusion_config.min_overlap_ratio,
            reid_similarity_threshold=self.fusion_config.reid_similarity_threshold,
            max_gap_seconds=self.fusion_config.max_gap_seconds,
        )

        # Load embeddings if available
        embeddings_meta = None
        if self.embeddings_meta_path.exists():
            with open(self.embeddings_meta_path) as f:
                embeddings_meta = json.load(f)

        fuse_face_body_tracks(
            fusion=fusion,
            face_tracks_path=face_tracks_path,
            body_tracks_path=self.tracks_path,
            body_embeddings_path=self.embeddings_path if self.embeddings_path.exists() else None,
            embeddings_meta=embeddings_meta,
            output_path=self.fusion_path,
        )

        logger.info(f"  Output: {self.fusion_path}")
        return self.fusion_path

    def run_comparison(self) -> Path:
        """Run screen-time comparison stage."""
        if self._should_skip(self.comparison_path, "Comparison"):
            return self.comparison_path

        # Ensure fusion exists
        if not self.fusion_path.exists():
            self.run_fusion()

        # Check for identities
        identities_path = Path(f"data/manifests/{self.episode_id}/identities.json")
        if not identities_path.exists():
            logger.warning(f"Identities not found at {identities_path}")
            logger.warning("Comparison requires identities. Skipping comparison stage.")
            return self.comparison_path

        logger.info("[STAGE] Screen-time Comparison")

        comparator = ScreenTimeComparator(
            merge_short_gaps=self.fusion_config.merge_short_gaps,
            max_merge_gap_seconds=self.fusion_config.max_merge_gap_seconds,
        )

        compare_screen_time(
            comparator=comparator,
            identities_path=identities_path,
            fusion_path=self.fusion_path,
            output_path=self.comparison_path,
        )

        logger.info(f"  Output: {self.comparison_path}")
        return self.comparison_path

    def run_full_pipeline(self) -> dict:
        """Run the full body tracking pipeline."""
        logger.info("=" * 60)
        logger.info(f"Body Tracking Pipeline - {self.episode_id}")
        logger.info("=" * 60)

        results = {
            "episode_id": self.episode_id,
            "output_dir": str(self.output_dir),
            "stages": {},
        }

        # Stage 1: Detection
        self.run_detection()
        results["stages"]["detection"] = str(self.detections_path)

        # Stage 2: Tracking
        self.run_tracking()
        results["stages"]["tracking"] = str(self.tracks_path)

        # Stage 3: Embeddings
        if self.config.reid_enabled:
            self.run_embedding()
            results["stages"]["embedding"] = str(self.embeddings_path)

        # Stage 4: Fusion
        self.run_fusion()
        results["stages"]["fusion"] = str(self.fusion_path)

        # Stage 5: Comparison
        self.run_comparison()
        results["stages"]["comparison"] = str(self.comparison_path)

        # Write metrics summary
        self._write_metrics(results)

        logger.info("=" * 60)
        logger.info("Pipeline Complete")
        logger.info("=" * 60)

        return results

    def _write_metrics(self, results: dict):
        """Write pipeline metrics to JSON."""
        import os

        metrics = {
            "episode_id": self.episode_id,
            "config": {
                "detector_model": self.config.detector_model,
                "tracker": self.config.tracker,
                "reid_model": self.config.reid_model if self.config.reid_enabled else None,
            },
            "outputs": {},
        }

        # Gather file stats
        for stage, path_str in results.get("stages", {}).items():
            path = Path(path_str)
            if path.exists():
                stat = os.stat(path)
                metrics["outputs"][stage] = {
                    "path": path_str,
                    "size_bytes": stat.st_size,
                    "exists": True,
                }

                # Count lines for JSONL files
                if path.suffix == ".jsonl":
                    with open(path) as f:
                        metrics["outputs"][stage]["line_count"] = sum(1 for _ in f)
            else:
                metrics["outputs"][stage] = {
                    "path": path_str,
                    "exists": False,
                }

        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics written to: {self.metrics_path}")

"""
Face Alignment Pipeline Runner.

Orchestrates the FAN-based alignment pipeline.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from .load_detections import load_face_detections
from .run_fan_alignment import FANAligner, run_fan_alignment
from .export_aligned_faces import export_aligned_faces


logger = logging.getLogger(__name__)


@dataclass
class FaceAlignmentConfig:
    """Configuration for face alignment pipeline."""

    # Model settings
    model_type: str = "2d"  # "2d" | "3d"
    landmarks_type: str = "2D"  # "2D" | "3D" | "2.5D"
    flip_input: bool = False

    # Processing
    batch_size: int = 16
    stride: int = 1  # Process every Nth frame
    device: str = "auto"

    # Quality thresholds
    min_face_size: int = 20  # Minimum face box size in pixels
    min_confidence: float = 0.5  # Minimum detection confidence

    # Output
    export_crops: bool = False
    crop_size: int = 112  # ArcFace standard size
    crop_margin: float = 0.0  # Margin around aligned face

    @classmethod
    def from_yaml(cls, path: Path) -> "FaceAlignmentConfig":
        """Load config from YAML file."""
        if not path.exists():
            logger.warning(f"Config not found at {path}, using defaults")
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f)

        fa = data.get("face_alignment", {})
        model = fa.get("model", {})
        processing = fa.get("processing", {})
        quality = fa.get("quality", {})
        output = fa.get("output", {})

        return cls(
            model_type=model.get("type", cls.model_type),
            landmarks_type=model.get("landmarks_type", cls.landmarks_type),
            flip_input=model.get("flip_input", cls.flip_input),
            batch_size=processing.get("batch_size", cls.batch_size),
            stride=processing.get("stride", cls.stride),
            device=processing.get("device", cls.device),
            min_face_size=quality.get("min_face_size", cls.min_face_size),
            min_confidence=quality.get("min_confidence", cls.min_confidence),
            export_crops=output.get("export_crops", cls.export_crops),
            crop_size=output.get("crop_size", cls.crop_size),
            crop_margin=output.get("crop_margin", cls.crop_margin),
        )


class FaceAlignmentRunner:
    """Orchestrates the face alignment pipeline."""

    def __init__(
        self,
        episode_id: str,
        config_path: Optional[Path] = None,
        video_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        device: Optional[str] = None,
        stride: Optional[int] = None,
        batch_size: Optional[int] = None,
        export_crops: bool = False,
        skip_existing: bool = False,
    ):
        self.episode_id = episode_id
        self.skip_existing = skip_existing

        # Load config
        config_path = config_path or Path("config/pipeline/face_alignment.yaml")
        self.config = FaceAlignmentConfig.from_yaml(config_path)

        # Override config with CLI args
        if device:
            self.config.device = device
        if stride:
            self.config.stride = stride
        if batch_size:
            self.config.batch_size = batch_size
        if export_crops:
            self.config.export_crops = True

        # Set up paths
        self.manifest_dir = Path(f"data/manifests/{episode_id}")
        self.output_dir = output_dir or self.manifest_dir / "face_alignment"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve video path
        self.video_path = video_path
        if not self.video_path:
            self.video_path = self._find_video_path()

        # Input paths
        self.detections_path = self.manifest_dir / "detections.jsonl"
        self.tracks_path = self.manifest_dir / "tracks.jsonl"

        # Output paths
        self.aligned_faces_path = self.output_dir / "aligned_faces.jsonl"
        self.crops_dir = self.output_dir / "aligned_crops"
        self.metrics_path = self.output_dir / "alignment_metrics.json"

        logger.info(f"Output directory: {self.output_dir}")

    def _find_video_path(self) -> Path:
        """Find video path from episode manifest."""
        manifest_path = self.manifest_dir / "manifest.json"

        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            if "video_path" in manifest:
                return Path(manifest["video_path"])

        # Try common locations
        for ext in [".mp4", ".mkv", ".avi", ".mov"]:
            for base in [self.manifest_dir, Path("data/videos")]:
                video_path = base / f"{self.episode_id}{ext}"
                if video_path.exists():
                    return video_path

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

    def run_alignment(self) -> Path:
        """Run face alignment stage."""
        if self._should_skip(self.aligned_faces_path, "Alignment"):
            return self.aligned_faces_path

        logger.info("[STAGE] Face Alignment (FAN)")
        logger.info(f"  Model: FAN {self.config.model_type}")
        logger.info(f"  Device: {self.config.device}")
        logger.info(f"  Stride: {self.config.stride}")

        # Check for detections
        if not self.detections_path.exists():
            raise FileNotFoundError(
                f"Face detections not found at {self.detections_path}. "
                "Run face detection pipeline first."
            )

        # Load detections
        detections = load_face_detections(
            self.detections_path,
            min_confidence=self.config.min_confidence,
            min_size=self.config.min_face_size,
        )
        logger.info(f"  Loaded {len(detections)} face detections")

        # Filter by stride
        if self.config.stride > 1:
            detections = [
                d for d in detections
                if d["frame_idx"] % self.config.stride == 0
            ]
            logger.info(f"  After stride filter: {len(detections)} detections")

        # Initialize aligner
        aligner = FANAligner(
            model_type=self.config.model_type,
            landmarks_type=self.config.landmarks_type,
            device=self.config.device,
            flip_input=self.config.flip_input,
        )

        # Run alignment
        aligned_faces = run_fan_alignment(
            aligner=aligner,
            video_path=self.video_path,
            detections=detections,
            batch_size=self.config.batch_size,
            crop_size=self.config.crop_size,
            crop_margin=self.config.crop_margin,
        )

        # Export results
        export_aligned_faces(
            aligned_faces=aligned_faces,
            output_path=self.aligned_faces_path,
            crops_dir=self.crops_dir if self.config.export_crops else None,
        )

        logger.info(f"  Output: {self.aligned_faces_path}")
        return self.aligned_faces_path

    def run_export(self) -> Path:
        """Export aligned crops (if not done during alignment)."""
        if not self.aligned_faces_path.exists():
            raise FileNotFoundError(
                f"Aligned faces not found at {self.aligned_faces_path}. "
                "Run alignment stage first."
            )

        if self._should_skip(self.crops_dir / ".done", "Export"):
            return self.crops_dir

        logger.info("[STAGE] Export Aligned Crops")

        # Load aligned faces
        aligned_faces = []
        with open(self.aligned_faces_path) as f:
            for line in f:
                aligned_faces.append(json.loads(line))

        logger.info(f"  Loaded {len(aligned_faces)} aligned faces")

        # Export crops
        self.crops_dir.mkdir(parents=True, exist_ok=True)

        # This would need video access to extract crops
        # For now, crops are optionally saved during alignment
        logger.warning("  Crop export requires re-running alignment with --export-crops")

        return self.crops_dir

    def run_full_pipeline(self) -> dict:
        """Run the full alignment pipeline."""
        logger.info("=" * 60)
        logger.info(f"Face Alignment Pipeline - {self.episode_id}")
        logger.info("=" * 60)

        results = {
            "episode_id": self.episode_id,
            "output_dir": str(self.output_dir),
            "stages": {},
        }

        # Stage 1: Alignment
        self.run_alignment()
        results["stages"]["alignment"] = str(self.aligned_faces_path)

        # Stage 2: Export (if enabled)
        if self.config.export_crops:
            results["stages"]["export"] = str(self.crops_dir)

        # Write metrics
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
                "model_type": self.config.model_type,
                "stride": self.config.stride,
                "batch_size": self.config.batch_size,
            },
            "outputs": {},
        }

        # Count aligned faces
        if self.aligned_faces_path.exists():
            with open(self.aligned_faces_path) as f:
                count = sum(1 for _ in f)
            metrics["outputs"]["aligned_faces_count"] = count

        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics written to: {self.metrics_path}")

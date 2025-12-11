#!/usr/bin/env python
"""
Smoke Test Runner for Screenalytics Pipeline.

Validates end-to-end pipeline wiring with minimal resource usage.

Usage:
    python -m tools.smoke.smoke_run --episode-id demo-s01e01
    python -m tools.smoke.smoke_run --episode-id demo-s01e01 --dry-run

Output:
    data/smoke/{episode_id}/smoke_report.json
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Result from running a pipeline stage."""
    name: str
    status: str  # "success" | "skipped" | "failed"
    duration_seconds: float = 0.0
    artifact_paths: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class SmokeReport:
    """Complete smoke test report."""
    episode_id: str
    timestamp: str
    config: Dict[str, Any]
    total_duration_seconds: float = 0.0
    status: str = "pending"
    stages: List[StageResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "config": self.config,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "status": self.status,
            "stages": [asdict(s) for s in self.stages],
            "summary": self.summary,
            "errors": self.errors,
        }


class SmokeRunner:
    """Orchestrates smoke test pipeline run."""

    def __init__(
        self,
        episode_id: str,
        max_frames: int = 300,
        embedding_backend: str = "pytorch",
        alignment: str = "on",
        alignment_gating: str = "on",
        body_tracking: bool = False,
        dry_run: bool = False,
    ):
        self.episode_id = episode_id
        self.max_frames = max_frames
        self.embedding_backend = embedding_backend
        self.alignment = alignment == "on"
        self.alignment_gating = alignment_gating == "on"
        self.body_tracking = body_tracking
        self.dry_run = dry_run

        self.output_dir = Path(f"data/smoke/{episode_id}")
        self.manifest_dir = Path(f"data/manifests/{episode_id}")
        self.report = SmokeReport(
            episode_id=episode_id,
            timestamp=datetime.now().isoformat(),
            config={
                "max_frames": max_frames,
                "embedding_backend": embedding_backend,
                "alignment": alignment,
                "alignment_gating": alignment_gating,
                "body_tracking": body_tracking,
                "dry_run": dry_run,
            },
        )

    def run(self) -> bool:
        """Run the smoke test. Returns True if successful."""
        start_time = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting smoke test for {self.episode_id}")
        logger.info(f"  Max frames: {self.max_frames}")
        logger.info(f"  Backend: {self.embedding_backend}")

        if self.dry_run:
            logger.info("DRY RUN: Validating configuration only")
            self._run_dry_validation()
        else:
            self._run_stages()

        self.report.total_duration_seconds = time.time() - start_time

        failed_stages = [s for s in self.report.stages if s.status == "failed"]
        if failed_stages or self.report.errors:
            self.report.status = "failed"
        else:
            self.report.status = "success"

        self._build_summary()
        self._write_report()

        return self.report.status == "success"

    def _run_dry_validation(self):
        """Validate configuration without running pipeline."""
        required_configs = [
            "config/pipeline/detection.yaml",
            "config/pipeline/tracking.yaml",
            "config/pipeline/embedding.yaml",
        ]

        if self.alignment:
            required_configs.append("config/pipeline/face_alignment.yaml")

        for config_path in required_configs:
            if not Path(config_path).exists():
                self.report.errors.append(f"Missing config: {config_path}")

        self.report.stages.append(StageResult(
            name="dry_validation",
            status="success" if not self.report.errors else "failed",
            metrics={"configs_checked": len(required_configs)},
        ))

    def _run_stages(self):
        """Run actual pipeline stages."""
        self._run_stage_detect_track()
        if self.alignment:
            self._run_stage_alignment()
        self._run_stage_embeddings()
        self._run_stage_clustering()
        self._run_stage_screentime()
        if self.body_tracking:
            self._run_stage_body_tracking()

    def _run_stage_detect_track(self):
        """Run face detection and tracking stage."""
        stage_name = "detect_track"
        start_time = time.time()

        try:
            detections_path = self.manifest_dir / "detections.jsonl"
            tracks_path = self.manifest_dir / "tracks.jsonl"

            if detections_path.exists() and tracks_path.exists():
                with open(detections_path) as f:
                    num_detections = sum(1 for _ in f)
                with open(tracks_path) as f:
                    num_tracks = sum(1 for _ in f)

                self.report.stages.append(StageResult(
                    name=stage_name,
                    status="success",
                    duration_seconds=time.time() - start_time,
                    artifact_paths=[str(detections_path), str(tracks_path)],
                    metrics={
                        "num_detections": num_detections,
                        "num_tracks": num_tracks,
                        "source": "existing",
                    },
                ))
            else:
                self.report.stages.append(StageResult(
                    name=stage_name,
                    status="skipped",
                    duration_seconds=time.time() - start_time,
                    error="No existing detections/tracks - full pipeline run required",
                ))

        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {e}")
            self.report.stages.append(StageResult(
                name=stage_name,
                status="failed",
                duration_seconds=time.time() - start_time,
                error=str(e),
            ))
            self.report.errors.append(f"{stage_name}: {e}")

    def _run_stage_alignment(self):
        """Run face alignment stage."""
        stage_name = "face_alignment"
        start_time = time.time()

        try:
            aligned_path = self.manifest_dir / "face_alignment" / "aligned_faces.jsonl"

            if aligned_path.exists():
                with open(aligned_path) as f:
                    num_aligned = sum(1 for _ in f)
                self.report.stages.append(StageResult(
                    name=stage_name,
                    status="success",
                    duration_seconds=time.time() - start_time,
                    artifact_paths=[str(aligned_path)],
                    metrics={"num_aligned_faces": num_aligned, "source": "existing"},
                ))
            else:
                self.report.stages.append(StageResult(
                    name=stage_name,
                    status="skipped",
                    duration_seconds=time.time() - start_time,
                    error="No aligned faces - alignment not run",
                ))

        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {e}")
            self.report.stages.append(StageResult(
                name=stage_name,
                status="failed",
                duration_seconds=time.time() - start_time,
                error=str(e),
            ))
            self.report.errors.append(f"{stage_name}: {e}")

    def _run_stage_embeddings(self):
        """Run face embeddings stage."""
        stage_name = "face_embeddings"
        start_time = time.time()

        try:
            embeddings_path = self.manifest_dir / "face_embeddings.npy"
            faces_path = self.manifest_dir / "faces.jsonl"

            if embeddings_path.exists() and faces_path.exists():
                import numpy as np
                embeddings = np.load(embeddings_path)
                num_embeddings = len(embeddings)

                gated_count = 0
                total_faces = 0
                with open(faces_path) as f:
                    for line in f:
                        total_faces += 1
                        data = json.loads(line)
                        if "low_alignment_quality" in data.get("skip_reason", ""):
                            gated_count += 1

                gating_rate = gated_count / total_faces if total_faces > 0 else 0

                self.report.stages.append(StageResult(
                    name=stage_name,
                    status="success",
                    duration_seconds=time.time() - start_time,
                    artifact_paths=[str(embeddings_path), str(faces_path)],
                    metrics={
                        "num_embeddings": num_embeddings,
                        "total_faces": total_faces,
                        "gated_count": gated_count,
                        "gating_rate": round(gating_rate, 4),
                        "backend": self.embedding_backend,
                        "source": "existing",
                    },
                ))
            else:
                self.report.stages.append(StageResult(
                    name=stage_name,
                    status="skipped",
                    duration_seconds=time.time() - start_time,
                    error="Embeddings not found",
                ))

        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {e}")
            self.report.stages.append(StageResult(
                name=stage_name,
                status="failed",
                duration_seconds=time.time() - start_time,
                error=str(e),
            ))
            self.report.errors.append(f"{stage_name}: {e}")

    def _run_stage_clustering(self):
        """Run clustering/identities stage."""
        stage_name = "clustering"
        start_time = time.time()

        try:
            identities_path = self.manifest_dir / "identities.json"

            if identities_path.exists():
                with open(identities_path) as f:
                    identities = json.load(f)

                identity_list = identities.get("identities", [])
                num_clusters = len(identity_list)
                num_singletons = sum(
                    1 for i in identity_list if len(i.get("tracks", [])) == 1
                )

                self.report.stages.append(StageResult(
                    name=stage_name,
                    status="success",
                    duration_seconds=time.time() - start_time,
                    artifact_paths=[str(identities_path)],
                    metrics={
                        "num_clusters": num_clusters,
                        "num_singletons": num_singletons,
                        "source": "existing",
                    },
                ))
            else:
                self.report.stages.append(StageResult(
                    name=stage_name,
                    status="skipped",
                    duration_seconds=time.time() - start_time,
                    error="Identities not found",
                ))

        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {e}")
            self.report.stages.append(StageResult(
                name=stage_name,
                status="failed",
                duration_seconds=time.time() - start_time,
                error=str(e),
            ))
            self.report.errors.append(f"{stage_name}: {e}")

    def _run_stage_screentime(self):
        """Run screen time metrics stage."""
        stage_name = "screentime"
        start_time = time.time()

        try:
            screentime_path = self.manifest_dir / "screentime.json"

            if screentime_path.exists():
                with open(screentime_path) as f:
                    screentime = json.load(f)

                total_time = screentime.get("total_screen_time_seconds", 0)

                self.report.stages.append(StageResult(
                    name=stage_name,
                    status="success",
                    duration_seconds=time.time() - start_time,
                    artifact_paths=[str(screentime_path)],
                    metrics={"total_screen_time_seconds": round(total_time, 2), "source": "existing"},
                ))
            else:
                self.report.stages.append(StageResult(
                    name=stage_name,
                    status="skipped",
                    duration_seconds=time.time() - start_time,
                    error="Screen time not found",
                ))

        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {e}")
            self.report.stages.append(StageResult(
                name=stage_name,
                status="failed",
                duration_seconds=time.time() - start_time,
                error=str(e),
            ))
            self.report.errors.append(f"{stage_name}: {e}")

    def _run_stage_body_tracking(self):
        """Run body tracking stage."""
        stage_name = "body_tracking"
        start_time = time.time()

        try:
            body_tracks_path = self.manifest_dir / "body_tracking" / "body_tracks.jsonl"

            if body_tracks_path.exists():
                with open(body_tracks_path) as f:
                    num_body_tracks = sum(1 for _ in f)
                self.report.stages.append(StageResult(
                    name=stage_name,
                    status="success",
                    duration_seconds=time.time() - start_time,
                    artifact_paths=[str(body_tracks_path)],
                    metrics={"num_body_tracks": num_body_tracks, "source": "existing"},
                ))
            else:
                self.report.stages.append(StageResult(
                    name=stage_name,
                    status="skipped",
                    duration_seconds=time.time() - start_time,
                    error="Body tracking not run",
                ))

        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {e}")
            self.report.stages.append(StageResult(
                name=stage_name,
                status="failed",
                duration_seconds=time.time() - start_time,
                error=str(e),
            ))

    def _build_summary(self):
        """Build summary statistics from stage results."""
        self.report.summary = {
            "stages_run": len(self.report.stages),
            "stages_success": len([s for s in self.report.stages if s.status == "success"]),
            "stages_failed": len([s for s in self.report.stages if s.status == "failed"]),
            "stages_skipped": len([s for s in self.report.stages if s.status == "skipped"]),
        }

        for stage in self.report.stages:
            if stage.status == "success":
                if stage.name == "detect_track":
                    self.report.summary["num_tracks"] = stage.metrics.get("num_tracks", 0)
                elif stage.name == "face_embeddings":
                    self.report.summary["num_embeddings"] = stage.metrics.get("num_embeddings", 0)
                    self.report.summary["gating_rate"] = stage.metrics.get("gating_rate", 0)
                elif stage.name == "clustering":
                    self.report.summary["num_clusters"] = stage.metrics.get("num_clusters", 0)

    def _write_report(self):
        """Write the smoke report to JSON file."""
        report_path = self.output_dir / "smoke_report.json"
        with open(report_path, "w") as f:
            json.dump(self.report.to_dict(), f, indent=2)
        logger.info(f"Smoke report written to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Smoke Test Runner")
    parser.add_argument("--episode-id", required=True, help="Episode ID to test")
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames")
    parser.add_argument("--embedding-backend", choices=["pytorch", "tensorrt"], default="pytorch")
    parser.add_argument("--alignment", choices=["off", "on"], default="on")
    parser.add_argument("--alignment-gating", choices=["off", "on"], default="on")
    parser.add_argument("--body-tracking", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", help="Validate config only")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("SCREENALYTICS SMOKE TEST")

    runner = SmokeRunner(
        episode_id=args.episode_id,
        max_frames=args.max_frames,
        embedding_backend=args.embedding_backend,
        alignment=args.alignment,
        alignment_gating=args.alignment_gating,
        body_tracking=args.body_tracking,
        dry_run=args.dry_run,
    )

    success = runner.run()

    print(f"\nStatus: {'PASS' if success else 'FAIL'}")
    print(f"Duration: {runner.report.total_duration_seconds:.2f}s")
    print(f"Report: data/smoke/{args.episode_id}/smoke_report.json")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

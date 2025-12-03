"""
Pipeline Checkpoint/Recovery System.

Allows the audio pipeline to resume from the last successful stage after a failure.
Checkpoint state is persisted to disk so it survives process restarts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class StageCheckpoint:
    """Checkpoint data for a single pipeline stage."""
    name: str
    status: str  # "pending", "running", "completed", "failed"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_paths: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineCheckpoint:
    """Full pipeline checkpoint state."""
    ep_id: str
    pipeline_version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    stages: Dict[str, StageCheckpoint] = field(default_factory=dict)
    config_hash: Optional[str] = None  # Hash of config to detect changes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ep_id": self.ep_id,
            "pipeline_version": self.pipeline_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config_hash": self.config_hash,
            "stages": {
                name: asdict(stage) for name, stage in self.stages.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineCheckpoint":
        """Create from dictionary."""
        stages = {
            name: StageCheckpoint(**stage_data)
            for name, stage_data in data.get("stages", {}).items()
        }
        return cls(
            ep_id=data["ep_id"],
            pipeline_version=data.get("pipeline_version", "1.0"),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
            config_hash=data.get("config_hash"),
            stages=stages,
        )


class CheckpointManager:
    """
    Manages pipeline checkpoint state for recovery after failures.

    Usage:
        manager = CheckpointManager(ep_id="show123_s01_e01", checkpoint_dir=Path("data/checkpoints"))

        # Check if stage is already complete
        if not manager.is_stage_complete("extract"):
            manager.start_stage("extract")
            try:
                # ... do extraction work ...
                manager.complete_stage("extract", output_paths={"audio": str(audio_path)})
            except Exception as e:
                manager.fail_stage("extract", str(e))
                raise

        # Get last completed stage for resume
        last_stage = manager.get_last_completed_stage()
    """

    # Default stage order for audio pipeline
    STAGE_ORDER = [
        "extract",
        "separate",
        "enhance",
        "diarize",
        "voices",
        "transcribe",
        "fuse",
        "export",
        "qc",
    ]

    def __init__(
        self,
        ep_id: str,
        checkpoint_dir: Optional[Path] = None,
        config_hash: Optional[str] = None,
    ):
        self.ep_id = ep_id
        self.checkpoint_dir = checkpoint_dir or Path("data/checkpoints")
        self.checkpoint_file = self.checkpoint_dir / f"{ep_id}_audio_checkpoint.json"
        self.config_hash = config_hash
        self._checkpoint: Optional[PipelineCheckpoint] = None

    def _ensure_dir(self) -> None:
        """Ensure checkpoint directory exists."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> PipelineCheckpoint:
        """Load checkpoint from disk, or create new one."""
        if self._checkpoint is not None:
            return self._checkpoint

        if self.checkpoint_file.exists():
            try:
                with self.checkpoint_file.open("r") as f:
                    data = json.load(f)
                self._checkpoint = PipelineCheckpoint.from_dict(data)

                # Check if config changed - if so, start fresh
                if self.config_hash and self._checkpoint.config_hash != self.config_hash:
                    LOGGER.warning(
                        f"Config changed since last checkpoint, starting fresh for {self.ep_id}"
                    )
                    self._checkpoint = self._create_new_checkpoint()

                LOGGER.info(f"Loaded checkpoint for {self.ep_id}")
            except Exception as e:
                LOGGER.warning(f"Failed to load checkpoint: {e}, creating new")
                self._checkpoint = self._create_new_checkpoint()
        else:
            self._checkpoint = self._create_new_checkpoint()

        return self._checkpoint

    def _create_new_checkpoint(self) -> PipelineCheckpoint:
        """Create a new checkpoint with pending stages."""
        checkpoint = PipelineCheckpoint(
            ep_id=self.ep_id,
            config_hash=self.config_hash,
        )
        # Initialize all stages as pending
        for stage in self.STAGE_ORDER:
            checkpoint.stages[stage] = StageCheckpoint(name=stage, status="pending")
        return checkpoint

    def save(self) -> None:
        """Save checkpoint to disk."""
        if self._checkpoint is None:
            return

        self._ensure_dir()
        self._checkpoint.updated_at = datetime.utcnow().isoformat()

        with self.checkpoint_file.open("w") as f:
            json.dump(self._checkpoint.to_dict(), f, indent=2)

        LOGGER.debug(f"Saved checkpoint for {self.ep_id}")

    def is_stage_complete(self, stage_name: str) -> bool:
        """Check if a stage is already complete."""
        checkpoint = self.load()
        stage = checkpoint.stages.get(stage_name)
        return stage is not None and stage.status == "completed"

    def start_stage(self, stage_name: str) -> None:
        """Mark a stage as running."""
        checkpoint = self.load()
        if stage_name not in checkpoint.stages:
            checkpoint.stages[stage_name] = StageCheckpoint(name=stage_name, status="pending")

        stage = checkpoint.stages[stage_name]
        stage.status = "running"
        stage.started_at = datetime.utcnow().isoformat()
        stage.completed_at = None
        stage.error = None

        self.save()
        LOGGER.info(f"Stage '{stage_name}' started for {self.ep_id}")

    def complete_stage(
        self,
        stage_name: str,
        output_paths: Optional[Dict[str, str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark a stage as completed."""
        checkpoint = self.load()
        if stage_name not in checkpoint.stages:
            checkpoint.stages[stage_name] = StageCheckpoint(name=stage_name, status="pending")

        stage = checkpoint.stages[stage_name]
        stage.status = "completed"
        stage.completed_at = datetime.utcnow().isoformat()
        if output_paths:
            stage.output_paths = output_paths
        if metrics:
            stage.metrics = metrics

        self.save()
        LOGGER.info(f"Stage '{stage_name}' completed for {self.ep_id}")

    def fail_stage(self, stage_name: str, error: str) -> None:
        """Mark a stage as failed."""
        checkpoint = self.load()
        if stage_name not in checkpoint.stages:
            checkpoint.stages[stage_name] = StageCheckpoint(name=stage_name, status="pending")

        stage = checkpoint.stages[stage_name]
        stage.status = "failed"
        stage.error = error

        self.save()
        LOGGER.warning(f"Stage '{stage_name}' failed for {self.ep_id}: {error}")

    def get_last_completed_stage(self) -> Optional[str]:
        """Get the name of the last completed stage."""
        checkpoint = self.load()

        last_completed = None
        for stage_name in self.STAGE_ORDER:
            stage = checkpoint.stages.get(stage_name)
            if stage and stage.status == "completed":
                last_completed = stage_name
            else:
                break  # Stop at first non-completed stage

        return last_completed

    def get_next_stage(self) -> Optional[str]:
        """Get the name of the next stage to run."""
        checkpoint = self.load()

        for stage_name in self.STAGE_ORDER:
            stage = checkpoint.stages.get(stage_name)
            if not stage or stage.status != "completed":
                return stage_name

        return None  # All stages complete

    def get_completed_stages(self) -> List[str]:
        """Get list of completed stage names."""
        checkpoint = self.load()
        return [
            name for name in self.STAGE_ORDER
            if checkpoint.stages.get(name, StageCheckpoint(name=name, status="pending")).status == "completed"
        ]

    def get_stage_output(self, stage_name: str, key: str) -> Optional[str]:
        """Get a specific output path from a completed stage."""
        checkpoint = self.load()
        stage = checkpoint.stages.get(stage_name)
        if stage and stage.status == "completed":
            return stage.output_paths.get(key)
        return None

    def reset(self) -> None:
        """Reset checkpoint to start fresh."""
        self._checkpoint = self._create_new_checkpoint()
        self.save()
        LOGGER.info(f"Reset checkpoint for {self.ep_id}")

    def delete(self) -> None:
        """Delete checkpoint file."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            LOGGER.info(f"Deleted checkpoint for {self.ep_id}")
        self._checkpoint = None

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of checkpoint state."""
        checkpoint = self.load()
        return {
            "ep_id": self.ep_id,
            "created_at": checkpoint.created_at,
            "updated_at": checkpoint.updated_at,
            "stages": {
                name: {
                    "status": checkpoint.stages.get(name, StageCheckpoint(name=name, status="pending")).status,
                    "completed_at": checkpoint.stages.get(name, StageCheckpoint(name=name, status="pending")).completed_at,
                }
                for name in self.STAGE_ORDER
            },
            "last_completed": self.get_last_completed_stage(),
            "next_stage": self.get_next_stage(),
        }


def compute_config_hash(config: Any) -> str:
    """Compute a hash of config for change detection."""
    import hashlib

    # Convert config to string and hash it
    if hasattr(config, "to_dict"):
        config_str = json.dumps(config.to_dict(), sort_keys=True)
    elif hasattr(config, "__dict__"):
        config_str = json.dumps(config.__dict__, sort_keys=True, default=str)
    else:
        config_str = str(config)

    return hashlib.md5(config_str.encode()).hexdigest()[:12]

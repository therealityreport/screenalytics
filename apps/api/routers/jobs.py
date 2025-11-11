from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from py_screenalytics.artifacts import ensure_dirs, get_path

router = APIRouter()


class DetectRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")
    video: str = Field(..., description="Source video path or URL")
    stride: int = Field(5, description="Frame stride for detection sampling")
    fps: float | None = Field(None, description="Optional target FPS for sampling")
    stub: bool = Field(False, description="Use stub pipeline (fast, no ML)")


class TrackRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")


class DetectTrackRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")
    stride: int = Field(5, description="Frame stride for detection sampling")
    fps: float | None = Field(None, description="Optional target FPS for sampling")
    stub: bool = Field(False, description="Use stub pipeline (fast, no ML)")


def _artifact_summary(ep_id: str) -> dict:
    ensure_dirs(ep_id)
    return {
        "video": str(get_path(ep_id, "video")),
        "detections": str(get_path(ep_id, "detections")),
        "tracks": str(get_path(ep_id, "tracks")),
        "frames_root": str(get_path(ep_id, "frames_root")),
    }


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


@router.post("/detect")
def enqueue_detect(req: DetectRequest) -> dict:
    artifacts = _artifact_summary(req.ep_id)

    command: List[str] = [
        "python",
        "tools/episode_run.py",
        "--ep-id",
        req.ep_id,
        "--video",
        req.video,
        "--stride",
        str(req.stride),
    ]
    if req.fps is not None:
        command += ["--fps", str(req.fps)]
    if req.stub:
        command.append("--stub")

    return {
        "job": "detect",
        "ep_id": req.ep_id,
        "queued": False,
        "command": command,
        "artifacts": artifacts,
        "note": "Queue integration pending; run command manually for now.",
    }


@router.post("/track")
def enqueue_track(req: TrackRequest) -> dict:
    artifacts = _artifact_summary(req.ep_id)
    command = [
        "python",
        "FEATURES/tracking/src/bytetrack_runner.py",
        "--ep-id",
        req.ep_id,
        "--detections",
        artifacts["detections"],
        "--output",
        artifacts["tracks"],
    ]
    return {
        "job": "track",
        "ep_id": req.ep_id,
        "queued": False,
        "command": command,
        "artifacts": artifacts,
        "note": "Tracking runs locally; detection must be prepared first.",
    }


@router.post("/detect_track")
def run_detect_track(req: DetectTrackRequest) -> dict:
    artifacts = _artifact_summary(req.ep_id)
    video_path = Path(artifacts["video"])
    if not video_path.exists():
        raise HTTPException(status_code=400, detail="Episode video not uploaded yet.")

    command: List[str] = [
        "python",
        "tools/episode_run.py",
        "--ep-id",
        req.ep_id,
        "--video",
        str(video_path),
        "--stride",
        str(req.stride),
    ]
    if req.fps is not None:
        command += ["--fps", str(req.fps)]
    if req.stub:
        command.append("--stub")

    env = os.environ.copy()
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "episode_run_failed",
                "stderr": completed.stderr.strip(),
            },
        )

    detections_count = _count_lines(Path(artifacts["detections"]))
    tracks_count = _count_lines(Path(artifacts["tracks"]))

    return {
        "job": "detect_track",
        "ep_id": req.ep_id,
        "command": command,
        "stub": req.stub,
        "detections_count": detections_count,
        "tracks_count": tracks_count,
        "artifacts": artifacts,
        "note": "TODO: enqueue via orchestrator when workers are online.",
    }

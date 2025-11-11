from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from fastapi import APIRouter
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
    fps: float | None = Field(None, description="Optional frame extraction FPS")
    stub: bool = Field(False, description="Force stubbed RetinaFace")


class TrackRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")


def _artifact_summary(ep_id: str) -> dict:
    ensure_dirs(ep_id)
    return {
        "video": str(get_path(ep_id, "video")),
        "detections": str(get_path(ep_id, "detections")),
        "tracks": str(get_path(ep_id, "tracks")),
        "frames_root": str(get_path(ep_id, "frames_root")),
    }


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

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from py_screenalytics.artifacts import ensure_dirs, get_path

from apps.api.services.jobs import JobNotFoundError, JobService
from apps.api.services.storage import artifact_prefixes, episode_context_from_id

router = APIRouter()
JOB_SERVICE = JobService()


class DetectRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")
    video: str = Field(..., description="Source video path or URL")
    stride: int = Field(5, description="Frame stride for detection sampling")
    fps: float | None = Field(None, description="Optional target FPS for sampling")
    stub: bool = Field(False, description="Use stub pipeline (fast, no ML)")
    device: Literal["auto", "cpu", "mps", "cuda"] = Field("auto", description="Execution device")


class TrackRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")


class DetectTrackRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")
    stride: int = Field(5, description="Frame stride for detection sampling")
    fps: float | None = Field(None, description="Optional target FPS for sampling")
    stub: bool = Field(False, description="Use stub pipeline (fast, no ML)")
    device: Literal["auto", "cpu", "mps", "cuda"] = Field("auto", description="Execution device")
    save_frames: bool = Field(False, description="Sample full-frame JPGs to S3/local frames root")
    save_crops: bool = Field(False, description="Save per-track crops (requires tracks)")
    jpeg_quality: int = Field(85, ge=1, le=100, description="JPEG quality for frame/crop exports")


class FacesEmbedRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")
    stub: bool = Field(True, description="Use stub faces pipeline")
    device: Literal["auto", "cpu", "mps", "cuda"] = Field("auto", description="Execution device")
    save_crops: bool = Field(False, description="Export crops to data/frames + S3")
    jpeg_quality: int = Field(85, ge=1, le=100, description="JPEG quality for face crops")


class ClusterRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")
    stub: bool = Field(True, description="Use stub clustering pipeline")
    device: Literal["auto", "cpu", "mps", "cuda"] = Field("auto", description="Execution device")


def _artifact_summary(ep_id: str) -> dict:
    ensure_dirs(ep_id)
    return {
        "video": str(get_path(ep_id, "video")),
        "detections": str(get_path(ep_id, "detections")),
        "tracks": str(get_path(ep_id, "tracks")),
        "frames_root": str(get_path(ep_id, "frames_root")),
    }


def _s3_prefixes(ep_id: str) -> dict | None:
    try:
        ctx = episode_context_from_id(ep_id)
    except ValueError:
        return None
    return artifact_prefixes(ctx)


def _progress_file_path(ep_id: str) -> Path:
    manifests_dir = get_path(ep_id, "detections").parent
    manifests_dir.mkdir(parents=True, exist_ok=True)
    return manifests_dir / "progress.json"


def _build_detect_track_command(req: DetectTrackRequest, video_path: Path, progress_path: Path) -> List[str]:
    command: List[str] = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_run.py"),
        "--ep-id",
        req.ep_id,
        "--video",
        str(video_path),
        "--stride",
        str(req.stride),
        "--device",
        req.device,
        "--progress-file",
        str(progress_path),
    ]
    if req.fps is not None and req.fps > 0:
        command += ["--fps", str(req.fps)]
    if req.stub:
        command.append("--stub")
    if req.save_frames:
        command.append("--save-frames")
    if req.save_crops:
        command.append("--save-crops")
    if req.jpeg_quality and req.jpeg_quality != 85:
        command += ["--jpeg-quality", str(req.jpeg_quality)]
    return command


def _build_faces_command(req: FacesEmbedRequest, progress_path: Path) -> List[str]:
    command: List[str] = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_run.py"),
        "--ep-id",
        req.ep_id,
        "--faces-embed",
        "--device",
        req.device,
        "--progress-file",
        str(progress_path),
    ]
    if req.stub:
        command.append("--stub")
    if req.save_crops:
        command.append("--save-crops")
    if req.jpeg_quality and req.jpeg_quality != 85:
        command += ["--jpeg-quality", str(req.jpeg_quality)]
    return command


def _build_cluster_command(req: ClusterRequest, progress_path: Path) -> List[str]:
    command: List[str] = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_run.py"),
        "--ep-id",
        req.ep_id,
        "--cluster",
        "--device",
        req.device,
        "--progress-file",
        str(progress_path),
    ]
    if req.stub:
        command.append("--stub")
    return command


def _format_sse(event_name: str, payload: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"


def _parse_progress_line(line: str) -> dict | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if "phase" not in payload:
        return None
    return payload


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _wants_sse(request: Request) -> bool:
    accept = (request.headers.get("accept") or "").lower()
    return "text/event-stream" in accept


def _run_job_with_optional_sse(command: List[str], request: Request):
    env = os.environ.copy()
    if _wants_sse(request):
        generator = _stream_progress_command(command, env, request)
        return StreamingResponse(generator, media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
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
    return completed


async def _stream_progress_command(command: List[str], env: dict, request: Request):
    proc = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(PROJECT_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    saw_terminal = False
    try:
        assert proc.stdout is not None
        async for raw_line in proc.stdout:
            line = raw_line.decode().strip()
            if not line:
                continue
            payload = _parse_progress_line(line)
            if payload is None:
                continue
            phase = str(payload.get("phase", "")).lower()
            if phase == "done":
                event_name = "done"
                saw_terminal = True
            elif phase == "error":
                event_name = "error"
                saw_terminal = True
            else:
                event_name = "progress"
            yield _format_sse(event_name, payload)
            if await request.is_disconnected():
                proc.terminate()
                break
        await proc.wait()
        if proc.returncode not in (0, None) and not saw_terminal:
            stderr_text = ""
            if proc.stderr:
                stderr_bytes = await proc.stderr.read()
                stderr_text = stderr_bytes.decode().strip()
            payload = {
                "phase": "error",
                "error": stderr_text or f"episode_run exited with code {proc.returncode}",
                "return_code": proc.returncode,
            }
            yield _format_sse("error", payload)
    finally:
        if proc.returncode is None:
            proc.kill()


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
    command += ["--device", req.device]

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
def run_detect_track(req: DetectTrackRequest, request: Request):
    artifacts = _artifact_summary(req.ep_id)
    video_path = Path(artifacts["video"])
    if not video_path.exists():
        raise HTTPException(status_code=400, detail="Episode video not uploaded yet.")
    progress_path = _progress_file_path(req.ep_id)
    try:
        progress_path.unlink()
    except FileNotFoundError:
        pass
    command = _build_detect_track_command(req, video_path, progress_path)
    result = _run_job_with_optional_sse(command, request)
    if isinstance(result, StreamingResponse):
        return result

    detections_count = _count_lines(Path(artifacts["detections"]))
    tracks_count = _count_lines(Path(artifacts["tracks"]))

    return {
        "job": "detect_track",
        "ep_id": req.ep_id,
        "command": command,
        "stub": req.stub,
        "device": req.device,
        "detections_count": detections_count,
        "tracks_count": tracks_count,
        "artifacts": artifacts,
        "progress_file": str(progress_path),
    }


@router.post("/faces_embed")
def run_faces_embed(req: FacesEmbedRequest, request: Request):
    track_path = get_path(req.ep_id, "tracks")
    if not track_path.exists():
        raise HTTPException(status_code=400, detail="tracks.jsonl not found; run detect/track first")
    progress_path = _progress_file_path(req.ep_id)
    command = _build_faces_command(req, progress_path)
    result = _run_job_with_optional_sse(command, request)
    if isinstance(result, StreamingResponse):
        return result

    manifests_dir = get_path(req.ep_id, "detections").parent
    faces_path = manifests_dir / "faces.jsonl"
    faces_count = _count_lines(faces_path)
    return {
        "job": "faces_embed",
        "ep_id": req.ep_id,
        "faces_count": faces_count,
        "artifacts": {
            "faces": str(faces_path),
            "tracks": str(track_path),
            "s3_prefixes": _s3_prefixes(req.ep_id),
        },
        "progress_file": str(progress_path),
    }


@router.post("/cluster")
def run_cluster(req: ClusterRequest, request: Request):
    manifests_dir = get_path(req.ep_id, "detections").parent
    faces_path = manifests_dir / "faces.jsonl"
    if not faces_path.exists():
        raise HTTPException(status_code=400, detail="faces.jsonl not found; run faces_embed first")
    progress_path = _progress_file_path(req.ep_id)
    command = _build_cluster_command(req, progress_path)
    result = _run_job_with_optional_sse(command, request)
    if isinstance(result, StreamingResponse):
        return result

    identities_path = manifests_dir / "identities.json"
    identities_count = 0
    faces_count = _count_lines(faces_path)
    if identities_path.exists():
        try:
            identities_doc = json.loads(identities_path.read_text(encoding="utf-8"))
            identities_count = len(identities_doc.get("identities", []))
            faces_count = int(identities_doc.get("stats", {}).get("faces", faces_count))
        except json.JSONDecodeError:
            pass
    return {
        "job": "cluster",
        "ep_id": req.ep_id,
        "identities_count": identities_count,
        "faces_count": faces_count,
        "artifacts": {
            "identities": str(identities_path),
            "faces": str(faces_path),
            "s3_prefixes": _s3_prefixes(req.ep_id),
        },
        "progress_file": str(progress_path),
    }


@router.post("/detect_track_async")
def enqueue_detect_track_async(req: DetectTrackRequest) -> dict:
    artifacts = _artifact_summary(req.ep_id)
    video_path = Path(artifacts["video"])
    if not video_path.exists():
        raise HTTPException(status_code=400, detail="Episode video not uploaded yet.")
    fps_value = req.fps if req.fps and req.fps > 0 else None
    try:
        job = JOB_SERVICE.start_detect_track_job(
            ep_id=req.ep_id,
            stride=req.stride,
            fps=fps_value,
            stub=req.stub,
            device=req.device,
            video_path=video_path,
            save_frames=req.save_frames,
            save_crops=req.save_crops,
            jpeg_quality=req.jpeg_quality,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "job_id": job["job_id"],
        "ep_id": req.ep_id,
        "state": job["state"],
        "started_at": job["started_at"],
        "progress_file": job["progress_file"],
        "requested": job.get("requested"),
        "artifacts": artifacts,
    }


@router.post("/faces_embed_async")
def enqueue_faces_embed_async(req: FacesEmbedRequest) -> dict:
    try:
        job = JOB_SERVICE.start_faces_embed_job(
            ep_id=req.ep_id,
            stub=req.stub,
            device=req.device,
            save_crops=req.save_crops,
            jpeg_quality=req.jpeg_quality,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "job_id": job["job_id"],
        "ep_id": req.ep_id,
        "state": job["state"],
        "started_at": job["started_at"],
        "progress_file": job.get("progress_file"),
        "requested": job.get("requested"),
    }


@router.post("/cluster_async")
def enqueue_cluster_async(req: ClusterRequest) -> dict:
    try:
        job = JOB_SERVICE.start_cluster_job(
            ep_id=req.ep_id,
            stub=req.stub,
            device=req.device,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "job_id": job["job_id"],
        "ep_id": req.ep_id,
        "state": job["state"],
        "started_at": job["started_at"],
        "progress_file": job.get("progress_file"),
        "requested": job.get("requested"),
    }


@router.get("/{job_id}/progress")
def get_job_progress(job_id: str) -> dict:
    try:
        job = JOB_SERVICE.get(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    progress = JOB_SERVICE.get_progress(job_id) or {}
    return {
        "job_id": job_id,
        "ep_id": job["ep_id"],
        "state": job["state"],
        "started_at": job["started_at"],
        "ended_at": job.get("ended_at"),
        "progress": progress,
    }


@router.get("/{job_id}")
def job_details(job_id: str) -> dict:
    try:
        job = JOB_SERVICE.get(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job["job_id"],
        "ep_id": job["ep_id"],
        "state": job["state"],
        "started_at": job["started_at"],
        "ended_at": job.get("ended_at"),
        "summary": job.get("summary"),
        "error": job.get("error"),
        "requested": job.get("requested"),
        "progress_file": job.get("progress_file"),
    }


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    try:
        job = JOB_SERVICE.cancel(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "state": job.get("state"),
        "ended_at": job.get("ended_at"),
        "error": job.get("error"),
    }

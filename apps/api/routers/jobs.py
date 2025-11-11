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

from apps.api.services.episodes import EpisodeStore
from apps.api.services import jobs as jobs_service
from apps.api.services.jobs import JobNotFoundError, JobService
from apps.api.services.storage import artifact_prefixes, episode_context_from_id

router = APIRouter()
JOB_SERVICE = JobService()
EPISODE_STORE = EpisodeStore()
DETECTOR_CHOICES = {"retinaface"}
TRACKER_CHOICES = {"bytetrack", "strongsort"}
_DEFAULT_DETECTOR_RAW = os.getenv("DEFAULT_DETECTOR", "retinaface").lower()
DEFAULT_DETECTOR_ENV = _DEFAULT_DETECTOR_RAW if _DEFAULT_DETECTOR_RAW in DETECTOR_CHOICES else "retinaface"
DEFAULT_TRACKER_ENV = os.getenv("DEFAULT_TRACKER", "bytetrack").lower()
SCENE_DETECTOR_CHOICES = getattr(jobs_service, "SCENE_DETECTOR_CHOICES", ("pyscenedetect", "internal", "off"))
SCENE_DETECTOR_DEFAULT = getattr(jobs_service, "SCENE_DETECTOR_DEFAULT", "pyscenedetect")
SCENE_THRESHOLD_DEFAULT = getattr(jobs_service, "SCENE_THRESHOLD_DEFAULT", 27.0)
SCENE_MIN_LEN_DEFAULT = getattr(jobs_service, "SCENE_MIN_LEN_DEFAULT", 12)
SCENE_WARMUP_DETS_DEFAULT = getattr(jobs_service, "SCENE_WARMUP_DETS_DEFAULT", 3)
CLEANUP_ACTIONS = ("split_tracks", "reembed", "recluster", "group_clusters")
CleanupAction = Literal["split_tracks", "reembed", "recluster", "group_clusters"]


LEGACY_SUFFIX = "st" "ub"
LEGACY_KEYS = ("use_" + LEGACY_SUFFIX, LEGACY_SUFFIX)


def _has_legacy_marker(payload: dict | None, query_params: dict[str, str]) -> bool:
    if any(key in query_params for key in LEGACY_KEYS):
        return True
    if payload is None:
        return False
    return any(key in payload for key in LEGACY_KEYS)


async def _reject_legacy_payload(request: Request) -> None:
    query_markers = dict(request.query_params)
    payload_data: dict | None = None
    try:
        body_bytes = await request.body()
    except Exception:
        body_bytes = b""
    if body_bytes:
        try:
            parsed = json.loads(body_bytes)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            payload_data = parsed
    if _has_legacy_marker(payload_data, query_markers):
        raise HTTPException(status_code=400, detail="Stub mode is not supported.")


class DetectRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")
    video: str = Field(..., description="Source video path or URL")
    stride: int = Field(5, description="Frame stride for detection sampling")
    fps: float | None = Field(None, description="Optional target FPS for sampling")
    device: Literal["auto", "cpu", "mps", "cuda"] = Field("auto", description="Execution device")


class TrackRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")


class DetectTrackRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")
    stride: int = Field(3, description="Frame stride for detection sampling")
    fps: float | None = Field(None, description="Optional target FPS for sampling")
    device: Literal["auto", "cpu", "mps", "cuda"] = Field("auto", description="Execution device")
    save_frames: bool = Field(False, description="Sample full-frame JPGs to S3/local frames root")
    save_crops: bool = Field(False, description="Save per-track crops (requires tracks)")
    jpeg_quality: int = Field(85, ge=1, le=100, description="JPEG quality for frame/crop exports")
    detector: str = Field(
        DEFAULT_DETECTOR_ENV,
        description="Face detector backend (retinaface)",
    )
    tracker: str = Field(
        DEFAULT_TRACKER_ENV,
        description="Tracker backend (bytetrack or strongsort)",
    )
    max_gap: int | None = Field(30, ge=1, description="Frame gap before forcing a new track")
    det_thresh: float | None = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="RetinaFace detection threshold (0-1)",
    )
    scene_detector: Literal[SCENE_DETECTOR_CHOICES] = Field(
        SCENE_DETECTOR_DEFAULT,
        description="Scene-cut detector backend (PySceneDetect default, internal fallback, or off)",
    )
    scene_threshold: float = Field(SCENE_THRESHOLD_DEFAULT, ge=0.0, description="Scene-cut threshold passed to the selected detector")
    scene_min_len: int = Field(
        SCENE_MIN_LEN_DEFAULT,
        ge=1,
        description="Minimum frames between detected cuts",
    )
    scene_warmup_dets: int = Field(
        SCENE_WARMUP_DETS_DEFAULT,
        ge=0,
        description="Frames forced to run detection immediately after a cut",
    )

class FacesEmbedRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")
    device: Literal["auto", "cpu", "mps", "cuda"] | None = Field(
        None,
        description="Execution device (defaults to server auto-detect)",
    )
    save_frames: bool = Field(False, description="Export sampled frames alongside crops")
    save_crops: bool = Field(False, description="Export crops to data/frames + S3")
    jpeg_quality: int = Field(85, ge=1, le=100, description="JPEG quality for face crops")
    thumb_size: int = Field(256, ge=64, le=512, description="Square thumbnail size")

class ClusterRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")
    device: Literal["auto", "cpu", "mps", "cuda"] | None = Field(
        None,
        description="Execution device (defaults to server auto-detect)",
    )
    cluster_thresh: float = Field(0.6, gt=0.0, le=1.0, description="Cosine distance threshold for clustering")
    min_cluster_size: int = Field(2, ge=1, description="Minimum tracks per identity")


class CleanupJobRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")
    stride: int = Field(3, ge=1, le=50)
    fps: float | None = Field(None, ge=0.0)
    device: Literal["auto", "cpu", "mps", "cuda"] = Field("auto", description="Detect/track device")
    embed_device: Literal["auto", "cpu", "mps", "cuda"] = Field("auto", description="Faces embed device")
    detector: str = Field(DEFAULT_DETECTOR_ENV, description="Detector backend (retinaface)")
    tracker: str = Field(DEFAULT_TRACKER_ENV, description="Tracker backend")
    max_gap: int = Field(30, ge=1, le=240)
    det_thresh: float | None = Field(None, ge=0.0, le=1.0)
    save_frames: bool = False
    save_crops: bool = False
    jpeg_quality: int = Field(85, ge=50, le=100)
    scene_detector: Literal[SCENE_DETECTOR_CHOICES] = Field(SCENE_DETECTOR_DEFAULT)
    scene_threshold: float = Field(SCENE_THRESHOLD_DEFAULT, ge=0.0)
    scene_min_len: int = Field(SCENE_MIN_LEN_DEFAULT, ge=1)
    scene_warmup_dets: int = Field(SCENE_WARMUP_DETS_DEFAULT, ge=0)
    cluster_thresh: float = Field(0.6, ge=0.05, le=1.0)
    min_cluster_size: int = Field(2, ge=1, le=50)
    thumb_size: int = Field(256, ge=64, le=512)
    actions: List[CleanupAction] = Field(default_factory=lambda: list(CLEANUP_ACTIONS))
    write_back: bool = Field(True, description="Update people.json with grouping assignments")

def _artifact_summary(ep_id: str) -> dict:
    ensure_dirs(ep_id)
    return {
        "video": str(get_path(ep_id, "video")),
        "detections": str(get_path(ep_id, "detections")),
        "tracks": str(get_path(ep_id, "tracks")),
        "frames_root": str(get_path(ep_id, "frames_root")),
    }


def _validate_episode_ready(ep_id: str) -> Path:
    record = EPISODE_STORE.get(ep_id)
    if not record:
        raise HTTPException(status_code=400, detail="Episode not tracked yet; create it via /episodes/upsert_by_id.")
    video_path = get_path(ep_id, "video")
    if not video_path.exists():
        raise HTTPException(status_code=400, detail="Episode video not mirrored locally; run Mirror from S3.")
    if not video_path.is_file():
        raise HTTPException(status_code=400, detail="Episode video path is invalid.")
    try:
        with video_path.open("rb"):
            # Just touching the file verifies readable bytes on disk.
            pass
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Episode video unreadable: {exc}") from exc
    return video_path


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


def _normalize_detector(detector: str | None) -> str:
    fallback = DEFAULT_DETECTOR_ENV or "retinaface"
    value = (detector or fallback).strip().lower()
    if value not in DETECTOR_CHOICES:
        raise HTTPException(status_code=400, detail=f"Unsupported detector '{detector}'")
    return value


def _normalize_tracker(tracker: str | None) -> str:
    fallback = DEFAULT_TRACKER_ENV or "bytetrack"
    value = (tracker or fallback).strip().lower()
    if value not in TRACKER_CHOICES:
        raise HTTPException(status_code=400, detail=f"Unsupported tracker '{tracker}'")
    return value


def _normalize_scene_detector(scene_detector: str | None) -> str:
    fallback = SCENE_DETECTOR_DEFAULT or "pyscenedetect"
    value = (scene_detector or fallback).strip().lower()
    if value not in SCENE_DETECTOR_CHOICES:
        raise HTTPException(status_code=400, detail=f"Unsupported scene detector '{scene_detector}'")
    return value


def _build_detect_track_command(
    req: DetectTrackRequest,
    video_path: Path,
    progress_path: Path,
    detector_value: str,
    tracker_value: str,
    det_thresh: float | None,
    scene_detector: str,
    scene_threshold: float,
    scene_min_len: int,
    scene_warmup_dets: int,
) -> List[str]:
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
    if req.save_frames:
        command.append("--save-frames")
    if req.save_crops:
        command.append("--save-crops")
    if req.jpeg_quality and req.jpeg_quality != 85:
        command += ["--jpeg-quality", str(req.jpeg_quality)]
    command += ["--detector", detector_value]
    command += ["--tracker", tracker_value]
    if req.max_gap:
        command += ["--max-gap", str(req.max_gap)]
    if det_thresh is not None:
        command += ["--det-thresh", str(det_thresh)]
    command += ["--scene-detector", scene_detector]
    command += ["--scene-threshold", str(scene_threshold)]
    command += ["--scene-min-len", str(max(scene_min_len, 1))]
    command += ["--scene-warmup-dets", str(max(scene_warmup_dets, 0))]
    return command


def _build_faces_command(req: FacesEmbedRequest, progress_path: Path) -> List[str]:
    device_value = req.device or "auto"
    command: List[str] = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_run.py"),
        "--ep-id",
        req.ep_id,
        "--faces-embed",
        "--device",
        device_value,
        "--progress-file",
        str(progress_path),
    ]
    if req.save_frames:
        command.append("--save-frames")
    if req.save_crops:
        command.append("--save-crops")
    if req.jpeg_quality and req.jpeg_quality != 85:
        command += ["--jpeg-quality", str(req.jpeg_quality)]
    return command


def _build_cluster_command(req: ClusterRequest, progress_path: Path) -> List[str]:
    device_value = req.device or "auto"
    command: List[str] = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_run.py"),
        "--ep-id",
        req.ep_id,
        "--cluster",
        "--device",
        device_value,
        "--progress-file",
        str(progress_path),
    ]
    command += ["--cluster-thresh", str(req.cluster_thresh)]
    command += ["--min-cluster-size", str(req.min_cluster_size)]
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
async def enqueue_detect(req: DetectRequest, request: Request) -> dict:
    await _reject_legacy_payload(request)
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
async def run_detect_track(req: DetectTrackRequest, request: Request):
    await _reject_legacy_payload(request)
    artifacts = _artifact_summary(req.ep_id)
    video_path = _validate_episode_ready(req.ep_id)
    detector_value = _normalize_detector(req.detector)
    tracker_value = _normalize_tracker(req.tracker)
    scene_detector_value = _normalize_scene_detector(req.scene_detector)
    try:
        JOB_SERVICE.ensure_retinaface_ready(detector_value, req.device, req.det_thresh)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    progress_path = _progress_file_path(req.ep_id)
    try:
        progress_path.unlink()
    except FileNotFoundError:
        # Missing progress files are normal when a prior run never started.
        pass
    command = _build_detect_track_command(
        req,
        video_path,
        progress_path,
        detector_value,
        tracker_value,
        req.det_thresh,
        scene_detector_value,
        req.scene_threshold,
        req.scene_min_len,
        req.scene_warmup_dets,
    )
    result = _run_job_with_optional_sse(command, request)
    if isinstance(result, StreamingResponse):
        return result

    detections_count = _count_lines(Path(artifacts["detections"]))
    tracks_count = _count_lines(Path(artifacts["tracks"]))

    return {
        "job": "detect_track",
        "ep_id": req.ep_id,
        "command": command,
        "device": req.device,
        "detector": detector_value,
        "tracker": tracker_value,
        "scene_detector": scene_detector_value,
        "detections_count": detections_count,
        "tracks_count": tracks_count,
        "artifacts": artifacts,
        "progress_file": str(progress_path),
    }


@router.post("/faces_embed")
async def run_faces_embed(req: FacesEmbedRequest, request: Request):
    await _reject_legacy_payload(request)
    track_path = get_path(req.ep_id, "tracks")
    if not track_path.exists():
        raise HTTPException(status_code=400, detail="tracks.jsonl not found; run detect/track first")
    device_value = req.device or "auto"
    try:
        JOB_SERVICE.ensure_arcface_ready(device_value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    progress_path = _progress_file_path(req.ep_id)
    command = _build_faces_command(req, progress_path)
    result = _run_job_with_optional_sse(command, request)
    if isinstance(result, StreamingResponse):
        return result

    manifests_dir = get_path(req.ep_id, "detections").parent
    faces_path = manifests_dir / "faces.jsonl"
    faces_count = _count_lines(faces_path)
    frames_dir = get_path(req.ep_id, "frames_root") / "frames"
    return {
        "job": "faces_embed",
        "ep_id": req.ep_id,
        "faces_count": faces_count,
        "artifacts": {
            "faces": str(faces_path),
            "tracks": str(track_path),
            "frames": str(frames_dir) if req.save_frames else None,
            "s3_prefixes": _s3_prefixes(req.ep_id),
        },
        "progress_file": str(progress_path),
    }


@router.post("/cluster")
async def run_cluster(req: ClusterRequest, request: Request):
    await _reject_legacy_payload(request)
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
            # A partially written identities.json should not break the API response.
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
async def enqueue_detect_track_async(req: DetectTrackRequest, request: Request) -> dict:
    await _reject_legacy_payload(request)
    artifacts = _artifact_summary(req.ep_id)
    video_path = _validate_episode_ready(req.ep_id)
    fps_value = req.fps if req.fps and req.fps > 0 else None
    detector_value = _normalize_detector(req.detector)
    tracker_value = _normalize_tracker(req.tracker)
    scene_detector_value = _normalize_scene_detector(req.scene_detector)
    try:
        job = JOB_SERVICE.start_detect_track_job(
            ep_id=req.ep_id,
            stride=req.stride,
            fps=fps_value,
            device=req.device,
            video_path=video_path,
            save_frames=req.save_frames,
            save_crops=req.save_crops,
            jpeg_quality=req.jpeg_quality,
            detector=detector_value,
            tracker=tracker_value,
            max_gap=req.max_gap,
            det_thresh=req.det_thresh,
            scene_detector=scene_detector_value,
            scene_threshold=req.scene_threshold,
            scene_min_len=req.scene_min_len,
            scene_warmup_dets=req.scene_warmup_dets,
        )
    except (FileNotFoundError, ValueError) as exc:
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
async def enqueue_faces_embed_async(req: FacesEmbedRequest, request: Request) -> dict:
    await _reject_legacy_payload(request)
    try:
        job = JOB_SERVICE.start_faces_embed_job(
            ep_id=req.ep_id,
            device=req.device or "auto",
            save_frames=req.save_frames,
            save_crops=req.save_crops,
            jpeg_quality=req.jpeg_quality,
            thumb_size=req.thumb_size,
        )
    except (FileNotFoundError, ValueError) as exc:
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
async def enqueue_cluster_async(req: ClusterRequest, request: Request) -> dict:
    await _reject_legacy_payload(request)
    try:
        job = JOB_SERVICE.start_cluster_job(
            ep_id=req.ep_id,
            device=req.device or "auto",
            cluster_thresh=req.cluster_thresh,
            min_cluster_size=req.min_cluster_size,
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


@router.post("/episode_cleanup_async")
async def enqueue_episode_cleanup_async(req: CleanupJobRequest, request: Request) -> dict:
    await _reject_legacy_payload(request)
    artifacts = _artifact_summary(req.ep_id)
    video_path = _validate_episode_ready(req.ep_id)
    fps_value = req.fps if req.fps and req.fps > 0 else None
    detector_value = _normalize_detector(req.detector)
    tracker_value = _normalize_tracker(req.tracker)
    scene_detector_value = _normalize_scene_detector(req.scene_detector)
    actions = req.actions or list(CLEANUP_ACTIONS)
    try:
        job = JOB_SERVICE.start_episode_cleanup_job(
            ep_id=req.ep_id,
            video_path=video_path,
            stride=req.stride,
            fps=fps_value,
            device=req.device,
            embed_device=req.embed_device or req.device,
            save_frames=req.save_frames,
            save_crops=req.save_crops,
            jpeg_quality=req.jpeg_quality,
            detector=detector_value,
            tracker=tracker_value,
            max_gap=req.max_gap,
            det_thresh=req.det_thresh,
            scene_detector=scene_detector_value,
            scene_threshold=req.scene_threshold,
            scene_min_len=req.scene_min_len,
            scene_warmup_dets=req.scene_warmup_dets,
            cluster_thresh=req.cluster_thresh,
            min_cluster_size=req.min_cluster_size,
            thumb_size=req.thumb_size,
            actions=actions,
            write_back=req.write_back,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "job_id": job["job_id"],
        "ep_id": req.ep_id,
        "state": job["state"],
        "started_at": job["started_at"],
        "progress_file": job.get("progress_file"),
        "requested": job.get("requested"),
        "artifacts": artifacts,
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

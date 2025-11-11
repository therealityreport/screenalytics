from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Dict, List

import json

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from py_screenalytics.artifacts import ensure_dirs, get_path

from apps.api.services.episodes import EpisodeStore
from apps.api.services.storage import StorageService

router = APIRouter()
EPISODE_STORE = EpisodeStore()
STORAGE = StorageService()


class EpisodeCreateRequest(BaseModel):
    show_slug_or_id: str = Field(..., min_length=1, description="Show slug or identifier")
    season_number: int = Field(..., ge=0, le=999, description="Season number")
    episode_number: int = Field(..., ge=0, le=999, description="Episode number within the season")
    title: str | None = Field(None, max_length=200)
    air_date: date | None = None


class EpisodeCreateResponse(BaseModel):
    ep_id: str


class EpisodeSummary(BaseModel):
    ep_id: str
    show_slug: str
    season_number: int
    episode_number: int
    title: str | None
    air_date: str | None


class EpisodeListResponse(BaseModel):
    episodes: List[EpisodeSummary]


class S3VideoItem(BaseModel):
    bucket: str
    key: str
    ep_id: str
    show: str | None = None
    season: int | None = None
    episode: int | None = None
    size: int | None = None
    last_modified: str | None = None
    etag: str | None = None
    exists_in_store: bool
    key_version: str | None = None


class S3VideosResponse(BaseModel):
    items: List[S3VideoItem]
    count: int


class EpisodeS3Status(BaseModel):
    bucket: str
    v2_key: str | None = None
    v2_exists: bool = False
    v1_key: str
    v1_exists: bool


class EpisodeLocalStatus(BaseModel):
    path: str
    exists: bool


class EpisodeDetailResponse(BaseModel):
    ep_id: str
    show_slug: str
    season_number: int
    episode_number: int
    title: str | None
    air_date: str | None
    s3: EpisodeS3Status
    local: EpisodeLocalStatus


class AssetUploadResponse(BaseModel):
    ep_id: str
    method: str
    bucket: str
    key: str
    object_key: str | None = None  # backwards compatibility
    upload_url: str | None
    expires_in: int | None
    headers: Dict[str, str]
    path: str | None = None
    local_video_path: str
    backend: str


class EpisodeMirrorResponse(BaseModel):
    ep_id: str
    local_video_path: str
    bytes: int | None = None
    etag: str | None = None
    used_key_version: str | None = None


class EpisodeVideoMeta(BaseModel):
    ep_id: str
    local_exists: bool
    local_video_path: str
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    duration_sec: float | None = None
    fps_detected: float | None = None


@router.get("/episodes", response_model=EpisodeListResponse, tags=["episodes"])
def list_episodes() -> EpisodeListResponse:
    records = EPISODE_STORE.list()
    episodes = [
        EpisodeSummary(
            ep_id=record.ep_id,
            show_slug=record.show_ref,
            season_number=record.season_number,
            episode_number=record.episode_number,
            title=record.title,
            air_date=record.air_date,
        )
        for record in records
    ]
    return EpisodeListResponse(episodes=episodes)


@router.get("/episodes/s3_videos", response_model=S3VideosResponse, tags=["episodes"])
def list_s3_videos(q: str | None = Query(None), limit: int = Query(200, ge=1, le=1000)) -> S3VideosResponse:
    raw_items = STORAGE.list_episode_videos_s3(limit=limit)
    items: List[S3VideoItem] = []
    for obj in raw_items:
        ep_id = obj.get("ep_id")
        if not isinstance(ep_id, str):
            continue
        if q and q.lower() not in ep_id.lower():
            continue
        items.append(
            S3VideoItem(
                bucket=obj.get("bucket", STORAGE.bucket),
                key=str(obj.get("key")),
                ep_id=ep_id,
                show=obj.get("show"),
                season=obj.get("season"),
                episode=obj.get("episode"),
                size=obj.get("size"),
                last_modified=str(obj.get("last_modified")) if obj.get("last_modified") else None,
                etag=obj.get("etag"),
                exists_in_store=EPISODE_STORE.exists(ep_id),
                key_version=obj.get("key_version"),
            )
        )
        if len(items) >= limit:
            break
    return S3VideosResponse(items=items, count=len(items))


@router.get("/episodes/{ep_id}", response_model=EpisodeDetailResponse, tags=["episodes"])
def episode_details(ep_id: str) -> EpisodeDetailResponse:
    record = EPISODE_STORE.get(ep_id)
    if not record:
        raise HTTPException(status_code=404, detail="Episode not found")

    local_path = get_path(ep_id, "video")
    v2_key = STORAGE.video_object_key_v2(record.show_ref, record.season_number, record.episode_number)
    v1_key = STORAGE.video_object_key_v1(ep_id)
    v2_exists = STORAGE.object_exists(v2_key)
    v1_exists = STORAGE.object_exists(v1_key)

    return EpisodeDetailResponse(
        ep_id=record.ep_id,
        show_slug=record.show_ref,
        season_number=record.season_number,
        episode_number=record.episode_number,
        title=record.title,
        air_date=record.air_date,
        s3=EpisodeS3Status(
            bucket=STORAGE.bucket,
            v2_key=v2_key,
            v2_exists=v2_exists,
            v1_key=v1_key,
            v1_exists=v1_exists,
        ),
        local=EpisodeLocalStatus(path=str(local_path), exists=local_path.exists()),
    )


@router.get("/episodes/{ep_id}/progress", tags=["episodes"])
def episode_progress(ep_id: str) -> dict:
    progress_path = get_path(ep_id, "detections").parent / "progress.json"
    if not progress_path.exists():
        raise HTTPException(status_code=404, detail="Progress not available")
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=503, detail="Progress file corrupt") from exc
    return {"ep_id": ep_id, "progress": payload}


@router.post("/episodes", response_model=EpisodeCreateResponse, tags=["episodes"])
def create_episode(payload: EpisodeCreateRequest) -> EpisodeCreateResponse:
    record = EPISODE_STORE.upsert(
        show_ref=payload.show_slug_or_id,
        season_number=payload.season_number,
        episode_number=payload.episode_number,
        title=payload.title,
        air_date=payload.air_date,
    )
    return EpisodeCreateResponse(ep_id=record.ep_id)


@router.post("/episodes/{ep_id}/assets", response_model=AssetUploadResponse, tags=["episodes"])
def presign_episode_assets(ep_id: str) -> AssetUploadResponse:
    record = EPISODE_STORE.get(ep_id)
    if not record:
        raise HTTPException(status_code=404, detail="Episode not found")

    ensure_dirs(ep_id)
    v2_key = STORAGE.video_object_key_v2(record.show_ref, record.season_number, record.episode_number)
    presigned = STORAGE.presign_episode_video(ep_id, object_key=v2_key)
    local_video_path = get_path(ep_id, "video")
    path = presigned.path or (str(local_video_path) if presigned.method == "FILE" else None)

    return AssetUploadResponse(
        ep_id=presigned.ep_id,
        method=presigned.method,
        bucket=presigned.bucket,
        key=presigned.object_key,
        object_key=presigned.object_key,
        upload_url=presigned.upload_url,
        expires_in=presigned.expires_in,
        headers=presigned.headers,
        path=path,
        local_video_path=str(local_video_path),
        backend=presigned.backend,
    )


@router.post(
    "/episodes/{ep_id}/mirror",
    response_model=EpisodeMirrorResponse,
    tags=["episodes"],
)
def mirror_episode_video(ep_id: str) -> EpisodeMirrorResponse:
    record = EPISODE_STORE.get(ep_id)
    if not record:
        raise HTTPException(status_code=404, detail="Episode not found")

    ensure_dirs(ep_id)
    try:
        result = STORAGE.ensure_local_mirror(
            ep_id,
            show_ref=record.show_ref,
            season_number=record.season_number,
            episode_number=record.episode_number,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return EpisodeMirrorResponse(
        ep_id=ep_id,
        local_video_path=str(result.get("local_video_path")),
        bytes=result.get("bytes"),
        etag=result.get("etag"),
        used_key_version=result.get("used_key_version"),
    )


@router.post(
    "/episodes/{ep_id}/hydrate",
    response_model=EpisodeMirrorResponse,
    tags=["episodes"],
)
def hydrate_episode_video(ep_id: str) -> EpisodeMirrorResponse:
    return mirror_episode_video(ep_id)


@router.get(
    "/episodes/{ep_id}/video_meta",
    response_model=EpisodeVideoMeta,
    tags=["episodes"],
)
def episode_video_meta(ep_id: str) -> EpisodeVideoMeta:
    video_path = get_path(ep_id, "video")
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Local video not found")

    width: int | None = None
    height: int | None = None
    frames: int | None = None
    fps_detected: float | None = None
    duration_sec: float | None = None

    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
            frames_val = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            frames = int(frames_val) if frames_val and frames_val > 0 else None
            fps_val = cap.get(cv2.CAP_PROP_FPS)
            fps_detected = float(fps_val) if fps_val and fps_val > 0 else None
            if frames and fps_detected:
                duration_sec = frames / fps_detected
        cap.release()
    except Exception as exc:  # pragma: no cover - best effort
        raise HTTPException(status_code=500, detail=f"Failed to analyze video: {exc}") from exc

    return EpisodeVideoMeta(
        ep_id=ep_id,
        local_exists=True,
        local_video_path=str(video_path),
        width=width,
        height=height,
        frames=frames,
        duration_sec=duration_sec,
        fps_detected=fps_detected,
    )

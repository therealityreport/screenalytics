from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, HTTPException
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


class AssetUploadResponse(BaseModel):
    ep_id: str
    method: str
    bucket: str
    object_key: str
    upload_url: str | None
    expires_in: int | None
    headers: Dict[str, str]
    path: str | None = None
    local_video_path: str


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
    presigned = STORAGE.presign_episode_video(ep_id)
    local_video_path = get_path(ep_id, "video")
    path = presigned.path or (str(local_video_path) if presigned.method == "FILE" else None)

    return AssetUploadResponse(
        ep_id=presigned.ep_id,
        method=presigned.method,
        bucket=presigned.bucket,
        object_key=presigned.object_key,
        upload_url=presigned.upload_url,
        expires_in=presigned.expires_in,
        headers=presigned.headers,
        path=path,
        local_video_path=str(local_video_path),
    )

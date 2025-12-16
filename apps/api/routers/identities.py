from __future__ import annotations

from pathlib import Path

import sys

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from apps.api.services import identities as identity_service
from py_screenalytics import run_layout

router = APIRouter(prefix="/identities", tags=["identities"])


def _normalize_run_id_or_422(run_id: str) -> str:
    try:
        return run_layout.normalize_run_id(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class RenameRequest(BaseModel):
    identity_id: str = Field(..., description="Identity to rename")
    new_label: str | None = Field(None, description="New label")


class MergeRequest(BaseModel):
    source_id: str
    target_id: str


class MoveTrackRequest(BaseModel):
    track_id: int
    target_identity_id: str | None = Field(None, description="Destination identity or null")


class DropTrackRequest(BaseModel):
    track_id: int


class DropFrameRequest(BaseModel):
    track_id: int
    frame_idx: int
    delete_assets: bool = False


@router.post("/{ep_id}/rename")
def rename_identity(ep_id: str, body: RenameRequest, run_id: str = Query(..., description="Run id scope (required)")) -> dict:
    try:
        run_id_norm = _normalize_run_id_or_422(run_id)
        identity = identity_service.rename_identity(ep_id, body.identity_id, body.new_label, run_id=run_id_norm)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"identity_id": body.identity_id, "label": identity.get("label")}


@router.post("/{ep_id}/merge")
def merge_identities(ep_id: str, body: MergeRequest, run_id: str = Query(..., description="Run id scope (required)")) -> dict:
    if body.source_id == body.target_id:
        raise HTTPException(status_code=400, detail="Source and target must be different")
    try:
        run_id_norm = _normalize_run_id_or_422(run_id)
        merged = identity_service.merge_identities(ep_id, body.source_id, body.target_id, run_id=run_id_norm)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"identity_id": body.target_id, "track_ids": merged.get("track_ids", [])}


@router.post("/{ep_id}/move_track")
def move_track(ep_id: str, body: MoveTrackRequest, run_id: str = Query(..., description="Run id scope (required)")) -> dict:
    try:
        run_id_norm = _normalize_run_id_or_422(run_id)
        result = identity_service.move_track(ep_id, body.track_id, body.target_identity_id, run_id=run_id_norm)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result


@router.post("/{ep_id}/drop_track")
def drop_track(ep_id: str, body: DropTrackRequest, run_id: str = Query(..., description="Run id scope (required)")) -> dict:
    try:
        run_id_norm = _normalize_run_id_or_422(run_id)
        return identity_service.drop_track(ep_id, body.track_id, run_id=run_id_norm)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{ep_id}/drop_frame")
def drop_frame(ep_id: str, body: DropFrameRequest, run_id: str = Query(..., description="Run id scope (required)")) -> dict:
    try:
        run_id_norm = _normalize_run_id_or_422(run_id)
        return identity_service.drop_frame(ep_id, body.track_id, body.frame_idx, body.delete_assets, run_id=run_id_norm)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

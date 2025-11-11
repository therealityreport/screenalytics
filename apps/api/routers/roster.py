from __future__ import annotations

from pathlib import Path
import sys

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from apps.api.services import roster as roster_service

router = APIRouter(prefix="/shows", tags=["shows"])


class CastNameRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


def _add_name(show: str, name: str) -> dict:
    cleaned = (name or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="name required")
    try:
        return roster_service.add_if_missing(show, cleaned)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{show}/roster")
def get_roster(show: str) -> dict:
    try:
        return roster_service.load_roster(show)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{show}/roster/names")
def add_roster_name(show: str, payload: CastNameRequest) -> dict:
    return _add_name(show, payload.name)


@router.get("/{show}/cast_names")
def list_cast_names(show: str) -> dict:
    roster = get_roster(show)
    return {"show": roster["show"], "names": roster.get("names", []), "updated_at": roster.get("updated_at")}


@router.post("/{show}/cast_names")
def add_cast_name(show: str, payload: CastNameRequest) -> dict:
    roster = _add_name(show, payload.name)
    return {"show": roster["show"], "names": roster.get("names", []), "updated_at": roster.get("updated_at")}

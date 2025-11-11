"""Cast management API endpoints."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from apps.api.services.cast import CastService

router = APIRouter()
cast_service = CastService()


class CastMemberResponse(BaseModel):
    cast_id: str
    show_id: str
    name: str
    role: str
    status: str
    aliases: List[str]
    seasons: List[str]
    social: dict
    created_at: str
    updated_at: str


class CastMemberCreateRequest(BaseModel):
    name: str = Field(..., description="Cast member name")
    role: str = Field("other", description="Role: main, friend, guest, other")
    status: str = Field("active", description="Status: active, past, inactive")
    aliases: Optional[List[str]] = Field(default_factory=list, description="Aliases/nicknames")
    seasons: Optional[List[str]] = Field(default_factory=list, description="Season IDs (e.g., S05)")
    social: Optional[dict] = Field(default_factory=dict, description="Social media handles")


class CastMemberUpdateRequest(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    status: Optional[str] = None
    aliases: Optional[List[str]] = None
    seasons: Optional[List[str]] = None
    social: Optional[dict] = None


@router.get("/shows/{show_id}/cast")
def list_cast(show_id: str, season: Optional[str] = None) -> dict:
    """Get all cast members for a show, optionally filtered by season."""
    cast_members = cast_service.list_cast(show_id, season=season)
    return {
        "show_id": show_id,
        "season": season,
        "cast": cast_members,
        "count": len(cast_members),
    }


@router.get("/shows/{show_id}/cast/{cast_id}")
def get_cast_member(show_id: str, cast_id: str) -> CastMemberResponse:
    """Get a specific cast member."""
    member = cast_service.get_cast_member(show_id, cast_id)
    if not member:
        raise HTTPException(status_code=404, detail=f"Cast member {cast_id} not found")
    return CastMemberResponse(**member)


@router.post("/shows/{show_id}/cast")
def create_cast_member(show_id: str, body: CastMemberCreateRequest) -> CastMemberResponse:
    """Create a new cast member."""
    member = cast_service.create_cast_member(
        show_id,
        name=body.name,
        role=body.role,
        status=body.status,
        aliases=body.aliases,
        seasons=body.seasons,
        social=body.social,
    )
    return CastMemberResponse(**member)


@router.patch("/shows/{show_id}/cast/{cast_id}")
def update_cast_member(show_id: str, cast_id: str, body: CastMemberUpdateRequest) -> CastMemberResponse:
    """Update a cast member."""
    member = cast_service.update_cast_member(
        show_id,
        cast_id,
        name=body.name,
        role=body.role,
        status=body.status,
        aliases=body.aliases,
        seasons=body.seasons,
        social=body.social,
    )
    if not member:
        raise HTTPException(status_code=404, detail=f"Cast member {cast_id} not found")
    return CastMemberResponse(**member)


@router.delete("/shows/{show_id}/cast/{cast_id}")
def delete_cast_member(show_id: str, cast_id: str) -> dict:
    """Delete a cast member."""
    success = cast_service.delete_cast_member(show_id, cast_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Cast member {cast_id} not found")
    return {"status": "deleted", "cast_id": cast_id}


class BulkImportRequest(BaseModel):
    members: List[Dict[str, Any]] = Field(..., description="List of cast member data")
    force_new: bool = Field(False, description="Always create new members (skip merge by name)")


@router.post("/shows/{show_id}/cast/import")
def bulk_import_cast(show_id: str, body: BulkImportRequest) -> dict:
    """Bulk import cast members.

    Members with matching names (case-insensitive) will be updated unless force_new=true.

    Example JSON:
    {
      "members": [
        {
          "name": "Kyle Richards",
          "role": "main",
          "status": "active",
          "aliases": ["Kyle", "Kyle R"],
          "seasons": ["S01", "S02", "S03"]
        }
      ],
      "force_new": false
    }
    """
    result = cast_service.bulk_import_cast(show_id, body.members, force_new=body.force_new)
    return result


__all__ = ["router"]

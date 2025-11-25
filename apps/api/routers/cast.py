"""Cast management API endpoints."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from apps.api.services.cast import CastService
from apps.api.services.facebank import FacebankService
from apps.api.services.people import PeopleService
from apps.api.services.storage import StorageService

router = APIRouter()


def _cast_service() -> CastService:
    data_root = os.environ.get("SCREENALYTICS_DATA_ROOT")
    return CastService(data_root=data_root) if data_root else CastService()


def _facebank_service() -> FacebankService:
    data_root = os.environ.get("SCREENALYTICS_DATA_ROOT")
    return FacebankService(data_root=data_root) if data_root else FacebankService()


def _storage_service() -> StorageService:
    return StorageService()


def _people_service() -> PeopleService:
    data_root = os.environ.get("SCREENALYTICS_DATA_ROOT")
    return PeopleService(data_root=data_root) if data_root else PeopleService()


class ShowEntry(BaseModel):
    show_id: str
    title: Optional[str] = None
    full_name: Optional[str] = None
    imdb_series_id: Optional[str] = None
    cast_count: int


class ShowListResponse(BaseModel):
    shows: List[ShowEntry]
    count: int


class ShowRegistrationResponse(ShowEntry):
    created: bool


class ShowCreateRequest(BaseModel):
    show_id: str = Field(..., description="Unique show identifier (slug)")
    title: Optional[str] = Field(None, description="Short display name")
    full_name: Optional[str] = Field(None, description="Full/legal show name")
    imdb_series_id: Optional[str] = Field(None, description="IMDb series identifier (e.g., tt2861424)")


class CastMemberResponse(BaseModel):
    cast_id: str
    show_id: str
    name: str
    full_name: Optional[str] = None
    role: str
    status: str
    aliases: List[str]
    seasons: List[str]
    social: dict
    imdb_id: Optional[str] = None
    created_at: str
    updated_at: str


class CastMemberCreateRequest(BaseModel):
    name: str = Field(..., description="Cast member name")
    full_name: Optional[str] = Field(None, description="Legal/full name")
    role: str = Field("other", description="Role: main, friend, guest, other")
    status: str = Field("active", description="Status: active, past, inactive")
    aliases: Optional[List[str]] = Field(default_factory=list, description="Aliases/nicknames")
    seasons: Optional[List[str]] = Field(default_factory=list, description="Season IDs (e.g., S05)")
    social: Optional[dict] = Field(default_factory=dict, description="Social media handles")
    imdb_id: Optional[str] = Field(None, description="IMDb person identifier (e.g., nm0000001)")


class CastMemberUpdateRequest(BaseModel):
    name: Optional[str] = None
    full_name: Optional[str] = None
    role: Optional[str] = None
    status: Optional[str] = None
    aliases: Optional[List[str]] = None
    seasons: Optional[List[str]] = None
    social: Optional[dict] = None
    imdb_id: Optional[str] = None


def _featured_thumbnail_url(show_id: str, cast_id: str) -> Optional[str]:
    """Resolve the featured thumbnail URL for a cast member if available."""
    facebank = _facebank_service().get_facebank(show_id, cast_id)
    featured_id = facebank.get("featured_seed_id")
    if not featured_id:
        return None

    seeds = facebank.get("seeds", [])
    for seed in seeds:
        if seed.get("fb_id") != featured_id:
            continue
        preferred_key = seed.get("display_s3_key") or seed.get("image_s3_key")
        if preferred_key:
            presigned = _storage_service().presign_get(preferred_key)
            if presigned:
                return presigned
        for url_key in ("display_url", "display_uri", "image_uri"):
            url_value = seed.get(url_key)
            if url_value:
                return url_value
    return None


@router.get("/shows")
def list_registered_shows() -> ShowListResponse:
    shows = _cast_service().list_registered_shows()
    entries = [ShowEntry(**entry) for entry in shows]
    return ShowListResponse(shows=entries, count=len(entries))


@router.post("/shows")
def register_show(body: ShowCreateRequest) -> ShowRegistrationResponse:
    try:
        entry = _cast_service().register_show(
            body.show_id,
            title=body.title,
            full_name=body.full_name,
            imdb_series_id=body.imdb_series_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ShowRegistrationResponse(**entry)


@router.get("/shows/{show_id}/cast")
def list_cast(
    show_id: str,
    season: Optional[str] = None,
    include_featured: bool = Query(
        False,
        description="Include featured thumbnail URLs sourced from facebank",
    ),
) -> dict:
    """Get all cast members for a show, optionally filtered by season."""
    cast_members = _cast_service().list_cast(show_id, season=season)

    if include_featured and cast_members:
        enriched: List[Dict[str, Any]] = []
        for member in cast_members:
            if not isinstance(member, dict):
                enriched.append(member)
                continue
            cast_entry = dict(member)
            cast_id = cast_entry.get("cast_id")
            if cast_id:
                thumb_url = _featured_thumbnail_url(show_id, cast_id)
                if thumb_url:
                    cast_entry["featured_thumbnail_url"] = thumb_url
            enriched.append(cast_entry)
        cast_members = enriched

    return {
        "show_id": show_id,
        "season": season,
        "cast": cast_members,
        "count": len(cast_members),
    }


@router.get("/shows/{show_id}/cast/{cast_id}")
def get_cast_member(show_id: str, cast_id: str) -> CastMemberResponse:
    """Get a specific cast member."""
    member = _cast_service().get_cast_member(show_id, cast_id)
    if not member:
        raise HTTPException(status_code=404, detail=f"Cast member {cast_id} not found")
    return CastMemberResponse(**member)


@router.post("/shows/{show_id}/cast")
def create_cast_member(show_id: str, body: CastMemberCreateRequest) -> CastMemberResponse:
    """Create a new cast member."""
    member = _cast_service().create_cast_member(
        show_id,
        name=body.name,
        full_name=body.full_name,
        role=body.role,
        status=body.status,
        aliases=body.aliases,
        seasons=body.seasons,
        social=body.social,
        imdb_id=body.imdb_id,
    )

    # Ensure a People record exists and carries this cast_id for UI/assignment flows
    try:
        people_service = _people_service()
        existing = people_service.find_person_by_name_or_alias(show_id, body.name, cast_id=member["cast_id"])
        if existing:
            # Persist cast_id (idempotent) and merge aliases when provided
            merged_aliases = existing.get("aliases") or []
            if body.aliases:
                for alias in body.aliases:
                    if alias and alias not in merged_aliases:
                        merged_aliases.append(alias)
            people_service.update_person(
                show_id,
                existing["person_id"],
                cast_id=member["cast_id"],
                aliases=merged_aliases,
            )
        else:
            people_service.create_person(show_id, name=body.name, aliases=body.aliases, cast_id=member["cast_id"])
    except Exception as exc:
        # Do not fail cast creation if people sync encounters an edge case
        LOGGER.warning("Failed to sync cast member %s to people: %s", body.name, exc)
    return CastMemberResponse(**member)


@router.patch("/shows/{show_id}/cast/{cast_id}")
def update_cast_member(show_id: str, cast_id: str, body: CastMemberUpdateRequest) -> CastMemberResponse:
    """Update a cast member."""
    member = _cast_service().update_cast_member(
        show_id,
        cast_id,
        name=body.name,
        full_name=body.full_name,
        role=body.role,
        status=body.status,
        aliases=body.aliases,
        seasons=body.seasons,
        social=body.social,
        imdb_id=body.imdb_id,
    )
    if not member:
        raise HTTPException(status_code=404, detail=f"Cast member {cast_id} not found")

    # Sync updates to People service to maintain consistency
    try:
        people_service = _people_service()
        existing = next(
            (p for p in people_service.list_people(show_id) if p.get("cast_id") == cast_id),
            None,
        )
        if existing:
            # Update the person with new name and aliases
            update_fields = {}
            if body.name is not None:
                update_fields["name"] = body.name
            if body.aliases is not None:
                update_fields["aliases"] = body.aliases
            if update_fields:
                people_service.update_person(show_id, existing["person_id"], **update_fields)
    except Exception as exc:
        # Do not fail cast update if people sync encounters an edge case
        LOGGER.warning("Failed to sync cast member %s update to people: %s", cast_id, exc)

    return CastMemberResponse(**member)


@router.delete("/shows/{show_id}/cast/{cast_id}")
def delete_cast_member(show_id: str, cast_id: str) -> dict:
    """Delete a cast member."""
    success = _cast_service().delete_cast_member(show_id, cast_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Cast member {cast_id} not found")

    # Clear cast_id from any linked People record (don't delete the person, just unlink)
    try:
        from apps.api.services.people import CLEAR_FIELD

        people_service = _people_service()
        existing = next(
            (p for p in people_service.list_people(show_id) if p.get("cast_id") == cast_id),
            None,
        )
        if existing:
            people_service.update_person(show_id, existing["person_id"], cast_id=CLEAR_FIELD)
            LOGGER.info("Unlinked person %s from deleted cast member %s", existing["person_id"], cast_id)
    except Exception as exc:
        LOGGER.warning("Failed to unlink person from deleted cast member %s: %s", cast_id, exc)

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
    result = _cast_service().bulk_import_cast(show_id, body.members, force_new=body.force_new)
    return result


__all__ = ["router"]

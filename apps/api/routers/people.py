"""People management endpoints for show-level person entities."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from apps.api.services.people import PeopleService
from apps.api.services.storage import StorageService

router = APIRouter()
people_service = PeopleService()
storage_service = StorageService()


class PersonResponse(BaseModel):
    person_id: str
    show_id: str
    name: Optional[str]
    aliases: List[str] = []
    prototype: List[float]
    cluster_ids: List[str]
    rep_crop: Optional[str]
    rep_crop_s3_key: Optional[str] = None
    created_at: str
    cast_id: Optional[str] = None


class PersonUpdateRequest(BaseModel):
    name: Optional[str] = None
    rep_crop: Optional[str] = None
    rep_crop_s3_key: Optional[str] = None
    aliases: Optional[List[str]] = None


class PersonCreateRequest(BaseModel):
    name: Optional[str] = None
    aliases: Optional[List[str]] = None


class PersonMergeRequest(BaseModel):
    source_person_id: str
    target_person_id: str


class PersonAddAliasRequest(BaseModel):
    alias: str


def _hydrate_rep_crop(person: dict) -> dict:
    record = dict(person)
    key = record.get("rep_crop_s3_key")
    if key:
        url = storage_service.presign_get(key)
        if url:
            record["rep_crop"] = url
    # Ensure backwards compatibility
    if "aliases" not in record:
        record["aliases"] = []
    return record


@router.get("/shows/{show_id}/people")
def list_people(show_id: str) -> dict:
    """Get all people for a show."""
    people = people_service.list_people(show_id)
    hydrated = [_hydrate_rep_crop(person) for person in people]
    return {
        "show_id": show_id,
        "people": hydrated,
        "count": len(hydrated),
    }


@router.get("/shows/{show_id}/people/{person_id}")
def get_person(show_id: str, person_id: str) -> PersonResponse:
    """Get a specific person."""
    person = people_service.get_person(show_id, person_id)
    if not person:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
    return PersonResponse(**_hydrate_rep_crop(person))


@router.post("/shows/{show_id}/people")
def create_person(show_id: str, body: PersonCreateRequest) -> PersonResponse:
    """Create a new person."""
    person = people_service.create_person(show_id, name=body.name)
    return PersonResponse(**_hydrate_rep_crop(person))


@router.patch("/shows/{show_id}/people/{person_id}")
def update_person(show_id: str, person_id: str, body: PersonUpdateRequest) -> PersonResponse:
    """Update a person."""
    person = people_service.update_person(
        show_id,
        person_id,
        name=body.name,
        rep_crop=body.rep_crop,
        rep_crop_s3_key=body.rep_crop_s3_key,
    )
    if not person:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
    return PersonResponse(**_hydrate_rep_crop(person))


@router.delete("/shows/{show_id}/people/{person_id}")
def delete_person(show_id: str, person_id: str) -> dict:
    """Delete a person."""
    success = people_service.delete_person(show_id, person_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
    return {"status": "deleted", "person_id": person_id}


@router.post("/shows/{show_id}/people/merge")
def merge_people(show_id: str, body: PersonMergeRequest) -> PersonResponse:
    """Merge source person into target person."""
    if body.source_person_id == body.target_person_id:
        raise HTTPException(status_code=400, detail="Source and target must be different")

    result = people_service.merge_people(show_id, body.source_person_id, body.target_person_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Source ({body.source_person_id}) or target ({body.target_person_id}) not found"
        )

    return PersonResponse(**_hydrate_rep_crop(result))


@router.post("/shows/{show_id}/people/{person_id}/add_alias")
def add_alias(show_id: str, person_id: str, body: PersonAddAliasRequest) -> PersonResponse:
    """Add an alias to a person."""
    person = people_service.add_alias_to_person(show_id, person_id, body.alias)
    if not person:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
    return PersonResponse(**_hydrate_rep_crop(person))


__all__ = ["router"]

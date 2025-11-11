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

router = APIRouter()
people_service = PeopleService()


class PersonResponse(BaseModel):
    person_id: str
    show_id: str
    name: Optional[str]
    prototype: List[float]
    cluster_ids: List[str]
    rep_crop: Optional[str]
    created_at: str


class PersonUpdateRequest(BaseModel):
    name: Optional[str] = None
    rep_crop: Optional[str] = None


class PersonCreateRequest(BaseModel):
    name: Optional[str] = None


@router.get("/shows/{show_id}/people")
def list_people(show_id: str) -> dict:
    """Get all people for a show."""
    people = people_service.list_people(show_id)
    return {
        "show_id": show_id,
        "people": people,
        "count": len(people),
    }


@router.get("/shows/{show_id}/people/{person_id}")
def get_person(show_id: str, person_id: str) -> PersonResponse:
    """Get a specific person."""
    person = people_service.get_person(show_id, person_id)
    if not person:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
    return PersonResponse(**person)


@router.post("/shows/{show_id}/people")
def create_person(show_id: str, body: PersonCreateRequest) -> PersonResponse:
    """Create a new person."""
    person = people_service.create_person(show_id, name=body.name)
    return PersonResponse(**person)


@router.patch("/shows/{show_id}/people/{person_id}")
def update_person(show_id: str, person_id: str, body: PersonUpdateRequest) -> PersonResponse:
    """Update a person."""
    person = people_service.update_person(
        show_id,
        person_id,
        name=body.name,
        rep_crop=body.rep_crop,
    )
    if not person:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
    return PersonResponse(**person)


@router.delete("/shows/{show_id}/people/{person_id}")
def delete_person(show_id: str, person_id: str) -> dict:
    """Delete a person."""
    success = people_service.delete_person(show_id, person_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
    return {"status": "deleted", "person_id": person_id}


__all__ = ["router"]

"""Face Review API endpoints for Improve Faces workflow.

This router exposes the FaceReviewService to the API, enabling:
1. Initial post-cluster suggestions for unassigned↔unassigned merges
2. Decision processing for merge/reject actions
3. Mixed queue for the Improve Faces feature
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from apps.api.services.face_review import face_review_service

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/episodes")


class DecisionRequest(BaseModel):
    """Request body for face review decisions."""

    pair_type: Literal["unassigned_unassigned", "unassigned_assigned"] = Field(
        ..., description="Type of comparison"
    )
    cluster_a_id: str = Field(..., description="First cluster ID")
    cluster_b_id: Optional[str] = Field(None, description="Second cluster ID (for merges)")
    person_id: Optional[str] = Field(None, description="Person ID (for assignments)")
    cast_id: Optional[str] = Field(None, description="Cast ID (for assignments)")
    decision: Literal["merge", "reject", "yes", "no"] = Field(
        ..., description="User decision"
    )
    execution_mode: Optional[Literal["redis", "local"]] = Field(
        "local", description="Execution mode (currently only local supported)"
    )


@router.get("/{ep_id}/face_review/initial_unassigned_suggestions")
def get_initial_unassigned_suggestions(
    ep_id: str,
    show_id: Optional[str] = Query(None, description="Show ID for people lookup"),
    limit: int = Query(20, description="Max suggestions to return"),
):
    """Get unassigned↔unassigned cluster pairs for initial post-cluster pass.

    Returns suggestions for clusters that are similar but weren't auto-merged,
    allowing users to manually review and merge if appropriate.
    """
    try:
        result = face_review_service.get_initial_unassigned_suggestions(
            ep_id=ep_id,
            show_id=show_id,
            limit=limit,
        )
        return result
    except Exception as e:
        LOGGER.error(f"[{ep_id}] Failed to get initial suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{ep_id}/face_review/decision/start")
def process_decision(ep_id: str, request: DecisionRequest):
    """Process a user decision (merge/reject) for a face review comparison.

    Accepts decisions from the Improve Faces modal:
    - "merge" or "yes": Merge the clusters (smaller into larger)
    - "reject" or "no": Record negative constraint to prevent re-suggestion

    Note: UI sends "merge"/"reject" but service expects "yes"/"no",
    so this endpoint translates between the two formats.
    """
    # Translate UI decision format to service format
    decision = request.decision
    if decision == "merge":
        decision = "yes"
    elif decision == "reject":
        decision = "no"

    try:
        result = face_review_service.process_decision(
            ep_id=ep_id,
            pair_type=request.pair_type,
            cluster_a_id=request.cluster_a_id,
            decision=decision,
            cluster_b_id=request.cluster_b_id,
            person_id=request.person_id,
            cast_id=request.cast_id,
            show_id=None,  # TODO: extract show_id from ep_id if needed
        )
        return result
    except Exception as e:
        LOGGER.error(f"[{ep_id}] Failed to process decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ep_id}/face_review/improve_faces_queue")
def get_improve_faces_queue(
    ep_id: str,
    show_id: Optional[str] = Query(None, description="Show ID"),
    limit: int = Query(30, description="Max suggestions"),
):
    """Get mixed queue of suggestions for Improve Faces feature.

    Returns both:
    1. Unassigned↔unassigned pairs (cluster merges)
    2. Unassigned↔assigned pairs (cast assignments)

    Sorted by similarity score for efficient review.
    """
    try:
        result = face_review_service.get_improve_faces_queue(
            ep_id=ep_id,
            show_id=show_id,
            limit=limit,
        )
        return result
    except Exception as e:
        LOGGER.error(f"[{ep_id}] Failed to get improve faces queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{ep_id}/face_review/mark_initial_pass_done")
def mark_initial_pass_done(ep_id: str):
    """Mark the initial unassigned pass as complete.

    This prevents the initial suggestions from being shown again
    on subsequent visits to the Faces Review page.
    """
    try:
        face_review_service.mark_initial_pass_done(ep_id)
        return {"status": "success", "ep_id": ep_id}
    except Exception as e:
        LOGGER.error(f"[{ep_id}] Failed to mark initial pass done: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{ep_id}/face_review/reset_initial_pass")
def reset_initial_pass(ep_id: str):
    """Reset the initial pass flag to allow re-running suggestions.

    Useful if the user wants to review initial suggestions again
    after previously completing them.
    """
    try:
        face_review_service.reset_initial_pass(ep_id)
        return {"status": "success", "ep_id": ep_id}
    except Exception as e:
        LOGGER.error(f"[{ep_id}] Failed to reset initial pass: {e}")
        raise HTTPException(status_code=500, detail=str(e))

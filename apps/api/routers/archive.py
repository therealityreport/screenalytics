"""Archive API endpoints for viewing and managing deleted/archived items."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from apps.api.services.archive import archive_service

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/archive", tags=["archive"])


class MatchRequest(BaseModel):
    """Request to find matching archived items."""

    centroid: list[float] = Field(..., description="Face centroid embedding")
    threshold: float = Field(0.70, ge=0.5, le=0.99, description="Minimum similarity threshold")
    item_type: Optional[str] = Field(None, description="Filter by type: person, cluster, track")


@router.get("/shows/{show_id}")
def list_archived_items(
    show_id: str,
    item_type: Optional[str] = Query(None, description="Filter: person, cluster, track"),
    episode_id: Optional[str] = Query(None, description="Filter by episode"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict:
    """List archived items for a show.

    Returns paginated list of archived people, clusters, and tracks.
    """
    return archive_service.list_archived(
        show_id,
        item_type=item_type,
        episode_id=episode_id,
        limit=limit,
        offset=offset,
    )


@router.get("/shows/{show_id}/stats")
def get_archive_stats(show_id: str) -> dict:
    """Get archive statistics for a show."""
    return archive_service.get_stats(show_id)


@router.post("/shows/{show_id}/match")
def find_matching_archived(show_id: str, request: MatchRequest) -> dict:
    """Find archived items matching a given face centroid.

    Used to check if a new face matches previously archived (rejected) faces.
    """
    matches = archive_service.find_matching_archived(
        show_id,
        request.centroid,
        threshold=request.threshold,
        item_type=request.item_type,
    )
    return {
        "matches": matches,
        "count": len(matches),
        "threshold": request.threshold,
    }


@router.post("/shows/{show_id}/restore/{archive_id}")
def restore_archived_item(show_id: str, archive_id: str) -> dict:
    """Restore an archived person.

    Returns the original person data. The caller should use this
    to recreate the person via the people API.
    """
    restored = archive_service.restore_person(show_id, archive_id)
    if not restored:
        raise HTTPException(status_code=404, detail=f"Archived item {archive_id} not found")

    return {
        "status": "restored",
        "archive_id": archive_id,
        "data": restored,
    }


@router.delete("/shows/{show_id}/{archive_id}")
def permanently_delete_archived(show_id: str, archive_id: str) -> dict:
    """Permanently delete an archived item.

    This cannot be undone - the item and its centroid will be gone forever.
    """
    deleted = archive_service.delete_archived(show_id, archive_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Archived item {archive_id} not found")

    return {
        "status": "deleted",
        "archive_id": archive_id,
    }


@router.get("/shows/{show_id}/centroids")
def get_archived_centroids(
    show_id: str,
    item_type: Optional[str] = Query(None, description="Filter: person, cluster, track"),
) -> dict:
    """Get all archived centroids for matching.

    Returns list of archived items with their centroid embeddings.
    Used for auto-archiving matching faces in future episodes.
    """
    centroids = archive_service.get_archived_centroids(show_id, item_type=item_type)
    return {
        "centroids": centroids,
        "count": len(centroids),
    }


class ArchiveClusterRequest(BaseModel):
    """Request to archive a cluster."""

    episode_id: str = Field(..., description="Episode ID")
    cluster_id: str = Field(..., description="Cluster ID")
    reason: str = Field("user_skipped", description="Reason for archiving")
    centroid: Optional[list[float]] = Field(None, description="Cluster centroid embedding")
    rep_crop_url: Optional[str] = Field(None, description="Representative crop URL")
    track_ids: Optional[list[int]] = Field(None, description="Track IDs in this cluster")
    face_count: int = Field(0, description="Number of faces in this cluster")


class ArchiveTrackRequest(BaseModel):
    """Request to archive a track."""

    episode_id: str = Field(..., description="Episode ID")
    track_id: int = Field(..., description="Track ID")
    reason: str = Field("user_archived", description="Reason for archiving")
    cluster_id: Optional[str] = Field(None, description="Parent cluster ID")
    centroid: Optional[list[float]] = Field(None, description="Track centroid embedding")
    rep_crop_url: Optional[str] = Field(None, description="Representative crop URL")
    frame_count: int = Field(0, description="Number of frames in this track")


@router.post("/shows/{show_id}/clusters")
def archive_cluster(show_id: str, request: ArchiveClusterRequest) -> dict:
    """Archive a cluster when user skips/dismisses it.

    Stores the cluster metadata for potential future matching or restoration.
    This allows auto-archiving similar faces in future episodes.
    """
    archived = archive_service.archive_cluster(
        show_id,
        request.episode_id,
        request.cluster_id,
        reason=request.reason,
        centroid=request.centroid,
        rep_crop_url=request.rep_crop_url,
        track_ids=request.track_ids,
        face_count=request.face_count,
    )
    return {
        "status": "archived",
        "archive_id": archived.get("archive_id"),
        "cluster_id": request.cluster_id,
    }


@router.post("/shows/{show_id}/tracks")
def archive_track(show_id: str, request: ArchiveTrackRequest) -> dict:
    """Archive a track before deletion.

    Stores the track metadata for potential future matching or restoration.
    """
    archived = archive_service.archive_track(
        show_id,
        request.episode_id,
        request.track_id,
        reason=request.reason,
        centroid=request.centroid,
        rep_crop_url=request.rep_crop_url,
        frame_count=request.frame_count,
        cluster_id=request.cluster_id,
    )
    return {
        "status": "archived",
        "archive_id": archived.get("archive_id"),
        "track_id": request.track_id,
    }


@router.delete("/shows/{show_id}/clear")
def clear_all_archived(
    show_id: str,
    episode_id: Optional[str] = Query(None, description="Filter by episode (only clear items from this episode)"),
) -> dict:
    """Clear archived items for a show (or specific episode).

    This permanently deletes archived people, clusters, and tracks.
    Use with caution - this cannot be undone.

    If episode_id is provided, only items from that episode are cleared.
    """
    result = archive_service.clear_all(show_id, episode_id=episode_id)
    return {
        "status": "cleared",
        "show_id": show_id,
        "episode_id": episode_id,
        "deleted_count": result.get("deleted_count", 0),
    }

"""Cluster grouping endpoints."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from apps.api.services.grouping import GroupingService

router = APIRouter()
grouping_service = GroupingService()


class GroupClustersRequest(BaseModel):
    strategy: Literal["auto", "manual"] = Field("auto", description="Grouping strategy")
    cluster_ids: Optional[List[str]] = Field(None, description="Cluster IDs for manual grouping")
    target_person_id: Optional[str] = Field(None, description="Target person ID for manual grouping")


@router.post("/episodes/{ep_id}/clusters/group")
def group_clusters(ep_id: str, body: GroupClustersRequest) -> dict:
    """Group clusters either automatically or manually.

    Auto mode: Compute centroids, run within-episode grouping, then across-episode matching.
    Manual mode: Assign specific clusters to a person (new or existing).
    """
    try:
        if body.strategy == "auto":
            result = grouping_service.group_clusters_auto(ep_id)
            return {
                "status": "success",
                "strategy": "auto",
                "ep_id": ep_id,
                "result": result,
            }
        elif body.strategy == "manual":
            if not body.cluster_ids:
                raise HTTPException(status_code=400, detail="cluster_ids required for manual grouping")

            result = grouping_service.manual_assign_clusters(
                ep_id,
                body.cluster_ids,
                body.target_person_id,
            )
            return {
                "status": "success",
                "strategy": "manual",
                "ep_id": ep_id,
                "result": result,
            }
        else:
            raise HTTPException(status_code=400, detail=f"Invalid strategy: {body.strategy}")

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grouping failed: {str(e)}")


@router.get("/episodes/{ep_id}/cluster_centroids")
def get_cluster_centroids(ep_id: str) -> dict:
    """Get cluster centroids for an episode."""
    try:
        centroids = grouping_service.load_cluster_centroids(ep_id)
        return centroids
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Cluster centroids not found for {ep_id}")


@router.post("/episodes/{ep_id}/cluster_centroids/compute")
def compute_cluster_centroids(ep_id: str) -> dict:
    """Compute cluster centroids for an episode."""
    try:
        result = grouping_service.compute_cluster_centroids(ep_id)
        return {"status": "success", "result": result}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Centroid computation failed: {str(e)}")


__all__ = ["router"]

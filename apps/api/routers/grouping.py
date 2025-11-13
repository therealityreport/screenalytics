"""Cluster grouping endpoints."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from apps.api.services.grouping import GroupingService
from apps.api.routers.episodes import _refresh_similarity_indexes

router = APIRouter()
grouping_service = GroupingService()


class GroupClustersRequest(BaseModel):
    strategy: Literal["auto", "manual", "facebank"] = Field("auto", description="Grouping strategy")
    cluster_ids: Optional[List[str]] = Field(None, description="Cluster IDs for manual grouping")
    target_person_id: Optional[str] = Field(None, description="Target person ID for manual grouping")


def _trigger_similarity_refresh(ep_id: str, cluster_ids: Iterable[str] | None) -> None:
    if not cluster_ids:
        return
    unique = [cluster_id for cluster_id in cluster_ids if cluster_id]
    if not unique:
        return
    _refresh_similarity_indexes(ep_id, identity_ids=unique)


@router.post("/episodes/{ep_id}/clusters/group")
def group_clusters(ep_id: str, body: GroupClustersRequest) -> dict:
    """Group clusters either automatically or manually.

    Auto mode: Compute centroids, run within-episode grouping, then across-episode matching.
    Manual mode: Assign specific clusters to a person (new or existing).
    """
    try:
        if body.strategy == "auto":
            result = grouping_service.group_clusters_auto(ep_id)
            affected_clusters = set()
            within = (result.get("within_episode") or {}).get("groups") if isinstance(result.get("within_episode"), dict) else None
            if isinstance(within, list):
                for group in within:
                    if isinstance(group, dict):
                        for cluster_id in group.get("cluster_ids") or []:
                            affected_clusters.add(cluster_id)
            across = result.get("across_episodes") or {}
            assignments = across.get("assigned") if isinstance(across, dict) else None
            if isinstance(assignments, list):
                for entry in assignments:
                    if isinstance(entry, dict):
                        cid = entry.get("cluster_id")
                        if cid:
                            affected_clusters.add(cid)
            _trigger_similarity_refresh(ep_id, affected_clusters)
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
            _trigger_similarity_refresh(ep_id, body.cluster_ids)
            return {
                "status": "success",
                "strategy": "manual",
                "ep_id": ep_id,
                "result": result,
            }
        elif body.strategy == "facebank":
            result = grouping_service.group_using_facebank(ep_id)
            assigned = []
            for entry in result.get("assigned", []):
                if isinstance(entry, dict) and entry.get("cluster_id"):
                    assigned.append(entry["cluster_id"])
            _trigger_similarity_refresh(ep_id, assigned)
            return {
                "status": "success",
                "strategy": "facebank",
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

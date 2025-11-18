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
    cast_id: Optional[str] = Field(None, description="Cast ID to link to the person (for new or existing)")
    name: Optional[str] = Field(None, description="Name for new person (when target_person_id is None)")


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
            # Track progress via callback
            progress_log = []
            
            def progress_callback(step: str, progress: float, message: str):
                progress_log.append({
                    "step": step,
                    "progress": progress,
                    "message": message,
                })
            
            result = grouping_service.group_clusters_auto(ep_id, progress_callback=progress_callback)
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
                "progress_log": progress_log,
            }
        elif body.strategy == "manual":
            if not body.cluster_ids:
                raise HTTPException(status_code=400, detail="cluster_ids required for manual grouping")

            result = grouping_service.manual_assign_clusters(
                ep_id,
                body.cluster_ids,
                body.target_person_id,
                cast_id=body.cast_id,
                name=body.name,
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


@router.get("/episodes/{ep_id}/cluster_suggestions")
def get_cluster_suggestions(ep_id: str) -> dict:
    """Get suggested cast member matches for episode clusters.
    
    Returns suggestions based on similarity to existing people without actually assigning.
    """
    try:
        # First check if centroids exist
        try:
            centroids_data = grouping_service.load_cluster_centroids(ep_id)
            if not centroids_data or not centroids_data.get("centroids"):
                return {
                    "status": "success",
                    "ep_id": ep_id,
                    "suggestions": [],
                    "message": "No centroids computed yet. Run clustering first.",
                }
        except FileNotFoundError:
            return {
                "status": "success",
                "ep_id": ep_id,
                "suggestions": [],
                "message": "No centroids found. Run clustering first.",
            }
        
        # Run group_across_episodes with auto_assign=False to get suggestions only
        result = grouping_service.group_across_episodes(ep_id, auto_assign=False)
        return {
            "status": "success",
            "ep_id": ep_id,
            "suggestions": result.get("suggestions", []),
        }
    except FileNotFoundError as e:
        return {
            "status": "success",
            "ep_id": ep_id,
            "suggestions": [],
            "message": str(e),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute suggestions: {str(e)}")


@router.get("/episodes/{ep_id}/cluster_suggestions_from_assigned")
def get_cluster_suggestions_from_assigned(ep_id: str) -> dict:
    """Get suggested matches for unassigned clusters by comparing with assigned clusters.
    
    Compares unassigned cluster centroids against assigned cluster centroids in the same episode.
    Returns suggestions based on which assigned person has the most similar cluster.
    """
    try:
        result = grouping_service.suggest_from_assigned_clusters(ep_id)
        return {
            "status": "success",
            "ep_id": ep_id,
            "suggestions": result.get("suggestions", []),
        }
    except FileNotFoundError as e:
        return {
            "status": "success",
            "ep_id": ep_id,
            "suggestions": [],
            "message": str(e),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute suggestions from assigned: {str(e)}")


@router.post("/episodes/{ep_id}/save_assignments")
def save_assignments(ep_id: str) -> dict:
    """Save all current cluster assignments to people.json and identities.json.
    
    This ensures all assignments made in the UI are persisted.
    """
    try:
        result = grouping_service.save_current_assignments(ep_id)
        return {
            "status": "success",
            "ep_id": ep_id,
            "saved_count": result.get("saved_count", 0),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save assignments: {str(e)}")


__all__ = ["router"]


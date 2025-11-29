"""Cluster grouping endpoints."""

from __future__ import annotations

import json
import logging
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

LOGGER = logging.getLogger(__name__)
router = APIRouter()
grouping_service = GroupingService()

# Lazy imports for Celery (only when needed)
_celery_available = None


def _check_celery_available() -> bool:
    """Check if Celery/Redis are available for async jobs."""
    global _celery_available
    if _celery_available is None:
        try:
            from apps.api.tasks import check_active_job
            import redis
            from apps.api.config import REDIS_URL
            # Try to ping Redis
            r = redis.from_url(REDIS_URL, socket_timeout=1)
            r.ping()
            _celery_available = True
        except Exception as e:
            LOGGER.warning(f"Celery/Redis not available: {e}")
            _celery_available = False
    return _celery_available


class GroupClustersRequest(BaseModel):
    strategy: Literal["auto", "manual", "facebank"] = Field("auto", description="Grouping strategy")
    cluster_ids: Optional[List[str]] = Field(None, description="Cluster IDs for manual grouping")
    target_person_id: Optional[str] = Field(None, description="Target person ID for manual grouping")
    cast_id: Optional[str] = Field(None, description="Cast ID to link to the person (for new or existing)")
    name: Optional[str] = Field(None, description="Name for new person (when target_person_id is None)")
    protect_manual: bool = Field(True, description="If True, don't merge manually assigned clusters to different people")
    facebank_first: bool = Field(True, description="If True, try facebank matching before people prototypes (more accurate)")
    execution_mode: Optional[Literal["redis", "local"]] = Field(
        "redis",
        description="Execution mode: 'redis' enqueues job via Celery, 'local' runs synchronously in-process"
    )


class BatchAssignmentItem(BaseModel):
    """Single assignment in a batch."""
    cluster_id: str = Field(..., description="Cluster ID to assign")
    target_cast_id: str = Field(..., description="Cast ID to assign the cluster to")


class BatchAssignRequest(BaseModel):
    """Batch assignment request for multiple clusters to multiple cast members."""
    assignments: List[BatchAssignmentItem] = Field(..., description="List of cluster-to-cast assignments")
    execution_mode: Optional[Literal["redis", "local"]] = Field(
        "redis",
        description="Execution mode: 'redis' enqueues job via Celery, 'local' runs synchronously in-process"
    )


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
                progress_log.append(
                    {
                        "step": step,
                        "progress": progress,
                        "message": message,
                    }
                )

            try:
                result = grouping_service.group_clusters_auto(
                    ep_id,
                    progress_callback=progress_callback,
                    protect_manual=body.protect_manual,
                    facebank_first=body.facebank_first,
                )
            except Exception as inner_exc:
                import traceback
                import logging
                logging.getLogger(__name__).error(
                    f"[{ep_id}] group_clusters_auto failed: {type(inner_exc).__name__}: {inner_exc}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
                raise
            affected_clusters = set()
            within = (
                (result.get("within_episode") or {}).get("groups")
                if isinstance(result.get("within_episode"), dict)
                else None
            )
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
            # NOTE: Skip _trigger_similarity_refresh for manual assignments.
            # The full regeneration of all track reps is expensive (60s+ timeout).
            # Manual assignments update identities and people directly, so similarity
            # indexes can be refreshed on next auto-cluster or explicit refresh.
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


@router.post("/episodes/{ep_id}/clusters/batch_assign")
def batch_assign_clusters(ep_id: str, body: BatchAssignRequest) -> dict:
    """Batch assign multiple clusters to cast members in a single operation.

    This endpoint is optimized for bulk assignments - it loads data once and
    processes all assignments together, significantly reducing latency compared
    to multiple individual calls.

    Request body:
        assignments: List of {cluster_id, target_cast_id} pairs

    Returns:
        - status: "success" or "partial" (if some failed)
        - results: List of assignment results
        - succeeded: Count of successful assignments
        - failed: Count of failed assignments
    """
    if not body.assignments:
        raise HTTPException(status_code=400, detail="No assignments provided")

    try:
        result = grouping_service.batch_assign_clusters(ep_id, body.assignments)
        status = "success" if result.get("failed", 0) == 0 else "partial"
        return {
            "status": status,
            "ep_id": ep_id,
            **result,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch assignment failed: {str(e)}")


# ============================================================================
# ASYNC ENDPOINTS - Return HTTP 202 with job_id for background processing
# ============================================================================


@router.post("/episodes/{ep_id}/clusters/batch_assign_async", status_code=202)
def batch_assign_clusters_async(ep_id: str, body: BatchAssignRequest) -> dict:
    """Enqueue batch cluster assignment as background job (non-blocking).

    Execution Mode:
        - execution_mode="redis" (default): Enqueues job via Celery, returns 202 with job_id
        - execution_mode="local": Runs job synchronously in-process, returns result when done

    If Celery/Redis are unavailable, falls back to synchronous execution.
    """
    if not body.assignments:
        raise HTTPException(status_code=400, detail="No assignments provided")

    execution_mode = body.execution_mode or "redis"

    # Handle local execution mode or Celery unavailability
    if execution_mode == "local" or not _check_celery_available():
        reason = "local mode requested" if execution_mode == "local" else "Celery unavailable"
        LOGGER.info(f"[{ep_id}] Running sync batch_assign ({reason})")
        try:
            result = grouping_service.batch_assign_clusters(ep_id, body.assignments)
            status = "success" if result.get("failed", 0) == 0 else "partial"
            return {
                "status": status,
                "ep_id": ep_id,
                "async": False,
                "execution_mode": "local",
                "message": f"Executed synchronously ({reason})",
                **result,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    try:
        from apps.api.tasks import run_manual_assign_task, check_active_job

        # Check for active job
        active = check_active_job(ep_id, "manual_assign")
        if active:
            raise HTTPException(
                status_code=409,
                detail=f"Job already in progress: {active}",
            )

        # Convert assignments to dict format for Celery
        payload = {
            "assignments": [
                {"cluster_id": a.cluster_id, "target_cast_id": a.target_cast_id}
                for a in body.assignments
            ]
        }

        # Enqueue the task
        task = run_manual_assign_task.delay(
            episode_id=ep_id,
            payload=payload,
        )

        return {
            "job_id": task.id,
            "status": "queued",
            "ep_id": ep_id,
            "async": True,
            "execution_mode": "redis",
        }
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception(f"[{ep_id}] Failed to enqueue batch_assign: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to enqueue job: {str(e)}")


@router.post("/episodes/{ep_id}/clusters/group_async", status_code=202)
def group_clusters_async(ep_id: str, body: GroupClustersRequest) -> dict:
    """Enqueue auto-grouping as background job (non-blocking).

    Execution Mode:
        - execution_mode="redis" (default): Enqueues job via Celery, returns 202 with job_id
        - execution_mode="local": Runs job synchronously in-process, returns result when done

    Only supports strategy="auto". For manual/facebank, use the synchronous endpoint.
    If Celery/Redis are unavailable, falls back to synchronous execution.
    """
    if body.strategy != "auto":
        raise HTTPException(
            status_code=400,
            detail=f"Async only supports strategy='auto', got '{body.strategy}'. "
            "Use /clusters/group for manual/facebank.",
        )

    execution_mode = body.execution_mode or "redis"

    # Handle local execution mode or Celery unavailability
    if execution_mode == "local" or not _check_celery_available():
        reason = "local mode requested" if execution_mode == "local" else "Celery unavailable"
        LOGGER.info(f"[{ep_id}] Running sync group_clusters ({reason})")
        try:
            progress_log = []

            def progress_callback(step: str, progress: float, message: str):
                progress_log.append({"step": step, "progress": progress, "message": message})

            result = grouping_service.group_clusters_auto(
                ep_id,
                progress_callback=progress_callback,
                protect_manual=body.protect_manual,
                facebank_first=body.facebank_first,
            )
            return {
                "status": "success",
                "strategy": "auto",
                "ep_id": ep_id,
                "async": False,
                "execution_mode": "local",
                "message": f"Executed synchronously ({reason})",
                "result": result,
                "progress_log": progress_log,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    try:
        from apps.api.tasks import run_auto_group_task, check_active_job

        # Check for active job
        active = check_active_job(ep_id, "auto_group")
        if active:
            raise HTTPException(
                status_code=409,
                detail=f"Job already in progress: {active}",
            )

        # Enqueue the task
        task = run_auto_group_task.delay(
            episode_id=ep_id,
            options={
                "protect_manual": body.protect_manual,
                "facebank_first": body.facebank_first,
            },
        )

        return {
            "job_id": task.id,
            "status": "queued",
            "ep_id": ep_id,
            "async": True,
            "execution_mode": "redis",
        }
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception(f"[{ep_id}] Failed to enqueue group_clusters: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to enqueue job: {str(e)}")


@router.get("/episodes/{ep_id}/clusters/group/progress")
def group_clusters_progress(ep_id: str) -> dict:
    """Return in-flight grouping progress for polling clients."""
    path = grouping_service._group_progress_path(ep_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="No grouping progress found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Invalid progress payload")
        return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read progress: {exc}")


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
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute suggestions from assigned: {str(e)}",
        )


@router.get("/episodes/{ep_id}/clusters/{cluster_id}/suggest_cast")
def suggest_cast_for_cluster(ep_id: str, cluster_id: str, min_similarity: float = 0.40, top_k: int = 5) -> dict:
    """Get cast member suggestions for a specific cluster (Enhancement #6).

    Compares the cluster's centroid against all cast member facebank seeds.
    Returns top-k cast member suggestions with confidence levels.

    This endpoint is designed for on-demand "Suggest for Me" button clicks.
    """
    try:
        from apps.api.services.facebank import cosine_similarity
        import numpy as np

        # Parse episode ID
        import re
        pattern = r"^(?P<show>.+)-s(?P<season>\d{2})e(?P<episode>\d{2})$"
        match = re.match(pattern, ep_id, re.IGNORECASE)
        if not match:
            raise HTTPException(status_code=400, detail=f"Invalid episode ID: {ep_id}")
        show_id = match.group("show").upper()

        # Load cluster centroid
        try:
            centroids_data = grouping_service.load_cluster_centroids(ep_id)
            centroids = centroids_data.get("centroids", {})
            cluster_data = centroids.get(cluster_id)
            if not cluster_data or not cluster_data.get("centroid"):
                return {
                    "status": "success",
                    "cluster_id": cluster_id,
                    "suggestions": [],
                    "message": f"No centroid found for cluster {cluster_id}",
                }
            centroid_vec = np.array(cluster_data["centroid"], dtype=np.float32)
        except FileNotFoundError:
            return {
                "status": "success",
                "cluster_id": cluster_id,
                "suggestions": [],
                "message": "No centroids file found. Run clustering first.",
            }

        # Get all seeds for the show
        from apps.api.services.facebank import FacebankService
        from apps.api.services.cast import CastService
        facebank_service = FacebankService()
        cast_service = CastService()

        seeds = facebank_service.get_all_seeds_for_show(show_id)
        if not seeds:
            return {
                "status": "success",
                "cluster_id": cluster_id,
                "suggestions": [],
                "message": f"No facebank seeds available for show {show_id}",
            }

        # Get cast member names
        cast_members = cast_service.list_cast(show_id)
        cast_lookup = {member["cast_id"]: member for member in cast_members if member.get("cast_id")}

        # Group seeds by cast_id
        seeds_by_cast = {}
        for seed in seeds:
            cast_id = seed.get("cast_id")
            if cast_id and seed.get("embedding"):
                seeds_by_cast.setdefault(cast_id, []).append(seed)

        # Find best similarity per cast member
        cast_matches = []
        for cast_id, cast_seeds in seeds_by_cast.items():
            best_sim = -1.0
            best_seed_id = None

            for seed in cast_seeds:
                seed_emb = np.array(seed["embedding"], dtype=np.float32)
                sim = cosine_similarity(centroid_vec, seed_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_seed_id = seed.get("fb_id")

            if best_sim >= min_similarity:
                cast_meta = cast_lookup.get(cast_id, {})
                cast_name = cast_meta.get("name", cast_id)

                # Determine confidence level
                if best_sim >= 0.80:
                    confidence = "high"
                elif best_sim >= 0.65:
                    confidence = "medium"
                else:
                    confidence = "low"

                cast_matches.append({
                    "cast_id": cast_id,
                    "name": cast_name,
                    "similarity": round(float(best_sim), 3),
                    "confidence": confidence,
                    "best_seed_id": best_seed_id,
                })

        # Sort by similarity (descending) and take top_k
        cast_matches.sort(key=lambda x: x["similarity"], reverse=True)
        top_matches = cast_matches[:top_k]

        return {
            "status": "success",
            "cluster_id": cluster_id,
            "suggestions": top_matches,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute suggestions for cluster: {str(e)}",
        )


@router.get("/episodes/{ep_id}/cast_suggestions")
def get_cast_suggestions(ep_id: str, min_similarity: float = 0.50, top_k: int = 3) -> dict:
    """Get cast member suggestions for unassigned clusters based on facebank similarity.

    Compares each unassigned cluster's centroid against all cast member facebank seeds.
    Returns top-k cast member suggestions per cluster with confidence levels.

    Query params:
        min_similarity: Minimum similarity threshold (default 0.50)
        top_k: Number of suggestions per cluster (default 3)
    """
    try:
        result = grouping_service.suggest_cast_for_unassigned_clusters(
            ep_id,
            min_similarity=min_similarity,
            top_k=top_k,
        )
        return {
            "status": "success",
            "ep_id": ep_id,
            "suggestions": result.get("suggestions", []),
            "message": result.get("message"),
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
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute cast suggestions: {str(e)}",
        )


@router.post("/episodes/{ep_id}/auto_link_cast")
def auto_link_cast(ep_id: str, min_confidence: float = 0.85) -> dict:
    """Auto-assign unassigned clusters to cast members with high confidence (Enhancement #8).

    Only assigns when facebank similarity is >= min_confidence.
    Called during Refresh Values to auto-link obvious matches.
    """
    try:
        result = grouping_service.auto_link_high_confidence_matches(
            ep_id,
            min_confidence=min_confidence,
        )
        return {
            "status": "success",
            "ep_id": ep_id,
            "auto_assigned": result.get("auto_assigned", 0),
            "assignments": result.get("assignments", []),
        }
    except FileNotFoundError as e:
        return {
            "status": "success",
            "ep_id": ep_id,
            "auto_assigned": 0,
            "assignments": [],
            "message": str(e),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Auto-link failed: {str(e)}",
        )


@router.get("/episodes/{ep_id}/cleanup_preview")
def cleanup_preview(ep_id: str) -> dict:
    """Get a preview of what would change if cleanup were run (Enhancement #3).

    Analyzes current state and estimates impact without making changes.
    Returns counts of affected clusters, manual assignments that could be impacted, etc.
    """
    try:
        from apps.api.services.identities import load_identities

        # Load current state
        identities_data = load_identities(ep_id)
        identities = identities_data.get("identities", [])

        # Count current clusters
        total_clusters = len(identities)

        # Load manual assignments
        manual_assignments = grouping_service._load_manual_assignments(ep_id)
        manual_cluster_ids = [
            cid for cid, data in manual_assignments.items()
            if data.get("assigned_by") == "user"
        ]

        # Load centroids if available
        centroids_count = 0
        try:
            centroids_data = grouping_service.load_cluster_centroids(ep_id)
            centroids_count = len(centroids_data.get("centroids", {}))
        except FileNotFoundError:
            pass

        # Estimate potential merges (clusters that could be merged based on similarity)
        potential_merges = 0
        if centroids_count > 1:
            try:
                within_result = grouping_service.group_within_episode(
                    ep_id,
                    protect_manual=False,  # Check without protection to see full impact
                )
                potential_merges = within_result.get("merged_count", 0)
            except Exception:
                pass

        # Count unassigned vs assigned clusters
        assigned_clusters = sum(1 for i in identities if i.get("person_id"))
        unassigned_clusters = total_clusters - assigned_clusters

        preview = {
            "total_clusters": total_clusters,
            "assigned_clusters": assigned_clusters,
            "unassigned_clusters": unassigned_clusters,
            "manual_assignments_count": len(manual_cluster_ids),
            "manual_cluster_ids": manual_cluster_ids[:10],  # First 10 for preview
            "potential_merges": potential_merges,
            "warning_level": "low",
            "warnings": [],
        }

        # Add warnings based on potential impact
        if potential_merges > 0 and len(manual_cluster_ids) > 0:
            preview["warning_level"] = "high"
            preview["warnings"].append(
                f"⚠️ {potential_merges} cluster group(s) could be merged, "
                f"potentially affecting {len(manual_cluster_ids)} manual assignment(s)."
            )
        elif potential_merges > 0:
            preview["warning_level"] = "medium"
            preview["warnings"].append(
                f"⚡ {potential_merges} cluster group(s) could be merged."
            )

        if unassigned_clusters > 0:
            preview["warnings"].append(
                f"ℹ️ {unassigned_clusters} unassigned cluster(s) may be grouped."
            )

        return {
            "status": "success",
            "ep_id": ep_id,
            "preview": preview,
        }
    except FileNotFoundError as e:
        return {
            "status": "success",
            "ep_id": ep_id,
            "preview": {
                "total_clusters": 0,
                "warning_level": "low",
                "warnings": [f"No data found: {e}"],
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")


@router.post("/episodes/{ep_id}/backup")
def create_backup(ep_id: str) -> dict:
    """Create backup before cleanup (Enhancement #7).

    Backs up identities.json, people.json, cluster_centroids.json.
    """
    try:
        result = grouping_service.backup_before_cleanup(ep_id)
        return {
            "status": "success",
            "ep_id": ep_id,
            "backup_id": result.get("backup_id"),
            "files_backed_up": len(result.get("files", [])),
            "timestamp": result.get("timestamp"),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backup failed: {str(e)}")


@router.post("/episodes/{ep_id}/restore/{backup_id}")
def restore_backup(ep_id: str, backup_id: str) -> dict:
    """Restore from a backup (Enhancement #7)."""
    try:
        result = grouping_service.restore_from_backup(ep_id, backup_id)
        return {
            "status": "success",
            "ep_id": ep_id,
            "backup_id": backup_id,
            "files_restored": result.get("restored", 0),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Restore failed: {str(e)}")


@router.get("/episodes/{ep_id}/backups")
def list_backups(ep_id: str) -> dict:
    """List available backups for an episode (Enhancement #7)."""
    try:
        backups = grouping_service.list_backups(ep_id)
        return {
            "status": "success",
            "ep_id": ep_id,
            "backups": backups,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list backups: {str(e)}")


@router.get("/episodes/{ep_id}/consistency_check")
def cross_episode_consistency(ep_id: str) -> dict:
    """Check for cross-episode inconsistencies (Enhancement #9).

    Finds clusters that might be the same person but are assigned differently.
    """
    try:
        import re
        pattern = r"^(?P<show>.+)-s(?P<season>\d{2})e(?P<episode>\d{2})$"
        match = re.match(pattern, ep_id, re.IGNORECASE)
        if not match:
            raise HTTPException(status_code=400, detail=f"Invalid episode ID: {ep_id}")
        show_id = match.group("show").upper()

        people = grouping_service.people_service.list_people(show_id)
        if not people:
            return {"status": "success", "inconsistencies": []}

        # Find cast members with multiple people records
        people_by_cast: dict = {}
        for person in people:
            cast_id = person.get("cast_id")
            if cast_id:
                people_by_cast.setdefault(cast_id, []).append(person)

        inconsistencies = []
        for cast_id, cast_people in people_by_cast.items():
            if len(cast_people) > 1:
                inconsistencies.append({
                    "cast_id": cast_id,
                    "people_count": len(cast_people),
                    "person_ids": [p.get("person_id") for p in cast_people],
                    "suggestion": "Consider merging - same cast member, multiple person records",
                })

        return {
            "status": "success",
            "inconsistencies": inconsistencies,
            "total_people": len(people),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consistency check failed: {str(e)}")


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

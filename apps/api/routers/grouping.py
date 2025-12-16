"""Cluster grouping endpoints."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from apps.api.services.grouping import GroupingService, _parse_ep_id
from apps.api.services.cast import CastService
from apps.api.services.people import PeopleService
from apps.api.services.dismissed_suggestions import dismissed_suggestions_service
from apps.api.routers.episodes import _refresh_similarity_indexes

LOGGER = logging.getLogger(__name__)
router = APIRouter()
grouping_service = GroupingService()
cast_service = CastService()
people_service = PeopleService()


def _scoped_grouping_service(run_id: str | None) -> GroupingService:
    """Create a run-scoped grouping service when run_id is provided."""
    candidate = str(run_id).strip() if run_id is not None else ""
    if not candidate:
        return grouping_service

    scoped = GroupingService(data_root=getattr(grouping_service, "data_root", None), run_id=candidate)

    # Preserve injected dependencies (tests and deployments may patch these on the router-level singleton).
    for attr in ("people_service", "facebank_service", "cast_service"):
        if hasattr(grouping_service, attr):
            setattr(scoped, attr, getattr(grouping_service, attr))
    for attr in ("load_cluster_centroids", "_update_identities_with_people"):
        if hasattr(grouping_service, attr):
            setattr(scoped, attr, getattr(grouping_service, attr))

    return scoped

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
    skip_cast_assignment: bool = Field(False, description="If False (default), auto-assign clusters to cast members based on facebank matches. Set to True to only group similar clusters without assigning.")
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


def _trigger_similarity_refresh(ep_id: str, cluster_ids: Iterable[str] | None, *, run_id: str | None = None) -> None:
    if not cluster_ids:
        return
    unique = [cluster_id for cluster_id in cluster_ids if cluster_id]
    if not unique:
        return
    _refresh_similarity_indexes(ep_id, identity_ids=unique, run_id=run_id)


def _queue_async_similarity_refresh(ep_id: str) -> Optional[str]:
    """Queue async similarity refresh (fire-and-forget, doesn't block response).

    Returns job_id if queued successfully, None if Celery unavailable.
    """
    if not _check_celery_available():
        LOGGER.debug(f"[{ep_id}] Skipping async similarity refresh (Celery unavailable)")
        return None
    try:
        from apps.api.tasks import run_refresh_similarity_task
        task = run_refresh_similarity_task.delay(episode_id=ep_id)
        LOGGER.info(f"[{ep_id}] Queued async similarity refresh: {task.id}")
        return task.id
    except Exception as e:
        LOGGER.warning(f"[{ep_id}] Failed to queue async similarity refresh: {e}")
        return None


@router.post("/episodes/{ep_id}/clusters/group")
def group_clusters(
    ep_id: str,
    body: GroupClustersRequest,
    run_id: str = Query(..., description="Run id scope (required for mutations)"),
) -> dict:
    """Group clusters either automatically or manually.

    Auto mode: Compute centroids, run within-episode grouping, then across-episode matching.
    Manual mode: Assign specific clusters to a person (new or existing).
    """
    try:
        scoped_service = _scoped_grouping_service(run_id)
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
                result = scoped_service.group_clusters_auto(
                    ep_id,
                    progress_callback=progress_callback,
                    protect_manual=body.protect_manual,
                    facebank_first=body.facebank_first,
                    skip_cast_assignment=body.skip_cast_assignment,
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
            _trigger_similarity_refresh(ep_id, affected_clusters, run_id=run_id)
            return {
                "status": "success",
                "strategy": "auto",
                "ep_id": ep_id,
                "run_id": run_id,
                "result": result,
                "progress_log": progress_log,
            }
        elif body.strategy == "manual":
            if not body.cluster_ids:
                raise HTTPException(status_code=400, detail="cluster_ids required for manual grouping")

            result = scoped_service.manual_assign_clusters(
                ep_id,
                body.cluster_ids,
                body.target_person_id,
                cast_id=body.cast_id,
                name=body.name,
            )
            _trigger_similarity_refresh(ep_id, body.cluster_ids, run_id=run_id)
            return {
                "status": "success",
                "strategy": "manual",
                "ep_id": ep_id,
                "run_id": run_id,
                "result": result,
            }
        elif body.strategy == "facebank":
            result = scoped_service.group_using_facebank(ep_id)
            assigned = []
            for entry in result.get("assigned", []):
                if isinstance(entry, dict) and entry.get("cluster_id"):
                    assigned.append(entry["cluster_id"])
            _trigger_similarity_refresh(ep_id, assigned, run_id=run_id)
            return {
                "status": "success",
                "strategy": "facebank",
                "ep_id": ep_id,
                "run_id": run_id,
                "result": result,
            }
        else:
            raise HTTPException(status_code=400, detail=f"Invalid strategy: {body.strategy}")

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
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

    # Validate cast IDs exist before enqueuing job
    parsed = _parse_ep_id(ep_id)
    if not parsed:
        raise HTTPException(status_code=400, detail=f"Invalid episode ID format: {ep_id}")
    show_id = parsed["show"]

    cast_ids_to_validate = {a.target_cast_id for a in body.assignments if a.target_cast_id}
    if cast_ids_to_validate:
        invalid_cast_ids = []
        for cast_id in cast_ids_to_validate:
            if not cast_service.get_cast_member(show_id, cast_id):
                invalid_cast_ids.append(cast_id)
        if invalid_cast_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid cast member ID(s) for show {show_id}: {', '.join(invalid_cast_ids)}"
            )

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
                skip_cast_assignment=body.skip_cast_assignment,
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
                "skip_cast_assignment": body.skip_cast_assignment,
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
def group_clusters_progress(
    ep_id: str,
    run_id: Optional[str] = Query(None, description="Optional run_id scope for progress"),
) -> dict:
    """Return in-flight grouping progress for polling clients."""
    try:
        scoped_service = _scoped_grouping_service(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    path = scoped_service._group_progress_path(ep_id)
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
def get_cluster_centroids(
    ep_id: str,
    run_id: Optional[str] = Query(None, description="Optional run_id scope for artifacts/state"),
) -> dict:
    """Get cluster centroids for an episode."""
    try:
        scoped_service = _scoped_grouping_service(run_id)
        centroids = scoped_service.load_cluster_centroids(ep_id)
        return centroids
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Cluster centroids not found for {ep_id}")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/episodes/{ep_id}/cluster_centroids/compute")
def compute_cluster_centroids(
    ep_id: str,
    run_id: str = Query(..., description="Run id scope (required for mutations)"),
) -> dict:
    """Compute cluster centroids for an episode."""
    try:
        scoped_service = _scoped_grouping_service(run_id)
        result = scoped_service.compute_cluster_centroids(ep_id)
        return {"status": "success", "result": result}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Centroid computation failed: {str(e)}")


@router.get("/episodes/{ep_id}/cluster_suggestions")
def get_cluster_suggestions(
    ep_id: str,
    run_id: Optional[str] = Query(None, description="Optional run_id scope for artifacts/state"),
) -> dict:
    """Get suggested cast member matches for episode clusters.

    Returns suggestions based on similarity to existing people without actually assigning.
    """
    try:
        scoped_service = _scoped_grouping_service(run_id)
        # First check if centroids exist
        try:
            centroids_data = scoped_service.load_cluster_centroids(ep_id)
            if not centroids_data or not centroids_data.get("centroids"):
                return {
                    "status": "success",
                    "ep_id": ep_id,
                    "run_id": run_id,
                    "suggestions": [],
                    "message": "No centroids computed yet. Run clustering first.",
                }
        except FileNotFoundError:
            return {
                "status": "success",
                "ep_id": ep_id,
                "run_id": run_id,
                "suggestions": [],
                "message": "No centroids found. Run clustering first.",
            }

        # Run group_across_episodes with auto_assign=False to get suggestions only
        result = scoped_service.group_across_episodes(ep_id, auto_assign=False)
        return {
            "status": "success",
            "ep_id": ep_id,
            "run_id": run_id,
            "suggestions": result.get("suggestions", []),
        }
    except FileNotFoundError as e:
        return {
            "status": "success",
            "ep_id": ep_id,
            "run_id": run_id,
            "suggestions": [],
            "message": str(e),
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute suggestions: {str(e)}")


@router.get("/episodes/{ep_id}/cluster_suggestions_from_assigned")
def get_cluster_suggestions_from_assigned(
    ep_id: str,
    run_id: Optional[str] = Query(None, description="Optional run_id scope for artifacts/state"),
) -> dict:
    """Get suggested matches for unassigned clusters by comparing with assigned clusters.

    Compares unassigned cluster centroids against assigned cluster centroids in the same episode.
    Returns suggestions based on which assigned person has the most similar cluster.
    """
    try:
        scoped_service = _scoped_grouping_service(run_id)
        result = scoped_service.suggest_from_assigned_clusters(ep_id)
        return {
            "status": "success",
            "ep_id": ep_id,
            "run_id": run_id,
            "suggestions": result.get("suggestions", []),
        }
    except FileNotFoundError as e:
        return {
            "status": "success",
            "ep_id": ep_id,
            "run_id": run_id,
            "suggestions": [],
            "message": str(e),
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute suggestions from assigned: {str(e)}",
        )


@router.get("/episodes/{ep_id}/clusters/{cluster_id}/suggest_cast")
def suggest_cast_for_cluster(
    ep_id: str,
    cluster_id: str,
    min_similarity: float = 0.40,
    top_k: int = 5,
    run_id: Optional[str] = Query(None, description="Optional run_id scope for artifacts/state"),
) -> dict:
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
            scoped_service = _scoped_grouping_service(run_id)
            centroids_data = scoped_service.load_cluster_centroids(ep_id)
            centroids = centroids_data.get("centroids", {})
            cluster_data = centroids.get(cluster_id)
            if not cluster_data or not cluster_data.get("centroid"):
                return {
                    "status": "success",
                    "cluster_id": cluster_id,
                    "run_id": run_id,
                    "suggestions": [],
                    "message": f"No centroid found for cluster {cluster_id}",
                }
            centroid_vec = np.array(cluster_data["centroid"], dtype=np.float32)
        except FileNotFoundError:
            return {
                "status": "success",
                "cluster_id": cluster_id,
                "run_id": run_id,
                "suggestions": [],
                "message": "No centroids file found. Run clustering first.",
            }
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

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
def get_cast_suggestions(
    ep_id: str,
    min_similarity: float = 0.50,
    top_k: int = 3,
    run_id: Optional[str] = Query(None, description="Optional run_id scope for artifacts/state"),
) -> dict:
    """Get cast member suggestions for unassigned clusters based on facebank similarity.

    Compares each unassigned cluster's centroid against all cast member facebank seeds.
    Returns top-k cast member suggestions per cluster with confidence levels.

    Query params:
        min_similarity: Minimum similarity threshold (default 0.50)
        top_k: Number of suggestions per cluster (default 3)

    Returns:
        suggestions: List of cluster suggestions
        mismatched_embeddings: List of clusters skipped due to embedding dimension mismatch
        message: Optional message (e.g., "No centroids found")
    """
    try:
        scoped_service = _scoped_grouping_service(run_id)
        result = scoped_service.suggest_cast_for_unassigned_clusters(
            ep_id,
            min_similarity=min_similarity,
            top_k=top_k,
        )
        response = {
            "status": "success",
            "ep_id": ep_id,
            "run_id": run_id,
            "suggestions": result.get("suggestions", []),
        }
        # Include dimension mismatch warnings if any clusters were skipped
        mismatched = result.get("mismatched_embeddings", [])
        if mismatched:
            response["mismatched_embeddings"] = mismatched
        # Include quality-only clusters (no embeddings, all faces skipped by quality gate)
        quality_only = result.get("quality_only_clusters", [])
        if quality_only:
            response["quality_only_clusters"] = quality_only
        # Include any message (e.g., "No assigned clusters available")
        if result.get("message"):
            response["message"] = result["message"]
        return response
    except FileNotFoundError as e:
        return {
            "status": "success",
            "ep_id": ep_id,
            "run_id": run_id,
            "suggestions": [],
            "message": str(e),
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute cast suggestions: {str(e)}",
        )


# =============================================================================
# Smart Suggestions Persistence (DB-backed, run-scoped)
# =============================================================================


class SmartSuggestionsGenerateRequest(BaseModel):
    min_similarity: float = Field(0.01, ge=0.0, le=1.0, description="Minimum similarity threshold")
    top_k: int = Field(3, ge=1, le=10, description="Top-k cast candidates per cluster")
    generator_version: str = Field("cast_suggestions_v1", description="Generator version identifier")


class SmartSuggestionsDismissRequest(BaseModel):
    batch_id: str = Field(..., description="Suggestion batch id (required)")
    suggestion_ids: List[str] = Field(..., description="Suggestion ids to dismiss/restore")
    dismissed: bool = Field(True, description="True=dismiss, False=restore")


class SmartSuggestionsApplyRequest(BaseModel):
    batch_id: str = Field(..., description="Suggestion batch id (required)")
    suggestion_id: str = Field(..., description="Suggestion id to apply")
    applied_by: str | None = Field(None, description="Optional actor identifier (email/username)")
    cast_id_override: str | None = Field(
        None,
        description="Optional cast_id override (when user selects a non-top suggestion)",
    )


class SmartSuggestionsApplyAllRequest(BaseModel):
    batch_id: str = Field(..., description="Suggestion batch id (required)")
    applied_by: str | None = Field(None, description="Optional actor identifier (email/username)")


def _identity_person_map(ep_id: str, *, run_id: str | None) -> dict[str, str | None]:
    from apps.api.services.identities import load_identities

    payload = load_identities(ep_id, run_id=run_id)
    identities = payload.get("identities", []) if isinstance(payload, dict) else []
    mapping: dict[str, str | None] = {}
    for identity in identities:
        if not isinstance(identity, dict):
            continue
        cid = identity.get("identity_id") or identity.get("id")
        if not cid:
            continue
        mapping[str(cid)] = identity.get("person_id")
    return mapping


def _people_by_id(ep_id: str) -> dict[str, dict[str, Any]]:
    parsed = _parse_ep_id(ep_id)
    show_id = parsed["show"] if parsed else None
    if not show_id:
        return {}
    people = people_service.list_people(show_id)
    return {p.get("person_id"): p for p in people if isinstance(p, dict) and p.get("person_id")}


@router.post("/episodes/{ep_id}/smart_suggestions/generate")
def generate_smart_suggestions(
    ep_id: str,
    body: SmartSuggestionsGenerateRequest,
    run_id: str = Query(..., description="Run id scope (required)"),
) -> dict:
    """Generate a new persisted Smart Suggestions batch (run-scoped)."""
    try:
        scoped_service = _scoped_grouping_service(run_id)
        result = scoped_service.suggest_cast_for_unassigned_clusters(
            ep_id,
            min_similarity=body.min_similarity,
            top_k=body.top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate suggestions: {exc}") from exc

    suggestions_entries = result.get("suggestions", []) if isinstance(result, dict) else []
    rows: list[dict[str, Any]] = []
    for entry in suggestions_entries:
        if not isinstance(entry, dict):
            continue
        cluster_id = entry.get("cluster_id")
        cast_suggestions = entry.get("cast_suggestions") or []
        if not cluster_id or not isinstance(cast_suggestions, list) or not cast_suggestions:
            continue
        best = cast_suggestions[0] if isinstance(cast_suggestions[0], dict) else None
        if not best:
            continue
        best_cast_id = best.get("cast_id")
        similarity = best.get("similarity")
        try:
            similarity_value = float(similarity) if similarity is not None else 0.0
        except (TypeError, ValueError):
            similarity_value = 0.0

        rows.append(
            {
                "type": "cluster_cast_assignment",
                "target_identity_id": str(cluster_id),
                "suggested_person_id": str(best_cast_id) if best_cast_id else "",
                "confidence": similarity_value,
                "evidence_json": {
                    "cluster_id": cluster_id,
                    "best": best,
                    "cast_suggestions": cast_suggestions,
                    "meta": {k: v for k, v in entry.items() if k not in {"cast_suggestions"}},
                },
            }
        )

    try:
        from py_screenalytics import run_layout
        from apps.api.services.run_persistence import run_persistence_service

        run_id_norm = run_layout.normalize_run_id(run_id)
        run_persistence_service.ensure_run(ep_id=ep_id, run_id=run_id_norm)
        batch_id = run_persistence_service.create_suggestion_batch(
            ep_id=ep_id,
            run_id=run_id_norm,
            generator_version=body.generator_version,
            generator_config_json={
                "min_similarity": body.min_similarity,
                "top_k": body.top_k,
                "source": "GroupingService.suggest_cast_for_unassigned_clusters",
            },
            summary_json={
                "clusters_with_suggestions": len(rows),
                "mismatched_embeddings": len(result.get("mismatched_embeddings", []) if isinstance(result, dict) else []),
                "quality_only_clusters": len(result.get("quality_only_clusters", []) if isinstance(result, dict) else []),
            },
        )

        inserted = run_persistence_service.insert_suggestions(batch_id=batch_id, ep_id=ep_id, run_id=run_id_norm, rows=rows)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to persist suggestions: {exc}") from exc

    return {
        "status": "success",
        "ep_id": ep_id,
        "run_id": run_id_norm,
        "batch_id": batch_id,
        "counts": {
            "suggestions_persisted": len(inserted),
            "clusters_with_suggestions": len(rows),
        },
        "suggestions": inserted,
        "mismatched_embeddings": result.get("mismatched_embeddings", []) if isinstance(result, dict) else [],
        "quality_only_clusters": result.get("quality_only_clusters", []) if isinstance(result, dict) else [],
    }


@router.get("/episodes/{ep_id}/smart_suggestions/batches")
def list_smart_suggestion_batches(
    ep_id: str,
    run_id: str = Query(..., description="Run id scope (required)"),
    limit: int = Query(25, ge=1, le=250),
) -> dict:
    try:
        from py_screenalytics import run_layout
        from apps.api.services.run_persistence import run_persistence_service

        run_id_norm = run_layout.normalize_run_id(run_id)
        batches = run_persistence_service.list_suggestion_batches(ep_id=ep_id, run_id=run_id_norm, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list suggestion batches: {exc}") from exc
    return {"status": "success", "ep_id": ep_id, "run_id": run_id_norm, "batches": batches}


@router.get("/episodes/{ep_id}/smart_suggestions")
def list_smart_suggestions(
    ep_id: str,
    run_id: str = Query(..., description="Run id scope (required)"),
    batch_id: str | None = Query(None, description="Optional batch_id (defaults to latest)"),
    include_dismissed: bool = Query(False, description="Include dismissed suggestions"),
) -> dict:
    try:
        from py_screenalytics import run_layout
        from apps.api.services.run_persistence import run_persistence_service

        run_id_norm = run_layout.normalize_run_id(run_id)
        effective_batch_id = batch_id
        if not effective_batch_id:
            batches = run_persistence_service.list_suggestion_batches(ep_id=ep_id, run_id=run_id_norm, limit=1)
            effective_batch_id = (batches[0].get("batch_id") if batches else None)
        if not effective_batch_id:
            return {"status": "success", "ep_id": ep_id, "run_id": run_id_norm, "batch_id": None, "suggestions": []}

        suggestions = run_persistence_service.list_suggestions(
            ep_id=ep_id,
            run_id=run_id_norm,
            batch_id=str(effective_batch_id),
            include_dismissed=include_dismissed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list suggestions: {exc}") from exc
    return {
        "status": "success",
        "ep_id": ep_id,
        "run_id": run_id_norm,
        "batch_id": str(effective_batch_id),
        "suggestions": suggestions,
    }


@router.post("/episodes/{ep_id}/smart_suggestions/dismiss")
def dismiss_smart_suggestions(
    ep_id: str,
    body: SmartSuggestionsDismissRequest,
    run_id: str = Query(..., description="Run id scope (required)"),
) -> dict:
    try:
        from py_screenalytics import run_layout
        from apps.api.services.run_persistence import run_persistence_service

        run_id_norm = run_layout.normalize_run_id(run_id)
        updated = 0
        for sid in body.suggestion_ids:
            if run_persistence_service.set_suggestion_dismissed(
                ep_id=ep_id,
                run_id=run_id_norm,
                batch_id=body.batch_id,
                suggestion_id=sid,
                dismissed=bool(body.dismissed),
            ):
                updated += 1
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to update dismissed state: {exc}") from exc
    return {
        "status": "success",
        "ep_id": ep_id,
        "run_id": run_id_norm,
        "batch_id": body.batch_id,
        "dismissed": bool(body.dismissed),
        "updated": updated,
        "requested": len(body.suggestion_ids),
    }


@router.post("/episodes/{ep_id}/smart_suggestions/apply")
def apply_smart_suggestion(
    ep_id: str,
    body: SmartSuggestionsApplyRequest,
    run_id: str = Query(..., description="Run id scope (required)"),
) -> dict:
    """Apply a single suggestion (run-scoped; enforces identity locks)."""
    try:
        from py_screenalytics import run_layout
        from apps.api.services.run_persistence import run_persistence_service

        run_id_norm = run_layout.normalize_run_id(run_id)
        suggestions = run_persistence_service.list_suggestions(
            ep_id=ep_id,
            run_id=run_id_norm,
            batch_id=body.batch_id,
            include_dismissed=True,
        )
        suggestion = next((s for s in suggestions if s.get("suggestion_id") == body.suggestion_id), None)
        if not suggestion:
            raise HTTPException(status_code=404, detail="suggestion_not_found")
        if suggestion.get("dismissed"):
            return {"status": "skipped", "reason": "dismissed", "ep_id": ep_id, "run_id": run_id_norm}

        target_identity_id = suggestion.get("target_identity_id")
        if not target_identity_id:
            raise HTTPException(status_code=400, detail="suggestion_missing_target_identity_id")

        if run_persistence_service.is_identity_locked(ep_id=ep_id, run_id=run_id_norm, identity_id=str(target_identity_id)):
            return {"status": "skipped", "reason": "locked", "ep_id": ep_id, "run_id": run_id_norm}

        people_by_id = _people_by_id(ep_id)
        before_map = _identity_person_map(ep_id, run_id=run_id_norm)
        before_person_id = before_map.get(str(target_identity_id))
        before_cast_id = (people_by_id.get(before_person_id or "", {}) or {}).get("cast_id") if before_person_id else None

        cast_id = body.cast_id_override or suggestion.get("suggested_person_id")
        if not cast_id:
            raise HTTPException(status_code=400, detail="suggestion_missing_cast_id")

        # Skip if already assigned to a cast member.
        if before_cast_id:
            return {
                "status": "skipped",
                "reason": "already_assigned",
                "ep_id": ep_id,
                "run_id": run_id_norm,
            }

        class _Assign:
            def __init__(self, cluster_id: str, target_cast_id: str) -> None:
                self.cluster_id = cluster_id
                self.target_cast_id = target_cast_id

        scoped_service = _scoped_grouping_service(run_id_norm)
        scoped_service.batch_assign_clusters(ep_id, [_Assign(str(target_identity_id), str(cast_id))])

        after_map = _identity_person_map(ep_id, run_id=run_id_norm)
        after_person_id = after_map.get(str(target_identity_id))
        people_after = _people_by_id(ep_id)
        after_cast_id = (people_after.get(after_person_id or "", {}) or {}).get("cast_id") if after_person_id else None

        apply_id = run_persistence_service.record_suggestion_apply(
            ep_id=ep_id,
            run_id=run_id_norm,
            batch_id=body.batch_id,
            suggestion_id=body.suggestion_id,
            applied_by=body.applied_by,
            changes_json={
                "target_identity_id": str(target_identity_id),
                "suggestion_cast_id": suggestion.get("suggested_person_id"),
                "applied_cast_id": str(cast_id),
                "before_person_id": before_person_id,
                "after_person_id": after_person_id,
                "before_cast_id": before_cast_id,
                "after_cast_id": after_cast_id,
            },
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to apply suggestion: {exc}") from exc

    return {
        "status": "success",
        "ep_id": ep_id,
        "run_id": run_id_norm,
        "batch_id": body.batch_id,
        "suggestion_id": body.suggestion_id,
        "apply_id": apply_id,
    }


@router.post("/episodes/{ep_id}/smart_suggestions/apply_all")
def apply_all_smart_suggestions(
    ep_id: str,
    body: SmartSuggestionsApplyAllRequest,
    run_id: str = Query(..., description="Run id scope (required)"),
) -> dict:
    """Apply all suggestions in a batch (run-scoped; requires batch_id)."""
    try:
        from py_screenalytics import run_layout
        from apps.api.services.run_persistence import run_persistence_service

        run_id_norm = run_layout.normalize_run_id(run_id)
        suggestions = run_persistence_service.list_suggestions(
            ep_id=ep_id,
            run_id=run_id_norm,
            batch_id=body.batch_id,
            include_dismissed=False,
        )

        locked_set = {
            row.get("identity_id")
            for row in run_persistence_service.list_identity_locks(ep_id=ep_id, run_id=run_id_norm)
            if isinstance(row, dict) and row.get("locked") and row.get("identity_id")
        }

        # Deduplicate by target_identity_id (keep highest confidence per cluster).
        best_by_cluster: dict[str, dict[str, Any]] = {}
        for s in suggestions:
            cid = s.get("target_identity_id")
            if not cid:
                continue
            if cid not in best_by_cluster or float(s.get("confidence") or 0.0) > float(best_by_cluster[cid].get("confidence") or 0.0):
                best_by_cluster[cid] = s

        people_before = _people_by_id(ep_id)
        before_map = _identity_person_map(ep_id, run_id=run_id_norm)

        to_apply: list[tuple[str, str, str]] = []  # (cluster_id, cast_id, suggestion_id)
        skipped_locked = 0
        skipped_already_assigned = 0

        for cid, s in best_by_cluster.items():
            if cid in locked_set:
                skipped_locked += 1
                continue
            before_person_id = before_map.get(cid)
            before_cast_id = (people_before.get(before_person_id or "", {}) or {}).get("cast_id") if before_person_id else None
            if before_cast_id:
                skipped_already_assigned += 1
                continue
            cast_id = s.get("suggested_person_id")
            if not cast_id:
                continue
            to_apply.append((cid, str(cast_id), str(s.get("suggestion_id"))))

        class _Assign:
            def __init__(self, cluster_id: str, target_cast_id: str) -> None:
                self.cluster_id = cluster_id
                self.target_cast_id = target_cast_id

        scoped_service = _scoped_grouping_service(run_id_norm)
        assignments = [_Assign(cid, cast_id) for (cid, cast_id, _sid) in to_apply]
        batch_result = scoped_service.batch_assign_clusters(ep_id, assignments) if assignments else {"succeeded": 0, "failed": 0, "results": []}

        after_map = _identity_person_map(ep_id, run_id=run_id_norm)
        people_after = _people_by_id(ep_id)

        applied = 0
        apply_ids: list[str] = []
        for cid, cast_id, sid in to_apply:
            before_person_id = before_map.get(cid)
            after_person_id = after_map.get(cid)
            after_cast_id = (people_after.get(after_person_id or "", {}) or {}).get("cast_id") if after_person_id else None
            if after_cast_id:
                applied += 1
            apply_ids.append(
                run_persistence_service.record_suggestion_apply(
                    ep_id=ep_id,
                    run_id=run_id_norm,
                    batch_id=body.batch_id,
                    suggestion_id=sid,
                    applied_by=body.applied_by,
                    changes_json={
                        "target_identity_id": cid,
                        "applied_cast_id": cast_id,
                        "before_person_id": before_person_id,
                        "after_person_id": after_person_id,
                        "before_cast_id": (people_before.get(before_person_id or "", {}) or {}).get("cast_id") if before_person_id else None,
                        "after_cast_id": after_cast_id,
                        "mode": "apply_all",
                    },
                )
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to apply all: {exc}") from exc

    return {
        "status": "success",
        "ep_id": ep_id,
        "run_id": run_id_norm,
        "batch_id": body.batch_id,
        "counts": {
            "applied": applied,
            "skipped_locked": skipped_locked,
            "skipped_dismissed": 0,
            "skipped_already_assigned": skipped_already_assigned,
            "requested": len(best_by_cluster),
        },
        "job": batch_result,
        "apply_ids": apply_ids,
    }


@router.get("/episodes/{ep_id}/unlinked_entities")
def list_unlinked_entities(
    ep_id: str,
    run_id: Optional[str] = Query(None, description="Optional run_id scope for artifacts/state"),
) -> dict:
    """Return clusters that are not linked to a cast member (auto-people + unassigned clusters)."""
    try:
        scoped_service = _scoped_grouping_service(run_id)
        result = scoped_service.list_unlinked_entities(ep_id)
        return {
            "status": "success",
            "ep_id": ep_id,
            "run_id": run_id,
            **result,
        }
    except FileNotFoundError as e:
        return {
            "status": "success",
            "ep_id": ep_id,
            "run_id": run_id,
            "entities": [],
            "counts": {"total": 0, "clusters": 0},
            "message": str(e),
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load unlinked entities: {str(e)}",
        )


@router.get("/episodes/{ep_id}/find_similar_unassigned")
def find_similar_unassigned(
    ep_id: str,
    person_id: str = Query(None, description="Person ID to find similar clusters for"),
    cast_id: str = Query(None, description="Cast ID to find similar clusters for"),
    min_similarity: float = Query(0.50, ge=0.3, le=1.0, description="Minimum similarity threshold"),
) -> dict:
    """Find unassigned clusters similar to a given person or cast member.

    Used after accepting a suggestion to find additional singletons that might
    belong to the same person but weren't in the original suggestion group.

    Args:
        ep_id: Episode ID
        person_id: Person ID to match against (uses person's assigned clusters' centroid)
        cast_id: Cast ID to match against (uses facebank seeds)
        min_similarity: Minimum cosine similarity threshold (default 0.50)

    Returns:
        similar_clusters: List of unassigned cluster IDs with their similarity scores
    """
    import numpy as np
    from apps.api.services.facebank import cosine_similarity, FacebankService
    import re

    if not person_id and not cast_id:
        raise HTTPException(status_code=400, detail="Either person_id or cast_id required")

    # Parse episode ID for show
    pattern = r"^(?P<show>.+)-s(?P<season>\d{2})e(?P<episode>\d{2})$"
    match = re.match(pattern, ep_id, re.IGNORECASE)
    if not match:
        raise HTTPException(status_code=400, detail=f"Invalid episode ID: {ep_id}")
    show_id = match.group("show").upper()

    try:
        # Load cluster centroids
        centroids_data = grouping_service.load_cluster_centroids(ep_id)
        centroids = centroids_data.get("centroids", {})
        if not centroids:
            return {"status": "success", "similar_clusters": [], "message": "No centroids found"}

        # Load identities to find assigned/unassigned
        from apps.api.services.identities import load_identities
        identities_data = load_identities(ep_id)
        identities = identities_data.get("identities", [])

        # Build lookup: cluster_id -> person_id
        cluster_to_person = {}
        for ident in identities:
            cid = ident.get("identity_id")
            pid = ident.get("person_id")
            if cid:
                cluster_to_person[cid] = pid

        # Get reference embedding(s) to compare against
        reference_embeddings = []

        if cast_id:
            # Use facebank seeds for this cast member
            facebank_service = FacebankService()
            seeds = facebank_service.list_seeds(show_id, cast_id=cast_id)
            for seed in seeds:
                emb = seed.get("embedding")
                if emb:
                    reference_embeddings.append(np.array(emb, dtype=np.float32))

        if person_id:
            # Use centroids of clusters assigned to this person
            for ident in identities:
                if ident.get("person_id") == person_id:
                    cid = ident.get("identity_id")
                    if cid and cid in centroids:
                        centroid = centroids[cid].get("centroid")
                        if centroid:
                            reference_embeddings.append(np.array(centroid, dtype=np.float32))

        if not reference_embeddings:
            return {"status": "success", "similar_clusters": [], "message": "No reference embeddings found"}

        # Average reference embeddings
        ref_vec = np.mean(reference_embeddings, axis=0)
        ref_vec = ref_vec / (np.linalg.norm(ref_vec) + 1e-8)

        # Find unassigned clusters similar to reference
        similar_clusters = []
        for cluster_id, cluster_data in centroids.items():
            # Skip if already assigned to a person with cast link
            pid = cluster_to_person.get(cluster_id)
            if pid:
                # Check if this person has a cast_id
                from apps.api.services.people import PeopleService
                people_service = PeopleService()
                person = people_service.get_person(show_id, pid)
                if person and person.get("cast_id"):
                    continue  # Already assigned to cast, skip

            centroid = cluster_data.get("centroid")
            if not centroid:
                continue

            cluster_vec = np.array(centroid, dtype=np.float32)
            cluster_vec = cluster_vec / (np.linalg.norm(cluster_vec) + 1e-8)

            sim = float(cosine_similarity(ref_vec, cluster_vec))
            if sim >= min_similarity:
                similar_clusters.append({
                    "cluster_id": cluster_id,
                    "similarity": sim,
                    "tracks": cluster_data.get("track_count", 1),
                    "faces": cluster_data.get("face_count", 0),
                })

        # Sort by similarity descending
        similar_clusters.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "status": "success",
            "ep_id": ep_id,
            "person_id": person_id,
            "cast_id": cast_id,
            "min_similarity": min_similarity,
            "similar_clusters": similar_clusters,
            "count": len(similar_clusters),
        }

    except FileNotFoundError as e:
        return {"status": "success", "similar_clusters": [], "message": str(e)}
    except Exception as e:
        LOGGER.exception(f"[{ep_id}] Failed to find similar unassigned: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/episodes/{ep_id}/analyze_unassigned")
def analyze_unassigned_clusters(
    ep_id: str,
    similarity_threshold: float = 0.70,
    min_cast_similarity: float = 0.50,
    top_k: int = 3,
) -> dict:
    """Analyze unassigned clusters: group similar ones and recommend cast members.

    Groups unassigned clusters by their similarity to each other, then provides
    cast member recommendations for each group. Useful for bulk review/assignment.

    Query params:
        similarity_threshold: Min similarity to group clusters together (default 0.70)
        min_cast_similarity: Min similarity for cast recommendations (default 0.50)
        top_k: Number of cast suggestions per group (default 3)

    Returns:
        - groups: List of cluster groups with cast recommendations
        - singletons: Clusters that don't match any others
        - summary: Stats about the analysis
    """
    try:
        result = grouping_service.analyze_unassigned_clusters(
            ep_id,
            similarity_threshold=similarity_threshold,
            min_cast_similarity=min_cast_similarity,
            top_k_cast=top_k,
        )
        return {
            "status": "success",
            "ep_id": ep_id,
            **result,
        }
    except FileNotFoundError as e:
        return {
            "status": "success",
            "ep_id": ep_id,
            "groups": [],
            "singletons": [],
            "message": str(e),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze unassigned clusters: {str(e)}",
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
        merge_estimate_error = None
        if centroids_count > 1:
            try:
                within_result = grouping_service.group_within_episode(
                    ep_id,
                    protect_manual=False,  # Check without protection to see full impact
                )
                potential_merges = within_result.get("merged_count", 0)
            except Exception as e:
                LOGGER.warning(f"[{ep_id}] Failed to estimate merges in cleanup preview: {e}")
                merge_estimate_error = str(e)

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

        # Add error warning if merge estimation failed
        if merge_estimate_error:
            preview["warning_level"] = "high"
            preview["warnings"].append(
                f"⚠️ Could not estimate merges: {merge_estimate_error}"
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
def save_assignments(
    ep_id: str,
    run_id: str = Query(..., description="Run id scope (required for mutations)"),
) -> dict:
    """Save all current cluster assignments to people.json and identities.json.

    This ensures all assignments made in the UI are persisted.
    """
    try:
        scoped_service = _scoped_grouping_service(run_id)
        result = scoped_service.save_current_assignments(ep_id)
        return {
            "status": "success",
            "ep_id": ep_id,
            "run_id": run_id,
            "saved_count": result.get("saved_count", 0),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save assignments: {str(e)}")


# =============================================================================
# Enhancement #3: Undo/Redo Stack Endpoints
# =============================================================================


@router.get("/episodes/{ep_id}/undo_stack")
def get_undo_stack(ep_id: str) -> dict:
    """Get the undo stack for an episode.

    Returns list of operations that can be undone, with id, type, description, timestamp.
    """
    try:
        operations = grouping_service.get_undo_stack(ep_id)
        return {
            "ep_id": ep_id,
            "operations": operations,
            "count": len(operations),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get undo stack: {str(e)}")


@router.post("/episodes/{ep_id}/undo")
def undo_last_operation(ep_id: str) -> dict:
    """Undo the last operation for an episode.

    Restores the state before the last undoable operation.
    """
    try:
        result = grouping_service.undo_last_operation(ep_id)
        if result is None:
            return {
                "status": "no_operations",
                "ep_id": ep_id,
                "message": "No operations to undo",
            }
        return {
            "status": "success",
            "ep_id": ep_id,
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to undo operation: {str(e)}")


# =============================================================================
# Enhancement #6: Confidence-Based Auto-Assignment Queue Endpoints
# =============================================================================


@router.get("/episodes/{ep_id}/tiered_suggestions")
def get_tiered_suggestions(
    ep_id: str,
    high_threshold: float = 0.85,
    medium_threshold: float = 0.68,
) -> dict:
    """Get cast suggestions tiered by confidence level.

    Returns suggestions in three tiers:
    - high_confidence: Auto-assignable (≥85% similarity)
    - medium_confidence: Review queue (68-85% similarity)
    - low_confidence: Manual review required (<68% similarity)
    """
    try:
        return grouping_service.get_tiered_suggestions(
            ep_id,
            high_threshold=high_threshold,
            medium_threshold=medium_threshold,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tiered suggestions: {str(e)}")


@router.post("/episodes/{ep_id}/auto_assign_high_confidence")
def auto_assign_high_confidence(ep_id: str, threshold: float = 0.85) -> dict:
    """Auto-assign all high-confidence suggestions.

    Automatically assigns clusters to cast members when similarity exceeds threshold.

    Args:
        threshold: Minimum similarity for auto-assignment (default 0.85)
    """
    try:
        result = grouping_service.auto_assign_high_confidence(ep_id, threshold=threshold)
        # Queue async similarity refresh
        _queue_async_similarity_refresh(ep_id)
        return {
            "status": "success",
            "ep_id": ep_id,
            **result,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to auto-assign: {str(e)}")


# =============================================================================
# Enhancement #10: Smart Merge Suggestions Endpoints
# =============================================================================


class MergeClustersRequest(BaseModel):
    """Request to merge multiple clusters."""
    cluster_ids: List[str] = Field(..., description="List of cluster IDs to merge")
    target_person_id: Optional[str] = Field(None, description="Optional person ID to merge into")


class MergeAllRequest(BaseModel):
    """Request to merge all high-similarity pairs."""
    similarity_threshold: float = Field(0.90, description="Minimum similarity for auto-merge")


@router.get("/episodes/{ep_id}/potential_duplicates")
def get_potential_duplicates(
    ep_id: str,
    similarity_threshold: float = 0.85,
    max_pairs: int = 20,
) -> dict:
    """Find clusters that might be duplicates (same person split across clusters).

    Returns pairs of clusters that exceed the similarity threshold.
    """
    try:
        return grouping_service.find_potential_duplicates(
            ep_id,
            similarity_threshold=similarity_threshold,
            max_pairs=max_pairs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to find duplicates: {str(e)}")


@router.post("/episodes/{ep_id}/merge_clusters")
def merge_clusters(ep_id: str, req: MergeClustersRequest) -> dict:
    """Merge multiple clusters into a single person.

    Args:
        cluster_ids: List of cluster IDs to merge
        target_person_id: Optional person ID to merge into (creates new if not provided)
    """
    try:
        result = grouping_service.merge_clusters(
            ep_id,
            cluster_ids=req.cluster_ids,
            target_person_id=req.target_person_id,
        )
        # Queue async similarity refresh
        _queue_async_similarity_refresh(ep_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to merge clusters: {str(e)}")


@router.post("/episodes/{ep_id}/merge_all_duplicates")
def merge_all_duplicates(ep_id: str, req: MergeAllRequest) -> dict:
    """Automatically merge all high-similarity cluster pairs.

    Uses transitive closure to group connected clusters before merging.

    Args:
        similarity_threshold: Minimum similarity for auto-merge (default 0.90)
    """
    try:
        result = grouping_service.merge_all_high_similarity_pairs(
            ep_id,
            similarity_threshold=req.similarity_threshold,
        )
        # Queue async similarity refresh
        _queue_async_similarity_refresh(ep_id)
        return {
            "status": "success",
            "ep_id": ep_id,
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to merge duplicates: {str(e)}")


# =============================================================================
# Dismissed Suggestions Endpoints
# =============================================================================


class DismissRequest(BaseModel):
    """Request to dismiss one or more suggestions."""

    suggestion_ids: List[str] = Field(..., description="List of suggestion IDs to dismiss")


class ResetDismissedSuggestionsRequest(BaseModel):
    """Request to reset dismissed suggestions state."""

    archive_existing: bool = Field(
        True,
        description="If true, archive the existing dismissed suggestions file before clearing.",
    )


@router.get("/episodes/{ep_id}/dismissed_suggestions")
def get_dismissed_suggestions(
    ep_id: str,
    run_id: str = Query(..., description="Run id scope (required)"),
) -> dict:
    """Get all dismissed suggestions for an episode.

    Returns:
        {
            "status": "success",
            "ep_id": "show-S01E01",
            "dismissed": ["cluster_id_1", "person:person_id_1", ...]
        }
    """
    try:
        dismissed = dismissed_suggestions_service.get_dismissed(ep_id, run_id=run_id)
        return {
            "status": "success",
            "ep_id": ep_id,
            "run_id": run_id,
            "dismissed": dismissed,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dismissed suggestions: {str(e)}")


@router.post("/episodes/{ep_id}/dismissed_suggestions")
def dismiss_suggestions(
    ep_id: str,
    req: DismissRequest,
    run_id: str = Query(..., description="Run id scope (required for mutations)"),
) -> dict:
    """Dismiss one or more suggestions.

    Request body:
        suggestion_ids: List of suggestion IDs (cluster_id or "person:{person_id}")

    Returns:
        {"status": "success", "dismissed_count": N}
    """
    try:
        if not req.suggestion_ids:
            raise HTTPException(status_code=400, detail="suggestion_ids required")

        success = dismissed_suggestions_service.dismiss_many(ep_id, req.suggestion_ids, run_id=run_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to persist dismissed suggestions")

        return {
            "status": "success",
            "ep_id": ep_id,
            "run_id": run_id,
            "dismissed_count": len(req.suggestion_ids),
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to dismiss suggestions: {str(e)}")


@router.delete("/episodes/{ep_id}/dismissed_suggestions/{suggestion_id}")
def restore_suggestion(
    ep_id: str,
    suggestion_id: str,
    run_id: str = Query(..., description="Run id scope (required for mutations)"),
) -> dict:
    """Restore a previously dismissed suggestion.

    Args:
        suggestion_id: The suggestion ID to restore (URL encoded if contains special chars)

    Returns:
        {"status": "success", "restored": "suggestion_id"}
    """
    try:
        success = dismissed_suggestions_service.restore(ep_id, suggestion_id, run_id=run_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to persist restored suggestion")

        return {
            "status": "success",
            "ep_id": ep_id,
            "run_id": run_id,
            "restored": suggestion_id,
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restore suggestion: {str(e)}")


@router.delete("/episodes/{ep_id}/dismissed_suggestions")
def clear_dismissed_suggestions(
    ep_id: str,
    run_id: str = Query(..., description="Run id scope (required for mutations)"),
) -> dict:
    """Clear all dismissed suggestions for an episode.

    Returns:
        {"status": "success", "message": "All dismissed suggestions cleared"}
    """
    try:
        success = dismissed_suggestions_service.clear_all(ep_id, run_id=run_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to clear dismissed suggestions")

        return {
            "status": "success",
            "ep_id": ep_id,
            "run_id": run_id,
            "message": "All dismissed suggestions cleared",
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear dismissed suggestions: {str(e)}")


@router.post("/episodes/{ep_id}/dismissed_suggestions/reset_state")
def reset_dismissed_suggestions_state(
    ep_id: str,
    req: ResetDismissedSuggestionsRequest,
    run_id: str = Query(..., description="Run id scope (required for mutations)"),
) -> dict:
    """Reset dismissed Smart Suggestions (explicit, user-controlled).

    By default, archives the prior file so user intent is preserved and can be restored if needed.
    """
    try:
        result = dismissed_suggestions_service.reset_state(ep_id, run_id=run_id, archive_existing=req.archive_existing)
        if not result.get("ok"):
            raise HTTPException(
                status_code=500,
                detail=result.get("error") or "Failed to reset dismissed suggestions",
            )

        return {
            "status": "success",
            "ep_id": ep_id,
            "run_id": run_id,
            "archived": result.get("archived"),
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset dismissed suggestions: {str(e)}")


@router.get("/episodes/{ep_id}/outlier_audit")
def get_outlier_audit(
    ep_id: str,
    cohesion_threshold: float = Query(0.70, description="Tracks below this cohesion are flagged as outliers"),
    min_tracks_for_comparison: int = Query(2, description="Minimum tracks needed to compute cohesion"),
) -> dict:
    """Audit all assigned tracks for potential outliers.

    Scans all cast members with assigned clusters/tracks and identifies tracks
    that have low cohesion with other tracks assigned to the same person.

    Returns:
        {
            "ep_id": str,
            "total_cast_members": int,
            "total_tracks_audited": int,
            "outliers": [
                {
                    "track_id": int,
                    "cluster_id": str,
                    "person_id": str,
                    "cast_id": str,
                    "cast_name": str,
                    "cohesion_score": float,
                    "track_similarity": float,
                    "thumb_url": str,
                    "faces": int,
                    "suggested_reassignment": {
                        "cast_id": str,
                        "cast_name": str,
                        "similarity": float
                    } | None
                }
            ],
            "summary": {
                "high_risk": int,  # cohesion < 0.50
                "medium_risk": int,  # cohesion 0.50-0.65
                "low_risk": int  # cohesion 0.65-threshold
            }
        }
    """
    import numpy as np
    from apps.api.services.people import PeopleService

    try:
        # Get show slug from episode using same pattern as other endpoints
        parsed = _parse_ep_id(ep_id)
        if not parsed:
            raise HTTPException(status_code=400, detail="Invalid episode ID format")
        show_slug = parsed["show"]

        # Get all people for this show
        people_service_instance = PeopleService()
        people = people_service_instance.list_people(show_slug)
        if not people:
            return {
                "ep_id": ep_id,
                "total_cast_members": 0,
                "total_tracks_audited": 0,
                "outliers": [],
                "summary": {"high_risk": 0, "medium_risk": 0, "low_risk": 0},
            }

        outliers = []
        total_tracks_audited = 0
        cast_members_with_tracks = 0

        # Process each person who has a cast_id (assigned to cast member)
        for person in people:
            person_id = person.get("person_id")
            cast_id = person.get("cast_id")
            if not cast_id:
                continue  # Skip people not linked to cast

            cast_name = person.get("name") or "(unnamed)"

            # Get this episode's clusters from person's cluster_ids
            # cluster_ids are in format "ep_id:cluster_id"
            all_cluster_ids = person.get("cluster_ids", [])
            episode_clusters = [
                cid.split(":", 1)[1] if ":" in cid else cid
                for cid in all_cluster_ids
                if isinstance(cid, str) and cid.startswith(f"{ep_id}:")
            ]
            if not episode_clusters:
                continue

            cast_members_with_tracks += 1

            # Fetch cluster details to get tracks and embeddings
            all_tracks = []
            all_embeddings = []
            track_to_cluster = {}

            for cluster_id in episode_clusters:
                try:
                    cluster_detail = grouping_service.get_cluster_detail(ep_id, cluster_id)
                    if not cluster_detail:
                        continue

                    tracks = cluster_detail.get("tracks", [])
                    for track in tracks:
                        track_id = track.get("track_id")
                        embedding = track.get("embedding")
                        if track_id is not None and embedding:
                            track_idx = len(all_tracks)
                            all_tracks.append({
                                "track_id": track_id,
                                "cluster_id": cluster_id,
                                "faces": track.get("faces", 0),
                                "thumb_url": track.get("thumb_url") or track.get("thumbnail_url"),
                                "track_similarity": track.get("track_similarity"),
                            })
                            all_embeddings.append(embedding)
                            track_to_cluster[track_idx] = cluster_id
                except Exception as e:
                    LOGGER.warning(f"[outlier_audit] Failed to get cluster {cluster_id}: {e}")
                    continue

            total_tracks_audited += len(all_tracks)

            # Need at least min_tracks_for_comparison to compute cohesion
            if len(all_embeddings) < min_tracks_for_comparison:
                continue

            # Compute pairwise similarities and per-track cohesion
            try:
                embeddings_array = np.array(all_embeddings)
                norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                normalized = embeddings_array / (norms + 1e-10)
                similarity_matrix = normalized @ normalized.T

                # Compute cohesion for each track (mean similarity to others)
                for idx, track_data in enumerate(all_tracks):
                    # Exclude self-similarity
                    other_sims = [similarity_matrix[idx, j] for j in range(len(all_tracks)) if j != idx]
                    cohesion = float(np.mean(other_sims)) if other_sims else 1.0

                    if cohesion < cohesion_threshold:
                        # This track is an outlier - find suggested reassignment
                        suggested = None
                        try:
                            # Get cast suggestions for this track's embedding
                            track_embedding = all_embeddings[idx]
                            suggestions = grouping_service.get_cast_suggestions_for_embedding(
                                ep_id, track_embedding, top_k=3, exclude_cast_id=cast_id
                            )
                            if suggestions:
                                top_suggestion = suggestions[0]
                                suggested = {
                                    "cast_id": top_suggestion.get("cast_id"),
                                    "cast_name": top_suggestion.get("name"),
                                    "similarity": top_suggestion.get("similarity"),
                                }
                        except Exception:
                            pass

                        outliers.append({
                            "track_id": track_data["track_id"],
                            "cluster_id": track_data["cluster_id"],
                            "person_id": person_id,
                            "cast_id": cast_id,
                            "cast_name": cast_name,
                            "cohesion_score": round(cohesion, 3),
                            "track_similarity": track_data.get("track_similarity"),
                            "thumb_url": track_data.get("thumb_url"),
                            "faces": track_data.get("faces", 0),
                            "suggested_reassignment": suggested,
                        })
            except Exception as e:
                LOGGER.warning(f"[outlier_audit] Failed to compute cohesion for {person_id}: {e}")
                continue

        # Sort outliers by cohesion (lowest first = worst outliers)
        outliers.sort(key=lambda x: x["cohesion_score"])

        # Compute summary
        high_risk = len([o for o in outliers if o["cohesion_score"] < 0.50])
        medium_risk = len([o for o in outliers if 0.50 <= o["cohesion_score"] < 0.65])
        low_risk = len([o for o in outliers if o["cohesion_score"] >= 0.65])

        return {
            "ep_id": ep_id,
            "cohesion_threshold": cohesion_threshold,
            "total_cast_members": cast_members_with_tracks,
            "total_tracks_audited": total_tracks_audited,
            "outliers": outliers,
            "summary": {
                "high_risk": high_risk,
                "medium_risk": medium_risk,
                "low_risk": low_risk,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        LOGGER.error(f"[outlier_audit] Failed for {ep_id}: {e}\n{tb}")
        raise HTTPException(status_code=500, detail=f"Failed to audit outliers: {str(e)}")


@router.post("/shows/{show_id}/fix_unnamed_people")
def fix_unnamed_people(show_id: str) -> dict:
    """Create cast members for people without cast_ids.

    This fixes the "Unknown" issue in Screentime where people were created
    during Faces Review without corresponding cast members.

    Returns:
        {
            "status": "success",
            "show_id": str,
            "fixed_count": int,
            "fixed_people": [{"person_id": str, "name": str, "new_cast_id": str}, ...]
        }
    """
    try:
        people = people_service.list_people(show_id)
        fixed_people = []

        for person in people:
            person_id = person.get("person_id")
            cast_id = person.get("cast_id")
            name = person.get("name")

            # Skip if already has cast_id
            if cast_id:
                continue

            # Skip if no person_id
            if not person_id:
                continue

            # Determine name to use for cast member
            display_name = name if name else person_id

            try:
                # Create cast member
                cast_member = cast_service.create_cast_member(
                    show_id=show_id,
                    name=display_name,
                    role="other",
                    status="active",
                )
                new_cast_id = cast_member["cast_id"]

                # Update person with cast_id
                people_service.update_person(show_id, person_id, cast_id=new_cast_id)

                fixed_people.append({
                    "person_id": person_id,
                    "name": display_name,
                    "new_cast_id": new_cast_id,
                })
                LOGGER.info(f"[{show_id}] Fixed person {person_id}: created cast member {new_cast_id} for '{display_name}'")

            except Exception as e:
                LOGGER.warning(f"[{show_id}] Failed to fix person {person_id}: {e}")
                continue

        return {
            "status": "success",
            "show_id": show_id,
            "fixed_count": len(fixed_people),
            "fixed_people": fixed_people,
        }

    except Exception as e:
        LOGGER.error(f"[{show_id}] Failed to fix unnamed people: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fix unnamed people: {str(e)}")


__all__ = ["router"]

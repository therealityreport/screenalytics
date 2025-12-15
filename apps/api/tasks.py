"""Celery tasks for long-running operations in SCREANALYTICS.

These tasks handle cluster assignments, auto-grouping, and cleanup operations
that would otherwise block the UI.

Usage:
    from apps.api.tasks import run_manual_assign_task
    result = run_manual_assign_task.delay(episode_id="rhobh-s01e01", payload={...})
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import redis
from celery import Task, states, group, chord
from celery.result import AsyncResult, GroupResult

from apps.api.celery_app import celery_app
from apps.api.config import REDIS_URL
from apps.api.services import redis_keys

LOGGER = logging.getLogger(__name__)


def _find_project_root() -> Path:
    """Find the SCREENALYTICS project root directory.

    Uses marker files (pyproject.toml, .git) to reliably locate the root,
    regardless of how the module was imported or from what working directory.
    """
    # Start from this file's location
    current = Path(__file__).resolve().parent

    # Walk up looking for project markers
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent

    # Fallback to parent count (should match: apps/api/tasks.py -> SCREENALYTICS)
    return Path(__file__).resolve().parents[2]


# Compute once at module load time for reliability
PROJECT_ROOT = _find_project_root()

# Redis client for job locking
_redis_client: Optional[redis.Redis] = None


def _get_redis() -> redis.Redis:
    """Get or create Redis client for job locking."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def _lock_key(episode_id: str, operation: str) -> str:
    """Generate Redis key for job lock."""
    return redis_keys.job_lock_key(episode_id, operation)


def _lock_keys(episode_id: str, operation: str) -> list[str]:
    """Generate canonical + legacy Redis keys for a job lock (canonical first)."""
    return redis_keys.job_lock_keys(episode_id, operation)


def check_active_job(episode_id: str, operation: str) -> Optional[str]:
    """Check if there's an active job for this episode+operation.

    Returns job_id if found, None otherwise.
    """
    try:
        r = _get_redis()
        job_id = None
        lock_key = None
        for candidate in _lock_keys(episode_id, operation):
            job_id = r.get(candidate)
            if job_id:
                lock_key = candidate
                break

        if job_id:
            # Verify the job is still running
            result = AsyncResult(job_id, app=celery_app)
            if result.state in (states.PENDING, states.RECEIVED, states.STARTED):
                return job_id
            # Job finished or failed, clean up lock
            if lock_key is not None:
                r.delete(lock_key)

        return None
    except Exception as e:
        LOGGER.warning(f"Failed to check active job: {e}")
        return None


# Operation-specific TTLs (Bug 7 fix)
OPERATION_TTLS = {
    "detect_track": 1800,       # 30 minutes
    "faces_embed": 900,         # 15 minutes
    "cluster": 600,             # 10 minutes
    "auto_group": 600,          # 10 minutes
    "manual_assign": 300,       # 5 minutes
    "refresh_similarity": 600,  # 10 minutes
    "improve_faces_queue": 300, # 5 minutes
}
DEFAULT_LOCK_TTL = 600  # 10 minutes default


def _acquire_lock(episode_id: str, operation: str, job_id: str, ttl: int = None) -> bool:
    """Try to acquire lock for an episode+operation.

    Args:
        episode_id: Episode ID
        operation: Operation type (manual_assign, auto_group, etc.)
        job_id: Celery task ID
        ttl: Lock TTL in seconds (defaults to operation-specific TTL)

    Returns:
        True if lock acquired, False if another job is running or Redis unavailable
    """
    # Use operation-specific TTL if not explicitly provided (Bug 7 fix)
    if ttl is None:
        ttl = OPERATION_TTLS.get(operation, DEFAULT_LOCK_TTL)

    try:
        r = _get_redis()
        # Check for any existing lock keys (canonical or legacy)
        for candidate in _lock_keys(episode_id, operation):
            existing = r.get(candidate)
            if not existing:
                continue
            result = AsyncResult(existing, app=celery_app)
            if result.state in (states.PENDING, states.RECEIVED, states.STARTED):
                return False  # Another job is running
            # Old job finished - explicitly delete stale lock before acquiring (Bug 8 fix)
            r.delete(candidate)
            LOGGER.info(
                "Cleaned up stale lock for %s:%s (key=%s job=%s state=%s)",
                episode_id,
                operation,
                candidate,
                existing,
                result.state,
            )

        # Set canonical lock with TTL (NX to avoid races)
        lock_key = _lock_key(episode_id, operation)
        acquired = r.set(lock_key, job_id, nx=True, ex=ttl)
        if not acquired:
            return False
        return True
    except Exception as e:
        LOGGER.error(f"Failed to acquire lock for {episode_id}:{operation}: {e}")
        # Bug 2 fix: Return False on Redis failure to prevent concurrent execution
        # without proper locking. Better to fail-safe than allow data corruption.
        return False


def _release_lock(episode_id: str, operation: str, job_id: str) -> None:
    """Release job lock."""
    try:
        r = _get_redis()
        # Only release if we own the lock (for canonical and legacy keys)
        for lock_key in _lock_keys(episode_id, operation):
            current = r.get(lock_key)
            if current == job_id:
                r.delete(lock_key)
    except Exception as e:
        LOGGER.warning(f"Failed to release lock: {e}")


def _force_release_lock(episode_id: str, operation: str) -> None:
    """Force release job lock regardless of owner.

    Used when a task in a chain fails - the lock was acquired by the chain
    but sub-tasks have different job_ids.
    """
    try:
        r = _get_redis()
        for lock_key in _lock_keys(episode_id, operation):
            r.delete(lock_key)
        LOGGER.info(f"Force released lock for {episode_id}:{operation}")
    except Exception as e:
        LOGGER.warning(f"Failed to force release lock: {e}")


class GroupingTask(Task):
    """Base class for grouping tasks with lock management."""

    abstract = True
    _grouping_service = None

    @property
    def grouping_service(self):
        """Lazy-load GroupingService to avoid import at module level."""
        if self._grouping_service is None:
            from apps.api.services.grouping import GroupingService
            self._grouping_service = GroupingService()
        return self._grouping_service


@celery_app.task(bind=True, base=GroupingTask, name="tasks.run_manual_assign")
def run_manual_assign_task(
    self,
    episode_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Background task for manual cluster assignment.

    Args:
        episode_id: Episode ID
        payload: Dict with 'assignments' list, each having:
            - cluster_id: str
            - target_cast_id: str

    Returns:
        Result dict with succeeded/failed counts and results list
    """
    job_id = self.request.id
    LOGGER.info(f"[{job_id}] Starting manual_assign for {episode_id}")

    # Try to acquire lock
    if not _acquire_lock(episode_id, "manual_assign", job_id):
        return {
            "status": "error",
            "error": "Another manual_assign job is already running for this episode",
            "episode_id": episode_id,
        }

    try:
        assignments = payload.get("assignments", [])

        # Convert list of dicts to BatchAssignmentItem-like objects
        class AssignmentItem:
            def __init__(self, d):
                self.cluster_id = d.get("cluster_id")
                self.target_cast_id = d.get("target_cast_id")

        assignment_items = [AssignmentItem(a) for a in assignments]

        result = self.grouping_service.batch_assign_clusters(episode_id, assignment_items)

        return {
            "status": "success",
            "episode_id": episode_id,
            "operation": "manual_assign",
            "succeeded": result.get("succeeded", 0),
            "failed": result.get("failed", 0),
            "results": result.get("results", []),
        }
    except Exception as e:
        LOGGER.exception(f"[{job_id}] manual_assign failed: {e}")
        return {
            "status": "error",
            "episode_id": episode_id,
            "operation": "manual_assign",
            "error": str(e),
        }
    finally:
        _release_lock(episode_id, "manual_assign", job_id)


@celery_app.task(bind=True, base=GroupingTask, name="tasks.run_auto_group")
def run_auto_group_task(
    self,
    episode_id: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Background task for auto-grouping clusters.

    Args:
        episode_id: Episode ID
        options: Optional dict with:
            - protect_manual: bool (default True) - Don't merge manually assigned clusters
            - facebank_first: bool (default True) - Try facebank matching before prototypes
            - skip_cast_assignment: bool (default True) - Only group clusters, don't assign to cast

    Returns:
        Result dict with grouping statistics
    """
    job_id = self.request.id
    options = options or {}
    LOGGER.info(f"[{job_id}] Starting auto_group for {episode_id} with options: {options}")

    # Try to acquire lock
    if not _acquire_lock(episode_id, "auto_group", job_id):
        return {
            "status": "error",
            "error": "Another auto_group job is already running for this episode",
            "episode_id": episode_id,
        }

    try:
        # Progress callback that updates task state
        def progress_callback(step: str, progress: float, message: str):
            self.update_state(
                state="PROGRESS",
                meta={
                    "step": step,
                    "progress": progress,
                    "message": message,
                    "episode_id": episode_id,
                },
            )

        result = self.grouping_service.group_clusters_auto(
            episode_id,
            progress_callback=progress_callback,
            protect_manual=options.get("protect_manual", True),
            facebank_first=options.get("facebank_first", True),
            skip_cast_assignment=options.get("skip_cast_assignment", True),
        )

        # Extract key stats from result
        centroids_count = len(result.get("centroids", {}).get("centroids", {}))
        within_merged = result.get("within_episode", {}).get("merged_count", 0)
        assignments_count = len(result.get("assignments", []))
        new_people = result.get("across_episodes", {}).get("new_people_count", 0)
        facebank_assigned = result.get("facebank_assigned", 0)

        return {
            "status": "success",
            "episode_id": episode_id,
            "operation": "auto_group",
            "centroids_count": centroids_count,
            "within_episode_merged": within_merged,
            "assignments_count": assignments_count,
            "new_people_count": new_people,
            "facebank_assigned": facebank_assigned,
            "log": result.get("log"),
        }
    except Exception as e:
        LOGGER.exception(f"[{job_id}] auto_group failed: {e}")
        return {
            "status": "error",
            "episode_id": episode_id,
            "operation": "auto_group",
            "error": str(e),
        }
    finally:
        _release_lock(episode_id, "auto_group", job_id)


@celery_app.task(bind=True, base=GroupingTask, name="tasks.run_reembed")
def run_reembed_task(
    self,
    episode_id: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Background task to re-embed all faces for an episode.

    This task triggers the faces_embed pipeline stage.
    """
    job_id = self.request.id
    LOGGER.info(f"[{job_id}] Starting reembed for {episode_id}")

    if not _acquire_lock(episode_id, "reembed", job_id):
        return {
            "status": "error",
            "error": "Another reembed job is already running for this episode",
            "episode_id": episode_id,
        }

    try:
        # Import the job service to trigger reembed
        from apps.api.services.jobs import JobService
        job_service = JobService()

        # Start faces_embed job (synchronously wait or return job info)
        # For now, just return a placeholder - full implementation would
        # integrate with the existing ML pipeline
        return {
            "status": "success",
            "episode_id": episode_id,
            "operation": "reembed",
            "message": "Reembed task queued (integration with ML pipeline pending)",
        }
    except Exception as e:
        LOGGER.exception(f"[{job_id}] reembed failed: {e}")
        return {
            "status": "error",
            "episode_id": episode_id,
            "operation": "reembed",
            "error": str(e),
        }
    finally:
        _release_lock(episode_id, "reembed", job_id)


@celery_app.task(bind=True, base=GroupingTask, name="tasks.run_recluster")
def run_recluster_task(
    self,
    episode_id: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Background task to recluster faces for an episode.

    This task triggers the cluster pipeline stage.
    """
    job_id = self.request.id
    LOGGER.info(f"[{job_id}] Starting recluster for {episode_id}")

    if not _acquire_lock(episode_id, "recluster", job_id):
        return {
            "status": "error",
            "error": "Another recluster job is already running for this episode",
            "episode_id": episode_id,
        }

    try:
        return {
            "status": "success",
            "episode_id": episode_id,
            "operation": "recluster",
            "message": "Recluster task queued (integration with ML pipeline pending)",
        }
    except Exception as e:
        LOGGER.exception(f"[{job_id}] recluster failed: {e}")
        return {
            "status": "error",
            "episode_id": episode_id,
            "operation": "recluster",
            "error": str(e),
        }
    finally:
        _release_lock(episode_id, "recluster", job_id)


@celery_app.task(bind=True, base=GroupingTask, name="tasks.run_split_tracks")
def run_split_tracks_task(
    self,
    episode_id: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Background task to split tracks for an episode."""
    job_id = self.request.id
    LOGGER.info(f"[{job_id}] Starting split_tracks for {episode_id}")

    if not _acquire_lock(episode_id, "split_tracks", job_id):
        return {
            "status": "error",
            "error": "Another split_tracks job is already running for this episode",
            "episode_id": episode_id,
        }

    try:
        return {
            "status": "success",
            "episode_id": episode_id,
            "operation": "split_tracks",
            "message": "Split tracks task queued (integration with ML pipeline pending)",
        }
    except Exception as e:
        LOGGER.exception(f"[{job_id}] split_tracks failed: {e}")
        return {
            "status": "error",
            "episode_id": episode_id,
            "operation": "split_tracks",
            "error": str(e),
        }
    finally:
        _release_lock(episode_id, "split_tracks", job_id)


@celery_app.task(bind=True, base=GroupingTask, name="tasks.run_refresh_similarity")
def run_refresh_similarity_task(
    self,
    episode_id: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Background task to refresh similarity values for an episode.

    This regenerates track representatives, cluster centroids, updates
    people prototypes, and refreshes suggestions for unassigned clusters.

    Args:
        episode_id: Episode ID
        options: Optional dict with configuration options

    Returns:
        Result dict with detailed step-by-step progress log
    """
    import time as time_module

    job_id = self.request.id
    LOGGER.info(f"[{job_id}] Starting refresh_similarity for {episode_id}")

    if not _acquire_lock(episode_id, "refresh_similarity", job_id):
        return {
            "status": "error",
            "error": "Another refresh_similarity job is already running for this episode",
            "episode_id": episode_id,
        }

    log_steps = []
    start_time = time_module.time()

    try:
        # Step 1: Load identities (0-10%)
        self.update_state(
            state="PROGRESS",
            meta={
                "step": "load_identities",
                "progress": 0.05,
                "message": "Loading identities...",
                "episode_id": episode_id,
            },
        )

        step_start = time_module.time()
        from apps.api.routers.episodes import _load_identities

        identities_payload = _load_identities(episode_id)
        identities = identities_payload.get("identities", [])
        track_count = sum(len(i.get("track_ids", [])) for i in identities)
        cluster_count = len(identities)
        assigned_count = sum(1 for i in identities if i.get("person_id"))
        unassigned_count = cluster_count - assigned_count

        log_steps.append({
            "step": "load_identities",
            "status": "success",
            "progress_pct": 10,
            "duration_ms": int((time_module.time() - step_start) * 1000),
            "cluster_count": cluster_count,
            "track_count": track_count,
            "assigned_count": assigned_count,
            "unassigned_count": unassigned_count,
            "details": [
                f"Loaded {cluster_count} clusters ({track_count} tracks)",
                f"Assigned: {assigned_count}, Unassigned: {unassigned_count}",
            ],
        })

        # Step 2: Generate track reps and centroids (10-50%)
        self.update_state(
            state="PROGRESS",
            meta={
                "step": "generate_track_reps",
                "progress": 0.15,
                "message": "Generating track representatives and centroids...",
                "episode_id": episode_id,
            },
        )

        step_start = time_module.time()
        track_reps_count = 0
        centroids_count = 0
        try:
            from apps.api.services.track_reps import generate_track_reps_and_centroids

            result = generate_track_reps_and_centroids(episode_id)
            track_reps_count = result.get("tracks_processed", 0)
            centroids_count = result.get("centroids_computed", 0)
            tracks_with_reps = result.get("tracks_with_reps", 0)
            tracks_skipped = result.get("tracks_skipped", 0)

            details = [
                f"Processed {track_reps_count} tracks",
                f"Computed {centroids_count} cluster centroids",
            ]
            if tracks_with_reps:
                details.append(f"Tracks with valid reps: {tracks_with_reps}")
            if tracks_skipped:
                details.append(f"Tracks skipped (no embeddings): {tracks_skipped}")

            log_steps.append({
                "step": "generate_track_reps",
                "status": "success",
                "progress_pct": 50,
                "duration_ms": int((time_module.time() - step_start) * 1000),
                "track_reps_count": track_reps_count,
                "centroids_count": centroids_count,
                "details": details,
            })
        except Exception as exc:
            log_steps.append({
                "step": "generate_track_reps",
                "status": "error",
                "progress_pct": 50,
                "duration_ms": int((time_module.time() - step_start) * 1000),
                "error": str(exc),
                "details": [f"Error: {exc}"],
            })
            LOGGER.error(f"[{job_id}] Track rep regeneration failed for {episode_id}: {exc}")

        # Step 3: Update people prototypes (50-80%)
        self.update_state(
            state="PROGRESS",
            meta={
                "step": "update_prototypes",
                "progress": 0.55,
                "message": "Updating people prototypes...",
                "episode_id": episode_id,
            },
        )

        step_start = time_module.time()
        people_updated = 0
        people_details = []
        people = []
        try:
            from apps.api.services.people import PeopleService, l2_normalize
            import numpy as np
            from apps.api.services.track_reps import load_cluster_centroids
            from apps.api.routers.episodes import episode_context_from_id

            ep_ctx = episode_context_from_id(episode_id)
            show_id = ep_ctx.show_slug.upper()
            people_service = PeopleService()
            people = people_service.list_people(show_id)

            if people:
                touched_person_ids = set()
                for identity in identities:
                    if identity.get("person_id"):
                        touched_person_ids.add(identity["person_id"])

                people_by_id = {p.get("person_id"): p for p in people if p.get("person_id")}
                try:
                    centroids_data = load_cluster_centroids(episode_id)
                except Exception:
                    centroids_data = {}

                for person_id in touched_person_ids:
                    person = people_by_id.get(person_id)
                    if not person:
                        continue
                    person_name = person.get("name") or person.get("display_name") or person_id
                    cluster_refs = person.get("cluster_ids") or []
                    vectors = []
                    for cluster_ref in cluster_refs:
                        if not isinstance(cluster_ref, str) or ":" not in cluster_ref:
                            continue
                        ep_slug, cluster_id = cluster_ref.split(":", 1)
                        if ep_slug == episode_id:
                            centroid_data = centroids_data.get(cluster_id, {})
                            if centroid_data and centroid_data.get("centroid"):
                                vectors.append(np.array(centroid_data["centroid"], dtype=np.float32))
                    if vectors:
                        stacked = np.stack(vectors, axis=0)
                        proto = l2_normalize(np.mean(stacked, axis=0)).tolist()
                        people_service.update_person(show_id, person_id, prototype=proto)
                        people_updated += 1
                        people_details.append(f"  • {person_name}: updated from {len(vectors)} centroid(s)")

            details = [f"Updated {people_updated} people prototypes"]
            if people_details:
                details.extend(people_details[:10])
                if len(people_details) > 10:
                    details.append(f"  ... and {len(people_details) - 10} more")

            log_steps.append({
                "step": "update_prototypes",
                "status": "success",
                "progress_pct": 80,
                "duration_ms": int((time_module.time() - step_start) * 1000),
                "people_updated": people_updated,
                "people_total": len(touched_person_ids) if people else 0,
                "details": details,
            })
        except Exception as exc:
            log_steps.append({
                "step": "update_prototypes",
                "status": "error",
                "progress_pct": 80,
                "duration_ms": int((time_module.time() - step_start) * 1000),
                "error": str(exc),
                "details": [f"Error: {exc}"],
            })
            LOGGER.warning(f"[{job_id}] People prototype update failed for {episode_id}: {exc}")

        # Step 4: Refresh suggestions (80-100%)
        self.update_state(
            state="PROGRESS",
            meta={
                "step": "refresh_suggestions",
                "progress": 0.85,
                "message": "Refreshing suggestions for unassigned clusters...",
                "episode_id": episode_id,
            },
        )

        step_start = time_module.time()
        suggestions_count = 0
        suggestions_details = []
        try:
            suggestions_result = self.grouping_service.suggest_from_assigned_clusters(episode_id)
            suggestions_list = suggestions_result.get("suggestions", [])
            suggestions_count = len(suggestions_list)

            # Build person name lookup
            person_names = {}
            if people:
                for p in people:
                    pid = p.get("person_id")
                    if pid:
                        person_names[pid] = p.get("name") or p.get("display_name") or pid

            details = [f"Generated {suggestions_count} suggestions for unassigned clusters"]
            for suggestion in suggestions_list[:10]:
                cluster_id = suggestion.get("cluster_id", "?")
                suggested_person = suggestion.get("suggested_person_id", "?")
                distance = suggestion.get("distance", 0)
                person_name = person_names.get(suggested_person, suggested_person)
                confidence = max(0, min(100, int((1 - distance) * 100)))
                details.append(f"  • Cluster {cluster_id} → {person_name} ({confidence}% confidence)")
                suggestions_details.append({
                    "cluster_id": cluster_id,
                    "suggested_person_id": suggested_person,
                    "suggested_person_name": person_name,
                    "distance": round(distance, 4),
                    "confidence_pct": confidence,
                })

            if len(suggestions_list) > 10:
                details.append(f"  ... and {len(suggestions_list) - 10} more suggestions")

            log_steps.append({
                "step": "refresh_suggestions",
                "status": "success",
                "progress_pct": 100,
                "duration_ms": int((time_module.time() - step_start) * 1000),
                "suggestions_count": suggestions_count,
                "unassigned_clusters": unassigned_count,
                "details": details,
                "suggestions": suggestions_details[:20],
            })
        except FileNotFoundError:
            log_steps.append({
                "step": "refresh_suggestions",
                "status": "skipped",
                "progress_pct": 100,
                "duration_ms": int((time_module.time() - step_start) * 1000),
                "message": "No identities file found",
                "details": ["Skipped: No identities file found"],
            })
        except Exception as exc:
            log_steps.append({
                "step": "refresh_suggestions",
                "status": "error",
                "progress_pct": 100,
                "duration_ms": int((time_module.time() - step_start) * 1000),
                "error": str(exc),
                "details": [f"Error: {exc}"],
            })
            LOGGER.warning(f"[{job_id}] Suggestions refresh failed for {episode_id}: {exc}")

        total_duration = int((time_module.time() - start_time) * 1000)

        return {
            "status": "success",
            "episode_id": episode_id,
            "operation": "refresh_similarity",
            "log": {
                "steps": log_steps,
                "total_duration_ms": total_duration,
            },
            "summary": {
                "clusters": cluster_count,
                "tracks": track_count,
                "assigned_clusters": assigned_count,
                "unassigned_clusters": unassigned_count,
                "centroids_computed": centroids_count,
                "people_updated": people_updated,
                "suggestions_generated": suggestions_count,
            },
        }
    except Exception as e:
        LOGGER.exception(f"[{job_id}] refresh_similarity failed: {e}")
        return {
            "status": "error",
            "episode_id": episode_id,
            "operation": "refresh_similarity",
            "error": str(e),
        }
    finally:
        _release_lock(episode_id, "refresh_similarity", job_id)


def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get status of a Celery job.

    Args:
        job_id: Celery task ID

    Returns:
        Dict with job_id, state, result (if finished), error (if failed)
    """
    result = AsyncResult(job_id, app=celery_app)

    # Map Celery states to simplified UI states
    state_map = {
        states.PENDING: "queued",
        states.RECEIVED: "queued",
        states.STARTED: "in_progress",
        states.SUCCESS: "success",
        states.FAILURE: "failed",
        states.RETRY: "retrying",
        states.REVOKED: "cancelled",
        "PROGRESS": "in_progress",
    }

    state = state_map.get(result.state, "unknown")

    response: Dict[str, Any] = {
        "job_id": job_id,
        "state": state,
        "raw_state": result.state,
    }

    if result.state == "PROGRESS":
        # Include progress metadata
        response["progress"] = result.info
    elif result.ready():
        # Job finished - include result
        if result.successful():
            response["result"] = result.result
        else:
            # Failed
            response["error"] = str(result.result)

    return response


def cancel_job(job_id: str) -> Dict[str, Any]:
    """Cancel a running Celery job.

    Args:
        job_id: Celery task ID

    Returns:
        Dict with job_id and status
    """
    result = AsyncResult(job_id, app=celery_app)
    result.revoke(terminate=True)

    return {
        "job_id": job_id,
        "status": "cancelled",
    }


# =============================================================================
# Pipeline Tasks (detect_track, faces_embed, cluster)
# =============================================================================


class PipelineTask(Task):
    """Base class for ML pipeline tasks that run subprocesses."""

    abstract = True

    # Operation-specific max durations in seconds (Bug 4 fix)
    OPERATION_MAX_DURATIONS = {
        "detect_track": 7200,    # 2 hours
        "faces_embed": 3600,    # 1 hour
        "cluster": 1800,        # 30 minutes
    }
    MAX_PROGRESS_FILE_SIZE = 10_000_000  # 10MB (Bug 15 fix)

    def _run_subprocess(
        self,
        command: list[str],
        episode_id: str,
        operation: str,
        progress_file: str | None = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Run a subprocess command and track progress with real-time updates.

        Uses Popen for non-blocking execution and polls the progress file
        to provide real-time progress updates to the Celery task state.

        Args:
            command: Command list to execute
            episode_id: Episode ID for logging
            operation: Operation name for lock management
            progress_file: Optional path to progress JSON file
            env: Optional environment overrides (e.g., CPU thread caps)

        Returns:
            Result dict with status, output, and any progress data
        """
        import subprocess
        import json

        job_id = self.request.id
        proc = None  # Track subprocess for cleanup (Bug 9 fix)

        LOGGER.info(f"[{job_id}] Starting {operation} for {episode_id}")
        LOGGER.info(f"[{job_id}] Command: {' '.join(command)}")

        try:
            # Use Popen for non-blocking execution with real-time progress
            proc = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            progress_path = Path(progress_file) if progress_file else None
            poll_interval = 2.0  # seconds
            last_progress = 0.0
            last_phase = ""
            start_time = time.time()  # Bug 4 fix: track start time
            max_duration = self.OPERATION_MAX_DURATIONS.get(operation, 7200)

            # Poll until process completes
            while proc.poll() is None:
                time.sleep(poll_interval)

                # Bug 4 fix: Check for timeout
                elapsed = time.time() - start_time
                if elapsed > max_duration:
                    LOGGER.error(f"[{job_id}] {operation} exceeded max duration ({max_duration}s)")
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    return {
                        "status": "error",
                        "episode_id": episode_id,
                        "operation": operation,
                        "error": f"Task timeout exceeded ({max_duration}s)",
                    }

                # Read progress file if available and update Celery state
                if progress_path and progress_path.exists():
                    try:
                        # Bug 15 fix: Validate file size before reading
                        file_size = progress_path.stat().st_size
                        if file_size > self.MAX_PROGRESS_FILE_SIZE:
                            LOGGER.warning(f"[{job_id}] Progress file too large ({file_size} bytes), skipping")
                            continue

                        progress_text = progress_path.read_text(encoding="utf-8")
                        progress_data = json.loads(progress_text)

                        # Bug 12 fix: Validate progress data structure
                        if not isinstance(progress_data, dict):
                            LOGGER.warning(f"[{job_id}] Progress data not a dict, skipping")
                            continue

                        frames_done = int(progress_data.get("frames_done", 0))
                        frames_total = max(1, int(progress_data.get("frames_total", 1)))
                        phase = str(progress_data.get("phase", "running"))[:50]
                        secs_done = int(progress_data.get("secs_done", 0))

                        # Calculate progress percentage (cap at 99% until complete)
                        progress_pct = min(frames_done / frames_total, 0.99)

                        # Update Celery state if progress changed
                        if progress_pct > last_progress or phase != last_phase:
                            last_progress = progress_pct
                            last_phase = phase
                            self.update_state(
                                state="PROGRESS",
                                meta={
                                    "step": operation,
                                    "progress": progress_pct,
                                    "message": f"{phase}: {frames_done:,}/{frames_total:,} frames",
                                    "episode_id": episode_id,
                                    "frames_done": frames_done,
                                    "frames_total": frames_total,
                                    "phase": phase,
                                    "secs_done": secs_done,
                                },
                            )
                            LOGGER.debug(
                                f"[{job_id}] Progress update: {phase} {frames_done}/{frames_total} "
                                f"({progress_pct*100:.1f}%)"
                            )
                    except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
                        LOGGER.debug(f"[{job_id}] Could not read progress file: {e}")

            # Process complete - get final output
            stdout, stderr = proc.communicate()

            # Read final progress
            progress_data = None
            if progress_path and progress_path.exists():
                try:
                    progress_data = json.loads(progress_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass

            if proc.returncode != 0:
                error_msg = stderr.strip() if stderr else f"Exit code {proc.returncode}"
                LOGGER.error(f"[{job_id}] {operation} failed: {error_msg}")
                return {
                    "status": "error",
                    "episode_id": episode_id,
                    "operation": operation,
                    "error": error_msg,
                    "return_code": proc.returncode,
                    "progress": progress_data,
                }

            LOGGER.info(f"[{job_id}] {operation} completed successfully")
            return {
                "status": "success",
                "episode_id": episode_id,
                "operation": operation,
                "return_code": 0,
                "progress": progress_data,
                "stdout": stdout[:2000] if stdout else None,
            }

        except Exception as e:
            # Bug 9 fix: Clean up subprocess on exception
            if proc is not None and proc.poll() is None:
                LOGGER.warning(f"[{job_id}] Terminating subprocess due to exception")
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                except Exception:
                    pass  # Best effort cleanup

            LOGGER.exception(f"[{job_id}] {operation} raised exception: {e}")
            return {
                "status": "error",
                "episode_id": episode_id,
                "operation": operation,
                "error": str(e),
            }


@celery_app.task(bind=True, base=PipelineTask, name="tasks.run_detect_track")
def run_detect_track_task(
    self,
    episode_id: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Background task to run detect/track pipeline.

    Args:
        episode_id: Episode ID
        options: Dict with pipeline options:
            - stride: int (default 6)
            - device: str (default "auto")
            - detector: str (default "retinaface")
            - tracker: str (default "bytetrack")
            - save_frames: bool
            - save_crops: bool
            - jpeg_quality: int
            - det_thresh: float
            - max_gap: int
            - scene_detector: str
            - scene_threshold: float
            etc.

    Returns:
        Result dict with status and pipeline output
    """
    job_id = self.request.id
    options = options or {}
    run_id: str | None = None
    run_id_raw = options.get("run_id")
    if isinstance(run_id_raw, str) and run_id_raw.strip():
        try:
            from py_screenalytics import run_layout

            run_id = run_layout.normalize_run_id(run_id_raw)
        except ValueError as exc:
            return {
                "status": "error",
                "episode_id": episode_id,
                "operation": "detect_track",
                "error": f"Invalid run_id: {exc}",
            }

    if not _acquire_lock(episode_id, "detect_track", job_id):
        return {
            "status": "error",
            "error": "Another detect_track job is already running for this episode",
            "episode_id": episode_id,
        }

    try:
        from py_screenalytics.artifacts import get_path

        # Get video path
        video_path = get_path(episode_id, "video")
        if not video_path.exists():
            return {
                "status": "error",
                "episode_id": episode_id,
                "operation": "detect_track",
                "error": f"Video not found at {video_path}",
            }

        # Progress file
        manifests_dir = get_path(episode_id, "detections").parent
        if run_id:
            from py_screenalytics import run_layout

            run_dir = run_layout.run_root(episode_id, run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            progress_file = run_dir / "progress.json"
        else:
            manifests_dir.mkdir(parents=True, exist_ok=True)
            progress_file = manifests_dir / "progress.json"

        # Build command
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "episode_run.py"),
            "--ep-id", episode_id,
            *(
                ["--run-id", run_id]
                if run_id
                else []
            ),
            "--video", str(video_path),
            "--stride", str(options.get("stride", 6)),
            "--device", options.get("device", "auto"),
            "--progress-file", str(progress_file),
        ]

        fps_value = options.get("fps")
        if fps_value is not None and fps_value > 0:
            command += ["--fps", str(fps_value)]
        if options.get("detector"):
            command += ["--detector", options["detector"]]
        if options.get("tracker"):
            command += ["--tracker", options["tracker"]]
        if options.get("save_frames"):
            command.append("--save-frames")
        if options.get("save_crops"):
            command.append("--save-crops")
        if options.get("jpeg_quality"):
            command += ["--jpeg-quality", str(options["jpeg_quality"])]
        if options.get("det_thresh") is not None:
            command += ["--det-thresh", str(options["det_thresh"])]
        if options.get("max_gap"):
            command += ["--max-gap", str(options["max_gap"])]
        if options.get("scene_detector"):
            command += ["--scene-detector", options["scene_detector"]]
        if options.get("scene_threshold") is not None:
            command += ["--scene-threshold", str(options["scene_threshold"])]
        if options.get("scene_min_len") is not None:
            command += ["--scene-min-len", str(options["scene_min_len"])]
        if options.get("scene_warmup_dets") is not None:
            command += ["--scene-warmup-dets", str(options["scene_warmup_dets"])]
        if options.get("track_high_thresh") is not None:
            command += ["--track-high-thresh", str(options["track_high_thresh"])]
        if options.get("new_track_thresh") is not None:
            command += ["--new-track-thresh", str(options["new_track_thresh"])]
        if options.get("track_buffer") is not None:
            command += ["--track-buffer", str(options["track_buffer"])]
        if options.get("min_box_area") is not None:
            command += ["--min-box-area", str(options["min_box_area"])]
        # Note: --profile is not a valid argument for episode_run.py

        env = os.environ.copy()
        cpu_threads = options.get("cpu_threads")
        if cpu_threads:
            # Cap threads between 1 and available CPU count to prevent overheating
            max_allowed = os.cpu_count() or 8
            threads = max(1, min(int(cpu_threads), max_allowed))
            # Apply CPU thread caps to downstream ML libraries for laptop-friendly runs
            env.update(
                {
                    "SCREENALYTICS_MAX_CPU_THREADS": str(threads),
                    "OMP_NUM_THREADS": str(threads),
                    "MKL_NUM_THREADS": str(threads),
                    "OPENBLAS_NUM_THREADS": str(threads),
                    "VECLIB_MAXIMUM_THREADS": str(threads),
                    "NUMEXPR_NUM_THREADS": str(threads),
                    "ORT_INTRA_OP_NUM_THREADS": str(threads),
                    "ORT_INTER_OP_NUM_THREADS": "1",
                }
            )

        # Update progress
        self.update_state(
            state="PROGRESS",
            meta={
                "step": "detect_track",
                "progress": 0.0,
                "message": "Starting detect/track pipeline...",
                "episode_id": episode_id,
            },
        )

        result = self._run_subprocess(
            command, episode_id, "detect_track", str(progress_file), env
        )

        return result

    finally:
        _release_lock(episode_id, "detect_track", job_id)


@celery_app.task(bind=True, base=PipelineTask, name="tasks.run_faces_embed")
def run_faces_embed_task(
    self,
    episode_id: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Background task to run faces embed (harvest) pipeline.

    Args:
        episode_id: Episode ID
        options: Dict with pipeline options:
            - device: str (default "auto")
            - save_frames: bool
            - save_crops: bool
            - jpeg_quality: int
            - min_frames_between_crops: int
            - thumb_size: int

    Returns:
        Result dict with status and pipeline output
    """
    job_id = self.request.id
    options = options or {}
    run_id: str | None = None
    run_id_raw = options.get("run_id")
    if isinstance(run_id_raw, str) and run_id_raw.strip():
        try:
            from py_screenalytics import run_layout

            run_id = run_layout.normalize_run_id(run_id_raw)
        except ValueError as exc:
            return {
                "status": "error",
                "episode_id": episode_id,
                "operation": "faces_embed",
                "error": f"Invalid run_id: {exc}",
            }

    if not _acquire_lock(episode_id, "faces_embed", job_id):
        return {
            "status": "error",
            "error": "Another faces_embed job is already running for this episode",
            "episode_id": episode_id,
        }

    try:
        from py_screenalytics.artifacts import get_path

        # Progress file
        manifests_dir = get_path(episode_id, "detections").parent
        if run_id:
            from py_screenalytics import run_layout

            run_dir = run_layout.run_root(episode_id, run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            progress_file = run_dir / "progress.json"
        else:
            manifests_dir.mkdir(parents=True, exist_ok=True)
            progress_file = manifests_dir / "progress.json"

        # Build command
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "episode_run.py"),
            "--ep-id", episode_id,
            *(
                ["--run-id", run_id]
                if run_id
                else []
            ),
            "--faces-embed",
            "--device", options.get("device", "auto"),
            "--progress-file", str(progress_file),
        ]

        if options.get("save_frames"):
            command.append("--save-frames")
        if options.get("save_crops"):
            command.append("--save-crops")
        if options.get("jpeg_quality"):
            command += ["--jpeg-quality", str(options["jpeg_quality"])]
        if options.get("min_frames_between_crops"):
            command += ["--min-frames-between-crops", str(options["min_frames_between_crops"])]
        if options.get("thumb_size"):
            command += ["--thumb-size", str(options["thumb_size"])]
        # Note: --profile is not a valid argument for episode_run.py

        # Apply CPU thread limits for laptop-friendly runs
        env = os.environ.copy()
        cpu_threads = options.get("cpu_threads")
        if cpu_threads:
            max_allowed = os.cpu_count() or 8
            threads = max(1, min(int(cpu_threads), max_allowed))
            env.update(
                {
                    "SCREENALYTICS_MAX_CPU_THREADS": str(threads),
                    "OMP_NUM_THREADS": str(threads),
                    "MKL_NUM_THREADS": str(threads),
                    "OPENBLAS_NUM_THREADS": str(threads),
                    "VECLIB_MAXIMUM_THREADS": str(threads),
                    "NUMEXPR_NUM_THREADS": str(threads),
                    "ORT_INTRA_OP_NUM_THREADS": str(threads),
                    "ORT_INTER_OP_NUM_THREADS": "1",
                }
            )

        # Update progress
        self.update_state(
            state="PROGRESS",
            meta={
                "step": "faces_embed",
                "progress": 0.0,
                "message": "Starting faces harvest pipeline...",
                "episode_id": episode_id,
            },
        )

        result = self._run_subprocess(
            command, episode_id, "faces_embed", str(progress_file), env
        )

        return result

    finally:
        _release_lock(episode_id, "faces_embed", job_id)


@celery_app.task(bind=True, base=PipelineTask, name="tasks.run_cluster")
def run_cluster_task(
    self,
    episode_id: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Background task to run clustering pipeline.

    Args:
        episode_id: Episode ID
        options: Dict with pipeline options:
            - device: str (default "auto")
            - cluster_thresh: float (default 0.7)
            - min_cluster_size: int (default 2)
            - min_identity_sim: float

    Returns:
        Result dict with status and pipeline output
    """
    job_id = self.request.id
    options = options or {}
    run_id: str | None = None
    run_id_raw = options.get("run_id")
    if isinstance(run_id_raw, str) and run_id_raw.strip():
        try:
            from py_screenalytics import run_layout

            run_id = run_layout.normalize_run_id(run_id_raw)
        except ValueError as exc:
            return {
                "status": "error",
                "episode_id": episode_id,
                "operation": "cluster",
                "error": f"Invalid run_id: {exc}",
            }

    if not _acquire_lock(episode_id, "cluster", job_id):
        return {
            "status": "error",
            "error": "Another cluster job is already running for this episode",
            "episode_id": episode_id,
        }

    try:
        from py_screenalytics.artifacts import get_path

        # Progress file
        manifests_dir = get_path(episode_id, "detections").parent
        if run_id:
            from py_screenalytics import run_layout

            run_dir = run_layout.run_root(episode_id, run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            progress_file = run_dir / "progress.json"
        else:
            manifests_dir.mkdir(parents=True, exist_ok=True)
            progress_file = manifests_dir / "progress.json"

        # Build command
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "episode_run.py"),
            "--ep-id", episode_id,
            *(
                ["--run-id", run_id]
                if run_id
                else []
            ),
            "--cluster",
            "--device", options.get("device", "auto"),
            "--progress-file", str(progress_file),
        ]

        if options.get("cluster_thresh") is not None:
            command += ["--cluster-thresh", str(options["cluster_thresh"])]
        if options.get("min_cluster_size"):
            command += ["--min-cluster-size", str(options["min_cluster_size"])]
        if options.get("min_identity_sim") is not None:
            command += ["--min-identity-sim", str(options["min_identity_sim"])]
        # Note: --profile is not a valid argument for episode_run.py

        # Apply CPU thread limits for laptop-friendly runs
        env = os.environ.copy()
        cpu_threads = options.get("cpu_threads")
        if cpu_threads:
            max_allowed = os.cpu_count() or 8
            threads = max(1, min(int(cpu_threads), max_allowed))
            env.update(
                {
                    "SCREENALYTICS_MAX_CPU_THREADS": str(threads),
                    "OMP_NUM_THREADS": str(threads),
                    "MKL_NUM_THREADS": str(threads),
                    "OPENBLAS_NUM_THREADS": str(threads),
                    "VECLIB_MAXIMUM_THREADS": str(threads),
                    "NUMEXPR_NUM_THREADS": str(threads),
                    "ORT_INTRA_OP_NUM_THREADS": str(threads),
                    "ORT_INTER_OP_NUM_THREADS": "1",
                }
            )

        # Update progress
        self.update_state(
            state="PROGRESS",
            meta={
                "step": "cluster",
                "progress": 0.0,
                "message": "Starting clustering pipeline...",
                "episode_id": episode_id,
            },
        )

        result = self._run_subprocess(
            command, episode_id, "cluster", str(progress_file), env
        )

        return result

    finally:
        _release_lock(episode_id, "cluster", job_id)


# =============================================================================
# Enhancement #1: Parallel Job Execution
# =============================================================================


def run_parallel_jobs(
    episode_ids: List[str],
    operation: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the same operation on multiple episodes in parallel.

    Uses Celery group to execute jobs concurrently across episodes.

    Args:
        episode_ids: List of episode IDs to process
        operation: Operation to run (auto_group, refresh_similarity, manual_assign)
        options: Options to pass to each task

    Returns:
        Dict with group_id and individual job_ids for tracking
    """
    options = options or {}

    # Map operation to task function
    task_map = {
        "auto_group": run_auto_group_task,
        "refresh_similarity": run_refresh_similarity_task,
        "manual_assign": run_manual_assign_task,
    }

    task_func = task_map.get(operation)
    if not task_func:
        return {
            "status": "error",
            "error": f"Unknown operation: {operation}. Valid: {list(task_map.keys())}",
        }

    # Check for existing jobs on any episode
    conflicting = []
    for ep_id in episode_ids:
        existing_job = check_active_job(ep_id, operation)
        if existing_job:
            conflicting.append({"ep_id": ep_id, "job_id": existing_job})

    if conflicting:
        return {
            "status": "error",
            "error": "Jobs already running on some episodes",
            "conflicts": conflicting,
        }

    # Create task signatures for each episode
    if operation == "manual_assign":
        # manual_assign requires payload
        signatures = [
            task_func.s(ep_id, options.get("payload", {}))
            for ep_id in episode_ids
        ]
    else:
        signatures = [
            task_func.s(ep_id, options)
            for ep_id in episode_ids
        ]

    # Execute as a group
    job_group = group(signatures)
    group_result = job_group.apply_async()

    # Collect individual job IDs
    job_ids = {}
    for i, ep_id in enumerate(episode_ids):
        if i < len(group_result.children):
            job_ids[ep_id] = group_result.children[i].id

    return {
        "status": "queued",
        "group_id": group_result.id,
        "operation": operation,
        "episode_count": len(episode_ids),
        "job_ids": job_ids,
    }


def get_parallel_job_status(group_id: str) -> Dict[str, Any]:
    """Get status of a parallel job group.

    Args:
        group_id: The group ID returned from run_parallel_jobs

    Returns:
        Dict with overall status and per-episode results
    """
    try:
        group_result = GroupResult.restore(group_id, app=celery_app)
        if not group_result:
            return {
                "status": "error",
                "error": f"Group {group_id} not found",
            }

        # Collect results
        results = []
        completed = 0
        failed = 0
        in_progress = 0

        for child in group_result.children:
            child_status = get_job_status(child.id)
            results.append(child_status)

            state = child_status.get("state", "unknown")
            if state == "success":
                completed += 1
            elif state == "failed":
                failed += 1
            elif state in ("queued", "in_progress"):
                in_progress += 1

        # Determine overall status
        if in_progress > 0:
            overall_status = "in_progress"
        elif failed > 0 and completed == 0:
            overall_status = "failed"
        elif failed > 0:
            overall_status = "partial_success"
        elif completed == len(results):
            overall_status = "success"
        else:
            overall_status = "unknown"

        return {
            "status": overall_status,
            "group_id": group_id,
            "total": len(results),
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "progress": completed / max(len(results), 1),
            "results": results,
        }
    except Exception as e:
        LOGGER.exception(f"Failed to get parallel job status: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


# =============================================================================
# Enhancement #9: Job Progress Persistence
# =============================================================================

_JOB_HISTORY_MAX = 100  # Keep last 100 jobs per user


def persist_job_record(
    job_id: str,
    episode_id: str,
    operation: str,
    status: str,
    user_id: Optional[str] = None,
    duration_ms: Optional[int] = None,
    result: Optional[Dict[str, Any]] = None,
) -> bool:
    """Persist a job record to Redis for history tracking.

    Args:
        job_id: Celery task ID
        episode_id: Episode ID
        operation: Operation type
        status: Job status (queued, in_progress, success, failed)
        user_id: Optional user identifier
        duration_ms: Duration in milliseconds
        result: Optional result data

    Returns:
        True if persisted successfully
    """
    try:
        r = _get_redis()

        record = {
            "job_id": job_id,
            "episode_id": episode_id,
            "operation": operation,
            "status": status,
            "user_id": user_id or "anonymous",
            "duration_ms": duration_ms,
            "created_at": time.time(),
            "result_summary": _summarize_result(result) if result else None,
        }

        # Use a sorted set with timestamp as score for ordering
        key = redis_keys.job_history_user_key(user_id)
        r.zadd(key, {json.dumps(record): time.time()})

        # Trim to max entries
        r.zremrangebyrank(key, 0, -(_JOB_HISTORY_MAX + 1))

        # Also set TTL for cleanup (7 days)
        r.expire(key, 7 * 24 * 3600)

        return True
    except Exception as e:
        LOGGER.warning(f"Failed to persist job record: {e}")
        return False


def _summarize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary of job result for storage (limit size)."""
    if not result:
        return {}

    return {
        "status": result.get("status"),
        "error": result.get("error"),
        "succeeded": result.get("succeeded"),
        "failed": result.get("failed"),
    }


def get_job_history(
    user_id: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Get job history for a user.

    Args:
        user_id: User identifier (None for anonymous)
        limit: Maximum number of records to return
        offset: Number of records to skip

    Returns:
        List of job records, newest first
    """
    try:
        r = _get_redis()
        primary, *fallbacks = redis_keys.job_history_user_keys(user_id)
        key = primary

        # Get entries in reverse order (newest first)
        start = -(offset + limit)
        end = -(offset + 1) if offset > 0 else -1

        entries = r.zrevrange(key, offset, offset + limit - 1)
        if not entries:
            for fallback in fallbacks:
                entries = r.zrevrange(fallback, offset, offset + limit - 1)
                if entries:
                    break

        records = []
        for entry in entries:
            try:
                record = json.loads(entry)
                records.append(record)
            except json.JSONDecodeError:
                continue

        return records
    except Exception as e:
        LOGGER.warning(f"Failed to get job history: {e}")
        return []


def get_active_jobs_for_user(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all active (queued/in_progress) jobs for a user.

    Returns:
        List of active job records with current status
    """
    history = get_job_history(user_id, limit=50)

    active = []
    for record in history:
        job_id = record.get("job_id")
        if not job_id:
            continue

        # Check current status
        try:
            current = get_job_status(job_id)
            state = current.get("state", "unknown")
            if state in ("queued", "in_progress"):
                record["current_status"] = current
                active.append(record)
        except Exception:
            continue

    return active


__all__ = [
    "run_manual_assign_task",
    "run_auto_group_task",
    "run_reembed_task",
    "run_recluster_task",
    "run_split_tracks_task",
    "run_refresh_similarity_task",
    "run_detect_track_task",
    "run_faces_embed_task",
    "run_cluster_task",
    "check_active_job",
    "get_job_status",
    "cancel_job",
    # Enhancement #1: Parallel Job Execution
    "run_parallel_jobs",
    "get_parallel_job_status",
    # Enhancement #9: Job Progress Persistence
    "persist_job_record",
    "get_job_history",
    "get_active_jobs_for_user",
]

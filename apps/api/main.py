from __future__ import annotations

# Apply global CPU limits BEFORE importing any ML libraries or heavy dependencies
# Uses centralized configuration from apps.common.cpu_limits (default: 3 threads = ~300% CPU)
# Override with env var: SCREENALYTICS_MAX_CPU_THREADS=N
from apps.common.cpu_limits import apply_global_cpu_limits
apply_global_cpu_limits()

import asyncio
import logging
import os
from typing import Any, Dict, Tuple

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.api.config import REDIS_URL
from apps.api.errors import install_error_handlers
from apps.api.routers import (
    archive,
    audio,
    cast,
    config,
    diagnostics,
    episodes,
    facebank,
    face_review,
    files,
    grouping,
    identities,
    jobs,
    metadata,
    people,
    roster,
)

app = FastAPI(title="Screenalytics API", version="0.1.0")
install_error_handlers(app)
LOGGER = logging.getLogger(__name__)

ui_origin = os.environ.get("UI_ORIGIN", "http://localhost:8501")
origins = {ui_origin, "http://localhost:8501"}
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)
app.include_router(config.router, tags=["config"])
app.include_router(episodes.router, tags=["episodes"])
app.include_router(identities.router)
app.include_router(roster.router)
app.include_router(cast.router, tags=["cast"])
app.include_router(facebank.router, tags=["facebank"])
app.include_router(files.router, tags=["files"])
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(people.router, tags=["people"])
app.include_router(grouping.router, tags=["grouping"])
app.include_router(face_review.router, tags=["face_review"])
app.include_router(metadata.router)
app.include_router(archive.router, tags=["archive"])
app.include_router(audio.router, tags=["audio"])
app.include_router(audio.edit_router, tags=["audio"])
app.include_router(diagnostics.router, tags=["diagnostics"])

# Celery is optional in local dev; guard the import so /healthz stays alive even if
# celery[redis] is not installed. Expose a 503 stub so callers see a clear error.
celery_import_error: Exception | None = None
celery_router = None
try:
    from apps.api.routers import celery_jobs

    celery_router = celery_jobs.router
except ImportError as exc:
    celery_import_error = exc
    LOGGER.warning("Celery router disabled (dependency missing): %s", exc)

if celery_router is not None:
    app.include_router(celery_router, tags=["celery_jobs"])
else:
    celery_disabled = APIRouter(prefix="/celery_jobs", tags=["celery_jobs"])
    _celery_detail = "Celery routes unavailable: install celery[redis] to enable background workers."
    if celery_import_error:
        _celery_detail = f"{_celery_detail} ({celery_import_error})"

    @celery_disabled.get("", include_in_schema=False)
    def celery_missing_root() -> None:
        raise HTTPException(status_code=503, detail=_celery_detail)

    @celery_disabled.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"], include_in_schema=False)
    def celery_missing(path: str) -> None:  # pragma: no cover - simple stub
        raise HTTPException(status_code=503, detail=_celery_detail)

    app.include_router(celery_disabled)


@app.on_event("startup")
async def _cleanup_stale_jobs() -> None:
    """Clean up stale locks, progress files, and zombie jobs on startup.

    When the API/worker crashes or restarts, locks can be left in Redis/filesystem
    and progress files can show stale "running" state. This clears them so new
    jobs can start fresh.
    """
    import os
    import json
    import subprocess
    from pathlib import Path
    from datetime import datetime, timezone

    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))
    manifests_dir = data_root / "manifests"
    jobs_dir = data_root / "jobs"

    # 1. Mark all in-progress audio_progress.json files as stale
    if manifests_dir.exists():
        for ep_dir in manifests_dir.iterdir():
            if not ep_dir.is_dir():
                continue
            progress_file = ep_dir / "audio_progress.json"
            if progress_file.exists():
                try:
                    data = json.loads(progress_file.read_text(encoding="utf-8"))
                    status = str(data.get("status", "")).lower()
                    # If it was "running" or missing status, mark as stale
                    if status not in {"completed", "complete", "succeeded", "error", "failed", "cancelled", "stale"}:
                        data["status"] = "stale"
                        data["message"] = "Marked stale on API restart"
                        progress_file.write_text(json.dumps(data), encoding="utf-8")
                        LOGGER.info(f"Marked stale progress file: {progress_file}")
                except Exception as e:
                    LOGGER.warning(f"Failed to check progress file {progress_file}: {e}")

    # 2. Clean up stale file-based locks (older than 1 hour or PID dead)
    stale_lock_threshold = 3600  # 1 hour in seconds
    locks_cleaned = 0
    if manifests_dir.exists():
        for ep_dir in manifests_dir.iterdir():
            if not ep_dir.is_dir():
                continue
            for lock_file in ep_dir.glob(".lock_*"):
                try:
                    lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
                    lock_pid = lock_data.get("pid")
                    lock_started = lock_data.get("started_at", "")

                    # Check if lock is stale by time
                    is_stale = False
                    if lock_started:
                        try:
                            started_dt = datetime.fromisoformat(lock_started.replace("Z", "+00:00"))
                            age_seconds = (datetime.now(timezone.utc) - started_dt).total_seconds()
                            if age_seconds > stale_lock_threshold:
                                is_stale = True
                                LOGGER.info(f"[startup] Lock {lock_file.name} is {age_seconds/3600:.1f}h old (stale)")
                        except (ValueError, TypeError, OverflowError) as exc:
                            LOGGER.debug(f"[startup] Failed to parse lock timestamp: {exc}")

                    # Check if PID is dead
                    if lock_pid and not is_stale:
                        try:
                            subprocess.run(["kill", "-0", str(lock_pid)], check=True, capture_output=True)
                            # Process is alive, skip
                            continue
                        except subprocess.CalledProcessError:
                            # Process is dead
                            is_stale = True
                            LOGGER.info(f"[startup] Lock {lock_file.name} PID {lock_pid} is dead (stale)")

                    if is_stale:
                        lock_file.unlink()
                        locks_cleaned += 1
                        LOGGER.info(f"[startup] Removed stale lock: {lock_file}")

                except Exception as e:
                    LOGGER.warning(f"[startup] Failed to check lock {lock_file}: {e}")

    if locks_cleaned > 0:
        LOGGER.info(f"[startup] Cleaned {locks_cleaned} stale file-based locks")

    # 3. Clean up zombie jobs (running state but PID dead)
    zombies_cleaned = 0
    if jobs_dir.exists():
        for job_file in jobs_dir.glob("*.json"):
            try:
                job_data = json.loads(job_file.read_text(encoding="utf-8"))
                if job_data.get("state") != "running":
                    continue

                job_pid = job_data.get("pid")
                if not job_pid:
                    continue

                # Check if PID is dead
                try:
                    subprocess.run(["kill", "-0", str(job_pid)], check=True, capture_output=True)
                    # Process is alive, skip
                    continue
                except subprocess.CalledProcessError:
                    # Process is dead - mark as failed
                    job_data["state"] = "failed"
                    job_data["error"] = "Marked as failed on startup - process no longer exists"
                    job_data["ended_at"] = datetime.now(timezone.utc).isoformat()
                    job_file.write_text(json.dumps(job_data, indent=2), encoding="utf-8")
                    zombies_cleaned += 1
                    LOGGER.info(f"[startup] Marked zombie job as failed: {job_file.name}")

            except Exception as e:
                LOGGER.warning(f"[startup] Failed to check job {job_file}: {e}")

    if zombies_cleaned > 0:
        LOGGER.info(f"[startup] Cleaned {zombies_cleaned} zombie jobs")

    # 4. Clear audio pipeline locks from Redis
    try:
        from apps.api.tasks import _get_redis
        from apps.api.services import redis_keys

        r = _get_redis()
        cleared = 0
        for lock_pattern in redis_keys.job_lock_patterns("audio_pipeline"):
            cursor = 0
            while True:
                cursor, keys = r.scan(cursor=cursor, match=lock_pattern, count=100)
                for key in keys:
                    r.delete(key)
                    cleared += 1
                if cursor == 0:
                    break
        if cleared > 0:
            LOGGER.info(f"Cleared {cleared} stale audio pipeline locks from Redis")
    except Exception as e:
        LOGGER.warning(f"Failed to clear stale locks from Redis: {e}")


@app.on_event("startup")
async def _initialize_safeguards() -> None:
    """Initialize manifest safeguards: checksums directory and backup rotation.

    This enables the D31 (checksums) and backup rotation features.
    """
    import os
    from pathlib import Path

    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))
    manifests_dir = data_root / "manifests"
    checksums_dir = data_root / "checksums"

    # 1. Create checksums directory to enable D31
    try:
        checksums_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"[startup] Checksums directory ready: {checksums_dir}")
    except Exception as e:
        LOGGER.warning(f"[startup] Failed to create checksums directory: {e}")

    # 2. Rotate old backup files (keep max 3 per manifest)
    max_backups = 3
    backups_deleted = 0

    if manifests_dir.exists():
        from apps.api.services.manifests import rotate_backups

        for ep_dir in manifests_dir.iterdir():
            if not ep_dir.is_dir():
                continue

            # Find all manifest files and rotate their backups
            for manifest_file in ep_dir.iterdir():
                if manifest_file.is_file() and manifest_file.suffix in (".json", ".jsonl"):
                    try:
                        deleted = rotate_backups(manifest_file, max_backups)
                        backups_deleted += deleted
                    except Exception as e:
                        LOGGER.warning(f"[startup] Failed to rotate backups for {manifest_file}: {e}")

    if backups_deleted > 0:
        LOGGER.info(f"[startup] Cleaned {backups_deleted} old backup files")


@app.on_event("startup")
async def _warmup_retinaface() -> None:
    """Launch RetinaFace warmup in background to avoid blocking health checks."""

    async def warmup():
        """Run the blocking warmup in a thread pool to avoid blocking event loop."""
        # Lazy import to avoid pulling in heavy ML deps at API startup
        from tools import episode_run

        loop = asyncio.get_running_loop()
        ready, detail, _ = await loop.run_in_executor(
            None, episode_run.ensure_retinaface_ready, "auto"
        )
        if not ready:
            LOGGER.warning("RetinaFace detector not ready: %s", detail)
        else:
            LOGGER.info("RetinaFace detector warmed up successfully")

    # Launch warmup in background without awaiting it
    # This allows the server to immediately respond to health checks
    asyncio.create_task(warmup())
    LOGGER.info("RetinaFace warmup launched in background")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/healthz")
def healthz() -> JSONResponse:
    """Lightweight health check for load balancers and uptime probes."""

    redis_status, redis_detail = _check_redis()
    storage_status, storage_detail, storage_backend = _check_storage()
    db_status, db_detail = _check_db()

    # Consider the service healthy when required dependencies respond quickly
    dependencies_ok = [
        redis_status == "ok",
        storage_status == "ok",
        db_status in {"ok", "unconfigured"},
    ]
    overall_ok = all(dependencies_ok)
    status_code = 200 if overall_ok else 503

    payload: Dict[str, object] = {
        "status": "ok" if overall_ok else "error",
        "version": app.version or "unknown",
        "redis": redis_status,
        "storage": storage_status,
        "db": db_status,
    }
    if storage_backend:
        payload["storage_backend"] = storage_backend

    details: Dict[str, str] = {}
    if redis_detail:
        details["redis"] = redis_detail
    if storage_detail:
        details["storage"] = storage_detail
    if db_detail:
        details["db"] = db_detail
    if details:
        payload["details"] = details

    return JSONResponse(status_code=status_code, content=payload)


@app.get("/storage/status")
def storage_status() -> Dict[str, Any]:
    """Get detailed storage configuration and status.

    Returns information about the current storage backend, S3 configuration,
    and overall status useful for the Settings UI.
    """
    result: Dict[str, Any] = {
        "backend": os.environ.get("STORAGE_BACKEND", "s3"),
        "s3_enabled": False,
        "bucket": "",
        "region": "",
        "endpoint": "",
        "write_enabled": False,
        "status": "unknown",
    }

    try:
        from apps.api.routers import episodes as episodes_router

        storage = getattr(episodes_router, "STORAGE", None)
    except Exception:
        result["status"] = "error"
        result["error"] = "Could not import storage service"
        return result

    if storage is None:
        result["status"] = "error"
        result["error"] = "Storage service not initialized"
        return result

    # Extract storage configuration
    result["backend"] = getattr(storage, "backend", "unknown")
    result["s3_enabled"] = getattr(storage, "s3_enabled", lambda: False)()
    result["bucket"] = getattr(storage, "bucket", "") or ""
    result["write_enabled"] = getattr(storage, "write_enabled", False)

    # Get endpoint/region from environment
    result["endpoint"] = os.environ.get("MINIO_ENDPOINT", "") or os.environ.get("AWS_ENDPOINT_URL", "")
    result["region"] = os.environ.get("AWS_REGION", "") or os.environ.get("AWS_DEFAULT_REGION", "")

    # Check for init errors
    init_error = getattr(storage, "init_error", None)
    if init_error:
        result["status"] = "degraded"
        result["error"] = str(init_error)
    elif result["s3_enabled"]:
        result["status"] = "ok"
    elif result["backend"] == "local":
        result["status"] = "ok"
    else:
        result["status"] = "degraded"
        result["error"] = "S3 not configured"

    return result


def _check_redis(timeout: float = 0.5) -> Tuple[str, str | None]:
    """Ping Redis with a short timeout; no dependency on Celery."""
    try:
        import redis  # type: ignore
    except Exception as exc:
        return "error", f"redis import error: {exc}"

    try:
        client = redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=timeout,
            socket_timeout=timeout,
        )
        client.ping()
        return "ok", None
    except Exception as exc:  # pragma: no cover - network dependent
        return "error", str(exc)


def _check_storage() -> Tuple[str, str | None, str | None]:
    """Validate storage configuration without heavy S3 operations."""
    storage_backend: str | None = None
    try:
        from apps.api.routers import episodes as episodes_router

        storage = getattr(episodes_router, "STORAGE", None)
    except Exception as exc:  # pragma: no cover - defensive import
        return "error", f"storage import failed: {exc}", storage_backend

    if storage is None:
        return "error", "storage service unavailable", storage_backend

    storage_backend = getattr(storage, "backend", None)
    init_error = getattr(storage, "init_error", None)
    if init_error:
        return "degraded", str(init_error), storage_backend

    if storage_backend not in {"s3", "minio", "local"}:
        return "error", f"unsupported backend {storage_backend}", storage_backend

    # For S3/MinIO, consider the client present and bucket configured as healthy
    if storage_backend in {"s3", "minio"}:
        client = getattr(storage, "_client", None)
        bucket = getattr(storage, "bucket", None)
        if client is None or not bucket:
            return "error", "storage client or bucket not initialized", storage_backend
    return "ok", None, storage_backend


def _check_db(timeout: int = 2) -> Tuple[str, str | None]:
    """Validate optional Postgres connectivity with a short SELECT 1."""
    db_url = os.environ.get("TRR_DB_URL")
    if not db_url:
        return "unconfigured", None

    try:
        import psycopg2  # type: ignore
    except Exception as exc:
        return "error", f"psycopg2 import error: {exc}"

    try:
        conn = psycopg2.connect(db_url, connect_timeout=timeout)  # type: ignore[arg-type]
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        finally:
            conn.close()
        return "ok", None
    except Exception as exc:  # pragma: no cover - network dependent
        return "error", str(exc)

from __future__ import annotations

# Apply global CPU limits BEFORE importing any ML libraries or heavy dependencies
# Uses centralized configuration from apps.common.cpu_limits (default: 3 threads = ~300% CPU)
# Override with env var: SCREANALYTICS_MAX_CPU_THREADS=N
from apps.common.cpu_limits import apply_global_cpu_limits
apply_global_cpu_limits()

import asyncio
import logging
import os
from typing import Dict, Tuple

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.api.config import REDIS_URL
from apps.api.errors import install_error_handlers
from apps.api.routers import (
    archive,
    audio,
    cast,
    episodes,
    facebank,
    files,
    grouping,
    identities,
    jobs,
    metadata,
    people,
    roster,
)
from tools import episode_run

app = FastAPI(title="Screenalytics API", version="0.1.0")
install_error_handlers(app)
LOGGER = logging.getLogger(__name__)

ui_origin = os.environ.get("UI_ORIGIN", "http://localhost:8501")
origins = {ui_origin, "http://localhost:8501"}
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(episodes.router, tags=["episodes"])
app.include_router(identities.router)
app.include_router(roster.router)
app.include_router(cast.router, tags=["cast"])
app.include_router(facebank.router, tags=["facebank"])
app.include_router(files.router, tags=["files"])
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(people.router, tags=["people"])
app.include_router(grouping.router, tags=["grouping"])
app.include_router(metadata.router)
app.include_router(archive.router, tags=["archive"])
app.include_router(audio.router, tags=["audio"])
app.include_router(audio.edit_router, tags=["audio"])

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
    """Clean up stale audio pipeline locks and progress files on startup.

    When the API/worker crashes or restarts, locks can be left in Redis and
    progress files can show stale "running" state. This clears them so new
    jobs can start fresh.
    """
    import os
    import json
    from pathlib import Path

    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))
    manifests_dir = data_root / "manifests"

    # Mark all in-progress audio_progress.json files as stale
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

    # Clear audio pipeline locks from Redis
    try:
        from apps.api.tasks import _get_redis
        r = _get_redis()
        # Find all audio_pipeline lock keys
        lock_pattern = "screanalytics:job_lock:*:audio_pipeline"
        cursor = 0
        cleared = 0
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
async def _warmup_retinaface() -> None:
    """Launch RetinaFace warmup in background to avoid blocking health checks."""

    async def warmup():
        """Run the blocking warmup in a thread pool to avoid blocking event loop."""
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

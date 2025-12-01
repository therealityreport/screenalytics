from __future__ import annotations

# Apply global CPU limits BEFORE importing any ML libraries or heavy dependencies
# Uses centralized configuration from apps.common.cpu_limits (default: 3 threads = ~300% CPU)
# Override with env var: SCREANALYTICS_MAX_CPU_THREADS=N
from apps.common.cpu_limits import apply_global_cpu_limits
apply_global_cpu_limits()

import asyncio
import logging
import os

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
    """Lightweight health check - must not block on heavy imports or GIL-holding operations."""

    import sys

    errors: list[str] = []
    coreml_available = None
    apple_silicon = None
    storage_backend = None
    storage_error = None

    # Avoid accessing episode_run during startup warmup - it can block due to GIL contention
    if "tools.episode_run" in sys.modules:
        try:
            er = sys.modules["tools.episode_run"]
            coreml_available = bool(getattr(er, "COREML_PROVIDER_AVAILABLE", False))
            apple_silicon = bool(getattr(er, "APPLE_SILICON_HOST", False))
        except Exception:
            coreml_available = None
            apple_silicon = None

    if celery_import_error:
        errors.append(f"celery: {celery_import_error}")

    if "apps.api.routers.episodes" in sys.modules:
        try:
            eps_mod = sys.modules["apps.api.routers.episodes"]
            storage = getattr(eps_mod, "STORAGE", None)
            storage_backend = getattr(storage, "backend", None) if storage is not None else None
            storage_error = getattr(storage, "init_error", None) if storage is not None else None
            if storage_error:
                errors.append(f"storage: {storage_error}")
        except Exception:
            storage_error = None

    ok = not errors
    payload = {
        "ok": ok,
        "coreml_available": coreml_available,
        "apple_silicon": apple_silicon,
        "celery_available": celery_router is not None,
        "storage_backend": storage_backend,
    }
    if storage_error:
        payload["storage_error"] = storage_error
    if celery_import_error:
        payload["celery_error"] = str(celery_import_error)
    if errors:
        payload["errors"] = errors

    status = 200 if ok else 503
    return JSONResponse(status_code=status, content=payload)

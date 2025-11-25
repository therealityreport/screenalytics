from __future__ import annotations

# Apply global CPU limits BEFORE importing any ML libraries or heavy dependencies
# Uses centralized configuration from apps.common.cpu_limits (default: 3 threads = ~300% CPU)
# Override with env var: SCREANALYTICS_MAX_CPU_THREADS=N
from apps.common.cpu_limits import apply_global_cpu_limits
apply_global_cpu_limits()

import asyncio
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routers import (
    cast,
    celery_jobs,
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
def healthz() -> dict:
    """Lightweight health check - must not block on heavy imports or GIL-holding operations."""
    # Avoid accessing episode_run during startup warmup - it can block due to GIL contention
    # These values are computed lazily only if warmup has completed
    try:
        # Only access these if the module attributes are already computed (non-blocking check)
        import sys
        if "tools.episode_run" in sys.modules:
            er = sys.modules["tools.episode_run"]
            coreml_available = bool(getattr(er, "COREML_PROVIDER_AVAILABLE", False))
            apple_silicon = bool(getattr(er, "APPLE_SILICON_HOST", False))
        else:
            coreml_available = None
            apple_silicon = None
    except Exception:
        coreml_available = None
        apple_silicon = None
    return {"ok": True, "coreml_available": coreml_available, "apple_silicon": apple_silicon}

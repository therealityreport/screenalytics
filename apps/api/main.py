from __future__ import annotations

from fastapi import FastAPI

from apps.api.routers import jobs

app = FastAPI(title="Screenalytics API", version="0.1.0")
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

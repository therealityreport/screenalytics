from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routers import episodes, jobs

app = FastAPI(title="Screenalytics API", version="0.1.0")

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
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

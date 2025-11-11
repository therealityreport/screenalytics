from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routers import cast, episodes, facebank, grouping, identities, jobs, people, roster

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
app.include_router(identities.router)
app.include_router(roster.router)
app.include_router(cast.router, tags=["cast"])
app.include_router(facebank.router, tags=["facebank"])
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(people.router, tags=["people"])
app.include_router(grouping.router, tags=["grouping"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}

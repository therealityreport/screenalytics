# API Reference — Screenalytics

Last Updated: 2025-12-13

This document is a **high-level map** of the Screenalytics FastAPI surface area. The authoritative source of truth is the OpenAPI schema:

- `GET /docs` (Swagger UI)
- `GET /openapi.json`

For pipeline behavior (what each stage produces/consumes), start with `docs/pipeline/overview.md`.

---

## Base URL

- Local dev: `http://localhost:8000`

---

## Running Locally

```bash
uvicorn apps.api.main:app --reload --port 8000
```

---

## Authentication

None by default (local/dev). If you add auth in your deployment, keep it enforced consistently across Streamlit + API clients.

---

## Errors

Most errors follow FastAPI’s default shape:

```json
{ "detail": "Episode not found" }
```

---

## Key Endpoints (Common UI + Pipeline)

### Episodes (`apps/api/routers/episodes.py`)

**List episodes (used by Workspace sidebar)**
- `GET /episodes`

**Create/update metadata**
- `POST /episodes`
- `PATCH /episodes/{ep_id}` (title/air_date only)

**Episode inspection**
- `GET /episodes/{ep_id}`
- `GET /episodes/{ep_id}/status`
- `GET /episodes/{ep_id}/progress` (polling helper)
- `GET /episodes/{ep_id}/events` (SSE status stream)

**Media/preview helpers (used by Timestamp Search + review pages)**
- `GET /episodes/{ep_id}/timestamp/{timestamp_s}/preview`
- `POST /episodes/{ep_id}/video_clip`
- `GET /episodes/{ep_id}/tracks/{track_id}/crops`

**Examples**

`GET /episodes` (response model: `EpisodeListResponse`)
```json
{
  "episodes": [
    {
      "ep_id": "rhobh-s05e02",
      "show_slug": "rhobh",
      "season_number": 5,
      "episode_number": 2,
      "title": "Reunion Part 2",
      "air_date": "2015-04-14",
      "created_at": "2025-11-18T12:34:56Z",
      "updated_at": "2025-11-19T09:10:11Z"
    }
  ]
}
```

`POST /episodes` (request model: `EpisodeCreateRequest`)
```json
{
  "show_slug_or_id": "rhobh",
  "season_number": 5,
  "episode_number": 2,
  "title": "Reunion Part 2",
  "air_date": "2015-04-14"
}
```

Response (model: `EpisodeCreateResponse`)
```json
{ "ep_id": "rhobh-s05e02" }
```

`PATCH /episodes/{ep_id}` (request model: `EpisodeUpdateRequest`)
```json
{ "title": "Reunion Part 2 (Edited)", "air_date": "2015-04-14" }
```

Response (shape returned by handler)
```json
{
  "ep_id": "rhobh-s05e02",
  "show_slug": "rhobh",
  "season_number": 5,
  "episode_number": 2,
  "title": "Reunion Part 2 (Edited)",
  "air_date": "2015-04-14",
  "updated_at": "2025-12-13T12:34:56Z"
}
```

`GET /episodes/{ep_id}` (response model: `EpisodeDetailResponse`)
```json
{
  "ep_id": "rhobh-s05e02",
  "show_slug": "rhobh",
  "season_number": 5,
  "episode_number": 2,
  "title": "Reunion Part 2",
  "air_date": "2015-04-14",
  "s3": {
    "bucket": "screenalytics",
    "v2_key": "shows/rhobh/episodes/s05/e02/video.mp4",
    "v2_exists": false,
    "v1_key": "rhobh-s05e02.mp4",
    "v1_exists": true
  },
  "local": {
    "path": "data/raw/rhobh-s05e02/video.mp4",
    "exists": true
  }
}
```

---

### Jobs (`apps/api/routers/jobs.py`, mounted at `/jobs`)

Jobs run pipeline phases and write artifacts under `data/manifests/{ep_id}/...`. Most job endpoints accept `ep_id` plus optional overrides; performance profiles are documented in `docs/reference/config/pipeline_configs.md`.

#### Profile Resolution Order

When a job request includes a `profile` and/or explicit overrides (e.g., `stride`, `fps`, thresholds), Screenalytics resolves effective settings in this precedence order:

1. **Explicit request parameters** (highest priority)
2. **Environment variables**
3. **Profile preset values**
4. **Stage config defaults**
5. **Hardcoded fallbacks** (lowest priority)

Core pipeline phases:
- `POST /jobs/detect_track` (SSE streaming)
- `POST /jobs/detect_track_async`
- `POST /jobs/faces_embed_async`
- `POST /jobs/cluster_async`
- `POST /jobs/episode_cleanup_async`
- `POST /jobs/screen_time/analyze` (screentime v2)

Job management:
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/progress`
- `POST /jobs/{job_id}/cancel`

`POST /jobs/screen_time/analyze` (request model: `AnalyzeScreenTimeRequest`)
```json
{
  "ep_id": "rhobh-s05e02",
  "preset": "default",
  "screen_time_mode": "tracks"
}
```

Response (shape returned by handler)
```json
{
  "job_id": "job-abc123",
  "ep_id": "rhobh-s05e02",
  "state": "running",
  "started_at": "2025-12-13T12:34:56Z",
  "progress_file": "data/manifests/rhobh-s05e02/jobs/job-abc123.progress.json",
  "requested": {
    "preset": "default",
    "screen_time_mode": "tracks"
  }
}
```

---

### Audio (`apps/api/routers/audio.py`, mounted at `/audio`)

The audio pipeline is optional and depends on external tooling/credentials.

- `GET /audio/prerequisites`
- See OpenAPI (`/docs`) for the exact run endpoints and request models.

Pipeline details: `docs/pipeline/audio_pipeline.md`.

---

## Related Docs

- `docs/reference/config/pipeline_configs.md` — performance profiles + config keys
- `docs/pipeline/overview.md` — pipeline phases and artifacts
- `docs/pipeline/screentime_analytics_optimization.md` — screentime meaning/tuning/troubleshooting
- `docs/reference/artifacts_faces_tracks_identities.md` — vision artifact schemas
- `ACCEPTANCE_MATRIX.md` — CI gates and acceptance thresholds

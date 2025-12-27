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

## Run-Scoped Pipeline + Faces Review Surface Area

This section maps the endpoints used by Detect/Track → Faces Embed → Cluster → Validator → Assignments → Screentime.

### Common IDs

- `ep_id`: Episode ID (e.g., `rhoslc-s06e11`)
- `run_id`: Run attempt ID (e.g., `Attempt3_2025-01-01_123456EST`)
- `track_id`: Track identifier (int)
- `identity_id`/`cluster_id`: Cluster identity ID
- `face_id`: Face record ID
- `cast_id`: Cast identity ID
- `job_id`: Async job handle

### Detect/Track Output (detections + tracks + track metrics)

Artifacts (run-scoped):
- `data/manifests/{ep_id}/runs/{run_id}/detections.jsonl`
- `data/manifests/{ep_id}/runs/{run_id}/tracks.jsonl`
- `data/manifests/{ep_id}/runs/{run_id}/track_metrics.json`
- `data/manifests/{ep_id}/runs/{run_id}/detect_track.json` (marker)

| Endpoint | Appears to be | Creates / writes | Reads / uses | IDs |
| --- | --- | --- | --- | --- |
| `POST /jobs/detect_track` | Sync detect+track pipeline | artifacts above | episode video | `ep_id`, optional `run_id` |
| `POST /jobs/detect_track_async` | Async detect+track (JobService) | artifacts above | episode video | `ep_id`, optional `run_id`, `job_id` |
| `POST /celery_jobs/detect_track` | Celery/local detect+track (streaming logs) | artifacts above | episode video | `ep_id`, optional `run_id`, `job_id` |
| `POST /episodes/{ep_id}/runs/{run_id}/jobs/detect_track` | Run-scoped trigger | artifacts above | episode video | `ep_id`, `run_id`, `job_id` |
| `GET /episodes/{ep_id}/status?run_id=...` | Stage readiness/status | none | run markers + manifests | `ep_id`, `run_id` |
| `GET /episodes/{ep_id}/runs/{run_id}/state` | Run state + artifact pointers | none | DB run_state + filesystem | `ep_id`, `run_id` |

### Crop/Embed Output (faces manifest + embeddings + crops)

Artifacts (run-scoped):
- `data/manifests/{ep_id}/runs/{run_id}/faces.jsonl`
- `data/embeds/{ep_id}/runs/{run_id}/faces.npy`
- `data/frames/{ep_id}/runs/{run_id}/crops/track_####/frame_######.jpg`

| Endpoint | Appears to be | Creates / writes | Reads / uses | IDs |
| --- | --- | --- | --- | --- |
| `POST /jobs/faces_embed` | Sync faces embed/harvest | artifacts above | tracks/detections | `ep_id`, optional `run_id` |
| `POST /jobs/faces_embed_async` | Async faces embed (JobService) | artifacts above | tracks/detections | `ep_id`, optional `run_id`, `job_id` |
| `POST /celery_jobs/faces_embed` | Celery/local faces embed (streaming logs) | artifacts above | tracks/detections | `ep_id`, optional `run_id`, `job_id` |
| `POST /episodes/{ep_id}/runs/{run_id}/jobs/faces_embed` | Run-scoped trigger | artifacts above | tracks/detections | `ep_id`, `run_id`, `job_id` |
| `GET /episodes/{ep_id}/tracks/{track_id}/crops` | Track crop browser | none | crops on disk/S3 | `ep_id`, `track_id` |
| `GET /episodes/{ep_id}/tracks/{track_id}/integrity` | Track faces vs crops integrity | none | faces.jsonl + crops | `ep_id`, `track_id` |

### Cluster Output (identities + reps + membership)

Artifacts (run-scoped):
- `data/manifests/{ep_id}/runs/{run_id}/identities.json`
- `data/manifests/{ep_id}/runs/{run_id}/cluster_centroids.json`
- `data/manifests/{ep_id}/runs/{run_id}/track_reps.jsonl`

| Endpoint | Appears to be | Creates / writes | Reads / uses | IDs |
| --- | --- | --- | --- | --- |
| `POST /jobs/cluster` | Sync clustering pipeline | artifacts above | faces + embeddings | `ep_id`, optional `run_id` |
| `POST /jobs/cluster_async` | Async clustering (JobService) | artifacts above | faces + embeddings | `ep_id`, optional `run_id`, `job_id` |
| `POST /celery_jobs/cluster` | Celery/local clustering (streaming logs) | artifacts above | faces + embeddings | `ep_id`, optional `run_id`, `job_id` |
| `POST /episodes/{ep_id}/runs/{run_id}/jobs/cluster` | Run-scoped trigger | artifacts above | faces + embeddings | `ep_id`, `run_id`, `job_id` |
| `GET /episodes/{ep_id}/cluster_tracks?run_id=...` | Cluster↔track summary | none | identities + tracks (+ faces count) | `ep_id`, `run_id` |
| `GET /episodes/{ep_id}/faces_review_bundle?run_id=...` | Faces Review bundle | none | identities + tracks + reps | `ep_id`, `run_id` |

### Validator Output (Run Health panel)

Computed on demand (no artifact written).

| Endpoint | Appears to be | Creates / writes | Reads / uses | IDs |
| --- | --- | --- | --- | --- |
| `GET /episodes/{ep_id}/runs/{run_id}/integrity` | Validator report | none | run artifacts + run_state pointers | `ep_id`, `run_id` |
| `GET /episodes/{ep_id}/runs/{run_id}/state` | Run state + artifact pointers | none | DB run_state + filesystem | `ep_id`, `run_id` |
| `GET /episodes/{ep_id}/faces_review_bundle?run_id=...` | Bundle includes validator | none | validator + artifacts | `ep_id`, `run_id` |

### Assignments/Corrections (run-scoped canonical store)

Stored in `data/manifests/{ep_id}/runs/{run_id}/identities.json` under:
`manual_assignments`, `track_overrides`, `face_exclusions`.

| Endpoint | Appears to be | Creates / writes | Reads / uses | IDs |
| --- | --- | --- | --- | --- |
| `GET /episodes/{ep_id}/assignments?run_id=...` | Read assignment state | none | identities.json | `ep_id`, `run_id` |
| `POST/PUT /episodes/{ep_id}/assignments/cluster` | Cluster→cast assignment | manual_assignments | identities.json | `ep_id`, `run_id`, `cluster_id`, `cast_id` |
| `POST/PUT /episodes/{ep_id}/assignments/track` | Track override | track_overrides | identities.json | `ep_id`, `run_id`, `track_id`, `cast_id` |
| `POST/PUT /episodes/{ep_id}/assignments/face_exclusion` | Face exclusion | face_exclusions | identities.json | `ep_id`, `run_id`, `face_id` |

### Screentime Compute (run-scoped)

Artifacts (run-scoped):
- `data/manifests/{ep_id}/runs/{run_id}/analytics/screentime.json`
- `data/manifests/{ep_id}/runs/{run_id}/analytics/screentime.csv`

| Endpoint | Appears to be | Creates / writes | Reads / uses | IDs |
| --- | --- | --- | --- | --- |
| `POST /jobs/screen_time/analyze` | Run screentime analysis | screentime.json/csv | faces + tracks + identities + assignments | `ep_id`, optional `run_id`, `job_id` |
| `POST /episodes/{ep_id}/runs/{run_id}/jobs/screentime` | Run-scoped trigger | screentime.json/csv | faces + tracks + identities + assignments | `ep_id`, `run_id`, `job_id` |
| `GET /jobs?ep_id=...&job_type=screen_time_analyze` | Job history | none | job store | `ep_id` |
| `GET /jobs/{job_id}/progress` | Job progress | none | job store | `job_id` |

### Screentime UI (Screen Time Analytics Page)

The Streamlit page currently reads legacy analytics paths:
- `data/analytics/{ep_id}/screentime.json`
- `data/analytics/{ep_id}/screentime.csv`

| Endpoint | Appears to be | Creates / writes | Reads / uses | IDs |
| --- | --- | --- | --- | --- |
| `POST /jobs/screen_time/analyze` | Launch screentime job | screentime.json/csv | faces + tracks + identities + assignments | `ep_id`, optional `run_id` |
| `GET /jobs?ep_id=...&job_type=screen_time_analyze` | Job history | none | job store | `ep_id` |
| `GET /jobs/{job_id}/progress` | Job progress | none | job store | `job_id` |

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

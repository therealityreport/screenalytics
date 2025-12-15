# Detect/Track Call Chain Documentation

## Overview

This document describes how Detect/Track operations flow through the system for both execution modes: **Local Worker (direct)** and **Redis/Celery (queued)**.

## Current Implementation (main)

### Local Mode Call Chain

Local mode is **truly synchronous**: a single blocking HTTP request that waits for the job to complete. No job IDs, no polling, and page refresh cancels the job.

```
User clicks "Run Detect/Track" button (Execution Mode = Local)
    │
    ▼
2_Episode_Detail.py (line ~541)
    └── helpers.run_pipeline_job_with_mode(ep_id, "detect_track", job_payload, ...)
            │
            ▼
ui_helpers.py::run_pipeline_job_with_mode (line ~2419)
    └── For execution_mode == "local":
        ├── Display: "⏳ [LOCAL MODE] Running {operation} (device=X)..."
        ├── Single blocking HTTP POST to /celery_jobs/detect_track
        │   (timeout=3600 seconds = 1 hour)
        ├── NO job_id returned
        ├── NO polling loop
        └── Response: {"status": "completed", "logs": [...], "elapsed_seconds": X}
                │
                ▼
celery_jobs.py::start_detect_track_celery (line ~1363)
    └── For execution_mode == "local":
        ├── Build command for tools/episode_run.py
        └── result = _run_local_subprocess_blocking(command, ep_id, "detect_track", options)
                │
                ▼
celery_jobs.py::_run_local_subprocess_blocking (line ~741)
    ├── Check for duplicate job via _running_local_jobs (prevents concurrent runs)
    ├── Set CPU thread limits to max 2 for thermal safety
    ├── Spawn subprocess with subprocess.Popen()
    ├── Register PID in _running_local_jobs
    ├── Call process.communicate(timeout=3600) - BLOCKS until complete
    ├── Unregister from _running_local_jobs
    └── Return: {"status": "completed" | "error", "logs": [...], "elapsed_seconds": X}
                │
                ▼
tools/episode_run.py
    └── Runs detection and tracking pipeline
        ├── Applies CPU thread limits (inherited from parent env)
        ├── Loads video via cv2
        ├── Runs RetinaFace detection
        ├── Runs ByteTrack/StrongSORT tracking
        └── Saves detections.parquet, tracks.parquet
```

### Redis/Celery Mode Call Chain

Redis mode uses **background job execution**: the job is queued, and the UI polls for status updates.

```
User clicks "Run Detect/Track" button (Execution Mode = Redis)
    │
    ▼
2_Episode_Detail.py (line ~541)
    └── helpers.run_pipeline_job_with_mode(ep_id, "detect_track", job_payload, ...)
            │
            ▼
ui_helpers.py::run_pipeline_job_with_mode (line ~2503)
    └── For execution_mode != "local":
        └── run_celery_job_with_progress(ep_id, operation, payload, ...)
                │
                ▼
ui_helpers.py::run_celery_job_with_progress
    ├── POST /celery_jobs/detect_track → returns job_id
    ├── Poll /celery_jobs/{job_id} every 2s
    └── Display progress until job completes
                │
                ▼
celery_jobs.py::start_detect_track_celery (line ~1387)
    └── For execution_mode == "redis":
        └── run_detect_track_task.delay(ep_id, options) → returns Celery AsyncResult
                │
                ▼
tasks.py::run_detect_track_task (Celery task)
    └── Runs in Celery worker process
        └── Spawns tools/episode_run.py subprocess
```

## Key Questions Answered

### Does `execution_mode="local"` create a job ID?

**NO** - Local mode does NOT create a job ID. The response contains:
- `status`: "completed" or "error"
- `logs`: List of log lines from the subprocess
- `elapsed_seconds`: Runtime in seconds
- `ep_id`, `operation`, `execution_mode`
- `device`, `profile`, `cpu_threads`

There is no `job_id` field in local mode responses.

### Does local mode use any polling loop?

**NO** - Neither client-side nor server-side polling.

- **Server-side**: Uses `subprocess.Popen().communicate()` which blocks until completion
- **Client-side**: Makes a single HTTP POST with a 1-hour timeout

### Does local mode continue if the page is refreshed?

**NO** - The subprocess is tied to the HTTP request lifecycle. When the request times out or is cancelled:
1. The subprocess and all its children are terminated via process group kill
2. The job is unregistered from `_running_local_jobs`

### What about thermal safety?

Local mode enforces **hard cap of 2 CPU threads** regardless of profile:
```python
local_max_threads = min(int(cpu_threads or 2), 2)  # Hard cap at 2
```

This prevents laptop overheating during synchronous local runs.

## Comparison: Local vs Redis Mode

| Aspect | Local Mode | Redis Mode |
|--------|------------|------------|
| HTTP call | Single blocking (up to 1 hour) | Quick POST + polling |
| Job ID | None | Celery task ID |
| UI feedback | Status + logs on completion | Progress polling |
| Page refresh | Kills subprocess | Job continues |
| CPU thread cap | 2 (hard limit) | Profile-based (up to 8) |
| Use case | Development, testing | Production, long runs |

## Historical Context

### Regression (commit 4badd97)

Commit `4badd97` introduced job-like behavior for local mode:
- Created "local-{uuid}" job IDs
- Started detached subprocesses with `start_new_session=True`
- UI polled for status like Redis mode
- Process continued after page refresh

### Original Fix (commit 45b5d96)

Commit `45b5d96` attempted to fix this but was incomplete:
- Made UI use blocking request
- But backend still started detached process

### Current Fix (main)

The current implementation properly fixes local mode:
- Backend: `_run_local_subprocess_blocking()` uses `communicate()` for true blocking
- UI: Single request with long timeout, no polling
- Thermal: Hard 2-thread cap for local mode
- Cleanup: Process killed on request cancellation

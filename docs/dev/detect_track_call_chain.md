# Detect/Track Call Chain Documentation

## Overview

This document describes how Detect/Track operations flow through the system for both execution modes: **Local Worker (direct)** and **Redis/Celery (queued)**.

## Current Implementation (as of nov-24 branch)

### Local Mode Call Chain

```
User clicks "Run Detect/Track" button (Execution Mode = Local)
    │
    ▼
2_Episode_Detail.py (line ~541)
    └── helpers.run_pipeline_job_with_mode(ep_id, "detect_track", job_payload, ...)
            │
            ▼
ui_helpers.py::run_pipeline_job_with_mode (line 2419-2472)
    └── For execution_mode == "local":
        ├── Display: "⏳ Running {operation} synchronously..."
        ├── Single blocking HTTP POST to /celery_jobs/detect_track
        │   (timeout=3600 seconds = 1 hour)
        └── Wait for response, display logs directly
                │
                ▼
celery_jobs.py::start_detect_track_celery (line 874-891)
    └── For execution_mode == "local":
        ├── Build command for tools/episode_run.py
        └── await _run_local_subprocess_async(command, ep_id, "detect_track", options)
                │
                ▼
celery_jobs.py::_run_local_subprocess_async (line 460-666)
    ├── Spawn subprocess with start_new_session=True (new process group)
    ├── Register in _running_local_jobs (prevents duplicates)
    ├── Poll every 0.5s via await asyncio.sleep(0.5)
    ├── On request cancel (page refresh): kill process group
    └── Return: {"status": "completed", "logs": [...], "elapsed_seconds": X}
                │
                ▼
tools/episode_run.py
    └── Runs detection and tracking pipeline
        ├── Applies CPU thread limits
        ├── Loads video via cv2
        ├── Runs RetinaFace detection
        ├── Runs ByteTrack/StrongSORT tracking
        └── Saves detections.parquet, tracks.parquet
```

### Redis/Celery Mode Call Chain

```
User clicks "Run Detect/Track" button (Execution Mode = Redis)
    │
    ▼
2_Episode_Detail.py (line ~541)
    └── helpers.run_pipeline_job_with_mode(ep_id, "detect_track", job_payload, ...)
            │
            ▼
ui_helpers.py::run_pipeline_job_with_mode (line 2490-2499)
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
celery_jobs.py::start_detect_track_celery (line 893-910)
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

**NO** - The current implementation does NOT create a job ID for local mode. The response contains:
- `status`: "completed" or "error"
- `logs`: List of log lines
- `elapsed_seconds`: Runtime in seconds
- `ep_id`, `operation`, `execution_mode`

There is no `job_id` field in local mode responses.

### Does local mode use any polling loop?

**Internally yes, externally no.**

- **Internal (server-side)**: The FastAPI handler uses `await asyncio.sleep(0.5)` to poll the subprocess status. This keeps the async event loop responsive.
- **External (client-side)**: The UI makes a single blocking HTTP request with a 1-hour timeout. No client-side polling.

### Does local mode use a jobs store or tracking map?

**Partially** - The `_running_local_jobs` dict tracks running jobs by `ep_id::operation` key to:
1. Prevent duplicate jobs for the same episode/operation
2. Store PID for process group cleanup

This is NOT a persistent job store - it's ephemeral in-memory tracking that clears on server restart.

### Does local mode continue if the page is refreshed?

**NO** - When the HTTP request is cancelled (page refresh), the `asyncio.CancelledError` is raised, which triggers `_kill_process_tree()` to terminate the subprocess and all its children.

## Historical Context

### Old Version (commit d499c64)

The older implementation had job-style behavior even for local mode:
- Created "local-{uuid}" job IDs
- Used `_local_jobs` in-memory dict for tracking
- UI showed "Job submitted: local-xxx" and "Polling for status..."
- Kept polling via client-side rerun

### Fix Commit (45b5d96)

Commit `45b5d96` attempted to fix this by:
- Removing job IDs from local mode
- Making UI use single blocking request
- Removing "Job submitted" / "Polling" log messages

### Current Working Directory

The current uncommitted changes refined the fix:
- Added `_run_local_subprocess_async` with process group management
- Added proper request cancellation handling
- Retained `_running_local_jobs` for duplicate prevention only (not job tracking)

## Remaining Issues

1. **Logging clarity**: Need to ensure local mode logs are clearly different from job-style logs
2. **Performance regression**: Local mode may be slower than the pre-Redis pipeline (investigation needed)
3. **Thermal behavior**: Need to verify CPU thread limits are properly applied

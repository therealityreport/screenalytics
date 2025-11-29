# Execution Mode: Local vs Redis/Celery

This document describes the Execution Mode feature which allows switching between local (synchronous) and Redis/Celery (queued) execution for pipeline jobs.

## Overview

The Execution Mode toggle controls how pipeline operations are executed:

- **Redis/Celery (queued)**: Jobs are enqueued via Celery to Redis and processed by background workers. The UI polls for status updates. Jobs continue running even if you refresh the page.

- **Local Worker (direct)**: Jobs run **synchronously** in-process. The API request blocks until the job completes. No job ID, no polling, no background task. If you refresh the page, the job stops.

## UI Controls

### Episode Detail Page

The Execution Mode selector is located in the **Execution Settings** expander near the top of the Episode Detail page.

Operations affected:
- Detect/Track
- Faces Harvest
- Cluster Identities
- Episode Cleanup
- And all other pipeline buttons

### Faces Review Page

The same Execution Mode selector appears in the **Execution Settings** expander on the Faces Review page.

Operations affected:
- Refresh Values
- Cluster Cleanup
- Batch Assign operations
- Auto Group Clusters
- And all assignment operations

### Shared State

The execution mode is shared between Episode Detail and Faces Review for the same episode:
- Changing the mode on Episode Detail updates it on Faces Review too
- The selection is stored in session state per episode

## API Parameter

All job endpoints now accept an optional `execution_mode` parameter:

```json
{
  "ep_id": "show-s01e01",
  "device": "auto",
  "execution_mode": "local"  // or "redis" (default)
}
```

### Supported Endpoints

Pipeline Jobs:
- `POST /celery_jobs/detect_track`
- `POST /celery_jobs/faces_embed`
- `POST /celery_jobs/cluster`

Grouping Operations:
- `POST /episodes/{ep_id}/clusters/batch_assign_async`
- `POST /episodes/{ep_id}/clusters/group_async`

Refresh Operations:
- `POST /episodes/{ep_id}/refresh_similarity_async`

### Response Differences

**Redis mode (execution_mode="redis")**:
```json
{
  "job_id": "abc-123-def",
  "state": "queued",
  "ep_id": "show-s01e01",
  "execution_mode": "redis",
  "message": "Detect/track job queued via Celery"
}
```
Poll `GET /celery_jobs/{job_id}` to check job status. The job runs in the background.

**Local mode (execution_mode="local")**:
```json
{
  "status": "completed",
  "ep_id": "show-s01e01",
  "operation": "detect_track",
  "execution_mode": "local",
  "elapsed_seconds": 432.5,
  "logs": [
    "[LOCAL MODE] Starting detect_track",
    "  Device: coreml, Stride: 6",
    "  This runs synchronously - page refresh will cancel the job.",
    "CPU threads limited to 2",
    "Process started (PID 12345)",
    "Loading models...",
    "Processing frames...",
    "[LOCAL MODE] detect_track completed successfully in 7m 12s"
  ]
}
```

Key differences in local mode:
- **No `job_id`**: The job is synchronous, no polling needed
- **`logs` array**: Execution logs returned directly in response
- **`elapsed_seconds`**: How long the job took
- **Blocking**: The HTTP request blocks until the job completes

## Performance and Safety

Local mode respects the same performance profiles and safety guardrails as Redis mode:

### CPU Thread Limits
- Thread limits (`cpu_threads`, `SCREENALYTICS_MAX_CPU_THREADS`) are honored
- Environment variables for BLAS/OMP threading are set appropriately

### Profile Resolution
- Performance profiles (low_power, balanced, performance) apply the same defaults
- Device-based profile auto-selection works identically
- "performance" profile auto-downgrades on CPU-only devices

### Timeouts
- Local mode has a 1-hour timeout per job (configurable)
- Long-running jobs will return an error if they exceed the timeout

## When to Use Each Mode

### Use Redis/Celery (queued) when:
- Running production workloads
- Processing multiple episodes concurrently
- You want jobs to continue if you navigate away or refresh the page
- Redis and Celery workers are running

### Use Local Worker (direct) when:
- Debugging pipeline issues (logs returned directly)
- Testing changes without starting workers
- Running on a single machine without Redis
- You want a simple, linear execution flow
- You prefer the old pre-Celery behavior

## Page Refresh Behavior

**Redis/Celery mode**: If you start a job and refresh the page:
- The job continues running in the background
- When you return, you can check job status
- Jobs are decoupled from the browser session

**Local Worker mode**: If you start a job and refresh the page:
- The job is tied to the HTTP request
- Refreshing cancels the in-flight request
- The subprocess is **killed** along with all child processes (using process groups)
- This is the expected behavior for synchronous execution - truly tied to the browser session

## Thermal Safety (Laptop Mode)

Local Worker mode is designed to be laptop-friendly with conservative thermal defaults:

### CPU Thread Limits
- Default: 2 threads for CPU/CoreML/MPS devices
- Environment variables set: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, etc.
- Override via: `cpu_threads` parameter or `SCREENALYTICS_MAX_CPU_THREADS` env var

### Profile Auto-Selection
- CoreML/MPS/CPU devices default to `low_power` profile (stride=12, fps=15)
- CUDA devices default to `balanced` profile (stride=6, fps=24)
- The `performance` profile auto-downgrades to `balanced` on non-CUDA devices

### Comparison to Pre-Celery Behavior
Local Worker mode is designed to match the old pre-Redis/Celery pipeline behavior:
- Direct subprocess execution
- Same thermal safety limits
- Same device/profile defaults
- Linear logging (no "Job submitted" or "Polling" messages)

If you experience different performance or thermal behavior in local mode compared to the old pipeline:
1. Restart the API server to pick up latest code
2. Check that CPU thread limits are being applied (look for "CPU threads limited to X" in logs)
3. Verify device is correctly detected (look for "CoreMLExecutionProvider" vs "CPUExecutionProvider")

## Troubleshooting

### "Background jobs unavailable" error
If you see this error in Redis mode, either:
1. Switch to Local Worker mode, or
2. Start Redis and Celery workers

### Jobs timing out in local mode
Long-running jobs may hit the 1-hour timeout. Consider:
- Using Redis mode for long jobs
- Breaking the work into smaller chunks
- Adjusting the timeout if running very large episodes

### Execution mode not syncing between pages
The mode is stored per-episode in session state. If pages show different modes:
- Refresh both pages
- Check that both pages are viewing the same episode

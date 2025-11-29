# Execution Mode: Local vs Redis/Celery

This document describes the Execution Mode feature which allows switching between local (synchronous) and Redis/Celery (queued) execution for pipeline jobs.

## Overview

The Execution Mode toggle controls how pipeline operations are executed:

- **Redis/Celery (queued)**: Default mode. Jobs are enqueued via Celery to Redis and processed by background workers. This allows the UI to remain responsive while jobs run asynchronously.

- **Local Worker (direct)**: Jobs run synchronously in-process. The API request blocks until the job completes. Useful for debugging, testing, or when Redis/Celery workers are unavailable.

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
  "status": "queued",
  "ep_id": "show-s01e01",
  "execution_mode": "redis"
}
```
Poll `/celery_jobs/{job_id}` to check job status.

**Local mode (execution_mode="local")**:
```json
{
  "status": "completed",
  "ep_id": "show-s01e01",
  "operation": "detect_track",
  "execution_mode": "local"
}
```
No job_id returned; the job is complete when the response returns.

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
- You want the UI to remain responsive during long operations
- Redis and Celery workers are running

### Use Local Worker (direct) when:
- Debugging pipeline issues
- Testing changes without starting workers
- Running on a single machine without Redis
- You want immediate feedback without polling

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

# Detect/Track Local Mode Regression - FIXED

## Issue Description

User reported that Local Worker mode for Detect/Track was:
1. Showing "Job submitted: local-xxx" and "Polling for status..." messages
2. Running slower and causing more thermal stress than expected
3. Continuing in background if page is refreshed (not truly synchronous)

## Root Cause

The regression was introduced in commit `4badd97` which changed local mode to behave like Redis/Celery mode:

**Problem code (commit 4badd97):**
```python
# In celery_jobs.py
def _start_local_subprocess(command, ep_id, operation, options):
    # Started detached subprocess
    process = subprocess.Popen(
        command,
        start_new_session=True,  # DETACHED - not tied to request
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Returned immediately with job_id
    return {"job_id": f"local-{uuid}", "state": "started", "pid": process.pid}
```

This caused:
- Job IDs for local mode (wrong)
- UI polling for local mode (wrong)
- Process continued after page refresh (wrong)
- No synchronous behavior (wrong)

## Fix Applied (nov-24 branch)

### 1. Backend: New Blocking Subprocess Runner

Created `_run_local_subprocess_blocking()` that:
- Uses `subprocess.Popen().communicate()` for true blocking
- Hard caps CPU threads at 2 for thermal safety
- Returns directly with status and logs (no job_id)
- Registers job for duplicate prevention only

```python
def _run_local_subprocess_blocking(command, ep_id, operation, options, timeout=3600):
    # Hard cap at 2 threads for local mode thermal safety
    local_max_threads = min(int(cpu_threads or 2), 2)

    process = subprocess.Popen(command, ...)
    stdout, stderr = process.communicate(timeout=timeout)  # BLOCKS

    return {
        "status": "completed",  # or "error"
        "logs": [line for line in stdout.split("\n")],
        "elapsed_seconds": elapsed,
        # NO job_id field
    }
```

### 2. Backend: Updated Endpoints

All three pipeline endpoints (detect_track, faces_embed, cluster) now use the blocking runner for local mode:

```python
@router.post("/detect_track")
async def start_detect_track_celery(req: DetectTrackCeleryRequest):
    if execution_mode == "local":
        result = _run_local_subprocess_blocking(command, req.ep_id, "detect_track", options)
        return result  # Synchronous response
    else:
        result = run_detect_track_task.delay(req.ep_id, options)
        return {"job_id": result.id, ...}  # Async with job_id
```

### 3. Frontend: Single Blocking Request

Updated `run_pipeline_job_with_mode()` to make a single blocking request:

```python
if execution_mode == "local":
    # Single blocking request - waits for job to complete
    resp = requests.post(
        f"{_api_base()}{endpoint}",
        json=payload,
        timeout=timeout,  # Full timeout (default 3600s)
    )
    result = resp.json()
    # Result contains status, logs, elapsed_seconds - no job_id

    if result.get("status") == "completed":
        return result, None  # Success
    else:
        return result, result.get("error")  # Error
```

### 4. UI: Removed Spinner for Local Mode

Updated Episode Detail page to not wrap local mode calls in `st.spinner()` since the helper handles its own UI:

```python
if execution_mode == "local":
    # Local mode handles its own UI - no spinner needed
    summary, error_message = helpers.run_pipeline_job_with_mode(...)
else:
    with st.spinner(f"Running detect/track via Celery..."):
        summary, error_message = helpers.run_pipeline_job_with_mode(...)
```

## Verification Checklist

After the fix, verify:

1. **No job IDs in local mode:**
   - Run Detect/Track with "Local Worker (direct)"
   - Response should NOT contain `job_id`
   - UI should NOT show "Job submitted: local-xxx"

2. **No polling in local mode:**
   - UI should NOT show "Polling for status..."
   - No network requests to `/celery_jobs/local`
   - Single blocking request until completion

3. **Page refresh cancels job:**
   - Start a Detect/Track run in local mode
   - Refresh the page before completion
   - Verify the subprocess is killed (check Activity Monitor)

4. **Thermal behavior:**
   - Local mode should cap CPU at 2 threads
   - Check `Detailed log` for "CPU threads capped at 2"
   - Compare thermal output vs previous runs

5. **Clear logs:**
   - Should see "[LOCAL MODE] Starting detect_track"
   - Should see device, profile, stride, cpu_threads
   - Should see "[LOCAL MODE] detect_track completed in X"

## Comparison: Before and After

| Aspect | Before (Broken) | After (Fixed) |
|--------|-----------------|---------------|
| Job ID | `local-{uuid}` | None |
| Subprocess | Detached (start_new_session=True) | Attached (communicate) |
| UI request | Quick POST + polling | Single blocking POST |
| Page refresh | Process continues | Process killed |
| CPU threads | Profile-based (2-8) | Hard cap at 2 |
| Log messages | "Job submitted", "Polling..." | "[LOCAL MODE] Starting..." |

## Related Commits

1. **d499c64**: Introduced async local mode with job IDs (problem started here)
2. **45b5d96**: Attempted fix but incomplete
3. **4badd97**: Regression - reintroduced async local mode
4. **Current (nov-24)**: Proper fix with truly synchronous local mode

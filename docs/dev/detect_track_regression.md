# Detect/Track Performance Regression Analysis

## Issue Description

User reports that Local Worker mode for Detect/Track is:
1. Showing "Job submitted: local-xxx" and "Polling for status..." messages
2. Running slower and causing more thermal stress than the pre-Redis/Celery implementation
3. Continuing in background if page is refreshed (not truly synchronous)

## Root Cause Analysis

### Issue 1: Stale Server Code

The "Job submitted: local-xxx" and "Polling for status" messages come from an **older version** of the code (commit `d499c64`).

**Resolution**: The current working directory (uncommitted changes after `45b5d96`) has fixed this:
- Local mode no longer creates job IDs
- Local mode no longer shows "Job submitted" or "Polling" messages
- UI makes a single blocking HTTP request

**Action Required**: Restart the API server to pick up the latest code changes:
```bash
# Kill existing API server
pkill -f "uvicorn apps.api.main:app" || true

# Restart
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Issue 2: Performance Regression - Investigation

Comparing old vs new execution paths:

| Aspect | Old (JobService via SSE) | New (Local mode via celery_jobs.py) |
|--------|--------------------------|-------------------------------------|
| Entry point | `/jobs/detect_track` | `/celery_jobs/detect_track` |
| Subprocess | `subprocess.Popen()` + cpulimit wrapper | `subprocess.Popen()` + env var limits |
| CPU limiting | `cpulimit` binary (250%) OR psutil affinity | Environment variables (OMP_NUM_THREADS, etc.) |
| Thread default | Via cpulimit: ~2.5 cores | Via env: 2 threads (profile-based) |
| Progress | SSE stream | HTTP response on completion |

#### Potential Performance Differences

1. **CPU limiting approach**:
   - Old: Used `cpulimit` binary which actively throttles CPU usage
   - New: Uses environment variables which limit thread parallelism but not actual CPU usage
   - Impact: Environment variables may be less effective at preventing thermal spikes

2. **Profile resolution**:
   - Old: `device=coreml` → `balanced` profile (stride=6, fps=24, cpu_threads=4)
   - New: `device=coreml` → `low_power` profile (stride=12, fps=15, cpu_threads=2)
   - Impact: New defaults are MORE conservative, should be FASTER and cooler

3. **Actual observed behavior**:
   - User selected: `stride=6`, `profile=balanced`, `device=coreml`
   - Expected: ~10,249 frames processed (61,494 / 6)
   - Observed: Slower and hotter than expected

### Issue 3: Thread Limits Not Applied?

The env var approach depends on `episode_run.py` calling `apply_global_cpu_limits()` **before** importing ML libraries. Let's verify:

```python
# In tools/episode_run.py (lines 28-32):
from apps.common.cpu_limits import apply_global_cpu_limits
apply_global_cpu_limits()  # Called before numpy import!

import numpy as np  # After limits applied
```

This is correct - limits ARE applied early. But the subprocess inherits different env vars:

```python
# In celery_jobs.py::_run_local_subprocess_async (lines 517-540):
env = os.environ.copy()
cpu_threads = options.get("cpu_threads")
if cpu_threads:
    env.update({
        "SCREENALYTICS_MAX_CPU_THREADS": str(threads),
        "OMP_NUM_THREADS": str(threads),
        ...
    })
```

**Issue Found**: The env vars are set but `cpu_threads` may be `None` if not explicitly provided, falling back to profile defaults. However, `_apply_profile_defaults()` does set `cpu_threads` before this.

## Recommendations

### Immediate Fix (Server Restart)

1. Restart the API server to pick up latest code
2. Clear any browser cache/stale Streamlit state

### Verification Steps

After restarting, verify local mode:
1. Select "Local Worker (direct)" execution mode
2. Run Detect/Track with stride=6, profile=balanced, device=coreml
3. Confirm UI shows:
   - "⏳ Running detect_track synchronously..."
   - NO "Job submitted" or "Polling" messages
4. Refresh page during run - process should be killed

### If Still Slow/Hot

Check these potential causes:

1. **CoreML/ONNX execution**:
   ```bash
   # In episode_run.py output, look for:
   # "Using CoreMLExecutionProvider" vs "Using CPUExecutionProvider"
   ```
   If falling back to CPU, that explains the heat.

2. **Stride not applied**:
   ```bash
   # Check progress.json for actual frames_done vs expected
   # Expected: ~10,249 (for 61,494 total / stride 6)
   # If much higher, stride is not being applied
   ```

3. **Resolution not reduced for CoreML**:
   ```python
   # episode_run.py defaults:
   RETINAFACE_DET_SIZE = (640, 640)
   RETINAFACE_COREML_DET_SIZE = (480, 480)  # Lower for thermal safety
   ```
   Verify CoreML is using 480x480.

## Historical Comparison

To establish baseline, compare old vs new runs on the SAME episode:

### Old Pipeline (if available)
```bash
# Run via old SSE endpoint
curl -X POST http://localhost:8000/jobs/detect_track/stream \
  -H "Content-Type: application/json" \
  -d '{"ep_id": "test-ep", "stride": 6, "device": "coreml"}'
```

### New Pipeline (local mode)
```bash
# Run via new endpoint
curl -X POST http://localhost:8000/celery_jobs/detect_track \
  -H "Content-Type: application/json" \
  -d '{"ep_id": "test-ep", "stride": 6, "device": "coreml", "execution_mode": "local"}'
```

Compare:
- Wall-clock runtime
- Peak CPU usage (Activity Monitor → CPU tab)
- Frames processed per second

## Conclusion

The primary issue appears to be **stale server code**. The current implementation (in working directory) should work correctly. Restart the server and retest before further investigation.

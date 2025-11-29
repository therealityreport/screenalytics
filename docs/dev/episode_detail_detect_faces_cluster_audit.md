# Episode Detail Pipeline Audit (Detect/Track, Faces Harvest, Cluster)

**Date:** 2025-11-29 (reconciled)  
**Branch:** nov-24  
**Status:** Merged audit + fixes todo

This document merges the two prior audits. Code on branch `nov-24` is the source of truth; issues marked NO ISSUE in the second audit were re-checked and downgraded only where the code actually fixed them.

---

## 1) Call Chains (confirmed)

### Detect/Track
UI (`pages/2_Episode_Detail.py`) → `helpers.run_pipeline_job_with_mode("detect_track")` → `/celery_jobs/detect_track` (local or redis) → Celery/local task → `tools/episode_run.py` detect/track → artifacts: `runs/detect_track.json`, `detections.jsonl`, `tracks.jsonl`, `progress.json`, frames/crops if enabled.

### Faces Harvest (faces_embed)
UI → `run_pipeline_job_with_mode("faces_embed")` → `/celery_jobs/faces_embed` → task → `tools/episode_run.py --faces-embed` → artifacts: `runs/faces_embed.json`, `faces.jsonl`, crops/thumbs/embeds if enabled.

### Cluster
UI → `run_pipeline_job_with_mode("cluster")` → `/celery_jobs/cluster` → task → `tools/episode_run.py --cluster` → artifacts: `runs/cluster.json`, `identities.json`, `track_metrics.json`.

Naming and ep_id propagation are consistent across UI/API/tasks/markers.

---

## 2) Merged Issue List (code-verified)

### Detect/Track
- **P1 – Silent CPU fallback even when accelerator requested.**  
  `_onnx_providers_for(...)` in `tools/episode_run.py` always allows CPUExecutionProvider; RetinaFace/ArcFace prep proceeds on CPU when CoreML/CUDA unavailable. No explicit opt-in flag.  
  **Fix:** Require explicit `allow_cpu_fallback` to add CPU EP when a non-CPU device is requested; otherwise fail with clear error. Log requested vs effective providers.

- **P1 – Heavy defaults in UI payload builder.**  
  `default_detect_track_payload` (`ui_helpers.py:1366+`) seeds `save_frames=True`, `save_crops=True`, `jpeg_quality=85` despite API models being light. Users can start heavy runs unintentionally.  
  **Fix:** Default to `save_frames=False`, `save_crops=False` (or True only if explicitly desired), `jpeg_quality≈70`. Align with `DetectTrackCeleryRequest` defaults.

- **P1 – Execution mode defaults to local.**  
  `EXECUTION_MODE_DEFAULT = "local"` (`ui_helpers.py:185-214`). If Redis is available, we still default local, which can peg CPU/thermals.  
  **Fix:** Default to `redis` when reachable; otherwise local with a strong warning. Persist per-episode choice.

- **P2 – Scene detector always on by default.**  
  `SCENE_DETECTOR_DEFAULT` is PySceneDetect (env default) and payload seeds it unconditionally. Local low-power runs pay an extra pass.  
  **Fix:** Gate scene detection by profile: off/internal for low_power/balanced, enable PySceneDetect only for performance profile or explicit toggle.

- **P2 – CPU thread caps not enforced on fallback.**  
  When an accelerator request falls back to CPU, thread caps from profile/env are not enforced in provider selection.  
  **Fix:** If effective device is CPU, apply a configurable cap (`SCREENALYTICS_LOCAL_MAX_THREADS`, default 2–4) and record in markers/logs.

### Faces Harvest (faces_embed)
- **P1 – Stale detection missing when manifest absent but marker exists.**  
  `_faces_phase_status` sets `last_run_at` from `faces.jsonl` mtime only; if marker exists but manifest missing, stale compare never fires.  
  **Fix:** Use marker `finished_at` as primary timestamp, manifest mtime as fallback. If detect/track is newer, mark faces `stale` even if manifest missing.

- **P2 – Zero-row UX missing.**  
  API sets `zero_rows`, but UI does not present a clear “0 faces found” state or block clustering by default.  
  **Fix:** In Episode Detail/Faces Review, surface zero-row explicitly and gate clustering unless acknowledged.

### Cluster
- **P1 – Cluster stale detection still weak for marker-only or metrics-only runs.**  
  `last_run_at` uses max of identities/track_metrics mtimes, but if marker exists without outputs, detect/track reruns won’t mark stale (no timestamp).  
  **Fix:** Include marker `finished_at` as timestamp. If detect/track or faces newer than marker/metrics, mark cluster `stale` and set `tracks_only_fallback` when identities missing.

- **P2 – Zero-identity UX missing.**  
  API sets `zero_rows`, but UI doesn’t warn when identities=0.  
  **Fix:** Add explicit zero-state warning and prompt rerun/merge guidance.

### Execution Mode / Jobs / Logs
- **P1 – Local vs Redis labeling and warnings.**  
  UI still presents local as default, with no strong warning for heavy runs when accelerators are absent.  
  **Fix:** When defaulting to local, warn if device will be CPU or if frames/crops are enabled at high JPEG.

- **P2 – Running-job recovery after reload depends on `/celery_jobs`.**  
  `/celery_jobs` now returns `ep_id`/`operation`, but UI only matches when both fields present; missing fields or non-standard task names can bypass detection.  
  **Fix:** Harden `get_running_job_for_episode` to match by stored job_id OR `(ep_id, operation)`; ensure backend always populates both.

- **P2 – Celery log persistence not guaranteed.**  
  Local logs are persisted; Celery-mode runs may skip `save_operation_logs`, so hydration can be empty after reload.  
  **Fix:** Ensure Celery handlers call `save_operation_logs` on completion/error for all three ops.

### Payloads / Defaults / Thermal Caps
- **P1 – Detect payload defaults heavy (see above).**
- **P2 – Scene detector default heavy (see above).**
- **P3 – Faces embed defaults**: `save_crops=True` by default may be fine; consider lighter option for laptop runs but not critical.

---

## 3) Final Fix Plan (implementation targets)

1) Device/Provider safety  
- Add explicit `allow_cpu_fallback` flag; fail fast when accelerator requested but unavailable.  
- Log requested vs effective providers; enforce CPU thread caps on fallback.

2) Execution mode + payload defaults  
- Default to `redis` when available; warn loudly on local heavy runs.  
- Lighten detect defaults: frames off, crops off (or explicit), JPEG ~70; gate scene detection by profile.

3) Stale detection & zero-row UX  
- Faces: use marker finished_at for last_run_at; stale when detect/track newer even without manifest; show 0-faces state.  
- Cluster: use marker finished_at as timestamp; stale when detect/track or faces newer even if outputs missing; warn on 0 identities.

4) Jobs/logs robustness  
- Ensure `/celery_jobs` always sets `ep_id` and `operation`; harden UI matching.  
- Persist Celery logs for hydration; improve log display to keep noisy output behind a toggle (short-term: tag noisy lines; long-term: quiet flag).

5) CPU caps  
- Configurable local CPU cap (env, default 2–4) applied whenever effective device is CPU.

---

## 4) Criticals to fix first (P1)
- Silent CPU fallback for accelerators.  
- Heavy detect payload defaults + scene detection gating.  
- Execution mode default (prefer redis) + warnings.  
- Faces/cluster stale detection gaps (marker-only cases).  
- Cluster stale + metrics/marker timestamps.  
- Celery log persistence (so hydration works).

## 5) Nice-to-have (P2/P3)
- Zero-row UX for faces/cluster.  
- Job detection hardening on reload.  
- Noisy log filtering / better progress bar smoothing.  
- Optional lighter faces_embed defaults on laptops.
**Suggested Fix:** Consider making default `False` and having UI explicitly enable when user wants crops.

---

## 4. Additional Issues Discovered

#### ISSUE-17: Session State Job ID Lost on Page Refresh (P2)
**Symptom:** If user refreshes page mid-job, stored job_id in session state is lost.
**Files:** `store_celery_job_id()` / `get_stored_celery_job_id()` in ui_helpers.py
**Root Cause:** Streamlit session state is browser-session-specific and reset on page refresh.
**Suggested Fix:** The fallback mechanisms (checking `/celery_jobs/local` and `/celery_jobs`) already handle this. However, progress display may be temporarily unavailable until next poll. Consider:
1. Persisting job_id to a file alongside progress.json
2. Or accepting current behavior as adequate (jobs are still detected via API polling)

#### ISSUE-18: Running Job Detection Timeout (P3)
**Symptom:** `get_running_job_for_episode()` makes up to 4 API calls with 10s timeout each.
**Files:** ui_helpers.py:2685-2813
**Root Cause:** Sequential API calls can add latency.
**Suggested Fix:**
1. Run checks in parallel (asyncio.gather)
2. Or implement a single unified `/jobs/running?ep_id=X` endpoint

#### ISSUE-19: Local Job PID Tracking Lost on API Restart (P2)
**Symptom:** `_running_local_jobs` dict in celery_jobs.py is in-memory only.
**Files:** celery_jobs.py:155-157
**Root Cause:** If API server restarts mid-job, the registry is lost.
**Existing Mitigation:** `_detect_running_episode_processes()` (lines 244-298) scans for orphan `episode_run.py` processes, so jobs are still detected.
**Status:** Adequately handled by orphan detection.

#### ISSUE-20: Duplicate Job Prevention Relies on In-Memory Registry (P2)
**Symptom:** Two browser tabs could start duplicate jobs if API restarts between requests.
**Files:** `_get_running_local_job()` in celery_jobs.py
**Existing Mitigation:** Orphan process detection catches this, but there's a race window.
**Suggested Fix:** Use file-based lock (`data/manifests/{ep_id}/.lock_{operation}`) for truly persistent locking.

---

## 5. Summary

### Critical Issues to Fix First (P1)
**None identified** - The core pipeline is functioning correctly.

### High Priority (P2)
| Issue | Description | Fix Complexity |
|-------|-------------|----------------|
| ISSUE-10 | Progress bar may not update smoothly | Low |
| ISSUE-12 | Noisy log output not filtered | Low |
| ISSUE-17 | Session state job ID lost on refresh | Medium |
| ISSUE-19 | Local job PID tracking lost on restart | Already mitigated |
| ISSUE-20 | Duplicate job prevention race condition | Medium |

### Nice-to-Have/Cleanup (P3)
| Issue | Description | Fix Complexity |
|-------|-------------|----------------|
| ISSUE-01 | Local mode CPU cap too aggressive | Low |
| ISSUE-16 | save_crops default in faces_embed | Low |
| ISSUE-18 | Running job detection timeout | Low |

---

## 6. Verification Checklist

- [x] Operation names consistent (detect_track, faces_embed, cluster)
- [x] ep_id passed correctly through all layers
- [x] Artifacts written to expected paths
- [x] Stale detection works for all three phases
- [x] Zero-row runs treated as success, not missing
- [x] Local mode uses CoreML/MPS when available
- [x] CPU thread limits applied in local mode
- [x] Celery job detection works after page reload
- [x] Buttons disabled when job running
- [x] Logs persisted and loadable after completion/crash
- [x] Default payloads are lightweight (no heavy exports by default)

---

## 7. Next Steps

1. Review this audit with stakeholder
2. Prioritize P2 issues based on user impact
3. Create implementation plan for selected fixes
4. Consider adding integration tests for job lifecycle

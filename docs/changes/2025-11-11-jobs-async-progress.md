# 2025-11-11 — Async detect/track progress

## Why
- Blocking `POST /jobs/detect_track` tied up the API/UI for multi-minute runs.
- We need resumable progress visibility (elapsed vs total time, ETA, cancel) for on-device experimentation and Ops dashboards.

## Highlights
- Introduced `apps/api/services/jobs.JobService` — filesystem-backed job records under `data/jobs/{job_id}.json`.
- `tools/episode_run.py` now accepts `--progress-file`, periodically writes `frames_done/frames_total/elapsed/analyzed_fps/device`, and finalizes with detection/track counts.
- New API surface:
  - `POST /jobs/detect_track_async` launches `tools/episode_run.py` via `subprocess.Popen` and returns `{job_id, ep_id, progress_file}`.
  - `GET /jobs/{job_id}/progress` streams the latest progress JSON plus job state.
  - `GET /jobs/{job_id}` exposes the terminal summary/error.
  - `POST /jobs/{job_id}/cancel` SIGTERMs the subprocess and flips `state=canceled`.
- Streamlit upload helper now defaults to the async run, shows a live progress bar with MM:SS readouts + ETA, surfaces cancel controls, and links directly to detections/tracks on success. A blocking fallback button remains for emergencies.

## Testing
- Added `tests/api/test_jobs_async_progress.py` to ensure `/jobs/{job_id}/progress` reflects the latest frames_done.
- `tests/ui/test_streamlit_pages_compile.py` still compiles every Streamlit page after the refactor.

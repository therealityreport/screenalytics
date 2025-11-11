# 2025-11-11 — Progress SSE + Artifact Exports

## Summary

- `episode_run.py` now emits structured JSON progress (phase, frames, seconds, device, fps) to stdout and `data/manifests/{ep}/progress.json`, adds `--save-frames`, `--save-crops`, and `--jpeg-quality`, and mirrors frames/crops/manifests to the v2 S3 hierarchy (`artifacts/frames|crops|manifests/...`) when `STORAGE_BACKEND=s3`.
- API: `POST /jobs/detect_track` supports Server-Sent Events (`event: progress/done/error`), `/episodes/{ep}/progress` surfaces the latest `progress.json`, and storage helpers expose `EpisodeContext`, `artifact_prefixes`, `put_artifact`, and `sync_tree_to_s3`.
- Streamlit Episode Detail page shows a live progress bar (SSE with `/episodes/{ep}/progress` polling fallback), exposes the export checkboxes + JPEG quality input, and renders the three S3 prefixes plus quick links into Faces Review and Screentime.
- Tests cover the new progress endpoint, SSE ratio/ETA math, and artifact path builders; README documents the workflow.

## Details

1. `tools/episode_run.py`
   - Computes totals from detected FPS (auto when `--fps=0`), pushes `phase` updates every ~25 frames, and writes a terminal `phase:"done"` record with detection/track counts + artifact prefixes.
   - Optional frame/crop sampling saves under `data/frames/{ep}/frames|crops/` using `cv2.imencode(..., IMWRITE_JPEG_QUALITY, quality)` and syncs to `artifacts/frames|crops/...`; manifests sync to `artifacts/manifests/...`. Failures log warnings, never fail the job.

2. API
   - `/jobs/detect_track` inspects `Accept: text/event-stream` and streams JSON lines straight from the runner; otherwise it returns the previous JSON payload (now with `progress_file`).
   - `/episodes/{ep}/progress` returns the most recent JSON snapshot for polling clients.
   - `StorageService` exposes `EpisodeContext`, `artifact_prefixes`, best-effort `put_artifact`, and `sync_tree_to_s3`.

3. UI
   - Episode Detail run form shows **Save frames to S3**, **Save face crops to S3**, and **JPEG quality** controls, streams SSE progress, and falls back to polling `/episodes/{ep}/progress` if SSE isn’t available.
   - Post-run summary lists detections/tracks, exported frame/crop counts, S3 prefixes, and buttons to jump to Faces Review / Screentime.
   - “Artifacts” section permanently lists the v2 prefixes so operators can inspect S3 even before exports run.

4. Docs & Tests
   - README documents the new CLI flags, SSE progress behavior, and UI controls.
   - Added tests: `/episodes/{ep}/progress`, SSE ratio/ETA monotonicity, and artifact prefix builder.

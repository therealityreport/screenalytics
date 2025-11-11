# 2025-11-11 — Faces Pipeline & Facebank

## Summary

- `tools/episode_run.py` grew `--faces-embed` and `--cluster` modes that reuse `ProgressEmitter`, write `faces.jsonl`/`identities.json`, sync optional crops + manifests to the v2 `artifacts/` prefixes, and stream SSE updates (with `stage` metadata) just like detect/track.
- FastAPI now exposes `POST /jobs/faces_embed` / `POST /jobs/cluster` (SSE + JSON fallback) plus async variants so the UI can poll `progress.json` when event streams are blocked.
- Streamlit’s Episode Detail page gained controls for faces + clustering (stub/device/crops/JPEG quality), reusing the shared SSE helper to show MM:SS progress, ETA, and artifact prefixes on completion.
- The Facebank view (page 3) loads `faces.jsonl` + `identities.json`, renders thumbnails (local or presigned from S3), and ships basic review tooling—rename identities, merge them, and move tracks—persisting updates locally and to S3.

## Details

1. **Runner**
   - `--faces-embed` replays `tracks.jsonl`, samples per-track crops (optional), generates deterministic embeddings, emits `phase:"faces_embed"` progress lines, and writes `faces.jsonl`. `--save-crops` mirrors JPEGs to `artifacts/crops/{show}/s{ss}/e{ee}/tracks/`.
   - `--cluster` ingests `faces.jsonl`, chunking tracks into lightweight identities with counts + representative crops, writing `identities.json` and syncing it to `artifacts/manifests/...`. Both phases reuse `_sync_artifacts_to_s3` and annotate `summary["stage"]`.

2. **API**
   - Added `/jobs/faces_embed` + `/jobs/cluster` (and `_async` companions) that wrap the shared subprocess/SSE helper, validate prerequisites, and return counts/S3 prefixes when clients prefer JSON.

3. **UI**
   - Episode Detail now lets operators run detect/track, faces harvest, and clustering with live progress bars, device pickers, crop export toggles, and instant links into Faces Review / Screentime.
   - Faces Review (Facebank) renders artifact paths, shows simple metrics, runs faces/cluster jobs with progress, and exposes rename/merge/move operations while syncing edits to S3.

4. **Tests**
   - Added `tests/api/test_jobs_faces_cluster_stub.py` for the happy-path stub flows + error cases.
   - Added `tests/api/test_progress_faces.py` to assert `progress.json` includes the `stage` transitions (`faces_embed` → `cluster` → `done`).

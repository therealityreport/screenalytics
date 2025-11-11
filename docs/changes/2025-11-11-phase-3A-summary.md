# 2025-11-11 — Phase-3A upload UI summary

## What shipped
- FastAPI `POST /episodes` (idempotent) plus `POST /episodes/{ep_id}/assets` presign endpoint with optional local backend.
- Streamlit uploader that creates episodes, PUTs the video (local or S3/MinIO), mirrors bytes to `data/videos/{ep_id}/episode.mp4`, and optionally calls the full detect/track pipeline.
- `POST /jobs/detect_track` shells out to `tools/episode_run.py`, counting detections/tracks written under `data/manifests/{ep_id}/`.
- README + change logs for dependency profiles, acceptance flow, and troubleshooting PyAV.

## How to run acceptance
```bash
export STORAGE_BACKEND=local
export SCREENALYTICS_API_URL=http://localhost:8000
export UI_ORIGIN=http://localhost:8501
uv run apps/api/main.py  # terminal 1
streamlit run apps/workspace-ui/streamlit_app.py  # terminal 2
```
In the UI, enter Show/Season/Episode, upload a small `.mp4`, and submit.

Verify:
- UI reports the `ep_id` and "Upload successful".
- `data/videos/{ep_id}/episode.mp4` exists.
- `data/manifests/{ep_id}/detections.jsonl` and `tracks.jsonl` are populated by the run.

## Acceptance checklist
- [x] `POST /episodes` returns deterministic `ep_id`
- [x] `POST /episodes/{ep_id}/assets` returns presigned PUT (or local path)
- [x] Streamlit flow handles upload + detect/track trigger
- [x] `pytest -q` → `15 passed`
- [x] README and troubleshooting docs updated
- [x] CORS allows `http://localhost:8501` (via `UI_ORIGIN`)

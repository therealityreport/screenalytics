# 2025-11-11 — Phase-3A acceptance wrap

## What shipped
- Streamlit upload workflow (episode create → presign → upload → detect/track stub) behind `feat-upload-ui-streamlit-presign-detect-track`.
- Idempotent `POST /episodes` plus presign responses that expose `method` (`PUT` or `FILE`) and headers/path for client uploads.
- `.env.example` and README instructions covering core vs ML installs, env exports, and artifact locations.

## How to run acceptance
```bash
pip install -r requirements-core.txt
cp .env.example .env
set -a && source .env && set +a
uv run apps/api/main.py
streamlit run apps/workspace-ui/streamlit_app.py
```
In the UI, fill Show/Season/Episode, upload a small `.mp4`, enable stub run, and submit.

Verify:
- UI shows the returned `ep_id` and “Upload successful”.
- Video mirrored at `data/videos/{ep_id}/episode.mp4`.
- Stub manifests at `data/manifests/{ep_id}/detections.jsonl` and `tracks.jsonl`.
- `pytest -q` (core profile) → `15 passed`.

## Checklist
- [x] Idempotent `POST /episodes`
- [x] Presign returns `Content-Type: video/mp4` for PUT, path for FILE
- [x] Streamlit upload flow (create → presign → upload → stub)
- [x] Artifacts under `data/manifests/{ep_id}/`
- [x] README + `.env.example` updated
- [x] Tests green under requirements-core profile

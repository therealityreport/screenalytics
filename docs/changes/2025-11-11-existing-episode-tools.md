# 2025-11-11 — Existing Episode tools (mirror + detect/track)

## Highlights
- API now exposes `GET /episodes`, `GET /episodes/{ep_id}`, and `POST /episodes/{ep_id}/mirror` so the UI can list selectable episodes, inspect S3/local status, and download the source video from S3 when needed.
- Streamlit UI adds an **Existing Episode** mode with a searchable dropdown, mirror button, and detect/track controls (stride, fps, stub toggle) for any `ep_id` that already lives in object storage.
- Storage service grows `ensure_local_mirror` + `s3_object_exists`, and new backend-aware tests cover local vs. S3 mirroring semantics.

## Why it matters
Operators can now revisit any uploaded episode, hydrate the local filesystem from S3 in one click, and run the detect/track stage without re-uploading the video. This closes the loop for reprocessing tasks, QA, and iterative tuning sessions.

## How to use it
1. Run the API + Streamlit UI.
2. In the sidebar, switch Mode → **Existing Episode**.
3. Select an `ep_id`, click **Mirror from S3**, then **Run detect/track** (YOLO or stub).
4. Manifests appear at `data/manifests/{ep_id}/` for inspection.

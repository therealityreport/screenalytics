# Master WEB To-Do List (Streamlit → Next.js)

Status: Phase 1 → Phase 2 transition  
Owner: Screenalytics eng (assign per feature)  
Scope: Next.js app under `web/` to reach feature parity with Streamlit workspace UI (Upload, Episode Detail, Faces Review) while keeping pipeline guardrails intact.

## 0) Cleanup & Branding
- [ ] Rebrand Next shell to Screenalytics: update `web/app/layout.tsx`, `web/package.json` name/metadata, and global copy.
- [ ] Quarantine or remove youth-league/Prisma demo pages; keep `web/app/screenalytics/**` as the entrypoint.
- [ ] Keep CSS Modules + light globals; codify lint/format defaults and Radix/toast provider usage.

## 1) API Contracts & Events
- [ ] Export OpenAPI from FastAPI and regenerate `web/api/schema.ts`; add a script/README note for refresh cadence.
- [x] Normalize API error envelope `{ code, message, details }` in `web/api/client.ts`.
- [ ] Finish SSE/WS endpoint wiring for episode events (phase start/finish/error, manifest mtime, metrics) and map into `useEpisodeEvents`.
- [ ] Document required endpoints (presign, status, job triggers, manifests/logs) and proxy config (`next.config.mjs`, `.env.local.example`).

## 2) Upload Parity
- [x] State machine in `web/lib/state/uploadMachine.ts` (idle → preparing → ready → uploading → verifying → processing → success/error).
- [ ] Expand UI to mirror Streamlit guardrails: show/season/episode selectors with replace lock, FILE/local-only success path, tracks-only + faces manifest fallback chips.
- [ ] Add cancel/retry + ETA, better progress smoothing, and clearer errors for presign/upload/verify.
- [ ] Trigger detect/track post-upload with actionable toasts; keep status polling keyed per episode.

## 3) Episode Detail Parity
- [ ] Mirror readiness rules: detections+tracks required; tracks_only disables harvest; faces manifest fallback shows “status stale, using manifest.”
- [ ] Render cluster metrics pre/post merge; show zero-row warnings for faces/identities.
- [ ] Add rerun buttons per phase with spinners; screentime job chip stored per-episode key; log drawer fed by SSE and status invalidations.
- [ ] Handle stale markers/manifest mtimes and manifest-fallback readiness when status is missing.

## 4) Faces Review Build-out
- [ ] Replace stub with virtualized faces.jsonl grid (filters/search, keyboard nav).
- [ ] Actions: merge/delete/move honoring cluster IDs and merged IDs alignment with screentime; respect tracks-only + manifest-fallback warnings.
- [ ] Use presigned thumbnails and paging to avoid loading all crops; support sample-every-N controls.

## 5) Observability & Tests
- [ ] MSW fixtures for Episode Detail and Faces (Upload already mocked) to enable `NEXT_PUBLIC_MSW=1`.
- [ ] Unit tests for upload/episode data mappers and SSE handling; Playwright smoke for Upload + Episode Detail.
- [ ] Backend contract tests for any new/changed endpoints; keep `python -m py_compile` on touched Python files.

## 6) Rollout & Operations
- [ ] Run Streamlit + Next in parallel; ship Upload first behind a beta flag, then Episode Detail, then Faces Review.
- [ ] Document local dev (`uvicorn` + `next dev`), env vars (`NEXT_PUBLIC_API_BASE`, `NEXT_PUBLIC_WS_BASE`, `NEXT_PUBLIC_MSW`), and troubleshooting (presign failures, SSE reconnect).
- [ ] Deprecate Streamlit pages once each Next page reaches parity; remove duplicate logic last.

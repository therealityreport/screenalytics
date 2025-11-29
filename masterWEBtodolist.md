# Master WEB To-Do List (Streamlit → Next.js)

## Foundation
- [ ] Confirm API surface for Upload/Episode/Jobs/logs in `apps/api/routers/episodes.py` et al; export OpenAPI spec and generate TypeScript client.
- [ ] Normalize API error envelope `{ code, message, details }` and document in `web/api/client.ts`.
- [ ] Add SSE/WS endpoint for job events (phase start/finish, errors, manifest mtime, metrics) with cache-bust tokens.
- [ ] Create `.env.local.example` with `NEXT_PUBLIC_API_BASE`, `NEXT_PUBLIC_WS_BASE`; add Next rewrites proxy to FastAPI.
- [ ] Decide CSS system (Tailwind vs CSS modules) and install Radix/Headless primitives; set up TanStack Query provider and toast/log drawer.

## API Client Layer
- [ ] Implement generated types in `web/api/schema.ts` and lightweight fetcher in `web/api/client.ts` (openapi-fetch or zodios).
- [ ] Hooks for core calls: create/replace episode upload (presign), job triggers by phase, status/manifests fetch, logs/events stream.
- [ ] Shared upload helper for presigned PUT/POST with progress + abort controller; optional MD5/ETag verification step.

## Upload Page
- [ ] Build state machine hook (`web/lib/state/uploadMachine.ts`): idle → preparing → ready → uploading → verifying → processing → success/error.
- [ ] UI: drag/drop + picker, size/MD5 display, replace-mode locking via `?ep_id`, progress bar with speed/ETA, cancel/retry controls.
- [ ] Wire to API: presign request, PUT upload with progress, optional ETag/MD5 verify, trigger detect/track job, start status polling + event stream.
- [ ] Error handling: actionable messages, retry from safe state, treat local-only uploads as success without cloud checks when flagged.

## Episode Detail Page
- [ ] Data model mirrors Streamlit guardrails: tracks-only sets `tracks_only_fallback` and disables harvest; faces manifest fallback sets `faces_manifest_fallback` with warning; cluster metrics pre/post merge.
- [ ] Poll status + subscribe to SSE/WS for job updates and manifest mtime invalidation.
- [ ] UI: manifests/status cards, rerun buttons per phase with spinners, log drawer, screentime job tracking stored per-episode key.
- [ ] Edge cases: missing/stale status but faces.jsonl present counts as ready; require detections + tracks for detect/track readiness.

## Faces Review Page
- [ ] Virtualized faces grid (faces.jsonl) with filters/search; keyboard navigation for selection/approval.
- [ ] Actions respect cluster IDs and merge rules; ensure merged IDs align with screentime outputs.
- [ ] Graceful handling when faces present but status stale (manifest fallback) or tracks-only; show warnings accordingly.

## Testing & QA
- [ ] MSW mocks for Upload/Episode APIs to allow `next dev` without backend.
- [ ] Unit tests for state machines (upload, episode detail data mapping).
- [ ] Playwright smoke: upload happy path, detect/track progress, faces grid load.
- [ ] Backend contract tests for any new/changed endpoints; keep `python -m py_compile` on touched Python files.

## Rollout
- [ ] Run Streamlit and Next in parallel; start with Upload page beta flag.
- [ ] Document dev workflow (`uvicorn` + `next dev`) and troubleshooting (presign failures, SSE connectivity).
- [ ] Deprecate Streamlit pages once parity confirmed; remove duplicate logic last.

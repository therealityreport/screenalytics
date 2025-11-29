# Master WEB To-Do List (Streamlit → Next.js)

## Foundation (Phase 1)
- [ ] Confirm API surface for Upload/Episode/Jobs/logs in `apps/api/routers/episodes.py` et al; export OpenAPI spec and generate TypeScript client.
- [x] Normalize API error envelope `{ code, message, details }` and document in `web/api/client.ts`.
- [x] Add SSE/WS endpoint for job events (phase start/finish, errors, manifest mtime, metrics) with cache-bust tokens.
- [x] Create `.env.local.example` with `NEXT_PUBLIC_API_BASE`, `NEXT_PUBLIC_WS_BASE`; add Next rewrites proxy to FastAPI.
- [x] Choose styling approach: CSS modules + lightweight globals (no Tailwind); QueryClient provider scaffolded (Radix/toast wiring still to add).

## API Client Layer
- [x] Implement generated types in `web/api/schema.ts` via OpenAPI output and align client wrappers.
- [x] Hooks for core calls: create/replace episode upload (presign), job triggers by phase, status/manifests fetch (events stream placeholder via SSE hook).
- [x] Shared upload helper for presigned PUT/POST with progress + abort controller; optional MD5/ETag verification step.

## Upload Page
- [x] Build state machine hook (`web/lib/state/uploadMachine.ts`): idle → preparing → ready → uploading → verifying → processing → success/error.
- [x] UI: drag/drop + picker, size display, replace-mode locking via `?ep_id`, progress bar, reset control.
- [x] Wire to API: presign request, PUT upload with progress, trigger detect/track job, start status polling (MSW mocks enabled).
- [x] Error handling: actionable messages (toast + inline), retry/reset, treat local-only uploads as success and kick detect/track when flagged.

## Episode Detail Page
- [ ] Data model mirrors Streamlit guardrails: tracks-only sets `tracks_only_fallback` and disables harvest; faces manifest fallback sets `faces_manifest_fallback` with warning; cluster metrics pre/post merge. (Stub UI at `/screenalytics/episodes/[id]`.)
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

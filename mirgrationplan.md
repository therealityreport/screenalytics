# Migration Plan: Streamlit to Next.js/FastAPI UI

## Goals
- Replace Streamlit workspace UI with a typed Next.js (App Router) front end backed by the existing FastAPI API and pipeline tooling.
- Preserve pipeline guardrails: detect/track readiness, faces manifest fallback, cluster merge semantics, screentime job tracking, and explicit error surfacing.
- Keep local usability (macOS) while enabling faster UX, richer state handling, and better observability.

## Target Architecture
- **Backend**: FastAPI remains the single API surface; expose job triggers and status/manifests (detect, track, faces, cluster, screentime) plus presigned upload endpoints. Add WebSocket/SSE channel for job progress and cache invalidation.
- **Frontend**: Next.js (TypeScript, App Router) in `web/`, using TanStack Query for data/polling, a small global store (zustand) for session state, and Radix/Headless primitives for accessibility. Keep Streamlit running until parity.
- **Communication**: REST for requests; SSE/WS for progress events. Standardized error envelope `{ code, message, details }` propagated to UI to avoid silent failures.

## Phased Approach
1) **Inventory & Contracts**
   - Document required endpoints from `apps/api/routers/episodes.py` and related routers (upload, status, job triggers, manifests, logs).
   - Generate OpenAPI schema and client types (`openapi-typescript` + `openapi-fetch` or `zodios`) to keep UI and API in lockstep.
   - Confirm presign flows: PUT/POST for cloud, local-only path that short-circuits cloud validation.

2) **Eventing & Caching**
   - Add/confirm SSE or WebSocket endpoint for episode job events (phase start/finish, errors, metric updates, manifest mtime changes). Include cache-bust tokens in payloads to invalidate status polling.
   - Standardize status objects to include `tracks_only_fallback`, `faces_manifest_fallback`, and merge-mode flags so UI can mirror Streamlit rules.

3) **Frontend Scaffold (web/)**
   - Baseline Next.js setup with ESLint/Prettier, Tailwind or CSS modules, Radix UI, TanStack Query provider, and toast/log drawer scaffolding.
   - Layout with left nav (Upload, Episodes, Faces Review), top connectivity chip, and a debug log drawer for job events.
   - Dev proxy in `next.config.mjs` rewriting `/api/**` to FastAPI (`http://localhost:8000`), `.env.local.example` with `NEXT_PUBLIC_API_BASE` and `NEXT_PUBLIC_WS_BASE`.

4) **Feature Parity by Page**
   - **Upload**: two modes (new vs replace via `?ep_id`). Drag/drop + picker, file metadata validation, presign request, PUT with progress, post-upload verification (ETag/MD5 when available), then trigger detect/track job and start progress stream.
   - **Episode Detail**: show manifests/status with the same fallback rules as Streamlit: tracks-only disables harvest, faces readiness via manifest fallback, cluster metrics pre/post merge, screentime job tracking via session-scoped keys. Provide rerun buttons with spinners and error surfacing.
   - **Faces Review**: virtualized grid for faces.jsonl; filters/search; keyboard nav; merge/delete actions respecting cluster IDs.

5) **Observability & Testing**
   - UI: MSW mocks for API, unit tests for state machines, Playwright smoke for Upload and Episode Detail flows.
   - Backend: contract tests for API responses used by UI; keep `python -m py_compile` on touched Python files; ensure SSE/WS handlers covered.

6) **Rollout**
   - Run Streamlit and Next in parallel; gate traffic to Next per-page as parity is reached (start with Upload, then Episode Detail, then Faces Review).
   - Add a deprecation notice in Streamlit once Next parity is confirmed; remove Streamlit-only logic after validation.

## Risks & Mitigations
- **Long-running jobs**: Without SSE/WS, polling may lag; prioritize event stream implementation and graceful fallbacks.
- **State divergence**: Mirror manifest-based readiness (tracks-only, faces manifest fallback) in UI models; include backend flags to prevent drift.
- **Auth/secrets**: Keep presign only on backend; never expose credentials in Next; use environment vars for base URLs.
- **Dual maintenance**: Minimize by incrementally turning off Streamlit pages once Next reaches parity.

## Success Criteria
- Upload, Episode Detail, and Faces Review flows run via Next.js with equivalent guardrails and error handling as Streamlit.
- Users can run locally (`uvicorn` + `next dev`) with clear `.env` configuration.
- Automated tests (API contracts, UI mocks, Playwright smoke) pass and prevent regressions.

## Detailed Next Steps (completed specs)

### 1) Next.js Scaffold + Typed API Client
- Keep Next.js App Router in `web/`; add providers for TanStack Query, Radix toast, and WS/SSE connectivity status in `web/app/layout.tsx`.
- Generate `web/api/schema.ts` from FastAPI OpenAPI via `openapi-typescript`; wrap with `openapi-fetch` in `web/api/client.ts` for typed calls.
- Rewrites in `next.config.mjs`: proxy `/api/:path*` to `http://localhost:8000/:path*`; configure `transpilePackages` only if needed.
- Env: `.env.local.example` with `NEXT_PUBLIC_API_BASE`, `NEXT_PUBLIC_WS_BASE`, optional `NEXT_PUBLIC_MSW=1` to enable mocks.
- Shared utilities: fetcher that normalizes errors to `{ code, message, details }`, attaches `AbortController`, and opts out of retries by default.

### 2) Upload Page UI + State Machine (mockable)
- State machine (`web/lib/state/uploadMachine.ts`):
  - `idle -> preparing`: load shows/seasons or locked episode metadata when `ep_id` query exists.
  - `preparing -> ready`: selections validated; replace mode locks selectors.
  - `ready -> uploading`: request presign (or local-only flag), start PUT/POST with progress + abort support.
  - `uploading -> verifying`: optional ETag/MD5 head check; cloud vs local-only branches.
  - `verifying -> processing`: trigger detect/track job; start status polling + event stream subscription.
  - `processing -> success`: detect+track ready; flag `tracks_only_fallback` if tracks exist without detections and disable harvest CTA.
  - Any -> `error`: carry code/message, allow retry from ready.
- UI (`web/app/upload/page.tsx`): drag/drop + picker, file metadata badge, progress with speed/ETA, cancel/retry, replace-mode banner, manifest-fallback warnings on return from processing.
- Mocks: MSW handlers for presign, PUT (simulated), job trigger, status polling to enable `next dev` without backend.

### 3) Episode Detail Polling/SSE Flow
- Data model mirrors Streamlit guardrails: `tracks_only_fallback` disables harvest; `faces_manifest_fallback` shows “status stale, using manifest”; require detections + tracks for ready.
- Fetch/poll: TanStack Query keyed by `episodeId`; staleTime short; refetch on focus/reconnect; revalidate when manifest mtime changes.
- SSE/WS: subscribe to `/episodes/{id}/events` for phase updates (start, finish, error), metric deltas (cluster pre/post merge), manifest mtime tokens, and screentime job IDs. Push events into a log drawer and invalidate relevant queries.
- UI (`web/app/episodes/[id]/page.tsx`): status cards per phase with rerun buttons, warnings for fallbacks, log drawer, screentime job chip keyed to `f"{id}::screentime_job"` to match Streamlit behavior.

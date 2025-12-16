# Web App Migration: Inventory & Triage

> **Created:** 2025-12-15
> **Purpose:** Audit all migration-related docs and code to identify what's current, outdated, or missing.

---

## Part 1: Document Inventory

### Migration Plans & TODOs

| Path | Type | Status | Action | Notes |
|------|------|--------|--------|-------|
| `docs/plans/in_progress/WEB_APP_MIGRATION_PLAN.md` | plan | **outdated** | update | Last updated 2024-12-02; percentages don't match reality; good structure to keep |
| `docs/_archive/new-features/masterWEBtodolist.md` | todo | **outdated** | archive | Moved to archive; checklist format useful but superseded by migration plan |
| `docs/plans/in_progress/product/prd.md` | spec | current | keep | References web UI integration in Phase 4; aligns with roadmap |
| `docs/plans/in_progress/infra/screenalytics_deploy_plan.md` | plan | current | keep | Mentions Vercel deployment for web app |

### API & Architecture References

| Path | Type | Status | Action | Notes |
|------|------|--------|--------|-------|
| `docs/reference/api.md` | reference | current | keep | Core API docs; web client must align |
| `docs/architecture/solution_architecture.md` | spec | current | keep | Shows web UI component in architecture |
| `docs/architecture/directory_structure.md` | reference | current | keep | Documents `web/` structure |
| `docs/reference/data_schema.md` | reference | current | keep | Artifact schemas web UI must consume |

### Pipeline Docs (UI Contract Dependencies)

| Path | Type | Status | Action | Notes |
|------|------|--------|--------|-------|
| `docs/pipeline/overview.md` | reference | current | keep | Defines pipeline stages web UI displays |
| `docs/pipeline/detect_track_faces.md` | reference | current | keep | Status fields Episode Detail needs |
| `docs/pipeline/cluster_identities.md` | reference | current | keep | Clustering data Faces Review needs |
| `docs/ops/faces_review_guide.md` | reference | current | keep | UX patterns to replicate |

---

## Part 2: Code Surface Inventory

### Streamlit UI (`apps/workspace-ui/`)

| Page | Lines | Priority | Notes |
|------|-------|----------|-------|
| `3_Faces_Review.py` | 6,832 | **P0** | Most complex; clusters, thumbnails, assignments |
| `2_Episode_Detail.py` | 4,913 | **P0** | Pipeline control, run_id, status |
| `3_Smart_Suggestions.py` | 3,393 | **P1** | Batch suggestions; can merge with Faces Review |
| `3_Voices_Review.py` | 2,814 | **P2** | Audio pipeline UI |
| `4_Screentime.py` | 1,785 | **P1** | Metrics display; depends on Episode Detail |
| `7_Timestamp_Search.py` | 1,608 | **P3** | Utility; lower priority |
| `0_Upload_Video.py` | 1,071 | **P0** | Upload flow; partially implemented in web |
| `4_Cast.py` | 867 | **P2** | Cast management |
| `8_Singletons_Review.py` | 830 | **P2** | Can merge with Faces Review |
| Other pages | ~3,889 | **P3** | Health, Settings, Docs, etc. |

**Total:** ~26,002 lines across 16 pages

### Web App (`web/`)

#### Framework & Config

| Item | Value | Status |
|------|-------|--------|
| Package name | `youth-league-web` | **Needs rebrand** to `screenalytics-web` |
| Next.js version | 14.2.3 | Current |
| TypeScript | Yes (`tsconfig.json`) | Good |
| Styling | CSS Modules | Basic; no component library |
| Data fetching | React Query 5.x | Good |
| API types | openapi-typescript | Good |
| Component library | None (raw Radix toast only) | **Gap** |
| Testing | None | **Gap** |

#### Implemented Pages

| Path | Lines | Status | Parity |
|------|-------|--------|--------|
| `web/app/screenalytics/upload/page.tsx` | 280 | **functional** | ~60% - missing S3 browser, audio trigger, ETA |
| `web/app/screenalytics/episodes/[id]/page.tsx` | 117 | **functional** | ~30% - basic status, missing job history, run_id |
| `web/app/screenalytics/faces/page.tsx` | 11 | **stub** | 0% - placeholder only |

#### API Client (`web/api/`)

| File | Purpose | Status |
|------|---------|--------|
| `client.ts` | Core fetch wrapper | Good |
| `hooks.ts` | React Query hooks | Basic set (create, presign, trigger, status) |
| `types.ts` | TypeScript types | Manual; should regenerate from OpenAPI |
| `schema.ts` | OpenAPI generated types | Exists but may be stale |
| `upload.ts` | Upload with progress | Good |

#### Other Code

| Path | Purpose | Status |
|------|---------|--------|
| `web/lib/state/uploadMachine.ts` | Upload state machine | Good |
| `web/mocks/handlers.ts` | MSW mocks | Basic coverage |
| `web/components/toast.tsx` | Toast notifications | Working |

### API Endpoints (FastAPI)

**Currently wired in web/:**
- `POST /episodes` - Create episode
- `POST /episodes/{ep_id}/assets` - Presign upload
- `POST /jobs/detect_track` - Trigger detect/track
- `POST /jobs/faces_embed` - Trigger faces embedding
- `POST /jobs/cluster` - Trigger clustering
- `POST /jobs/screen_time/analyze` - Trigger screentime
- `GET /episodes/{ep_id}/status` - Poll status

**Needed for parity (not yet wired):**
- Episodes list: `GET /episodes`
- Clusters: `GET /episodes/{ep_id}/cluster_tracks`
- Track reps: `GET /episodes/{ep_id}/clusters/{id}/track_reps`
- Tracks: `GET /episodes/{ep_id}/tracks`
- People/Cast: `GET /shows/{id}/cast`, `GET /shows/{id}/people`
- Smart suggestions: `GET /episodes/{ep_id}/suggestions`
- Assignments: `PATCH /episodes/{ep_id}/identities/{id}`
- Audio: `POST /audio/episode_audio_pipeline`
- Jobs: `GET /jobs/{id}/progress`, `POST /jobs/{id}/cancel`

---

## Part 3: Current State Assessment

### Streamlit Parity Requirements

#### Upload Page (`0_Upload_Video.py`)
**Key flows:**
1. New episode creation (show/season/episode selection)
2. Replace existing episode (locked ep_id)
3. S3 presigned upload with progress
4. Local-only fallback (`method=FILE`)
5. Auto-trigger detect/track after upload
6. Audio pipeline trigger option

**Web parity:** ~60%
- Has: Episode creation, presign, upload progress, local fallback
- Missing: S3 video browser, audio pipeline toggle, upload ETA, cancel/retry

#### Episode Detail (`2_Episode_Detail.py`)
**Key flows:**
1. Show phase status (detect/track, faces, cluster)
2. Trigger/rerun any phase
3. Display run_id and active run semantics
4. Show screentime metrics
5. Job progress and history
6. Navigate to Faces Review

**Web parity:** ~30%
- Has: Basic status display, trigger detect/track and cluster, SSE events
- Missing: run_id display, full job history, screentime panel, audio phases, video metadata

#### Faces Review (`3_Faces_Review.py`)
**Key flows:**
1. Virtualized thumbnail grid (faces.jsonl)
2. Cluster view with merge/split
3. Cast assignment with suggestions
4. Track timeline view
5. Quality/similarity badges
6. Keyboard navigation (arrows, M for merge, A for assign)

**Web parity:** 0% (stub only)
- Has: Nothing
- Missing: Everything

### Current web/ Implementation Status

**What exists:**
- Next.js 14 app with App Router
- React Query for data fetching
- CSS Modules for styling
- OpenAPI type generation (needs refresh)
- Working upload flow with presign
- Basic episode status polling with SSE
- Toast notifications

**Gaps:**
1. **Branding:** Package still named `youth-league-web`
2. **Component library:** No shared UI primitives (Modal, Dialog, Button, Badge)
3. **Faces Review:** Only a stub
4. **Smart Suggestions:** Not started
5. **Screentime display:** Not implemented
6. **Audio pipeline UI:** Not started
7. **Testing:** No tests exist
8. **OpenAPI types:** May be stale; needs regeneration

---

## Part 4: Reconciliation Summary

### Documents to Update

| Document | Action | Reason |
|----------|--------|--------|
| `WEB_APP_MIGRATION_PLAN.md` | **Update** | Percentages outdated; add run_id semantics; fix status |
| `masterWEBtodolist.md` | **Keep archived** | Superseded by migration plan |

### Documents to Create

| Document | Purpose |
|----------|---------|
| `MIGRATION_ROADMAP.md` | Phased implementation with acceptance criteria |
| This document | Inventory and triage |

### Immediate Blockers

1. **No Modal component** - Need real centered dialogs for better UX
2. **Package naming** - Still called `youth-league-web`
3. **OpenAPI types stale** - Need to regenerate from FastAPI
4. **No test coverage** - Risk of regressions

### API Endpoint Gaps

The web app needs these endpoints that aren't yet wired:
1. `GET /episodes` - Episode list
2. `GET /episodes/{ep_id}/cluster_tracks` - Cluster data for Faces Review
3. `GET /episodes/{ep_id}/suggestions` - Smart suggestions
4. `GET /shows/{id}/cast` - Cast for assignment dropdown

---

## Next Steps

1. Create `MIGRATION_ROADMAP.md` with phased plan and acceptance criteria
2. Update `WEB_APP_MIGRATION_PLAN.md` with corrected status
3. Rebrand `web/package.json` to `screenalytics-web`
4. Add Modal/Dialog component for better UX
5. Start Phase 1: Foundation + Episode Detail hardening

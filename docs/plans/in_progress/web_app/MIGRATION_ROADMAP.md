# Web App Migration Roadmap

> **Created:** 2025-12-15
> **Goal:** Migrate Streamlit UI to React/Next.js for richer UX (real modals, complex layouts, better state) and async operation.

---

## Architecture Decisions

### Framework & Stack

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Framework | **Next.js 14** (App Router) | Already in place; good DX |
| Language | **TypeScript** | Type safety required |
| Data fetching | **React Query** | Already in place; handles caching, refetching |
| API types | **openapi-typescript** | Generate types from FastAPI OpenAPI spec |
| Styling | **CSS Modules** | Already in place; simple and scoped |
| Component library | **Radix UI primitives** | For Modal/Dialog; already have Toast |
| Real-time | **SSE only** | Already wired; no WebSocket complexity |

### Typed API Strategy

```bash
# Regenerate types from FastAPI OpenAPI
cd web
npm run api:gen  # runs: openapi-typescript http://localhost:8000/openapi.json -o api/schema.ts
```

**Requirement:** Run this whenever API changes. Add to CI or pre-commit.

### Job Progress & SSE

- Episode events stream: `GET /episodes/{ep_id}/events` (already wired)
- Use `useEpisodeEvents` hook with reconnect logic
- Map events to UI state via `mapEventStream`

---

## Phase 0: Foundation

**Branch:** `screentime-improvements/web-foundation`

### Deliverables

1. **Rebrand package**
   - `web/package.json`: Change name to `screenalytics-web`
   - `web/app/layout.tsx`: Update root title/description

2. **Add Modal/Dialog component**
   - Install `@radix-ui/react-dialog`
   - Create `web/components/ui/dialog.tsx`
   - Create `web/components/ui/button.tsx`
   - Centered overlay with ESC close

3. **Refresh OpenAPI types**
   - Run `npm run api:gen` with running FastAPI
   - Commit updated `api/schema.ts`

4. **Add env config documentation**
   - Create `web/.env.local.example`
   - Document `NEXT_PUBLIC_API_BASE`, `NEXT_PUBLIC_MSW`

### Files to Modify/Create

```
web/
├── package.json                    # Rebrand name
├── .env.local.example              # Create
├── app/
│   ├── layout.tsx                  # Update title
│   └── screenalytics/
│       └── layout.tsx              # Keep as-is
├── components/
│   └── ui/
│       ├── dialog.tsx              # Create
│       └── button.tsx              # Create
└── api/
    └── schema.ts                   # Regenerate
```

### Acceptance Criteria

- [ ] Can run `npm run dev` and see "Screenalytics" branding
- [ ] Modal component renders centered with overlay
- [ ] ESC key closes modal
- [ ] `api/schema.ts` regenerated from current FastAPI spec
- [ ] `.env.local.example` documents required env vars

---

## Phase 1: Episode Detail Parity

**Branch:** `screentime-improvements/web-episode-detail`
**Depends on:** Phase 0

### Deliverables

1. **Run ID display and semantics**
   - Show active `run_id` in header
   - Load status scoped to `run_id` (use `?run_id=` query param)
   - "New Attempt" button to generate new run_id

2. **Full phase status panel**
   - Detect/Track: tracks, detections, detector/tracker names
   - Faces: faces count, quality metrics
   - Cluster: identities, singleton stats (before/after merge)
   - Screentime: visual_s, speaking_s per cast

3. **Job history panel**
   - List recent jobs (detect_track, faces_embed, cluster, screentime)
   - Show status, timestamps, duration
   - Collapsible log viewer per job

4. **Trigger all phases**
   - Detect/Track (existing)
   - Faces Embed (existing)
   - Cluster (existing)
   - Screentime (add)
   - Audio pipeline (add)

5. **Video metadata display**
   - Duration, FPS, resolution from status response

### API Hooks to Add

```typescript
// web/api/hooks.ts
useEpisodeStatus(epId, { runId })    // Add run_id param
useScreentimeMetrics(epId)           // GET /episodes/{ep_id}/screentime
useJobHistory(epId)                  // GET /jobs?ep_id={ep_id}
useTriggerAudio()                    // POST /audio/episode_audio_pipeline
```

### Files to Modify/Create

```
web/app/screenalytics/episodes/[id]/
├── page.tsx                         # Major rewrite
├── components/
│   ├── PhaseStatusCard.tsx          # Create
│   ├── JobHistoryPanel.tsx          # Create
│   ├── ScreentimePanel.tsx          # Create
│   └── VideoMetadata.tsx            # Create
└── episode.module.css               # Update styles
```

### Acceptance Criteria

- [ ] Run ID displayed in episode header
- [ ] Status loads scoped to active run_id
- [ ] Can trigger all 5 phases (detect, faces, cluster, screentime, audio)
- [ ] Job history shows last 10 jobs with status
- [ ] Screentime panel shows per-cast visual/speaking time
- [ ] Video metadata (duration, FPS) displayed
- [ ] SSE events update UI in real-time

---

## Phase 2: Faces Review Parity

**Branch:** `screentime-improvements/web-faces-review`
**Depends on:** Phase 1

### Deliverables

1. **Virtualized thumbnail grid**
   - Load faces.jsonl via API (paged)
   - Virtualized rendering for performance
   - Lazy load thumbnails

2. **Cluster view**
   - Group faces by cluster_id
   - Show cluster metrics (similarity, track count)
   - Collapse/expand clusters

3. **Smart Suggestions**
   - Generate suggestions via API
   - Show suggested cast match with confidence
   - Accept/dismiss individual suggestions
   - Batch apply button (scoped, not auto-apply)

4. **Assignments**
   - Dropdown to assign cluster to cast member
   - Mutation updates identities.json

5. **Similarity badges**
   - Identity (blue), Cast (purple), Track (orange)
   - Ambiguity warning (red) when margin < 0.15

6. **Keyboard navigation**
   - Arrow keys for grid navigation
   - M for merge, A for assign, D for delete

### API Hooks to Add

```typescript
// web/api/hooks.ts
useClusters(epId)                    // GET /episodes/{ep_id}/cluster_tracks
useClusterTrackReps(epId, clusterId) // GET /clusters/{id}/track_reps
useSuggestions(epId)                 // GET /episodes/{ep_id}/suggestions
useApplySuggestion()                 // POST /episodes/{ep_id}/suggestions/apply
useAssignCluster()                   // PATCH /episodes/{ep_id}/identities/{id}
useCast(showId)                      // GET /shows/{id}/cast
```

### Files to Create

```
web/app/screenalytics/faces/
├── page.tsx                         # Main page
├── components/
│   ├── FacesGrid.tsx                # Virtualized grid
│   ├── ClusterCard.tsx              # Cluster display
│   ├── ThumbnailCard.tsx            # Face thumbnail
│   ├── SimilarityBadge.tsx          # Colored badges
│   ├── SuggestionsPanel.tsx         # Smart suggestions
│   ├── AssignmentDropdown.tsx       # Cast assignment
│   └── FilterBar.tsx                # Sort/filter
├── hooks/
│   └── useFacesKeyboard.ts          # Keyboard nav
└── faces.module.css
```

### Acceptance Criteria

- [ ] Grid renders 1000+ faces without lag (virtualized)
- [ ] Clusters collapse/expand properly
- [ ] Suggestions show with accept/dismiss buttons
- [ ] Can assign cluster to cast via dropdown
- [ ] Similarity badges render with correct colors
- [ ] Keyboard navigation works (arrows, M, A, D)
- [ ] Smart suggestions are scoped (no auto-apply)

---

## Phase 3: Upload Parity

**Branch:** `screentime-improvements/web-upload-parity`
**Depends on:** Phase 1

### Deliverables

1. **S3 video browser**
   - List existing videos by show/season
   - Select existing video to process

2. **Audio pipeline toggle**
   - Checkbox to trigger audio pipeline after detect/track

3. **Upload enhancements**
   - Cancel button with AbortController
   - ETA calculation
   - Retry logic on failure
   - Upload speed display

### API Hooks to Add

```typescript
useS3Videos(showId?)                 // GET /episodes/s3_videos
useS3Shows()                         // GET /episodes/s3_shows
useCancelJob(jobId)                  // POST /jobs/{id}/cancel
```

### Files to Modify

```
web/app/screenalytics/upload/
├── page.tsx                         # Add S3 browser, audio toggle
├── components/
│   ├── S3VideoBrowser.tsx           # Create
│   └── UploadProgress.tsx           # Enhance
└── upload.module.css
```

### Acceptance Criteria

- [ ] Can browse existing S3 videos
- [ ] Audio pipeline toggle works
- [ ] Cancel button aborts upload
- [ ] ETA displays during upload
- [ ] Upload speed shows in MB/s

---

## Phase 4: Decommission Plan

**After all phases complete:**

1. **Parallel operation period**
   - Run both Streamlit and Next.js
   - Streamlit at `/workspace-ui`
   - Next.js at `/screenalytics`

2. **Feature parity verification**
   - Checklist of all Streamlit features
   - Manual testing against Streamlit

3. **Migration**
   - Announce deprecation date
   - Redirect `/workspace-ui` to `/screenalytics`

4. **Cleanup**
   - Archive `apps/workspace-ui/`
   - Remove Streamlit from deployment

---

## Backlog: Concrete Tasks

### Phase 0 Tasks

| Task | Files | API | Tests |
|------|-------|-----|-------|
| Rebrand package.json | `web/package.json` | - | - |
| Update root layout title | `web/app/layout.tsx` | - | - |
| Create Dialog component | `web/components/ui/dialog.tsx` | - | Manual |
| Create Button component | `web/components/ui/button.tsx` | - | Manual |
| Create .env.local.example | `web/.env.local.example` | - | - |
| Regenerate OpenAPI types | `web/api/schema.ts` | All | - |

### Phase 1 Tasks

| Task | Files | API Endpoints |
|------|-------|---------------|
| Add run_id to status display | `episodes/[id]/page.tsx` | `GET /episodes/{ep_id}/status?run_id=` |
| Create PhaseStatusCard | `PhaseStatusCard.tsx` | - |
| Create JobHistoryPanel | `JobHistoryPanel.tsx` | `GET /jobs?ep_id=` |
| Create ScreentimePanel | `ScreentimePanel.tsx` | `GET /episodes/{ep_id}/screentime` |
| Add screentime trigger | `page.tsx`, `hooks.ts` | `POST /jobs/screen_time/analyze` |
| Add audio trigger | `page.tsx`, `hooks.ts` | `POST /audio/episode_audio_pipeline` |
| Video metadata display | `VideoMetadata.tsx` | Via status response |

### Phase 2 Tasks

| Task | Files | API Endpoints |
|------|-------|---------------|
| Virtualized FacesGrid | `FacesGrid.tsx` | `GET /episodes/{ep_id}/faces` |
| ClusterCard component | `ClusterCard.tsx` | `GET /cluster_tracks` |
| ThumbnailCard component | `ThumbnailCard.tsx` | - |
| SimilarityBadge component | `SimilarityBadge.tsx` | - |
| SuggestionsPanel | `SuggestionsPanel.tsx` | `GET /suggestions` |
| Apply suggestion mutation | `hooks.ts` | `POST /suggestions/apply` |
| AssignmentDropdown | `AssignmentDropdown.tsx` | `GET /shows/{id}/cast` |
| Keyboard navigation hook | `useFacesKeyboard.ts` | - |

### Phase 3 Tasks

| Task | Files | API Endpoints |
|------|-------|---------------|
| S3VideoBrowser component | `S3VideoBrowser.tsx` | `GET /episodes/s3_videos` |
| Audio pipeline toggle | `upload/page.tsx` | `POST /audio/pipeline` |
| Cancel upload button | `upload/page.tsx` | `POST /jobs/{id}/cancel` |
| ETA calculation | `UploadProgress.tsx` | - |

---

## Rollback Strategy

Each phase maintains backward compatibility:

1. **Streamlit remains operational** throughout migration
2. **Feature flags** can disable new web pages
3. **API changes are additive** (new endpoints, not breaking changes)
4. **No data migrations required** (same manifests/artifacts)

If a phase causes issues:
1. Redirect traffic back to Streamlit
2. Debug in isolation
3. Re-deploy when fixed

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Episode Detail parity | 100% feature coverage |
| Faces Review parity | 100% feature coverage |
| Upload parity | 100% feature coverage |
| Load time (Episode Detail) | < 2s |
| Faces grid render (1000 faces) | < 500ms |
| User feedback | No regressions reported |

---

## References

- [MIGRATION_INVENTORY_TRIAGE.md](./MIGRATION_INVENTORY_TRIAGE.md) - Document/code inventory
- [WEB_APP_MIGRATION_PLAN.md](../WEB_APP_MIGRATION_PLAN.md) - Original plan (outdated)
- [docs/reference/api.md](../../reference/api.md) - API reference
- [docs/ops/faces_review_guide.md](../../ops/faces_review_guide.md) - Faces Review UX patterns

# Screenalytics Web App Migration Plan

> **Goal:** Migrate the Streamlit UI to a standalone Next.js web application that runs async without requiring VS Code to be open.

**Last Updated:** 2025-12-09

---

## Why Migrate? Streamlit Limitations vs. React/Next.js Capabilities

### Current Streamlit Pain Points

| Limitation | Impact | How React/Next.js Solves It |
|------------|--------|----------------------------|
| **Full-page reruns** | Every interaction re-runs the entire script (~6400 lines in Faces Review). Slow, state resets. | Component-level state + React Query caching. Only changed components re-render. |
| **No true routing** | URL params are fragile; deep-linking is hacky | Next.js App Router with proper URLs, back/forward navigation, bookmarkable views |
| **Widget key collisions** | Cryptic duplicate widget key errors | React keys are natural; no global namespace pollution |
| **Session state fragility** | State lost on refresh, tab switching, or server restart | React Query persistence, localStorage, URL state |
| **Single-threaded rendering** | Can't parallelize UI work | React concurrent rendering, Suspense boundaries |
| **No real-time without polling** | Expensive `time.sleep()` loops | Native SSE/WebSocket support, React Query subscriptions |
| **No offline capability** | Requires active server connection | Service workers, optimistic updates, offline-first patterns |
| **Limited mobile UX** | Touch interactions are clunky | Responsive CSS, touch events, gesture support |
| **No keyboard shortcuts** | Limited hotkey support | Full keyboard event handling, accessibility |
| **No virtualization** | Loads all thumbnails at once (OOM risk) | react-window/TanStack Virtual for 1000s of items |

### What You'll Be Able to Do After Migration

#### 1. **Real-time Collaboration & Multi-Device Access**
- Open Faces Review on your phone while processing runs on desktop
- Share episode URLs with collaborators
- Work from any browser without VS Code running
- Multiple tabs without state conflicts

#### 2. **Dramatically Faster Faces Review**
- **Virtualized grid**: Render 10,000+ faces smoothly (currently crashes Streamlit)
- **Instant view switching**: Cast → Cluster → Frames with no reload
- **Optimistic mutations**: Merge clusters instantly, sync in background
- **Prefetching**: Preload adjacent clusters while you review current one

#### 3. **True Keyboard-First Workflow**
- `J/K` or arrows to navigate grid
- `Space` to select, `Shift+Space` for range select
- `M` to merge selected clusters
- `A` to assign to cast
- `Cmd+Z` to undo
- `/` to search/filter
- Tab through views without mouse

#### 4. **Advanced Selection & Batch Operations**
- Multi-select with `Shift+Click` range, `Cmd+Click` toggle
- Drag-select lasso tool
- Select all in current filter
- Batch operations with preview before commit

#### 5. **Rich Visual Timeline**
- Scrubbing timeline showing when faces appear in episode
- Click timeline to jump to that frame
- Visual gaps showing track dropout
- Side-by-side track comparison

#### 6. **Offline-First Resilience**
- Queue operations when offline
- Sync when reconnected
- Never lose work to network blips

#### 7. **Custom Dashboards & Layouts**
- Resizable panels (like VS Code)
- Save layout preferences
- Drag/drop column reordering
- Hide/show metric badges per preference

#### 8. **Better Error Recovery**
- Errors don't crash the whole UI
- Error boundaries catch failures per component
- Retry buttons on failed requests
- Detailed error messages with context

#### 9. **Progressive Enhancement**
- Basic functionality works on slow connections
- Images lazy-load with blur placeholders
- Skeleton loaders instead of spinners
- Background sync for heavy operations

#### 10. **Integration-Ready**
- Embed analytics widgets elsewhere
- API for external tools
- Webhook triggers
- Slack/Discord notifications

---

## Current State Summary (Dec 2025)

| Component | Status | Notes |
|-----------|--------|-------|
| **Next.js App** | Exists at `web/` | React Query, SSE, OpenAPI types |
| **Layout/Providers** | Complete | Sidebar nav, Toast, QueryClient |
| **API Client** | Solid foundation | `apiFetch`, error normalization, types |
| **Upload Page** | 80% complete | Needs S3 browser, audio pipeline trigger |
| **Episode Detail** | 40% complete | Missing job history, audio, cluster UI |
| **Faces Review** | Stub only | Full implementation needed (6400 lines to port) |
| **Voices Review** | Not started | New page |
| **Cast Management** | Not started | New page |
| **Screen Time Dashboard** | Not started | Charts, export, comparisons |

## Revised Migration Priority Order

1. **Phase 1: Foundation** - Harden Upload + Episode Detail + Episodes List
2. **Phase 2: Faces Review** - Most critical (virtualized grid, keyboard nav, batch ops)
3. **Phase 3: Screen Time** - Charts and export (high business value)
4. **Phase 4: Cast Management** - Facebank seeds, identity prototypes
5. **Phase 5: Voices Review** - Audio pipeline UI (lower priority)

## Architecture Decisions

- **Extend existing Next.js app** in `web/` (not a new app)
- **All pages under** `/screenalytics/*`
- **Real-time:** SSE only (no WebSockets) initially; WebSocket upgrade optional
- **Deployment:** Vercel (frontend), existing server (FastAPI + Celery)
- **Branch strategy:** `feature/web-app/<feature-name>`
- **Component library:** Radix UI primitives + custom CSS modules
- **State management:** React Query for server state, Zustand for UI state
- **Virtualization:** TanStack Virtual for large grids

---

## Phase 1: Harden Upload + Episode Detail

**Branch:** `feature/web-app/upload-episode-hardening`

### 1.1 Upload Page Enhancements

**Current:** Basic upload with S3 presign + detect/track trigger

**Missing Features:**
- [ ] S3 video browser (browse existing videos by show/season)
- [ ] Audio pipeline trigger checkbox
- [ ] ASR provider selection (Whisper/Gemini)
- [ ] Job cancellation with abort controller
- [ ] Better error states and retry logic
- [ ] Upload speed display
- [ ] ETA calculation

**API Endpoints to Add:**
```typescript
// New hooks needed
useS3Videos(showId?: string)     // GET /episodes/s3_videos
useS3Shows()                      // GET /episodes/s3_shows
useCancelJob()                    // POST /jobs/{job_id}/cancel
```

**Files to Modify:**
- `web/app/screenalytics/upload/page.tsx`
- `web/api/hooks.ts`
- `web/api/client.ts`
- `web/api/types.ts`

### 1.2 Episode Detail Enhancements

**Current:** Basic status polling + SSE events

**Missing Features:**
- [ ] Full job history panel (detect/track, faces_embed, cluster, audio)
- [ ] Video metadata display (FPS, duration, resolution)
- [ ] Mirror from S3 button
- [ ] All job phase triggers (faces_embed, audio phases, screentime)
- [ ] Progress bars for running jobs
- [ ] Job logs viewer (collapsible)
- [ ] Backup/restore controls
- [ ] Episode deletion
- [ ] Navigation to Faces Review / Voices Review

**API Endpoints to Add:**
```typescript
useEpisodeDetail(episodeId)       // GET /episodes/{ep_id}
useEpisodeJobs(episodeId)         // GET /jobs?ep_id={ep_id}
useJobProgress(jobId)             // GET /jobs/{job_id}/progress
useMirrorEpisode()                // POST /episodes/{ep_id}/mirror
useDeleteEpisode()                // POST /episodes/{ep_id}/delete
useEpisodeBackups(episodeId)      // GET /episodes/{ep_id}/backups
useRestoreBackup()                // POST /episodes/{ep_id}/restore/{backup_id}
```

**New Components:**
- `JobHistoryPanel.tsx` - List of recent jobs with status
- `JobProgressCard.tsx` - Individual job progress display
- `VideoMetadata.tsx` - FPS, duration, resolution display
- `EpisodeActions.tsx` - Delete, mirror, backup controls

### 1.3 Episodes List Page

**New Page:** `/screenalytics/episodes`

**Features:**
- [ ] List all episodes with search/filter
- [ ] Show/season grouping
- [ ] Quick status indicators
- [ ] Click to navigate to Episode Detail
- [ ] Bulk operations (delete all)

---

## Phase 2: Faces Review Implementation

**Branch:** `feature/web-app/faces-review`

This is the most complex page (259KB in Streamlit).

### 2.1 File Structure

```
web/app/screenalytics/faces/
├── page.tsx                      # Main page with view routing
├── components/
│   ├── FacesGrid.tsx            # Virtualized thumbnail grid
│   ├── ThumbnailCard.tsx        # Individual face thumbnail
│   ├── SimilarityBadge.tsx      # Colored similarity badges
│   ├── QualityIndicator.tsx     # Consistency, confidence trend
│   ├── ClusterCard.tsx          # Cluster summary card
│   ├── PersonCard.tsx           # Person summary card
│   ├── TrackTimeline.tsx        # Track frame timeline
│   └── FilterBar.tsx            # Sort/filter controls
├── hooks/
│   ├── useClusters.ts
│   ├── useTracks.ts
│   ├── usePeople.ts
│   └── useAssignments.ts
└── state/
    └── facesViewMachine.ts      # View state machine
```

### 2.2 View Modes (6 views)

1. **Cast Members** - List all people linked to cast
2. **Person View** - Auto-generated people (not in cast)
3. **Cast Tracks** - Tracks assigned to a cast member
4. **Cluster View** - Unassigned clusters
5. **Frames View** - Individual track frames
6. **Smart Suggestions** - Batch accept/dismiss

### 2.3 API Hooks

```typescript
// Clusters
useClusters(episodeId)            // GET /episodes/{ep_id}/cluster_tracks
useClusterTrackReps(clusterId)    // GET /episodes/{ep_id}/clusters/{cluster_id}/track_reps
useClusterSuggestions(clusterId)  // POST /episodes/{ep_id}/clusters/{cluster_id}/suggest_cast

// Tracks
useTracks(episodeId)              // GET /episodes/{ep_id}/tracks
useTrackFrames(trackId)           // GET /episodes/{ep_id}/tracks/{track_id}/frames

// People & Cast
usePeople(showId)                 // GET /shows/{show_id}/people
useCast(showId)                   // GET /shows/{show_id}/cast
useCastSuggestions(episodeId)     // GET /episodes/{ep_id}/cast_suggestions

// Mutations
useMergeClusters()                // POST /episodes/{ep_id}/clusters/group
useBulkAssignTracks()             // POST /episodes/{ep_id}/tracks/bulk_assign
useAssignClusterToCast()          // PATCH /shows/{show_id}/people/{person_id}
useRefreshSimilarity()            // POST /episodes/{ep_id}/refresh_similarity
```

### 2.4 Similarity Badges

| Type | Color | Description |
|------|-------|-------------|
| IDENTITY | Blue | Auto-generated people |
| CAST | Purple | Cast member facebank matches |
| TRACK | Orange | Frame consistency within track |
| PERSON_COHESION | Teal | Track to other tracks same person |
| CLUSTER | Green | Track cohesion within cluster |
| TEMPORAL | Cyan | Consistency over time |
| AMBIGUITY | Red | Margin between top 2 matches |

### 2.5 Keyboard Navigation

- Arrow keys: Grid navigation
- Enter: Select/expand
- Escape: Deselect
- M: Merge selected
- D: Delete selected
- A: Assign to cast
- /: Search focus

---

## Phase 3: Voices Review Implementation

**Branch:** `feature/web-app/voices-review`

### 3.1 File Structure

```
web/app/screenalytics/voices/
├── page.tsx                      # Main voices review page
├── components/
│   ├── VoiceClusterGrid.tsx     # Voice cluster display
│   ├── AudioSegmentPlayer.tsx   # Audio playback
│   ├── TranscriptView.tsx       # Timestamped transcript
│   ├── SpeakerAssignment.tsx    # Assign speaker to cast
│   └── ClusterSimilarityMatrix.tsx
└── hooks/
    ├── useVoiceClusters.ts
    ├── useAudioSegments.ts
    └── useTranscript.ts
```

### 3.2 API Hooks

```typescript
useAudioPipelineStatus(episodeId)  // GET /audio/episode_audio_status
useTriggerAudioPipeline()          // POST /audio/episode_audio_pipeline
useVoiceClusters(episodeId)        // GET /jobs/episodes/{ep_id}/audio/clusters
useClusterSimilarityMatrix()       // GET similarity matrix
useTranscript(episodeId)           // GET /audio/episodes/{ep_id}/audio/transcript.vtt
useAssignSegmentToCast()           // POST assign segment
```

### 3.3 Features

- [ ] Voice cluster grid with audio playback
- [ ] Cluster merge/split preview
- [ ] Speaker assignment to cast
- [ ] Transcript display with speaker labels
- [ ] Diarization comparison (Pyannote vs Silero)
- [ ] Audio QC flags display

---

## Phase 4: Cast Management Implementation

**Branch:** `feature/web-app/cast-management`

### 4.1 File Structure

```
web/app/screenalytics/cast/
├── page.tsx                      # Cast list page
├── [castId]/page.tsx            # Cast member detail
├── components/
│   ├── CastMemberForm.tsx       # Add/edit cast member
│   ├── FacebankSeeds.tsx        # Upload/manage seeds
│   ├── VoiceReference.tsx       # Voice reference upload
│   └── ShowSelector.tsx         # Show selection
└── hooks/
    ├── useShows.ts
    ├── useCastMembers.ts
    └── useFacebank.ts
```

### 4.2 API Hooks

```typescript
useShows()                         // GET /shows
useCastMembers(showId)             // GET /shows/{show_id}/cast
useCreateCastMember()              // POST /shows/{show_id}/cast
useUpdateCastMember()              // PATCH /shows/{show_id}/cast/{cast_id}
useFacebankSeeds(castId)           // GET /cast/{cast_id}/facebank
useUploadFacebankSeed()            // POST multipart form
useVoiceReferences(showId)         // GET /shows/{show_id}/voice_references
```

### 4.3 Features

- [ ] Show management (create, select)
- [ ] Cast member CRUD
- [ ] Facebank seed upload with face detection preview
- [ ] Voice reference upload
- [ ] Metadata fields (name, role, status, aliases, seasons)

---

## Shared Components

```
web/components/
├── Card.tsx                      # Generic card container
├── Badge.tsx                     # Status/label badge
├── ProgressBar.tsx               # Progress indicator
├── Spinner.tsx                   # Loading spinner
├── Alert.tsx                     # Info/warning/error alerts
├── Modal.tsx                     # Modal dialog
├── ConfirmDialog.tsx             # Confirmation modal
├── SearchInput.tsx               # Search with debounce
└── VirtualizedGrid.tsx           # Virtualized grid for large lists
```

---

## Deployment

### Vercel Configuration

```json
{
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "https://your-backend.example.com/:path*"
    }
  ]
}
```

### Environment Variables

```
NEXT_PUBLIC_API_BASE=https://your-backend.example.com
NEXT_PUBLIC_MSW=0
```

---

## Next Steps

1. Create branch `feature/web-app/upload-episode-hardening`
2. Implement S3 video browser on Upload page
3. Add audio pipeline trigger option
4. Enhance Episode Detail with job history panel
5. Add Episodes list page

Start with Phase 1 and iterate based on feedback.

---

## Implementation Notes for Future Reference

### Key Streamlit Patterns to Replicate

**Session State Keys (from `ui_helpers.py`):**
```python
ep_id                          # Current episode
api_base                       # API base URL
backend, bucket                # Storage config
_execution_mode::{ep_id}       # Redis vs local per episode
_async_job                     # Current async job metadata
_local_logs_cache              # Cached operation logs
_custom_show_registry          # User-added shows
```

**Polling Intervals:**
- Upload status: 1.2s (`refetchInterval: 1200`)
- Episode detail: 1.8s (`refetchInterval: 1800`)
- Job progress: 0.5s (Streamlit uses `time.sleep(0.5)` in loop)

**Thumbnail Sizing:**
- Fixed frame: 147x184px (4:5 aspect ratio)
- CSS: `object-fit: cover` with placeholder silhouette for missing

### API Endpoint Reference

**Core endpoints already wired in `web/api/client.ts`:**
- `POST /episodes` - Create episode
- `POST /episodes/{ep_id}/assets` - Presign upload
- `POST /jobs/detect_track` - Trigger detect/track
- `POST /jobs/faces_embed` - Trigger faces embedding
- `POST /jobs/cluster` - Trigger clustering
- `GET /episodes/{ep_id}/status` - Poll status

**Endpoints still needed (70+ total in Streamlit):**
- Episode: `/episodes`, `/episodes/{ep_id}`, `/episodes/{ep_id}/delete`, `/episodes/s3_videos`
- Clusters: `/episodes/{ep_id}/cluster_tracks`, `/episodes/{ep_id}/clusters/{id}/track_reps`
- Tracks: `/episodes/{ep_id}/tracks`, `/episodes/{ep_id}/tracks/{id}/frames`
- Cast: `/shows/{id}/cast`, `/cast/{id}/facebank`
- Audio: `/audio/episode_audio_pipeline`, `/jobs/episodes/{ep_id}/audio/clusters`
- Jobs: `/jobs/{id}/progress`, `/jobs/{id}/cancel`, `/celery_jobs`

### State Machine Patterns

**Upload Machine (`web/lib/state/uploadMachine.ts`):**
```
idle → ready → preparing → uploading → verifying → processing → success
                                                              → error
```

**Faces View Machine (to build):**
```
cast → person → cast-tracks → cluster → frames → suggestions
       ↑___________↓____________↑__________↓
```

### Known Edge Cases

1. **tracks_only_fallback** - When detections missing, harvest disabled
2. **faces_manifest_fallback** - Status stale, using manifest file
3. **Celery vs Local mode** - Some jobs run in Celery, others as subprocess
4. **S3 vs Local storage** - Upload flow differs based on `STORAGE_BACKEND`

### Dependencies Between Features

```
Upload → Episode Detail → Faces Review
                       ↘ Voices Review
                       ↘ Cast Management (depends on Faces Review for assignments)
```

### CSS Module Files (existing)

- `web/app/screenalytics/layout.module.css`
- `web/app/screenalytics/upload/upload.module.css`
- `web/app/screenalytics/episode.module.css`

### MSW Handlers (for dev mocking)

Located in `web/mocks/handlers.ts` - already mocks:
- Episode creation
- Asset presigning
- Job triggers
- Status polling with progressive state transitions

---

---

## Phase 3: Screen Time Dashboard (NEW)

**Branch:** `feature/web-app/screen-time`

### 3.1 File Structure

```
web/app/screenalytics/screentime/
├── page.tsx                      # Main dashboard
├── [episodeId]/page.tsx          # Episode-specific view
├── components/
│   ├── ScreenTimeChart.tsx       # Bar/line chart for cast
│   ├── CastRanking.tsx           # Ranked list with percentages
│   ├── TimelineHeatmap.tsx       # Episode timeline heatmap
│   ├── ComparisonView.tsx        # Compare across episodes
│   ├── ExportPanel.tsx           # CSV/JSON export
│   └── FilterBar.tsx             # Date range, show, season filters
└── hooks/
    ├── useScreenTime.ts
    ├── useShowStats.ts
    └── useExport.ts
```

### 3.2 Features

- [ ] Episode screen time breakdown (bar chart per cast member)
- [ ] Season aggregation (total minutes per cast across episodes)
- [ ] Timeline heatmap (when each person appears)
- [ ] Export to CSV/JSON for external analysis
- [ ] Compare episodes side-by-side
- [ ] Filter by show, season, date range
- [ ] Percentage vs absolute time toggle
- [ ] Chart annotations for key moments

### 3.3 API Endpoints

```typescript
useEpisodeScreenTime(episodeId)     // GET /episodes/{ep_id}/screentime
useShowScreenTime(showId, season?)  // GET /shows/{show_id}/screentime
useExportScreenTime()               // GET /episodes/{ep_id}/screentime/export
```

### 3.4 Charts Library

Recommend: **Recharts** (lightweight, React-native) or **Nivo** (more features)

---

## Revised TODO Checklist

### Phase 1: Foundation (Weeks 1-2)
- [ ] **Upload Page**
  - [ ] S3 video browser with show/season filter
  - [ ] Audio pipeline trigger checkbox
  - [ ] ASR provider selection (Whisper/Gemini)
  - [ ] Upload speed display + ETA
  - [ ] Job cancellation support
- [ ] **Episodes List Page** (NEW)
  - [ ] Paginated list with search
  - [ ] Show/season grouping
  - [ ] Status badges (processed, pending, error)
  - [ ] Quick actions (delete, reprocess)
- [ ] **Episode Detail Page**
  - [ ] Job history panel with all phases
  - [ ] Progress bars for running jobs
  - [ ] Video metadata display
  - [ ] Navigation to Faces Review

### Phase 2: Faces Review (Weeks 3-5)
- [ ] **Core Grid**
  - [ ] TanStack Virtual grid (10k+ faces)
  - [ ] Thumbnail lazy loading with blurhash
  - [ ] Similarity badge component (7 types)
  - [ ] Quality indicator component
- [ ] **View Modes** (6 views)
  - [ ] Cast Members list
  - [ ] Person (auto-generated) view
  - [ ] Cast Tracks view
  - [ ] Cluster view
  - [ ] Frames view (track deep-dive)
  - [ ] Smart Suggestions batch panel
- [ ] **Interactions**
  - [ ] Keyboard navigation (J/K, Space, M, A, D)
  - [ ] Multi-select (Shift+Click, Cmd+Click)
  - [ ] Merge clusters modal with preview
  - [ ] Assign to cast dropdown
  - [ ] Bulk operations toolbar
- [ ] **State Machine**
  - [ ] View transition logic
  - [ ] Selection state
  - [ ] Undo/redo stack
  - [ ] URL sync for deep-linking

### Phase 3: Screen Time (Week 6)
- [ ] Episode screen time bar chart
- [ ] Cast ranking table
- [ ] Season aggregation view
- [ ] Export to CSV

### Phase 4: Cast Management (Week 7)
- [ ] Cast member CRUD
- [ ] Facebank seed upload with preview
- [ ] Voice reference upload
- [ ] Show management

### Phase 5: Voices Review (Week 8+)
- [ ] Voice cluster grid
- [ ] Audio segment player
- [ ] Transcript display
- [ ] Speaker assignment

---

## Migration Strategy

### Incremental Approach

1. **Keep Streamlit running** alongside Next.js during migration
2. **Feature flags** to enable Next.js pages incrementally
3. **Shared API** - both UIs use same FastAPI backend
4. **Redirect when ready** - update links to point to Next.js

### Data Migration

No data migration needed - both UIs read same manifests/S3 buckets.

### Testing Strategy

1. **Visual regression**: Screenshot comparison Streamlit vs Next.js
2. **E2E tests**: Playwright for critical flows
3. **API contract tests**: Ensure endpoint compatibility

---

## Dependencies to Install

```bash
cd web
npm install @tanstack/react-virtual  # Virtualization
npm install recharts                  # Charts
npm install zustand                   # UI state
npm install cmdk                      # Command palette
npm install @radix-ui/react-dialog   # Modals
npm install @radix-ui/react-dropdown-menu
npm install @radix-ui/react-tooltip
npm install react-hotkeys-hook        # Keyboard shortcuts
npm install blurhash                  # Image placeholders
```

---

## Status

**Current:** Ready to begin Phase 1

**Next Action:** Create branch `feature/web-app/upload-episode-hardening`

**Last updated:** 2025-12-09

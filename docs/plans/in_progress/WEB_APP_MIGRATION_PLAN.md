# Screenalytics Web App Migration Plan

> **Status:** PARTIALLY OUTDATED - See updated docs below
> **Last audit:** 2025-12-15

## Updated Documentation

This document contains useful reference material but the percentages and status are outdated.

**For current status and roadmap, see:**
- [web_app/MIGRATION_INVENTORY_TRIAGE.md](web_app/MIGRATION_INVENTORY_TRIAGE.md) - Document/code inventory and triage
- [web_app/MIGRATION_ROADMAP.md](web_app/MIGRATION_ROADMAP.md) - Phased implementation plan with acceptance criteria

---

> **Goal:** Migrate the Streamlit UI to a standalone Next.js web application that runs async without requiring VS Code to be open.

## Current State Summary (as of 2025-12-15 audit)

| Component | Status | Notes |
|-----------|--------|-------|
| **Next.js App** | Exists at `web/` | React Query, SSE, OpenAPI types |
| **Layout/Providers** | Complete | Sidebar nav, Toast, QueryClient |
| **API Client** | Solid foundation | `apiFetch`, error normalization, types |
| **Upload Page** | **~60% complete** | Missing S3 browser, audio trigger, ETA |
| **Episode Detail** | **~30% complete** | Missing run_id, job history, screentime panel |
| **Faces Review** | **Stub only (0%)** | Full implementation needed |
| **Voices Review** | Not started | New page |
| **Cast Management** | Not started | New page |

**Key gaps identified:**
1. Package still named `youth-league-web` (needs rebrand)
2. No Modal/Dialog component for better UX
3. OpenAPI types may be stale
4. No test coverage

## Migration Priority Order

1. **Upload + Episode Detail** - Harden existing pages
2. **Faces Review** - Most critical UI (clustering, cast assignment)
3. **Voices Review** - Audio pipeline UI
4. **Cast Management** - Facebank seeds, identity prototypes

## Architecture Decisions

- **Extend existing Next.js app** in `web/` (not a new app)
- **All pages under** `/screenalytics/*`
- **Real-time:** SSE only (no WebSockets)
- **Deployment:** Vercel (frontend), existing server (FastAPI + Celery)
- **Branch strategy:** `feature/web-app/<feature-name>`

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

## Status

**Current:** Active - see [MIGRATION_ROADMAP.md](web_app/MIGRATION_ROADMAP.md) for phased plan

**Last updated:** 2025-12-15

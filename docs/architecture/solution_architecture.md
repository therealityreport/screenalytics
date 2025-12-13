# Solution Architecture — Screenalytics

**Version:** 2.0
**Status:** Active
**Last Updated:** 2025-11-18

---

## 1. Purpose

Screenalytics transforms raw video and audio into structured cast-level analytics, delivering per-person screen presence, speaking time, and visual engagement metrics for every episode. It powers TRR's analytics, design tooling, and merchandise systems through a unified data pipeline.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      WORKSPACE UI (Next.js)                      │
│  ┌─────────────┬──────────────┬──────────────┬─────────────────┐│
│  │ SHOWS       │ PEOPLE       │ Episode      │ Facebank        ││
│  │ Directory   │ Directory    │ Workspace    │ Management      ││
│  └─────────────┴──────────────┴──────────────┴─────────────────┘│
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP/REST + SSE
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        API (FastAPI)                             │
│  ┌──────────┬──────────────┬─────────────┬───────────────────┐ │
│  │ Episodes │ Jobs         │ Identities  │ Facebank          │ │
│  │ CRUD     │ Orchestrator │ Moderation  │ Seed Management   │ │
│  └──────────┴──────────────┴─────────────┴───────────────────┘ │
└────┬───────────────┬──────────────────────────────┬─────────────┘
     │               │                              │
     │ Job Queue     │ Metadata/Vectors             │ Artifacts
     ▼               ▼                              ▼
┌─────────┐   ┌──────────────────┐    ┌────────────────────────┐
│ Redis   │   │ Postgres         │    │ Object Storage         │
│ Queue   │   │ + pgvector       │    │ (S3/R2/GCS/MinIO)      │
└────┬────┘   │                  │    │                        │
     │        │ • Episodes       │    │ • Videos               │
     │        │ • Tracks         │    │ • Frames               │
     │        │ • Identities     │    │ • Crops                │
     │        │ • Embeddings     │    │ • Manifests (JSONL)    │
     │        │ • Assignments    │    │ • Facebank thumbs      │
     │        │ • Screentime     │    │ • Analytics reports    │
     ▼        └──────────────────┘    └────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                    WORKERS (Pipeline Stages)                     │
│  ┌──────────┬──────────┬──────────┬──────────┬───────────────┐ │
│  │ Detect   │ Track    │ Faces    │ Cluster  │ Episode       │ │
│  │          │          │ Embed    │          │ Cleanup       │ │
│  └──────────┴──────────┴──────────┴──────────┴───────────────┘ │
│  ┌──────────┬──────────┬──────────────────────────────────────┐│
│  │ Audio    │ A/V      │ Screentime Aggregate                 ││
│  │ Diarize  │ Fusion   │                                      ││
│  └──────────┴──────────┴──────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                         ▲
                         │
            ┌────────────┴────────────┐
            │                         │
    ┌───────┴────────┐    ┌──────────┴────────┐
    │ MCP Servers    │    │ Agents            │
    │ • screenalytics│    │ • Codex           │
    │ • storage      │    │ • Claude          │
    │ • postgres     │    │   (relabel, docs) │
    └────────────────┘    └───────────────────┘
```

---

## 3. Core Components

### 3.1 Workspace UI
- **Technology:** Next.js, TypeScript, TailwindCSS
- **Purpose:** Upload episodes, run pipeline jobs, moderate identities, review analytics
- **Key Pages:**
  - **SHOWS:** Create/edit shows, seasons, episodes
  - **PEOPLE:** Manage cast directory and featured thumbnails
  - **Episode Workspace:** Multi-tab interface for detections, tracks, identities, A/V fusion, results, QA queue
  - **Facebank:** Identity grid, cluster drill-down, track moderation (merge/split/move/delete)

### 3.2 API
- **Technology:** FastAPI, Python 3.11+
- **Purpose:** CRUD operations, job orchestration, signed URL generation
- **Key Endpoints:**
  - `/episodes/*` — Episode metadata and artifacts
  - `/jobs/*` — Async job submission and progress tracking
  - `/identities/*` — Identity moderation (assign, merge, split, delete)
  - `/facebank/*` — Seed upload, embedding, thumbnail management

### 3.3 Workers
- **Technology:** Python, Redis queue, multi-stage pipeline
- **Purpose:** Execute ML pipeline stages (detect, track, embed, cluster, cleanup)
- **Stages:**
  1. **Detect:** RetinaFace face detection (InsightFace)
  2. **Track:** ByteTrack association with appearance gating
  3. **Faces Embed:** ArcFace embedding extraction with quality gating
  4. **Cluster:** Agglomerative clustering of track-level embeddings
  5. **Episode Cleanup:** Track splitting, re-embedding, re-clustering, grouping
  6. **Audio Pipeline:** NeMo MSDD diarization + OpenAI Whisper ASR (+ optional Gemini cleanup)
  7. **A/V Fusion:** Link face tracks with speech segments
  8. **Screentime Aggregate:** Compute visual/speaking/overlap metrics per person

### 3.4 Storage
- **Technology:** S3-compatible (AWS S3, Cloudflare R2, GCS, MinIO)
- **Layout (v2):**
  ```
  raw/videos/{show_slug}/s{season}/e{episode}/episode.mp4
  artifacts/frames/{show_slug}/s{season}/e{episode}/frames/
  artifacts/crops/{show_slug}/s{season}/e{episode}/tracks/
  artifacts/manifests/{show_slug}/s{season}/e{episode}/
  artifacts/thumbs/{show_slug}/s{season}/e{episode}/identities/{id}/rep.jpg
  facebank/{person_id}/{seed_id}_d.png (display derivative)
  facebank/{person_id}/{seed_id}_e.png (embedding derivative)
  ```

### 3.5 Database
- **Technology:** Postgres 15+ with pgvector extension
- **Purpose:** Structured metadata, vector embeddings, assignment history
- **Key Tables:**
  - `show`, `season`, `episode`, `person`, `cast_membership`
  - `track`, `detection`, `embedding` (with HNSW index)
  - `assignment`, `assignment_history` (audit trail)
  - `speech_segment`, `transcript`, `av_link`
  - `screen_time` (visual_s, speaking_s, both_s, confidence)

### 3.6 Agents & MCP
- **Technology:** Codex SDK, Claude API, MCP protocol
- **Purpose:** Autonomous relabeling, documentation updates, data exports
- **MCP Servers:**
  - `screanalytics` — List low-confidence tracks, assign identities, export screentime
  - `storage` — Signed URLs, list/purge artifacts
  - `postgres` — Safe analytics queries
- **Policies:** During feature work, agents write to `FEATURES/<name>/docs/`; root `/docs/**` updates occur only on promotion

---

## 4. Pipeline Flow

The complete pipeline transforms a raw `.mp4` episode into structured screentime analytics through these stages:

### 4.1 Episode Lifecycle

```
1. INGEST
   Input:  episode.mp4
   Output: local/S3 video, episode metadata in DB
   Tool:   UI Upload or API POST /episodes

2. DETECT & TRACK
   Input:  episode.mp4
   Output: detections.jsonl, tracks.jsonl, track_metrics.json
   Tool:   tools/episode_run.py or POST /jobs/detect_track_async
   Models: RetinaFace (detection), ByteTrack (tracking)

3. FACES HARVEST
   Input:  episode.mp4, tracks.jsonl
   Output: faces.jsonl, faces.npy, crops (JPG)
   Tool:   tools/episode_run.py --faces-embed or POST /jobs/faces_embed
   Models: RetinaFace (cropping), ArcFace (embedding)

4. CLUSTER IDENTITIES
   Input:  faces.jsonl, faces.npy
   Output: identities.json, Facebank thumbnails
   Tool:   tools/episode_run.py --cluster or POST /jobs/cluster
   Algorithm: Agglomerative clustering on track-level embeddings

5. EPISODE CLEANUP (optional)
   Input:  tracks.jsonl, faces.jsonl, identities.json
   Output: Updated tracks/faces/identities, cleanup_report.json
   Tool:   tools/episode_cleanup.py or POST /jobs/episode_cleanup_async
   Actions: split_tracks, reembed, recluster, group_clusters

6. AUDIO PIPELINE (optional)
   Input:  episode.mp4 (audio stream)
   Output: audio_diarization.jsonl, audio_asr_raw.jsonl, episode_transcript.jsonl, audio_voice_mapping.json
   Tool:   POST /jobs/episode_audio_pipeline
   Models: NeMo MSDD diarization, OpenAI Whisper ASR (+ optional Gemini cleanup)

7. SCREENTIME ANALYZE (optional)
   Input:  faces.jsonl, tracks.jsonl, identities.json, shows/{SHOW}/people.json
           + (optional) episode_transcript.jsonl + audio_voice_mapping.json (speaking_s)
           + (optional) body_tracking/screentime_comparison.json (body metrics)
   Output: screentime.json, screentime.csv
   Tool:   POST /jobs/screen_time/analyze (or python -m tools.analyze_screen_time)
   Metrics: face_visible_seconds (+ legacy visual_s), speaking_s, (optional) body_visible_seconds/body_only_seconds/gap_bridged_seconds
```

### 4.2 Artifact Dependencies

```
episode.mp4
    ├─► detections.jsonl ────┐
    ├─► tracks.jsonl ────────┼─► faces.jsonl ──► faces.npy ──► identities.json
    │                        │
    └─► track_metrics.json   │
                             │
                             └─► crops/{track_id}/*.jpg
```

---

## 5. Data Model (Simplified)

### 5.1 Catalog
```
show (id, slug, title)
  └─► season (id, show_id, number)
       └─► episode (id, ep_id, season_id, number, title, air_date)
            └─► track (id, track_id, ep_id, start_s, end_s, frame_span)
                 ├─► detection (bbox, landmarks, conf, frame_idx)
                 └─► embedding (vec VECTOR(512), owner_type='track')
                      └─► assignment (track_id, person_id, cluster_id, confidence, locked)
                           └─► assignment_history (audit trail)
```

### 5.2 People & Facebank
```
person (id, name, slug)
  ├─► cast_membership (person_id, show_id, season_num)
  ├─► facebank_seed (person_id, seed_id, display_path, embed_path, embedding VECTOR(512))
  └─► person_featured (person_id, media_asset_id, context)
```

### 5.3 Audio & Screentime
```
episode
  └─► speech_segment (id, start_s, end_s, speaker_label)
       ├─► transcript (text, confidence)
       └─► av_link (segment_id, track_id, overlap_s, spatial_score)
            └─► screen_time (person_id, visual_s, speaking_s, both_s, confidence)
```

---

## 6. Configuration Layers

All pipeline behavior is config-driven (no hardcoded thresholds).

### 6.1 Pipeline Configs
- **Location:** `config/pipeline/*.yaml`
- **Files:**
  - `detection.yaml` — RetinaFace model ID, min_size, confidence_th, iou_th, adaptive settings
  - `tracking.yaml` — ByteTrack: track_thresh, match_thresh, buffer, appearance gate params
  - `recognition.yaml` — ArcFace model ID, similarity_th, hysteresis (TBD)
  - `faces_embed_sampling.yaml` — Quality gating, max_crops_per_track, sampling strategy
  - `performance_profiles.yaml` — Device-aware profiles (fast_cpu, balanced, high_accuracy)
  - `audio.yaml` — NeMo diarization + ASR settings
  - `screen_time_v2.yaml` — DAG stage toggles, presets

### 6.2 Agents & Automation
- **Location:** `config/`
- **Files:**
  - `codex.config.toml` — Codex models, MCP servers, write policies
  - `agents.sdk.yaml` — Agents SDK graph, profiles, tasks
  - `agents/policies.yaml` + `claude.policies.yaml` — Doc/write rules

See [docs/reference/config/pipeline_configs.md](../reference/config/pipeline_configs.md) for key-by-key reference.

---

## 7. Performance & Scalability

### 7.1 Performance Targets
- **1-hour episode** ≤ 10 minutes on single mid-tier GPU (or ≤ 3× realtime CPU-only)
- **Visual ID accuracy** ≥ 90% on validation clips with occlusion/side-profiles
- **Speaking match precision** ≥ 85% with diarization alignment
- **CI integration test** ≤ 15 minutes full pipeline run

### 7.2 Scalability Patterns
- **Horizontal workers:** Redis queue supports multiple worker instances
- **S3 lifecycle:** Expire `frames/` after N days; retain `crops/`, `facebank/`, `manifests/`
- **pgvector HNSW:** Scales to millions of embeddings with approximate nearest neighbor search

### 7.3 Resource Controls
- **Threading limits:** PyTorch, ONNX Runtime, OpenCV thread counts configurable via env/config
- **Device selection:** Auto-detection (CUDA → MPS → CPU) with explicit override
- **Performance profiles:** CPU-safe defaults prevent thermal issues on fanless devices
- **Quality-gating:** Limits crop volume per track/episode to avoid embedding thousands of near-identical faces

---

## 8. Security & Privacy

### 8.1 Access Control
- **Signed URLs:** Short-lived (15-minute) presigned URLs for all media access
- **Row-level security (optional):** Postgres RLS by show_id for multi-tenant deployments
- **Audit trail:** Full history via `assignment_history` table

### 8.2 Data Protection
- **No secrets in repo:** All credentials via environment variables only
- **No PII storage:** Only embeddings and metadata; raw face crops are ephemeral (lifecycle-expired)
- **Encryption at rest:** S3 bucket encryption enabled by default

---

## 9. Observability

### 9.1 Logging & Metrics
- **Structured logs:** JSON to stdout for centralized collection
- **Progress events:** Real-time updates via `progress.json` and Redis PubSub
- **Metrics tracked:**
  - Throughput: frames/sec, crops/sec, embeddings/sec
  - Quality: track_count, short_track_fraction, id_switch_rate, singleton_fraction, largest_cluster_fraction
  - Runtime: per-stage elapsed time, total wall-clock

### 9.2 QA & Monitoring
- **QA view:** Aggregated low-confidence tracks for manual review
- **Episode Detail:** Per-run metrics surfaced in UI (tracks_born, tracks_lost, id_switches, longest_tracks)
- **Cleanup reports:** Before/after metrics in `cleanup_report.json`

---

## 10. Promotion & CI Flow

### 10.1 Feature Workflow
1. Develop in `FEATURES/<name>/` (`src/`, `tests/`, `docs/`, `TODO.md`)
2. CI enforces:
   - 30-day TTL
   - Tests + docs present
   - No imports from `FEATURES/**` in production code
3. Promote via `tools/promote-feature.py`
4. Post-promotion CI re-runs end-to-end sample clip
5. Agents auto-update docs (README, PRD, Solution Architecture, Directory Structure)

### 10.2 CI Gates
- **Lint:** Pre-commit hooks (black, ruff, mypy)
- **Tests:** Pytest unit + integration tests
- **Acceptance:** Checklist in `ACCEPTANCE_MATRIX.md` must be ✅
- **Docs:** Config keys documented in `docs/reference/config/pipeline_configs.md`

---

## 11. Agents & Automation Hooks

### 11.1 Triggers
- **File changes:** Any `add`, `delete`, `rename` under `apps/`, `web/`, `packages/`, `db/`, `config/`, `FEATURES/`, `agents/`, `mcps/`, `docs/`, `infra/`, `tests/`, `tools/`

### 11.2 Actions
- **Playbook:** `agents/playbooks/update-docs-on-change.yaml`
- **Updates:**
  - `docs/architecture/solution_architecture.md` — Adjust affected components/paths
  - `docs/architecture/directory_structure.md` — Update tree and descriptions
  - `docs/product/prd.md` — Mark feature addition/removal
  - `README.md` — Reflect new/removed directories

### 11.3 MCP Integration
- Codex and Claude agents use MCP servers to:
  - Query low-confidence tracks
  - Assign identities programmatically
  - Export screentime to external systems (Sheets, Airtable, etc.)

---

## 12. Future Enhancements

- **Scene-type classification:** VLM-based tagging (LLaVA, Qwen-VL)
- **Emotion/sentiment tagging:** Face expression analysis
- **Multi-camera handling:** Synchronized multi-angle episodes
- **Real-time analytics stream:** Live episode processing during broadcast
- **Cross-season trends dashboard:** Historical analytics across seasons

---

## 13. References

- [Directory Structure](directory_structure.md) — Repo layout and promotion policy
- [Pipeline Overview](../pipeline/overview.md) — Stage-by-stage pipeline details
- [Config Reference](../reference/config/pipeline_configs.md) — Key-by-key config documentation
- [Performance Tuning](../ops/performance_tuning_faces_pipeline.md) — Speed vs accuracy tuning
- [Troubleshooting](../ops/troubleshooting_faces_pipeline.md) — Common issues and fixes

---

**Maintained by:** Screenalytics Engineering
**Next Review:** After Phase 3 completion

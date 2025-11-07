# Screenalytics

> **Automated face, voice, and screen‑time intelligence for reality TV.**

Screenalytics ingests an episode, detects and tracks faces, recognizes cast members (even under partial occlusion), fuses voice activity, and outputs per-person **visual**, **speaking**, and **overlap** screen time. It powers internal tools (Shows/People directory, Socializer, Shop metadata) through a single source of truth.

---

## Highlights

- **Accurate under occlusion**: RetinaFace + ArcFace with quality gating and track-level pooling.
- **End-to-end pipeline**: Detect → Align → Track → Embed → Identify → Audio Fuse → Aggregate → Export.
- **Manual control**: Reassign at **cluster / track / frame** with full audit and undo.
- **Cloud-first**: S3-compatible blobs, Postgres + pgvector for metadata and embeddings.
- **Catalog built-in**: SHOWS and PEOPLE views unify cast metadata and featured thumbnails across products.
- **Agents & automation**: Codex/Agents SDK + MCP servers for low-confidence relabel, exports, and syncs.
- **Feature staging**: Isolate work in `FEATURES/` with promotion gates to keep `apps/` and `workers/` clean.

---

## Architecture

**Core stack**
- **Detection**: RetinaFace (InsightFace) with MediaPipe fallback  
- **Tracking**: ByteTrack  
- **Recognition**: ArcFace ONNX (AuraFace optional)  
- **Audio**: Pyannote diarization + Faster-Whisper ASR  
- **Storage**: S3-compatible (R2 / S3 / GCS)  
- **Database**: Postgres + **pgvector**  
- **Jobs**: Redis queue, idempotent workers  
- **UI**: Next.js workspace (SHOWS, PEOPLE, Episode Workspace)  
- **Agents**: Codex/Agents SDK + MCP servers (`screenalytics`, `storage`, `postgres`)  
- **Automation**: Zapier/n8n webhooks for notifications and exports

**Data flow**
```

[Workspace UI] → [API] → [Redis] → [Workers] → S3 (videos/frames/chips/reports)
└──────────→ Postgres + pgvector (episodes/tracks/embeddings/results)
Agents (Codex/Claude) ⇄ MCP servers (screenalytics/storage/postgres)
Webhooks → Zapier/n8n (notify, export to Sheets/Drive)

```

---

## Repository layout (summary)

```

apps/
api/              # FastAPI (CRUD, jobs, signed URLs, results)
workspace-ui/     # Next.js (SHOWS, PEOPLE, Workspace tabs)
workers/            # Pipeline stages (detect, track, id, audio, fuse, metrics, qa)
packages/           # Shared libs (py-screenalytics/, ts-sdk/)
db/                 # migrations, seeds, views
config/             # pipeline YAMLs + codex + agents sdk
FEATURES/           # feature sandboxes (src/tests/docs/TODO.md) with TTL + promotion gates
agents/             # AGENTS.md, playbooks, profiles, tasks
mcps/               # MCP servers (screenalytics/storage/postgres)
docs/               # architecture, pipeline, data model, API, ops, config reference
infra/              # docker compose, Dockerfiles, IaC
tests/              # api/workers/pipelines
tools/              # new-feature.py, promote-feature.py, lint-status.py

````

---

## Getting started

### Prerequisites
- Python 3.11+
- Node 20+ and **pnpm**
- Docker + Docker Compose
- **ffmpeg** on PATH
- GPU optional (recommended for speed)

### Bootstrap (local)
```bash
git clone https://github.com/<you>/screenalytics.git
cd screenalytics

# Bring up infra (Postgres, Redis, MinIO)
docker compose -f infra/docker/compose.yaml up -d

# Python deps (uv) and UI deps
uv sync
pnpm install

# Seed env
cp .env.example .env

# Migrate DB
psql "$DB_URL" -f db/migrations/0001_init_core.sql

# Run services
uv run apps/api/main.py               # API
uv run workers/orchestrator.py        # Workers (pipeline)
pnpm --filter workspace-ui dev        # UI
````

### Minimal .env example

```
DB_URL=postgresql://user:pass@localhost:5432/screenalytics
REDIS_URL=redis://localhost:6379/0
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=minio
S3_SECRET_KEY=miniosecret
S3_BUCKET=screenalytics
OPENAI_API_KEY=sk-***
```

---

## Product surface

### SHOWS view

* Create/edit **shows**; add **seasons**; manage **episode** metadata.
* Each episode links into the Workspace.

### PEOPLE view

* Select a **show** to see the cast grid with **featured thumbnail** per person.
* Click to view **episode presence** and manage featured media (shared with other tools).

### Episode Workspace (tabs)

1. **Detections**: RetinaFace boxes + landmarks, threshold controls.
2. **Tracks**: ByteTrack timelines; merge/split; thumbnails per span.
3. **Identities**: ArcFace assignments, occlusion badges, **reassign cluster/track/frame**, lock + propagate.
4. **A/V Fusion**: pyannote speakers overlaid with face tracks; “speaking appearances.”
5. **Results**: Visual/Speaking/Overlap totals; export CSV/JSON.
6. **QA Queue**: Low-confidence and disagreements; inline replay; audit trail.

---

## Data model (essentials)

**Catalog**

* `show`, `season`, `episode`, `person`, `cast_membership`
* `media_asset` + `person_featured` (featured thumbnails)

**Pipeline**

* `shot`, `detection`, `track`
* `embedding(vec VECTOR, owner_type in ['facebank','track'])` with HNSW index
* `assignment` (+ `locked`, `label_source`, `method`, `score`, `threshold`)
* `assignment_history` (full audit)
* `speech_segment`, `transcript`, `av_link`
* `screen_time` (visual_s, speaking_s, both_s, confidence)

---

## Storage layout (S3/R2/GCS)

```
videos/{show}/{SxxEyy}/episode.mp4
audio/{show}/{SxxEyy}/episode.wav
frames/{show}/{SxxEyy}/{shot_id}/{ts}.jpg
chips/{show}/{SxxEyy}/{track_id}/{n}.jpg
facebank/{person_id}/{hash}.jpg
manifests/{show}/{SxxEyy}/tracks.jsonl
reports/{show}/{SxxEyy}/screen_time.csv
```

Lifecycle: expire `frames/` and `thumbnails/` after N days; keep `chips/`, `facebank/`, `manifests/`.

---

## Configuration

All behavior is config-driven (no hardcoded thresholds).

```
config/
  pipeline/
    detection.yaml      # model ids, min_size, confidence_th, iou_th
    tracking.yaml       # ByteTrack: track_thresh, match_thresh, buffer
    recognition.yaml    # ArcFace model id, similarity_th, hysteresis
    audio.yaml          # diarization/asr knobs
    screen_time_v2.yaml # DAG + stage toggles
  storage.yaml          # buckets, prefixes, lifecycles
  services.yaml         # timeouts, external endpoints
  codex.config.toml     # Codex + MCP servers + write policies
  agents.sdk.yaml       # Agents SDK graph and profiles
```

---

## Agents & automation

* **MCP servers**

  * `screenalytics`: list low-confidence tracks, assign identity, export screen time, promote to facebank
  * `storage`: signed URLs, list/purge
  * `postgres`: safe analytics queries
* **Policies**: During feature work, agents write docs to `FEATURES/<feature>/docs/`. Root `/docs/**` updates occur only on promotion (CI-enforced).
* **Examples**

```bash
codex exec --config ./config/codex.config.toml \
  --task ./agents/tasks/aggregate-screen-time.json
```

---

## Feature staging & promotion

* Build in `FEATURES/<name>/` (`src/`, `tests/`, `docs/`, `TODO.md`) with 30-day TTL.
* Promotion requires: lint, tests, config docs, acceptance checks, and PR review.
* CI blocks imports from `FEATURES/**` in production code.

---

## Common tasks (CLI)

```bash
# Create a new Show, Season, Episode (API)
curl -X POST http://localhost:8000/shows -d '{"slug":"rhoslc","title":"RHOSLC"}' -H "Content-Type: application/json"

# Issue signed URLs for episode assets
curl -X POST http://localhost:8000/episodes/<ep_id>/assets

# Enqueue pipeline stages
curl -X POST http://localhost:8000/jobs/detect -d '{"ep_id":"<ep>"}'
curl -X GET  http://localhost:8000/jobs/<job_id>/status

# Read results
curl http://localhost:8000/episodes/<ep>/screen_time
```

---

## Performance & accuracy targets (v1)

* Visual ID accuracy ≥ **90%** on validation clips with sunglasses/side-profiles.
* Speaking match precision ≥ **85%** with diarization alignment.
* 1-hour episode ≤ **10 minutes** on 1× mid-tier GPU (or ≤ 3× realtime CPU-only).

---

## Security & privacy

* Short-lived signed URLs for all media access.
* Row-level security (optional per show).
* Full audit via `assignment_history`.
* No secrets in repo; env-only.

---

## Contributing

* Open a draft PR early; link relevant items in `MASTER_TODO.md`.
* Add tests and docs for any promoted module.
* Follow `CODING_STANDARDS.md`.
* Use `FEATURES/` for spikes and new components.

---

## License

Choose **MIT** or **Apache-2.0** before first public release. Include model-specific license notes where applicable.

---

## Status

Planning/Scaffolding phase. See:

* `MANIFEST.md` — origin and principles
* `PRD.md` — product requirements
* `SOLUTION_ARCHITECTURE.md` — system diagram and data model
* `DIRECTORY_STRUCTURE.md` — repo map and promotion policy
* `ACCEPTANCE_MATRIX.md` — promotion acceptance gates

See `ACCEPTANCE_MATRIX.md` for feature acceptance and promotion gates.

```
::contentReference[oaicite:0]{index=0}
```

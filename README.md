# Screenalytics

> **Automated face, voice, and screen‑time intelligence for reality TV.**

Screenalytics ingests an episode, detects and tracks faces, recognizes cast members, fuses voice activity, and outputs per-person **visual**, **speaking**, and **overlap** screen time.

---

## Quick Links

**Architecture & Design**
- [Solution Architecture](SOLUTION_ARCHITECTURE.md) → [Full Details](docs/architecture/solution_architecture.md)
- [Directory Structure](DIRECTORY_STRUCTURE.md) → [Full Details](docs/architecture/directory_structure.md)
- [Configuration Guide](CONFIG_GUIDE.md) → [Full Reference](docs/reference/config/pipeline_configs.md)

**Pipeline Documentation**
- [Pipeline Overview](docs/pipeline/overview.md) - High-level pipeline stages and data flow
- [Detect & Track](docs/pipeline/detect_track_faces.md) - Face detection and tracking
- [Faces Harvest](docs/pipeline/faces_harvest.md) - Face embedding extraction
- [Cluster Identities](docs/pipeline/cluster_identities.md) - Identity clustering
- [Episode Cleanup](docs/pipeline/episode_cleanup.md) - Track refinement and cleanup

**Operations & Troubleshooting**
- [Performance Tuning](docs/ops/performance_tuning_faces_pipeline.md) - Speed vs accuracy optimization
- [Troubleshooting](docs/ops/troubleshooting_faces_pipeline.md) - Common issues and fixes
- [Hardware Sizing](docs/ops/hardware_sizing.md) - Hardware recommendations

**Reference**
- [Artifact Schemas](docs/reference/artifacts_faces_tracks_identities.md) - JSONL/NPY file formats
- [Facebank](docs/reference/facebank.md) - Facebank structure and management
- [API Reference](API.md) - HTTP endpoints
- [Acceptance Matrix](ACCEPTANCE_MATRIX.md) - Quality gates and metrics

---

## Project Layout

This repository now hosts two main code paths:

- **Python ML/Streamlit** (existing): pipelines, API, and tools at the repo root.
- **Next.js + TypeScript + Prisma** web app under `web/` for the Youth League Team Builder (event/division admin and agents).

Quickstart for the web app:

```bash
cd web
npm install
cp .env.example .env.local    # provide DATABASE_URL, OPENAI_API_KEY, etc.
npx prisma generate
npx prisma migrate dev        # creates local schema; requires DATABASE_URL
npm run dev                   # http://localhost:3000
```

API stubs available at:
- `GET /api/events`
- `GET /api/divisions?eventId=...`
- `POST /api/agent-run`
- `GET /api/agent-status`

UI stubs:
- Home (`/`) lists events.
- Request Analysis (`/request-analysis`) can start a dummy agent run and poll status.

---

## Core Stack

- **Detection**: RetinaFace (InsightFace)
- **Tracking**: ByteTrack + Appearance Gate
- **Recognition**: ArcFace (512-d embeddings)
- **Storage**: S3-compatible (R2/S3/GCS) + Postgres + pgvector
- **Jobs**: Redis queue, idempotent workers
- **UI**: Next.js workspace + Streamlit tools

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node 20+ and pnpm
- Docker + Docker Compose
- ffmpeg on PATH
- GPU optional (recommended for speed)

### Bootstrap (Local)

```bash
git clone https://github.com/<you>/screenalytics.git
cd screenalytics

# Bring up infra (Postgres, Redis, MinIO)
docker compose -f infra/docker/compose.yaml up -d

# Python deps (uv) and UI deps
uv sync
source .venv/bin/activate
pnpm install

# Seed env
cp .env.example .env

# Dev infra (DB/Redis/S3 + migrations + env exports)
source ./tools/dev-up.sh

# Run services
python -m uvicorn apps.api.main:app --reload  # API
uv run workers/orchestrator.py               # Workers
pnpm --filter workspace-ui dev               # UI
```

### Minimal Episode Run

```bash
source .venv/bin/activate
./tools/dev-up.sh

# Detect + track faces (default: balanced profile)
python tools/episode_run.py --ep-id ep_demo --video samples/demo.mp4 --stride 3

# Harvest faces and embed
python tools/episode_run.py --ep-id ep_demo --faces-embed --save-crops

# Cluster identities
python tools/episode_run.py --ep-id ep_demo --cluster
```

**Artifacts land under `data/`:**
- `data/videos/{ep_id}/episode.mp4`
- `data/manifests/{ep_id}/detections.jsonl`
- `data/manifests/{ep_id}/tracks.jsonl`
- `data/manifests/{ep_id}/faces.jsonl`
- `data/embeds/{ep_id}/faces.npy` + `tracks.npy`
- `data/manifests/{ep_id}/identities.json`
- `data/manifests/{ep_id}/track_metrics.json`

See [SETUP.md](SETUP.md) for complete installation and artifact details.

---

## Performance Profiles & Limits

Screenalytics provides **hardware-aware performance profiles** to prevent overheating and optimize for your device:

### Profiles

Use `--profile` to select a preset:

```bash
# For fanless devices (MacBook Air, low-power laptops)
python tools/episode_run.py --ep-id demo --video test.mp4 --profile fast_cpu

# Standard local dev (balanced recall/speed)
python tools/episode_run.py --ep-id demo --video test.mp4 --profile balanced

# GPU production (maximum accuracy)
python tools/episode_run.py --ep-id demo --video test.mp4 --profile high_accuracy --device cuda
```

### Profile Defaults

| Profile | frame_stride | detection_fps_limit | min_size | Use Case |
|---------|--------------|---------------------|----------|----------|
| **fast_cpu** | 10 | 15 FPS | 120px | Fanless devices (Air, low-power) |
| **balanced** | 5 | 24 FPS | 90px | Standard local dev |
| **high_accuracy** | 1 | 30 FPS | 64px | GPU production |

### ⚠️ Thermal Warnings

**Avoid these combinations on CPU-only laptops (especially fanless):**

- ❌ `--stride 1` + `--fps 30` + `--save-frames --save-crops` (will overheat)
- ❌ `--profile high_accuracy` without `--device cuda` (CPU will thermal throttle)
- ❌ Exporting frames+crops on CPU for 1-hour episodes (disk I/O bottleneck)

**Safe CPU defaults:**
- ✅ Use `--profile fast_cpu` or `--profile balanced`
- ✅ Limit thread count: `export SCREENALYTICS_MAX_CPU_THREADS=2`
- ✅ Disable exporters when not needed: `--no-save-frames --no-save-crops`

**GPU recommendations:**
- ✅ Use `--profile high_accuracy --device cuda` for production runs
- ✅ RTX 3080 or better for 1-hour episodes (~5-10 minutes)
- ✅ T4 (AWS g4dn.2xlarge) for cloud workloads

See [docs/ops/performance_tuning_faces_pipeline.md](docs/ops/performance_tuning_faces_pipeline.md) and [docs/ops/hardware_sizing.md](docs/ops/hardware_sizing.md) for complete tuning guidance.

---

## Pipeline Metrics & Guardrails

The pipeline now tracks **derived metrics** and emits **warnings** when thresholds are exceeded:

**Track Metrics** (in `track_metrics.json`):
- `tracks_per_minute`: Tracks born per minute (⚠️ warn if > 50)
- `short_track_fraction`: Percentage of ghost tracks (⚠️ warn if > 0.3)
- `id_switch_rate`: ID switches / tracks_born (⚠️ warn if > 0.1)

**Cluster Metrics** (in `identities.json`):
- `singleton_fraction`: Single-track identities (⚠️ warn if > 0.5)
- `largest_cluster_fraction`: Largest identity's share (⚠️ warn if > 0.6)

All warnings include **actionable recommendations** and link to [docs/ops/troubleshooting_faces_pipeline.md](docs/ops/troubleshooting_faces_pipeline.md).

---

## Configuration

All thresholds are **config-driven** (no hardcoded magic numbers):

- **Detection**: `config/pipeline/detection.yaml`
- **Tracking**: `config/pipeline/tracking.yaml`
- **Quality Gating**: `config/pipeline/faces_embed_sampling.yaml`
- **Clustering**: `config/pipeline/clustering.yaml`
- **Performance Profiles**: `config/pipeline/performance_profiles.yaml`

**Override precedence:** CLI args > Environment variables > Performance profile > Stage config > Defaults

See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for quick reference or [docs/reference/config/pipeline_configs.md](docs/reference/config/pipeline_configs.md) for complete documentation.

---

## API Usage

```bash
# Detect + track
POST /jobs/detect_track
{
  "ep_id": "rhobh-s05e02",
  "profile": "balanced",
  "save_frames": true,
  "save_crops": false
}

# Faces embed
POST /jobs/faces_embed
{
  "ep_id": "rhobh-s05e02"
}

# Cluster identities
POST /jobs/cluster
{
  "ep_id": "rhobh-s05e02",
  "cluster_thresh": 0.58
}

# Episode cleanup
POST /jobs/episode_cleanup_async
{
  "ep_id": "rhobh-s05e02",
  "actions": ["split_tracks", "reembed", "recluster"],
  "write_back": true
}
```

See [API.md](API.md) for complete endpoint documentation.

---

## Feature Staging & Promotion

- Develop in `FEATURES/<name>/` with 30-day TTL
- Promotion requires: tests, docs, acceptance checks, PR review
- CI blocks imports from `FEATURES/**` in production code

See [FEATURES_GUIDE.md](FEATURES_GUIDE.md) and [ACCEPTANCE_MATRIX.md](ACCEPTANCE_MATRIX.md).

---

## Agents & Automation

**MCP Servers:**
- `screenalytics`: List low-confidence tracks, assign identities, export screen time
- `storage`: Signed URLs, list/purge
- `postgres`: Safe analytics queries

**Policies:**
- Agents cannot modify detection/tracking/clustering thresholds
- Write access to `FEATURES/` during development, `docs/` only on promotion

See [agents/AGENTS.md](agents/AGENTS.md) for complete policies.

---

## Data Model (Essentials)

**Catalog:** `show`, `season`, `episode`, `person`, `cast_membership`

**Pipeline:** `shot`, `detection`, `track`, `embedding(vec VECTOR)`, `assignment`, `speech_segment`, `screen_time`

**Storage Layout (S3):**
```
raw/videos/{show}/s{season}/e{episode}/episode.mp4
artifacts/frames/{show}/s{season}/e{episode}/frames/
artifacts/crops/{show}/s{season}/e{episode}/tracks/
artifacts/manifests/{show}/s{season}/e{episode}/
artifacts/thumbs/{show}/s{season}/e{episode}/identities/
```

See [docs/architecture/solution_architecture.md](docs/architecture/solution_architecture.md) for complete data model.

---

## Testing

```bash
# ML integration tests (requires RUN_ML_TESTS=1)
RUN_ML_TESTS=1 pytest tests/ml/test_detect_track_real.py -v
RUN_ML_TESTS=1 pytest tests/ml/test_faces_embed.py -v
RUN_ML_TESTS=1 pytest tests/ml/test_cluster.py -v

# Unit tests
pytest tests/ml/test_quality_gating.py -v
pytest tests/ml/test_performance_profiles.py -v

# API tests
pytest tests/api/ -v
```

See [ACCEPTANCE_MATRIX.md](ACCEPTANCE_MATRIX.md) for acceptance criteria.

---

## Common Errors & Troubleshooting

- `Connection refused` → Start API: `python -m uvicorn apps.api.main:app --reload`
- `NoSuchBucket` → Run `bash scripts/s3_bootstrap.sh` or set `S3_AUTO_CREATE=1`
- `RetinaFace init failed` → Download models: `python scripts/fetch_models.py`
- `Blank/gray crops` → See [docs/ops/troubleshooting_faces_pipeline.md#blank-crops](docs/ops/troubleshooting_faces_pipeline.md)
- `Too many tracks` → See [docs/ops/troubleshooting_faces_pipeline.md#track-explosion](docs/ops/troubleshooting_faces_pipeline.md)

Full troubleshooting guide: [docs/ops/troubleshooting_faces_pipeline.md](docs/ops/troubleshooting_faces_pipeline.md)

---

## Contributing

- Open a draft PR early
- Add tests and docs for promoted features
- Use `FEATURES/` for experimental work
- Follow `CODING_STANDARDS.md`

See [FEATURES_GUIDE.md](FEATURES_GUIDE.md) for the promotion workflow.

---

## License

MIT or Apache-2.0 (choose before public release)

---

## Status

**Production-ready pipeline** with comprehensive documentation and testing.

**Recent Updates:**
- ✅ Phase 1: 70,000+ words of documentation across 14 new docs
- ✅ Phase 2: Config-driven refactoring with metrics, guardrails, and performance profiles
- 🚧 Phase 3: Hardening, API integration, and comprehensive testing (in progress)

See [REFACTORING_COMPLETE_SUMMARY.md](REFACTORING_COMPLETE_SUMMARY.md) and [PHASE_2_IMPLEMENTATION_SUMMARY.md](PHASE_2_IMPLEMENTATION_SUMMARY.md) for complete change history.

---

**Maintained by:** Screenalytics Engineering

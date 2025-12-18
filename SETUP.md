# SETUP.md — Screenalytics Environment Guide

---

## 1️⃣ Prerequisites
- Python 3.11+
- Docker + Docker Compose
- ffmpeg installed
- (Optional GPU) CUDA 12+ for PyTorch
- GitHub CLI or SSH key configured
- Node 20+ (only if you want to run the optional Next.js app under `web/`)

---

## 2️⃣ Clone & bootstrap
```bash
git clone https://github.com/<your-org>/screenalytics.git
cd screenalytics
```

### Virtual environment

> **macOS (Apple Silicon):** install `pyenv`, add the following to `~/.zshrc`, then install + select Python 3.11.9 before creating the venv.
> ```bash
> brew install pyenv
> if command -v pyenv >/dev/null; then
>   eval "$(pyenv init --path)"
> fi
> if [[ $- == *i* ]]; then
>   eval "$(pyenv init -)"
> fi
> pyenv install 3.11.9
> pyenv local 3.11.9
> ```

Now create the environment with the pinned interpreter:

```bash
python -m venv .venv
source .venv/bin/activate # or .\.venv\Scripts\activate on Windows
pip install -U pip wheel setuptools
pip install -r requirements.txt
# For pipeline runs (detect/track, embeddings, body tracking), also install:
pip install -r requirements-ml.txt
```

### Node deps (optional Next.js app)
```bash
cd web
npm install
cd -
```

### Environment file
Copy `.env.example` → `.env` and fill in credentials:
```
DB_URL=postgresql://user:pass@localhost:5432/screenalytics
REDIS_URL=redis://localhost:6379/0
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=minio
S3_SECRET_KEY=miniosecret
S3_BUCKET=screenalytics
OPENAI_API_KEY=sk-xxxx
```

### Audio Pipeline Setup (NeMo MSDD + OpenAI Whisper)

The default audio pipeline uses:
- **NeMo MSDD** for overlap-aware speaker diarization (GPU recommended)
- **OpenAI Whisper** for transcription (requires `OPENAI_API_KEY`)

**Required (for ASR):**
```
OPENAI_API_KEY=sk-xxxx
```

**Optional (audio enhancement):**
```
RESEMBLE_API_KEY=...
```

**Legacy (deprecated):** pyannote diarization workflows are still supported for older episodes/backfills, but are no longer the recommended path. If you need them, set `PYANNOTEAI_API_KEY` (cloud) or `PYANNOTE_AUTH_TOKEN` (local OSS model) and configure `config/pipeline/audio.yaml` accordingly.

---

## 3️⃣ Local services (Docker)
```bash
docker compose -f infra/docker/compose.yaml up -d
```
Starts Postgres, Redis, and MinIO.

---

## 4️⃣ Run core components
| Component | Command |
| ----------- | -------------------------------- |
| **API** | `python -m uvicorn apps.api.main:app --reload` |
| **Workers (Celery)** | `celery -A apps.api.celery_app:celery_app worker -l info` |
| **Streamlit UI** | `streamlit run apps/workspace-ui/Upload_Video.py` |
| **Optional Next.js** | `cd web && npm run dev` |
| **Tests** | `pytest -q` |

---

## 5️⃣ Optional: initialize database
```bash
psql "$DB_URL" -f db/migrations/0001_init_core.sql
```

---

## 6️⃣ Verify install
Visit Streamlit at [http://localhost:8501](http://localhost:8501) → Upload tab should appear.
API health check: [http://localhost:8000/health](http://localhost:8000/health) returns `{"status":"ok"}`.
Next.js (optional) runs at [http://localhost:3000](http://localhost:3000).

---

## 7️⃣ Hardware Requirements & Performance

`tools/episode_run.py` does **not** accept `--profile`; pass explicit stride/FPS to mirror the presets.

**CPU-only (Apple Silicon M1/M2/M3):**
- ✅ Suitable for development and testing
- ✅ Use `--stride 5-8 --fps 8-24 --device auto` to avoid thermal throttling
- ✅ Limit threads if needed: `SCREENALYTICS_MAX_CPU_THREADS=2`
- **Avoid:** `--stride 1` + `--save-frames` on fanless laptops

**GPU (CUDA):**
- ✅ Recommended for production and high-accuracy work
- Use `--device cuda --stride 1 --fps 30`
- Expected runtime: ~3-5 min for 24-minute episode (stride=1)

**For detailed performance tuning**, see:
- [docs/ops/performance_tuning_faces_pipeline.md](docs/ops/performance_tuning_faces_pipeline.md)
- [docs/reference/config/pipeline_configs.md](docs/reference/config/pipeline_configs.md)

---

## 8️⃣ Artifact Pipeline

The faces pipeline produces a chain of artifacts that link together via `ep_id`, `track_id`, and `identity_id`:

```
detect/track → faces_embed → cluster → cleanup
```

### Pipeline Stages & Outputs

**Stage 1: Detection & Tracking**
- `data/manifests/{ep_id}/detections.jsonl` – Raw face detections per frame
- `data/manifests/{ep_id}/tracks.jsonl` – Temporal face tracks with metadata
- `data/manifests/{ep_id}/track_metrics.json` – Derived metrics (tracks_per_minute, short_track_fraction, etc.)

**Stage 2: Face Embedding & Sampling**
- `data/manifests/{ep_id}/faces.jsonl` – Quality-gated face crops with embeddings
- `data/embeds/{ep_id}/faces.npy` – Face embeddings as NumPy array
- `data/embeds/{ep_id}/tracks.npy` – Track-level pooled embeddings (used for clustering)

**Stage 3: Clustering**
- `data/manifests/{ep_id}/identities.json` – Cluster assignments mapping `track_id` → `identity_id`
- Includes cluster metrics: `singleton_fraction`, `largest_cluster_fraction`, `cluster_count`

**Stage 4: Cleanup (Optional)**
- `data/cleanup/{ep_id}/cleanup_report.json` – Before/after metrics from outlier removal
- Cleaned versions of `tracks.jsonl`, `faces.jsonl`, `identities.json`

### Artifact Relationships

```
detections.jsonl
  ↓ (grouped by track_id)
tracks.jsonl ──────────────┐
  ↓ (quality gating)        │ track_id links
faces.jsonl ───────────────┤
  ↓ (pooling)               │
tracks.npy ────────────────┤
  ↓ (clustering)            │
identities.json ───────────┘
  (track_id → identity_id mapping)
```

**Key Fields:**
- `ep_id` – Links all artifacts for a single episode
- `track_id` – Unique ID for a face track (e.g., `track-00001`)
- `identity_id` – Cluster assignment (e.g., `identity-00001`)
- `frame_idx` – Zero-based frame number in source video
- `ts_s` – Timestamp in seconds

### Schema Documentation

For schema definitions and field descriptions, see:
- [docs/reference/artifacts_faces_tracks_identities.md](docs/reference/artifacts_faces_tracks_identities.md)
- [docs/audio/diarization_manifest.md](docs/audio/diarization_manifest.md)
- [ACCEPTANCE_MATRIX.md](ACCEPTANCE_MATRIX.md) (thresholds and CI gates)

---

## 9️⃣ Documentation Index

Canonical documentation lives under `docs/`:
- [docs/README.md](docs/README.md)

**Pipeline Overview:**
- [docs/pipeline/overview.md](docs/pipeline/overview.md) – End-to-end pipeline architecture
- [docs/pipeline/detect_track_faces.md](docs/pipeline/detect_track_faces.md) – Detection & tracking details
- [docs/pipeline/faces_harvest.md](docs/pipeline/faces_harvest.md) – Face sampling & embedding
- [docs/pipeline/cluster_identities.md](docs/pipeline/cluster_identities.md) – Identity clustering
- [docs/pipeline/episode_cleanup.md](docs/pipeline/episode_cleanup.md) – Track refinement & post-processing
- [docs/pipeline/audio_pipeline.md](docs/pipeline/audio_pipeline.md) – Audio pipeline (diarization/ASR)

**Configuration & Tuning:**
- [docs/reference/config/pipeline_configs.md](docs/reference/config/pipeline_configs.md) – All config parameters
- [docs/ops/performance_tuning_faces_pipeline.md](docs/ops/performance_tuning_faces_pipeline.md) – Speed vs accuracy tuning

**Schemas & Metrics:**
- [docs/reference/artifacts_faces_tracks_identities.md](docs/reference/artifacts_faces_tracks_identities.md) – Vision artifact schemas (detections/tracks/faces/identities)
- [docs/audio/diarization_manifest.md](docs/audio/diarization_manifest.md) – Audio diarization/ASR manifests
- [ACCEPTANCE_MATRIX.md](ACCEPTANCE_MATRIX.md) – Quality thresholds and verification
- [docs/reference/api.md](docs/reference/api.md) – API endpoints and request/response shapes

**Operations:**
- [docs/ops/troubleshooting_faces_pipeline.md](docs/ops/troubleshooting_faces_pipeline.md) – Debugging & common fixes
- [docs/ops/ARTIFACTS_STORE.md](docs/ops/ARTIFACTS_STORE.md) – Storage layout and artifact handling

---

## 🔟 Agents & automation
* Codex config: `config/codex.config.toml`
* Claude policy: `config/claude.policies.yaml`
* Agents auto-update `README.md`, `docs/architecture/solution_architecture.md`, `docs/architecture/directory_structure.md`, and `docs/product/prd.md` when files change.

To run Codex locally:
```bash
codex exec --config config/codex.config.toml --task agents/tasks/aggregate-screen-time.json
```

---

## 1️⃣1️⃣ Promotion workflow
1. Create a feature sandbox under `FEATURES/<name>/` (see `docs/features/feature_sandboxes.md`)
2. Work inside `FEATURES/<name>/`
3. Pass CI (tests + docs)
4. Promote via PR: move code/tests/docs out of `FEATURES/` into `apps/` / `web/` / `packages/`
5. CI + Agents update docs automatically

---

## 1️⃣2️⃣ Troubleshooting
| Symptom | Fix |
| --------------------- | ------------------------------------------------ |
| `ModuleNotFoundError` | Activate `.venv` |
| `supervision_missing` / `ModuleNotFoundError: supervision` | Install ML deps: `pip install -r requirements-ml.txt` (macOS: `install supervision` is not `pip install supervision`) |
| `torchreid_missing` / `ModuleNotFoundError: torchreid` | Install ML deps: `pip install -r requirements-ml.txt` |
| Dependency doctor | `python -c "import supervision, torchreid; print('ok')"` |
| `pgvector` not found | Run migrations, ensure Postgres ≥15 |
| UI blank | Run `pnpm build` inside `apps/workspace-ui` |
| Codex writes blocked | Ensure correct profile (`promote` for root docs) |

---

**Screenalytics – “Every frame tells a story.”**

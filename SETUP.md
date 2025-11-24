# SETUP.md — Screenalytics Environment Guide

---

## 1️⃣ Prerequisites
- Python 3.11+
- Node 20+ and pnpm
- Docker + Docker Compose
- ffmpeg installed
- (Optional GPU) CUDA 12+ for PyTorch
- GitHub CLI or SSH key configured

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
```

### Node deps
```bash
pnpm install --filter workspace-ui
```

### Install Codex CLI
```bash
npm i -g @openai/codex   # or: brew install --cask codex
codex                    # sign in with your ChatGPT plan
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
| **API** | `uv run apps/api/main.py` |
| **Workers** | `uv run workers/orchestrator.py` |
| **UI** | `pnpm --filter workspace-ui dev` |
| **Tests** | `pytest -q` |

---

## 5️⃣ Optional: initialize database
```bash
psql "$DB_URL" -f db/migrations/0001_init_core.sql
```

---

## 6️⃣ Verify install
Visit [http://localhost:3000](http://localhost:3000) → Upload tab should appear.
API health check: [http://localhost:8000/health](http://localhost:8000/health) returns `{"status":"ok"}`.

---

## 7️⃣ Hardware Requirements & Performance

### CPU vs GPU

**CPU-only (Apple Silicon M1/M2/M3):**
- ✅ Suitable for development and testing
- ⚠️ Use `--profile fast_cpu` to avoid thermal throttling
- Expected runtime: ~10-15 min for 24-minute episode (stride=10)
- **Avoid:** `--stride 1` + `--save-frames` on fanless laptops

**GPU (CUDA):**
- ✅ Recommended for production and high-accuracy work
- Use `--profile high_accuracy --device cuda`
- Expected runtime: ~3-5 min for 24-minute episode (stride=1)

### Performance Profiles

See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for full details. Quick reference:

```bash
# Fanless devices (MacBook Air, low-power)
python tools/episode_run.py --ep-id demo --video test.mp4 --profile fast_cpu

# Standard local dev (balanced recall/speed)
python tools/episode_run.py --ep-id demo --video test.mp4 --profile balanced

# GPU production (maximum accuracy)
python tools/episode_run.py --ep-id demo --video test.mp4 --profile high_accuracy --device cuda
```

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
- `data/detections/{ep_id}/detections.jsonl` – Raw face detections per frame
- `data/tracks/{ep_id}/tracks.jsonl` – Temporal face tracks with metadata
- `data/tracks/{ep_id}/track_metrics.json` – Derived metrics (tracks_per_minute, short_track_fraction, etc.)

**Stage 2: Face Embedding & Sampling**
- `data/faces/{ep_id}/faces.jsonl` – Quality-gated face crops with embeddings
- `data/faces/{ep_id}/faces.npy` – Face embeddings as NumPy array
- `data/embeds/{ep_id}/tracks.npy` – Track-level pooled embeddings (used for clustering)

**Stage 3: Clustering**
- `data/identities/{ep_id}/identities.json` – Cluster assignments mapping `track_id` → `identity_id`
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

For complete schema definitions and field descriptions, see:
- [docs/reference/schemas/artifacts_schemas.md](docs/reference/schemas/artifacts_schemas.md)
- [docs/reference/schemas/identities_v1_spec.md](docs/reference/schemas/identities_v1_spec.md)

---

## 9️⃣ Documentation Index

All comprehensive documentation from Phase 1:

**Pipeline Overview:**
- [docs/pipeline/overview.md](docs/pipeline/overview.md) – End-to-end pipeline architecture
- [docs/pipeline/detect_track_stage.md](docs/pipeline/detect_track_stage.md) – Detection & tracking details
- [docs/pipeline/faces_embed_stage.md](docs/pipeline/faces_embed_stage.md) – Face sampling & embedding
- [docs/pipeline/clustering_stage.md](docs/pipeline/clustering_stage.md) – Identity clustering
- [docs/pipeline/cleanup_stage.md](docs/pipeline/cleanup_stage.md) – Outlier removal & post-processing

**Configuration & Tuning:**
- [CONFIG_GUIDE.md](CONFIG_GUIDE.md) – Quick config reference
- [docs/reference/config/pipeline_configs.md](docs/reference/config/pipeline_configs.md) – All config parameters
- [docs/ops/performance_tuning_faces_pipeline.md](docs/ops/performance_tuning_faces_pipeline.md) – Speed vs accuracy tuning

**Schemas & Metrics:**
- [docs/reference/schemas/artifacts_schemas.md](docs/reference/schemas/artifacts_schemas.md) – All artifact schemas
- [docs/reference/metrics/derived_metrics.md](docs/reference/metrics/derived_metrics.md) – Calculated metrics & guardrails
- [docs/reference/metrics/acceptance_matrix.md](docs/reference/metrics/acceptance_matrix.md) – Quality thresholds

**Operations:**
- [docs/ops/monitoring_logging_faces_pipeline.md](docs/ops/monitoring_logging_faces_pipeline.md) – Logging & debugging
- [docs/ops/episode_cleanup.md](docs/ops/episode_cleanup.md) – Cleanup workflow

---

## 🔟 Agents & automation
* Codex config: `config/codex.config.toml`
* Claude policy: `config/claude.policies.yaml`
* Agents auto-update docs (README, PRD, SolutionArchitecture, DirectoryStructure) when files change.

To run Codex locally:
```bash
codex exec --config config/codex.config.toml --task agents/tasks/aggregate-screen-time.json
```

---

## 1️⃣1️⃣ Promotion workflow
1. Create feature via `python tools/new-feature.py <name>`
2. Work inside `FEATURES/<name>/`
3. Pass CI (tests + docs)
4. Promote: `python tools/promote-feature.py <name>`
5. CI + Agents update docs automatically

---

## 1️⃣2️⃣ Troubleshooting
| Symptom | Fix |
| --------------------- | ------------------------------------------------ |
| `ModuleNotFoundError` | Activate `.venv` |
| `pgvector` not found | Run migrations, ensure Postgres ≥15 |
| UI blank | Run `pnpm build` inside `apps/workspace-ui` |
| Codex writes blocked | Ensure correct profile (`promote` for root docs) |

---

**Screenalytics – “Every frame tells a story.”**

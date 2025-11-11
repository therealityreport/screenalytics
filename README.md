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
source .venv/bin/activate
pnpm install

# Seed env
cp .env.example .env

# Dev infra (DB/Redis/S3 + migrations + env exports)
source ./tools/dev-up.sh

# Run services
python -m uvicorn apps.api.main:app --reload  # API
uv run workers/orchestrator.py               # Workers (pipeline)
pnpm --filter workspace-ui dev               # UI
````

> `tools/dev-up.sh` exports `DB_URL`, `REDIS_URL`, and `S3_*` for the current shell while bringing the Docker stack online. Source it (`source ./tools/dev-up.sh`) whenever you open a new terminal.

### Dev quick run

```bash
source .venv/bin/activate
./tools/dev-up.sh
# Add --stub for the fast path; omit it for the YOLOv8+ByteTrack pipeline.
python tools/episode_run.py --ep-id ep_demo --video samples/demo.mp4 --stride 5 --stub
```

Artifacts land under `data/` mirroring the future object-storage layout:

- `data/videos/ep_demo/episode.mp4`
- `data/manifests/ep_demo/detections.jsonl`
- `data/manifests/ep_demo/tracks.jsonl`
- `data/frames/ep_demo/` (only when `--fps` is provided)

### Dependency profiles

- **Core (API/UI/tests):**

  ```bash
  pip install -r requirements-core.txt
  ```

- **Full ML pipeline (optional / RetinaFace + ByteTrack + Whisper):**

  ```bash
  pip install -r requirements-ml.txt
  ```

### Upload via UI

**Quickstart**

- Default: `bash scripts/dev.sh` (runs API, waits on `/healthz`, then opens Streamlit).
- With Make: `make dev`.

1. Install dependencies:

   ```bash
   pip install -r requirements-core.txt
   # Optional full ML stack
   # pip install -r requirements-ml.txt
   ```

2. Copy and source env vars:

   ```bash
   cp .env.example .env
   set -a && source .env && set +a
   ```

3. Set your environment (pick one):

   **Local filesystem (default)**
   ```bash
   export STORAGE_BACKEND=local
   export SCREENALYTICS_API_URL=http://localhost:8000
   export UI_ORIGIN=http://localhost:8501
   ```

   **MinIO/S3-compatible**
   ```bash
   export STORAGE_BACKEND=s3
   export SCREENALYTICS_OBJECT_STORE_ENDPOINT=http://localhost:9000
   export SCREENALYTICS_OBJECT_STORE_BUCKET=screenalytics
   export SCREENALYTICS_OBJECT_STORE_ACCESS_KEY=minio
   export SCREENALYTICS_OBJECT_STORE_SECRET_KEY=miniosecret
   export SCREENALYTICS_API_URL=http://localhost:8000
   export UI_ORIGIN=http://localhost:8501
   ```

4. Start the API: `python -m uvicorn apps.api.main:app --reload` (or `uv run apps/api/main.py` if you prefer `uv`).
5. Launch the Streamlit upload helper: `streamlit run apps/workspace-ui/streamlit_app.py` (set `SCREENALYTICS_API_URL` if the API isn’t on `localhost:8000`).
6. Fill in Show, Season, Episode #, Title, optional Air date, choose an `.mp4`, and decide whether to enable **Use stub (fast, no ML)** before submitting. Leave it unchecked to run the real YOLOv8 + ByteTrack pass, or check it for the light stub.
7. The UI creates/returns the episode via the API, requests a presigned MinIO PUT for `videos/{ep_id}/episode.mp4`, mirrors the bytes locally, and calls `POST /jobs/detect_track` with the selected mode, showing counts once the job finishes.

Artifacts for any uploaded `ep_id` are written via `py_screenalytics.artifacts`:

- `data/videos/{ep_id}/episode.mp4`
- `data/manifests/{ep_id}/detections.jsonl`
- `data/manifests/{ep_id}/tracks.jsonl`
- `data/frames/{ep_id}/` (when jobs run with an `fps` override)

**Note:** Stub mode (`Use stub (fast, no ML)`) keeps the flow dependency-light and does not require the ML stack from `requirements-ml.txt`.

#### Run detection+tracking (real)

Need the full YOLOv8 + ByteTrack pass outside the UI?

1. Install the ML extras:

   ```bash
   pip install -r requirements-ml.txt
   ```

2. Run the episode helper without `--stub`:

   ```bash
   python tools/episode_run.py --ep-id ep_demo --video samples/demo.mp4 --stride 3 --fps 8
   ```

   - Lower `--stride` (for example `1` or `2`) and higher `--fps` increase recall but also GPU/CPU time.
   - Higher `--stride` or smaller `--fps` are useful for exploratory passes on long episodes.

3. (Optional) Verify the real pipeline via the ML test:

   ```bash
   RUN_ML_TESTS=1 pytest tests/ml/test_detect_track_real.py -q
   ```

The CLI and UI both emit manifests under `data/manifests/{ep_id}/` with YOLO metadata, so you can diff stub vs. real outputs before pushing upstream.

**Common errors**

- `Connection refused` — Start the API via `python -m uvicorn apps.api.main:app --reload` (or rerun `scripts/dev.sh`) and confirm it responds: `curl http://localhost:8000/healthz`.
- `API_BASE_URL mismatch` — Export the correct `SCREENALYTICS_API_URL` (or update `.env`) so the UI hits the right host/port.
- `File not found (Streamlit path)` — Ensure you launch commands from the repo root so `apps/workspace-ui/streamlit_app.py` resolves.
- `NoSuchBucket` — Run `bash scripts/s3_bootstrap.sh` or set `S3_AUTO_CREATE=1` before restarting the API.

### AWS S3 setup

1. Confirm AWS CLI auth:

   ```bash
   aws sts get-caller-identity
   ```

2. Bootstrap the global `screenalytics` bucket with lifecycle + encryption:

   ```bash
   bash scripts/s3_bootstrap.sh
   ```

3. Copy `.env.example`, keep `STORAGE_BACKEND=s3` (defaults to `AWS_S3_BUCKET=screenalytics`), and source it.

4. Run the dev flow (uploads will land in S3):

   ```bash
   STORAGE_BACKEND=s3 bash scripts/dev.sh
   ```

Artifacts live under the standardized prefixes:

- `s3://screenalytics/raw/videos/{ep_id}/episode.mp4`
- `s3://screenalytics/artifacts/manifests/{ep_id}/detections.jsonl`
- `s3://screenalytics/artifacts/faces/{ep_id}/...`

Verify uploads:

```bash
aws s3 ls s3://screenalytics/raw/videos/ --recursive | tail -n 5
```

#### Troubleshooting (macOS / FFmpeg / PyAV)

`faster-whisper` depends on PyAV, which may try to build against Homebrew’s FFmpeg 8.x headers on Apple Silicon. If you only need the API, UI, or stub detect/track flow, stick to `requirements-core.txt`—these features do **not** require PyAV or the rest of the ML stack. Install `requirements-ml.txt` only when you plan to run the full pipeline and have a working FFmpeg toolchain.

#### macOS (Apple Silicon) Python environment

If you're on Apple Silicon and rely on the system `zsh`, use `pyenv` to guarantee that `python` points at 3.11.9 before creating the virtual environment:

```bash
brew install pyenv

# ~/.zshrc
if command -v pyenv >/dev/null; then
  eval "$(pyenv init --path)"
fi
case $- in *i*) : ;; *) return ;; esac  # keep existing guard if you have one
if command -v pyenv >/dev/null; then
  eval "$(pyenv init -)"
fi

# back in the repo
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

`pyenv local 3.11.9` writes `.python-version`, so new shells automatically pick up the right interpreter before activating `.venv`.

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

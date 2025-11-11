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
# Export sampled frames/crops with RetinaFace + ByteTrack (face-only) and JPEG controls
python tools/episode_run.py --ep-id ep_demo --video samples/demo.mp4 --stride 3 \
  --detector retinaface --max-gap 30 \
  --save-frames --save-crops --jpeg-quality 90
# Faces pipeline
python tools/episode_run.py --ep-id ep_demo --faces-embed --save-crops
python tools/episode_run.py --ep-id ep_demo --cluster
```

Artifacts land under `data/` mirroring the future object-storage layout:

- `data/videos/ep_demo/episode.mp4`
- `data/manifests/ep_demo/detections.jsonl`
- `data/manifests/ep_demo/tracks.jsonl`
- `data/frames/ep_demo/frames/` + `crops/` (only when `--save-frames`/`--save-crops` are provided)
- `data/manifests/ep_demo/progress.json` (live `phase`/ETA snapshots written every ~25 frames)

`episode_run.py` now auto-detects FPS whenever `--fps` is unset/`0`, emits structured progress JSON (frames, seconds, inferred FPS, device, detector, tracker, phase) to stdout **and** to `progress.json`, and finishes with a `phase:"done"` payload that includes stage-specific counts, per-run track metrics (`tracks_born`, `tracks_lost`, `id_switches`, `longest_tracks`), plus local + v2 S3 prefixes (`artifacts/frames/{show}/s{ss}/e{ee}/frames/`, `artifacts/crops/{show}/s{ss}/e{ee}/tracks/`, `artifacts/manifests/{show}/s{ss}/e{ee}/`). `--detector {retinaface,yolov8face}` toggles between the high-quality InsightFace RetinaFace weights and the faster YOLOv8-face variant, while `--tracker {bytetrack,strongsort}` chooses either the default face-only ByteTrack gates or a StrongSORT/BoT-SORT ReID path for occlusions. Both trackers operate purely on face boxes (track_thresh≈0.6, match_thresh≈0.8, buffer≈30, min box area≈20 px), so there’s no longer a person-class detector. `--max-gap` still controls how many skipped frames cause a track split. `--faces-embed` now runs ArcFace (`arcface_r100_v1`) via ONNX Runtime to produce 512-d unit-norm embeddings (plus optional crops/frames), and `--cluster` averages those per-track vectors before agglomerative clustering so the Facebank can relabel/merge tracks.

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
6. Fill in Show, Season, Episode #, Title, optional Air date, choose an `.mp4`, then submit to mirror the footage locally and kick off the full YOLOv8 + ByteTrack pass (using the same defaults as Episode Detail—frames + crops exports enabled, JPEG quality 85, scene-cut detection on).
7. The UI creates/returns the episode via the API, requests a presigned MinIO PUT for `videos/{ep_id}/episode.mp4`, mirrors the bytes locally, and calls `POST /jobs/detect_track_async` (with a synchronous fallback button if you really need it). Progress is streamed from `/jobs/{job_id}/progress`, so you can cancel or watch counts materialize without blocking the page.

**Episode Detail live progress & exports**

- “Run detect/track” negotiates `text/event-stream` against `POST /jobs/detect_track`, so Streamlit renders a live bar (`mm:ss / MM:SS • phase=<detect|track> • device=<…> • fps=<…>`). If SSE isn’t available (corporate proxies, etc.) the button falls back to `/jobs/detect_track_async` and polls the new `GET /episodes/{ep_id}/progress` every 500 ms until the `phase:"done"` payload lands.
- Detector dropdown toggles between **RetinaFace (recommended)** and **YOLOv8-face (alt)**—both wired to the CLI’s `--detector {retinaface,yolov8face}` flag—while the new Tracker dropdown lets you pick **ByteTrack (default)** or **StrongSORT (ReID)** before launching a run. ByteTrack parameters (e.g. `--max-gap`, JPEG export toggles) stay available alongside the frame/crop exporter controls, and both “Save frames to S3” + “Save face crops to S3” are pre-selected so exports always land unless you opt out. The live status bar now renders `mm:ss / MM:SS • phase=<…> • detector=<…> • tracker=<…> • device=<…> • fps=<…>` so it’s obvious which models are currently running. Successful runs summarize counts, exporter stats, v2 prefixes, **and** the track quality metrics surfaced by the runner (`tracks born/lost`, `id switches`, `top longest tracks`, highlighting any track that exceeds 500 frames). Quick links jump straight into Faces Review (page 3) and Screentime (page 4) for follow-up QA.

Artifacts for any uploaded `ep_id` are written via `py_screenalytics.artifacts`:

- `data/videos/{ep_id}/episode.mp4`
- `data/manifests/{ep_id}/detections.jsonl`
- `data/manifests/{ep_id}/tracks.jsonl`
- `data/frames/{ep_id}/` (when jobs run with an `fps` override)

### Live progress (async detect/track)

![Live progress demo](docs/assets/live-progress.gif)

- `POST /jobs/detect_track_async` launches `tools/episode_run.py` in the background and records a job file under `data/jobs/{job_id}.json`.
- `GET /jobs/{job_id}/progress` streams the latest JSON snapshot (`frames_done`, `frames_total`, `elapsed_sec`, `fps_detected`, `analyzed_fps`), which the Streamlit UI renders as a progress bar with cancel + ETA controls.
- **Elapsed** is the real wall-clock runtime, **Total** is derived from `frames_total / analyzed_fps` (falling back to the detected FPS when no override is provided), and **ETA** is computed from the remaining frames and the effective processing FPS (or the observed ratio of frames_done/elapsed as a fallback).
- A `Cancel job` button wires into `POST /jobs/{job_id}/cancel`, cleanly SIGTERM-ing the worker and persisting `state=canceled`.
- Once `state` flips to `succeeded` the UI surfaces detections/tracks counts plus direct links to `data/manifests/{ep_id}/detections.jsonl` and `tracks.jsonl`; failures bubble up the captured stderr/error message inline.

**Faces pipeline + Facebank**

- Use `Run Faces Harvest` (Episode Detail or the Facebank page) to trigger `POST /jobs/faces_embed` with SSE updates. The CLI exports square thumbnails plus ArcFace (`arcface_r100_v1`) embeddings via ONNX Runtime, so every face row in `faces.jsonl` stores a 512-d unit-norm vector (and the same matrix is materialized under `data/embeds/{ep_id}/faces.npy`). When legacy tracks are detected (old “person” detectors), the UI blocks harvest/cluster buttons until a fresh RetinaFace/YOLOv8-face run completes.
- `Run Cluster` streams `POST /jobs/cluster`, which averages the per-track embeddings, clusters them with Agglomerative (tunable via `--cluster-thresh` and `--min-cluster-size`), and emits `identities.json` alongside identity-level thumbnails under `artifacts/thumbs/{show}/s{ss}/e{ee}/identities/{id}/rep.jpg`.
- When SSE can’t be established the UI falls back to the corresponding `..._async` endpoint and polls `GET /episodes/{ep_id}/progress` every 500 ms until `phase:"done"` arrives.
- The revamped Facebank page opens with a tracked-vs-S3 episode selector, shows a uniform identity grid (presigned thumbnails, inline Rename/Delete controls, and a **Merge into…** dropdown), and lets you drill down into clusters → tracks via a filmstrip UI. Cluster detail now renders one row per track: left/right arrows scroll a 4:5 thumbnail rail (no image rewriting; pure CSS `aspect-ratio: 4 / 5`), and the default “Sample every N crops” value (3) prevents the browser from loading thousands of JPGs at once. Each row surfaces **View track**, **Move to identity**, **Remove from identity**, and **Delete track** actions; “Load more” streams the next page via `GET /episodes/{ep_id}/tracks/{track_id}/crops?sample=N&limit=40&start_after={cursor}`, where `next_start_after` is treated as an opaque cursor. View Track opens the same horizontal scroller at sample=1 with paginator arrows so moderators can flip between clusters and dedicated track rails without reloading the page, plus inline **Move/Remove/Delete** controls and a “Run Faces Harvest (save crops to S3)” button whenever a track only has S3 crops. Every thumbnail rail relies on presigned URLs from `/episodes/{ep_id}/tracks/{track_id}/crops`, so face strips render even when crops only exist in S3. All moderation actions call the dedicated `/episodes/{ep_id}/identities/...` endpoints so `faces.jsonl`, `tracks.jsonl`, `identities.json`, and the corresponding S3 manifests stay in sync.

#### Run detection+tracking (real)

Need the full YOLOv8 + ByteTrack pass outside the UI?

1. Install the ML extras:

   ```bash
   pip install -r requirements-ml.txt
   ```

2. Run the episode helper (defaults to CPU):

   ```bash
   python tools/episode_run.py --ep-id ep_demo --video samples/demo.mp4 --stride 3 --fps 8 --device cpu
   ```

   - Lower `--stride` (for example `1` or `2`) and higher `--fps` increase recall but also GPU/CPU time.
   - Higher `--stride` or smaller `--fps` are useful for exploratory passes on long episodes.
   - Device selection order: `auto` → CUDA GPU → Apple `mps` → CPU. Override with `--device cpu` if you need to stay on CPU.
   - `--scene-detect` (default `on`) runs a histogram prepass to detect hard cuts, resets ByteTrack at each cut, and forces `--scene-warmup-dets` consecutive detections to reacquire IDs. Tune `--scene-threshold` (0–2) and `--scene-min-len` (frames between cuts) when you need a stricter or looser cut detector.
   - Faces harvesting accepts `--face-detector`, `--min-face-size`, `--min-face-conf`, `--face-validate`, and `--thumb-size` so you can choose SCRFD (default), fall back to MTCNN, or run the YOLOv8 face head before embeddings are written.

3. (Optional) Verify the real pipeline via the ML test:

   ```bash
   RUN_ML_TESTS=1 pytest tests/ml/test_detect_track_real.py -q
   ```

The CLI and UI both emit manifests under `data/manifests/{ep_id}/` with YOLO metadata, so you can diff runs before pushing upstream.

#### Re-run on an existing episode

Already uploaded footage to S3 and just need to run detect/track again?

1. Start the API + Streamlit UI, then choose **Existing Episode** in the sidebar.
2. Browse the **S3 videos** list to pick any `raw/videos/{show}/s{season}/e{episode}/episode.mp4` object. If it isn’t in the EpisodeStore yet, hit **Create episode in store** – the UI parses the v2 key, POSTs `/episodes/upsert_by_id`, and immediately shows whether it created or reused the record (along with the derived `ep_id`).
3. Click **Mirror from S3** to download the v2 object into `data/videos/{ep_id}/episode.mp4`. When only a legacy `raw/videos/{ep_id}/episode.mp4` exists, the UI falls back to it and shows a note until you migrate.
4. Adjust `Stride`, optional `FPS` (set `0` to reuse the detected FPS shown in the UI), **Device**, frame/crop export toggles (pre-checked to save frames + crops by default), and the advanced scene-cut settings, then hit **Run detect/track**. The progress widget always renders `mm:ss / mm:ss`, device, FPS, and the target S3 prefixes (frames/crops/manifests) so you know exactly where artifacts land once the run completes, and the completion toast now includes a `Scene cuts: N` pill so you can quickly gauge how many tracker resets fired.

 Manifest links for that episode stay visible so you can open `detections.jsonl` / `tracks.jsonl` directly after each pass.

#### Delete episodes & data

- **Delete episode** (Episodes page) opens a confirmation body for the currently selected `ep_id`. Local caches (`data/videos`, `data/manifests`, `data/frames`, `data/analytics`) are always removed, and an optional checkbox toggles whether the S3 artifact prefixes for that episode should be deleted. The UI POSTs `/episodes/{ep_id}/delete` and reports how many directories/files were removed locally plus the number of S3 objects deleted.
- **Delete ALL episodes & data** requires typing `DELETE ALL` and then POSTs `/episodes/delete_all`. Local EpisodeStore entries and caches are removed for every tracked episode; checking “Also delete S3 artifacts” wipes every episode’s `artifacts/{frames|crops|manifests|thumbs}` prefixes. The API response includes the episode count plus aggregate deletion stats so the UI can show a concise toast.
- Safety guardrails refuse to delete empty prefixes, so even the “delete raw video” option is scoped to the episode-specific folders rather than the entire bucket.

**Common errors**

- `Connection refused` — Start the API via `python -m uvicorn apps.api.main:app --reload` (or rerun `scripts/dev.sh`) and confirm it responds: `curl http://localhost:8000/healthz`.
- `API_BASE_URL mismatch` — Export the correct `SCREENALYTICS_API_URL` (or update `.env`) so the UI hits the right host/port.
- `File not found (Streamlit path)` — Ensure you launch commands from the repo root so `apps/workspace-ui/streamlit_app.py` resolves.
- `NoSuchBucket` — Run `bash scripts/s3_bootstrap.sh` or set `S3_AUTO_CREATE=1` before restarting the API.

### Navigation and pages

The Streamlit workspace ships as a multipage app with sidebar navigation:

1. **Upload & Run** — create/upload episodes, kick detect/track immediately.
2. **Episodes** — searchable browser across all `ep_id` entries with quick detect/track controls.
3. **Episode Detail** — hydrate from S3, re-run detect/track, launch faces/cluster/screentime jobs, and jump to local artifacts.
4. **Faces Review** — trigger faces/cluster jobs and inspect `faces.jsonl`/`identities.json`.
5. **Screentime** — run the screentime job and review/download analytics (`screentime.json` / `.csv`).
6. **Health** — quick API/S3/disk checks plus context on the current backend + data root.

Use the **Open in Episode Detail** buttons sprinkled throughout to sync the `ep_id` query param and jump between pages without re-entering the selection.

> The Device selector defaults to **Auto**, which resolves to CUDA → MPS → CPU. Force **CPU**, **MPS**, or **CUDA** explicitly if you need to pin a device.

### S3 layout (v2)

Episode videos now land under `raw/videos/{show_slug}/s{season}/e{episode}/episode.mp4` (“v2”). The Episode Detail page shows whether v2 or the legacy v1 (`raw/videos/{ep_id}/episode.mp4`) exists, and mirroring/detect-track automatically falls back to v1 if needed. Use `scripts/s3_migrate_v1_to_v2.py --apply` to copy existing v1 objects over once you’re ready (dry-run by default).

#### Deep links and state

The UI keeps the selected episode ID in both `st.session_state` and the URL query params, so you can deep link directly: `http://localhost:8501/?ep_id=rhobh-s05e17`. Uploading an episode automatically sets this `ep_id` so when you switch to Episodes, Episode Detail, or Screentime, the context follows you.

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

4. Run the dev flow (the script prints `API STORAGE_BACKEND=… BUCKET=…` so you can verify it latched onto the right backend before the UI launches):

   ```bash
   STORAGE_BACKEND=s3 bash scripts/dev.sh
   ```

Artifacts live under the standardized v2 prefixes:

- `s3://screenalytics/raw/videos/{show_slug}/s{season}/e{episode}/episode.mp4`
- `s3://screenalytics/artifacts/frames/{show_slug}/s{season}/e{episode}/frames/`
- `s3://screenalytics/artifacts/crops/{show_slug}/s{season}/e{episode}/tracks/`
- `s3://screenalytics/artifacts/manifests/{show_slug}/s{season}/e{episode}/`

Verify uploads:

```bash
aws s3 ls s3://screenalytics/raw/videos/ --recursive | tail -n 5
```

#### Troubleshooting (macOS / FFmpeg / PyAV)

`faster-whisper` depends on PyAV, which may try to build against Homebrew’s FFmpeg 8.x headers on Apple Silicon. Stick to `requirements-core.txt` unless you plan to run the full ML stack; install `requirements-ml.txt` when you need RetinaFace/ArcFace locally and have a working FFmpeg toolchain.

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

#### InsightFace models

RetinaFace + ArcFace ship as InsightFace ONNX weights. Download them once per machine to avoid the “RetinaFace init failed / NoneType has no attribute 'prepare'” crash:

```bash
python scripts/fetch_models.py
```

By default we expect the bundled `retinaface_r50_v1` and `arcface_r100_v1` names. Override via `RETINAFACE_MODEL=/path/to/model.onnx` and/or `ARCFACE_MODEL=/path/to/model.onnx` if you maintain a local cache.

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

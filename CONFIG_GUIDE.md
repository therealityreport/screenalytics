# CONFIG_GUIDE.md — Screenalytics

Version: 1.0

## Files
- `config/pipeline/detection.yaml` — RetinaFace ids, thresholds
- `config/pipeline/tracking.yaml` — ByteTrack params
- `config/pipeline/recognition.yaml` — ArcFace model id, similarity_th, hysteresis
- `config/pipeline/audio.yaml` — diarization/asr settings
- `config/pipeline/screen_time_v2.yaml` — DAG toggles and stage order
- `config/storage.yaml` — buckets, prefixes, lifecycle rules
- `config/services.yaml` — external endpoints/timeouts
- `config/codex.config.toml` — models, MCP servers, write policies
- `config/agents/policies.yaml` + `config/claude.policies.yaml` — doc/write rules

## Screen time presets
- `config/pipeline/screen_time_v2.yaml` now exposes `screen_time_presets` with a default `bravo_default` profile for track-based aggregation plus a stricter fallback.
- Update the `preset` key (or pass `--preset` / API overrides) to switch between presets without editing code.
- Each preset can set `quality_min`, `gap_tolerance_s`, `screen_time_mode`, `edge_padding_s`, `track_coverage_min`, and `use_video_decode`; keep unrelated flows (facebank, embeddings) on their own presets if the thresholds diverge.

## Conventions
- All thresholds and ids configured here; no magic numbers in code.
- Code reads configs via helper `packages/py-screenalytics/config.py` with override support.
- Promotion PR must include config diffs and doc updates.

## RetinaFace health + downloads
- The API now calls `tools.episode_run.ensure_retinaface_ready()` on startup and before facebank seed uploads. If InsightFace models are missing it serves a WARN banner (`detector="simulated"`) and the Streamlit Cast page shows a yellow notice.
- Install the InsightFace bundle and warm up the Buffalo-L profile once per machine to fetch RetinaFace + ArcFace weights:

```bash
pip install -U insightface onnxruntime opencv-python
python - <<'PY'
from insightface.app import FaceAnalysis
FaceAnalysis(name="buffalo_l").prepare(ctx_id=0, det_size=(640, 640))
print("RetinaFace ready")
PY
```

- When running completely offline, copy the downloaded `~/.insightface/models/buffalo_l` directory into your build image or shared cache before launching the API.

## Facebank seed derivatives
- Use `SEED_DISPLAY_SIZE`/`SEED_EMBED_SIZE` to control the square crops written during `/cast/{cast_id}/seeds/upload`. Defaults are `512` (display) and `112` (ArcFace input) and must stay >= 64.
- `SEED_DISPLAY_FORMAT` and `SEED_EMBED_FORMAT` accept `png|jpg`. The default is `png` for lossless UI thumbnails; set to `jpg` only when bandwidth or storage is constrained.
- `SEED_JPEG_QUALITY` is applied whenever a derivative is stored as JPEG (default `92`).
- `SEED_DISPLAY_MIN`/`SEED_DISPLAY_MAX` bound the aspect-preserving resize for the display derivative. Uploads larger than `SEED_DISPLAY_MAX` are clamped down; smaller uploads are never upscaled beyond their original size.
- `FACEBANK_KEEP_ORIG=1` persists the uploaded original (post EXIF transpose) so future re-crops/backfills can regenerate higher quality displays without asking the user to re-upload.
- Changing any of the above will alter the derivative filenames (`*_d.<ext>` / `*_e.<ext>`). Run the facebank refresh job after changing formats so cached presigns and the UI rebuild using the new extensions.
- `/jobs/facebank/backfill_display` will regenerate missing or <=128 px displays (preferring the saved original, falling back to the embed crop and marking it `low_res=true` when no better source exists).

## Stub jobs

## Codex Actions (CI)
- Required secret: add `OPENAI_API_KEY` in GitHub > Settings > Secrets and variables > Actions.
- Workflows:
  - `.github/workflows/on-push-doc-sync.yml` — runs Codex playbook to sync docs on push.
  - `.github/workflows/codex-manual.yml` — run Codex on demand via the Actions tab.
- Manual run: choose `codex-manual` workflow, set `mode` to `prompt` (enter freeform text) or `playbook` (e.g. `agents/playbooks/update-docs-on-change.yaml`).

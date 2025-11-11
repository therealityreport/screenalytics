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

## Conventions
- All thresholds and ids configured here; no magic numbers in code.
- Code reads configs via helper `packages/py-screenalytics/config.py` with override support.
- Promotion PR must include config diffs and doc updates.

## Stub jobs

## Codex Actions (CI)
- Required secret: add `OPENAI_API_KEY` in GitHub > Settings > Secrets and variables > Actions.
- Workflows:
  - `.github/workflows/on-push-doc-sync.yml` — runs Codex playbook to sync docs on push.
  - `.github/workflows/codex-manual.yml` — run Codex on demand via the Actions tab.
- Manual run: choose `codex-manual` workflow, set `mode` to `prompt` (enter freeform text) or `playbook` (e.g. `agents/playbooks/update-docs-on-change.yaml`).

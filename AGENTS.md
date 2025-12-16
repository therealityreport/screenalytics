<!--
This file defines how automated agents (Codex, etc.) must behave when working in this repository.
It merges legacy Codex review guidance with current operational guardrails.
-->

# AGENTS - SCREENALYTICS

## 0. Repository and Branch

- Repo: `github.com/therealityreport/screenalytics`
- Primary branch: `main` - stable, always deployable
- Default working base: `origin/main`

### Single Directory Workflow (REQUIRED)

**All work happens in ONE location:** `/Volumes/HardDrive/SCREENALYTICS`

**Agents must NOT:**
- Create git worktrees
- Work in multiple directories simultaneously
- Use `/Volumes/HardDrive/wt/` or any other worktree folders

**Agents must:**
- Work only in the repo root directory
- Create/continue work on a single branch per task
- Use branch pattern: `screentime-improvements/<task-slug>`
- Commit frequently (small commits)
- Push regularly to keep remote updated
- Not switch branches without committing first
- Not open a PR until user asks OR checklist is satisfied

### Branch Naming

**Required pattern:** `screentime-improvements/<task-slug>`

Examples:
- `screentime-improvements/run-isolation-screentime`
- `screentime-improvements/web-modal-dialogs`
- `screentime-improvements/workflow-docs-update`

Never push directly to `main`; open a PR targeting `main`.

### Hard Boundaries

- Do not touch or reference other repos (THB-BBL, basketball projects, etc.)
- Do not create cross-repo folders or assumptions
- Keep all changes within this codebase and the branch named in the task

## 1. Codex Review Priorities (legacy rules, still apply)

- P0: Incorrect behavior, crashes, security exposure, data loss, secrets in code/logs.
- P1: Logic bugs, flaky paths, major perf issues, unsafe defaults, schema drift, migrations without rollback.
- P2: Minor perf or style inconsistencies, typos in docs, non-blocking refactors.
- Scope: Review only the PR diff; prefer minimal, targeted fixes over broad refactors.
- Repo-specific checks:
  - ML pipeline: RetinaFace detector; ByteTrack over face boxes; StrongSORT/ReID optional. Embeddings: ArcFace 512-d; consistent normalization/versioning. Validate all CLI/Streamlit params; fail fast on missing models.
  - Artifacts/storage: S3 v2 layout respected; no silent overwrites; set content-type/cache headers where needed. Facebank updates are atomic; emit refresh jobs after seed-image changes.
  - Secrets/keys: No tokens or keys committed; no printing secrets; env vars via config, not inline.
  - Web/Streamlit: Avoid blocking UI thread; guard file paths; sanitize filenames; unique widget keys; correct thumbnail cropping and EXIF orientation.
  - Testing/safety: Add regression tests for fixed bugs; actionable errors; no raising on benign states; explicit None checks over attribute access on None.
  - Style: Python type hints, pydantic/dataclasses for configs, no bare except, log at INFO/WARN/ERROR with context keys. Git: conventional commits in PR title/description; clear migration notes when schema/embeddings change.
  - Output format: Leave inline comments where code is affected; summarize P0/P1 required changes; provide minimal code suggestions when safe.

## 2. Branching and Git Usage

See [docs/reference/branching/BRANCHING_STRATEGY.md](docs/reference/branching/BRANCHING_STRATEGY.md) for full workflow.

### Standard Task Lifecycle

```bash
# 1. Start task
git switch main
git pull --ff-only origin main
git switch -c screentime-improvements/<task-slug>

# 2. Work loop (repeat)
git add <files>
git commit -m "<type>: <description>"
git push -u origin screentime-improvements/<task-slug>

# 3. When ready for PR
gh pr create --base main

# 4. After merge
git switch main
git pull --ff-only origin main
git branch -d screentime-improvements/<task-slug>
```

### Required Checkpoints

- Before switching branches: MUST commit or explicitly stash (prefer commit)
- Before opening PR: MUST push branch and confirm it exists on GitHub
- Always keep main clean (no direct commits)

### Key Rules

- Default base: `origin/main`
- Branch pattern: `screentime-improvements/<task-slug>`
- Keep changes scoped; prefer small, focused edits
- Commit frequently with descriptive messages
- Push regularly to keep remote updated

## 3. Task Types and Severity (operational)

- P0 Critical: runtime errors in Streamlit/UI/API/pipeline; data corruption/loss; broken Upload/Episode Detail/pipeline flows.
- P1 High: incorrect behavior without crash; misleading or blocking UI states.
- P2 Medium: noticeable perf issues; expensive UI operations on typical workloads.
- P3 Low: styling/formatting/minor cleanup.
- Address P0/P1 first; only touch P2/P3 when trivial/low risk or explicitly requested.

## 4. Coding Conventions

### General Python
- Python 3.11 compatible.
- Prefer type hints on public functions, explicit exception handling, and logging over silent failures.
- Avoid new heavy dependencies without explicit instruction.

### Streamlit (workspace UI)
- Key files: `apps/workspace-ui/Upload_Video.py`, `apps/workspace-ui/pages/0_Upload_Video.py`, `apps/workspace-ui/pages/2_Episode_Detail.py`, `apps/workspace-ui/pages/3_Faces_Review.py`, `apps/workspace-ui/ui_helpers.py`.
- `st.set_page_config()` via `helpers.init_page(...)` once per page and as the first Streamlit call.
- Render layout before heavy/network work. Wrap heavy operations in `with st.spinner(...)` plus `try/except` that surfaces errors (`st.error` + `st.exception`) instead of blank pages.
- Avoid `st.experimental_rerun` in core logic; avoid blanket `st.stop()`—prefer returning an error outcome.
- Namespace `st.session_state` keys per episode (e.g., `f"{ep_id}::detect_device_choice"`).

### FastAPI / API
- Key file: `apps/api/routers/episodes.py`.
- Use Pydantic models; preserve response shapes; add new fields as optional and documented.

### Pipeline / Tools
- Key files: `tools/episode_run.py`, `py_screenalytics/...`, `config/pipeline/*.yaml`.
- Preserve phase structure (faces-harvest, faces-embed, cluster, screentime).
- Logging prefixes like `[PHASE]`, `[JOB]`, `[GUARDRAIL]`.
- Guardrails: report pre/post merge metrics; do not block unnecessarily when manifests show success.

## 5. Streamlit UI: Focused Rules

### Episode Detail (`2_Episode_Detail.py`)
- Do not assume optional fields; use `.get(..., default)` or `or {}`.
- Singleton/cluster stats: distinguish before vs after and merge-enabled vs merge-disabled; message precisely.
- Faces readiness: `faces.jsonl` present (even zero rows) counts as ready when status is missing/stale; show “manifest fallback”.
- Detect/Track: require detections + tracks for ready; tracks-only sets `tracks_only_fallback` and disables Faces Harvest.
- Screentime: track jobs via `f"{ep_id}::screentime_job"`.

### Upload Video (`Upload_Video.py` + `0_Upload_Video.py`)
- Two modes:
  - Replace existing episode: URL `?ep_id=<ep_id>`, lock upload to that episode.
  - New episode: no `ep_id`; user picks show/season/episode; API creates episode; upload attaches to that `ep_id`.
- Presign/AWS:
  - Presigned PUT: do not call `_validate_s3_credentials()`.
  - Local-only (`method == "FILE"`): successful local write is success (not cloud failure).
  - Boto3/direct: call `_validate_s3_credentials()` with bounded timeouts; surface clear errors if creds are missing.
- Always render title/uploader before network/S3 work; wrap uploads in spinner + try/except.

## 6. Readiness and Manifest Fallback

- Detect/Track: ready when detections and tracks exist; tracks-only sets `tracks_only_fallback` + warning + disables harvest.
- Faces: if status missing/stale but `faces.jsonl` readable, treat as ready; set `faces_manifest_fallback`; show “status stale, using manifest”.
- Cluster: use post-merge singleton metrics when merge enabled; keep pre/post metrics in manifests/logs; ensure Screentime uses merged IDs only.
- Status caching: invalidate when manifest/run marker mtimes change, even without job flags.

## 7. Testing and Validation

- Static: `python -m py_compile` for touched Python files (e.g., `apps/api/routers/episodes.py apps/workspace-ui/pages/2_Episode_Detail.py apps/workspace-ui/Upload_Video.py`).
- Targeted tests:
  - Cluster/singleton/stats: `tests/ml/test_cluster_singleton_merge.py`
  - Episodes/status/cluster summary: `tests/api/test_cluster_tracks_summary.py`, `tests/api/test_episode_status.py`
  - Add new tests for new behaviors when feasible.
- If pytest cannot run due to local infra (e.g., missing psycopg2), note it and ensure `py_compile` passes.

## 8. Reporting Back (agent responses)

### Standard Report Format

When completing a task, report:

```
Branch: screentime-improvements/<task-slug>
Commits: <list of commit hashes with short descriptions>
Tests: <tests run and results>
Pushed: yes/no
PR: <link if created, otherwise "not yet">
```

### Report Contents

- Summarize root causes for fixed bugs with file references
- List files changed with short bullets
- Describe tests run (e.g., `python -m py_compile ...`, `pytest ...`)
- Include branch name and commit info
- Note whether pushed to origin
- Include PR link only when created

## 9. Things You Must Not Do

- Do not modify other repos when working here.
- Do not silently swallow exceptions; surface via logs/UI errors.
- Do not rewrite history or change core branch names unless explicitly instructed.
- Avoid large refactors unless requested.

## 10. Legacy Codex Output Requirements

- Inline comments where code is affected.
- Summarize P0/P1 with required changes.
- Provide minimal code suggestions when safe.

---

## 11. Related Documentation

- **[Agent Catalog](agents/AGENTS.md)** — Feature agent ownership map and automation bots
- **[Branching Strategy](docs/reference/branching/BRANCHING_STRATEGY.md)** — Full git workflow details
- **[Acceptance Matrix](ACCEPTANCE_MATRIX.md)** — Quality gates and acceptance criteria

---

Agents must respect these rules for all tasks in this repository.

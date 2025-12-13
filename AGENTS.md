<!--
This file defines how automated agents (Codex, etc.) must behave when working in this repository.
It merges legacy Codex review guidance with current operational guardrails.
-->

# AGENTS - SCREENALYTICS

## 0. Repository and Branch

- Repo: `github.com/therealityreport/screenalytics`
- Primary branches:
  - `main` - stable
  - `nov-24` - active working branch (default unless a task says otherwise)
- Hard boundaries:
  - Do not touch or reference other repos (THB-BBL, basketball projects, etc.).
  - Do not create cross-repo folders or assumptions.
  - Keep all changes within this codebase and the branch named in the task.

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

- Default working branch: `nov-24` (unless a task specifies another).
- Keep changes scoped and coherent; prefer small, focused edits over broad refactors.
- Agents do not auto-run shell commands. When the user should run git, provide a command block at the end of the report, for example:
  ```bash
  git checkout nov-24
  git status
  git diff
  git add -A
  git commit -m "Short, precise message"
  git push origin nov-24
  ```

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

- Summarize root causes for fixed bugs with file references.
- List files changed with short bullets.
- Describe tests run (e.g., `python -m py_compile ...`, `pytest ...`).
- Provide git commands as a final block only; do not auto-run them.

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

## 11. Feature-Specific Agents

### BodyTrackingAgent

**Scope:** Person detection, tracking, Re-ID, and face↔body fusion.

**Owns:**
- `FEATURES/body_tracking/src/` - Body tracking implementation
- `config/pipeline/body_detection.yaml` - Detection and Re-ID config
- `config/pipeline/track_fusion.yaml` - Fusion rules

**Key Tasks:**
- YOLO person detection (class 0 = person)
- ByteTrack temporal tracking with ID offset (100000+)
- OSNet Re-ID embedding computation (256-d)
- Face↔body IoU association
- Re-ID handoff when face disappears
- Screen-time comparison (face-only vs face+body)

**Artifacts:**
- `data/manifests/{ep_id}/body_tracking/body_detections.jsonl`
- `data/manifests/{ep_id}/body_tracking/body_tracks.jsonl`
- `data/manifests/{ep_id}/body_tracking/body_embeddings.npy`
- `data/manifests/{ep_id}/body_tracking/track_fusion.json`
- `data/manifests/{ep_id}/body_tracking/screentime_comparison.json`

**Config Flags (rollback levers):**
- `body_tracking.enabled: false` - Disable entire feature
- `person_reid.enabled: false` - Disable Re-ID embeddings
- `track_fusion.enabled: false` - Disable face↔body fusion

**Acceptance Matrix:** Sections 3.10-3.12

---

### FaceAlignmentAgent

**Scope:** Face alignment, landmark extraction, and quality gating.

**Owns:**
- `FEATURES/face_alignment/src/` - Alignment implementation
- `config/pipeline/face_alignment.yaml` - Alignment config

**Key Tasks:**
- FAN 68-point 2D landmark extraction
- Aligned face crop generation (ArcFace normalization)
- Quality scoring (future: LUVLi uncertainty)
- 3D head pose estimation (future: 3DDFA_V2)

**Artifacts:**
- `data/manifests/{ep_id}/face_alignment/aligned_faces.jsonl`
- `data/manifests/{ep_id}/face_alignment/aligned_crops/` (optional)

**Config Flags (rollback levers):**
- `face_alignment.enabled: false` - Disable alignment
- Disable embedding gating: `config/pipeline/embedding.yaml` → `face_alignment.enabled: false` (or set `min_alignment_quality: 0.0`)

**Acceptance Matrix:** Sections 3.7-3.9

---

Agents must respect these rules for all tasks in this repository.

# Review Guidelines for Codex

## Priorities
- **P0**: Incorrect behavior, crashes, security exposure, data loss, secrets in code or logs.
- **P1**: Logic bugs, flaky paths, major perf issues, unsafe defaults, schema drift, migrations without rollback.
- **P2**: Minor perf or style inconsistencies, typos in docs, non-blocking refactors.

## Scope
Review only the PR diff. Favor minimal, targeted fixes over broad refactors.

## Repo-specific checks
- **ML pipeline**
  - Detector: RetinaFace; no generic COCO/person models in face path.
  - Tracker: ByteTrack over face boxes; StrongSORT/ReID optional.
  - Embeddings: ArcFace 512-d; ensure consistent normalization and versioning.
  - Input validation on all CLI/Streamlit params. Fail fast on missing models.
- **Artifacts and storage**
  - S3 v2 layout respected. No silent overwrites. Add content-type and cache headers where needed.
  - Facebank updates are atomic. Emit refresh jobs after seed-image changes.
- **Secrets and keys**
  - No tokens or keys committed. No printing secrets. Env vars pulled via config, not inline.
- **Web/Streamlit**
  - Avoid blocking the UI thread for long runs. Guard file paths. Sanitize filenames.
  - Unique keys for UI widgets. Correct thumbnail cropping and EXIF orientation.
- **Testing and safety**
  - Add regression tests for fixed bugs. Validate error messages are actionable.
  - Don’t raise on benign states. Prefer explicit None checks over attribute access on None.

## Style
- Python: type hints, pydantic/dataclasses for configs, no bare except, log at INFO/WARN/ERROR with context keys.
- Git: conventional commits in PR title or description. Clear migration notes when schema or embeddings change.

## Output format
- Leave inline comments where code is affected.
- Summarize P0/P1 with a short list of required changes.
- Provide minimal code suggestions when safe.


# FEATURES_GUIDE.md — Screenalytics

Version: 2.0
Last Updated: 2025-11-18

---

## What is a Feature Sandbox?

`FEATURES/<name>/` is a temporary area for in-progress modules with a **30-day TTL** from creation.

---

## Structure

```
FEATURES/<feature-name>/
├── src/              # Throwaway implementation
├── tests/            # Focused tests for this feature
├── docs/             # Working notes (agents write here)
└── TODO.md           # Status, owner, plan, promotion checklist
```

---

## Rules

1. **TTL: 30 days** from creation
2. **No production imports:** Code in `apps/`, `web/`, `packages/` **cannot** import from `FEATURES/**` (CI enforced)
3. **Promotion requirements:**
   - ✅ Tests present and passing
   - ✅ Docs written (what it does, config keys, metrics)
   - ✅ Config-driven (no hardcoded thresholds)
   - ✅ CI green (lint, tests, acceptance checks)
   - ✅ Row in `ACCEPTANCE_MATRIX.md` marked ✅ Accepted

---

## Creating a New Feature

```bash
python tools/new-feature.py <feature-name>
```

This creates:
```
FEATURES/<feature-name>/
├── src/
├── tests/
├── docs/
└── TODO.md (status: IN_PROGRESS, owner: $USER, created: today)
```

---

## Developing in a Feature Sandbox

### 1. Implement in `src/`
```python
# FEATURES/my-feature/src/my_module.py
def my_function():
    # Implementation
    pass
```

### 2. Write Tests in `tests/`
```python
# FEATURES/my-feature/tests/test_my_module.py
from FEATURES.my_feature.src.my_module import my_function

def test_my_function():
    assert my_function() == expected_result
```

### 3. Document in `docs/`
```markdown
# FEATURES/my-feature/docs/README.md

## Overview
This feature does X, Y, Z.

## Config
- `config/pipeline/my_feature.yaml`
- Key: `my_param` (default: 42)

## Metrics
- `my_metric`: Measures X (target: < 10)
```

### 4. Update `TODO.md`
```markdown
# TODO: my-feature

**Status:** IN_PROGRESS
**Owner:** jane-doe
**Created:** 2025-11-18
**TTL:** 2025-12-18

## Tasks
- [x] Implement core logic
- [x] Write unit tests
- [x] Document config keys
- [ ] Integration test with pipeline
- [ ] Add to ACCEPTANCE_MATRIX.md
```

---

## Running Pipeline Tests

To validate your feature against the full pipeline on a test episode:

### 1. Detect/Track Test
```bash
RUN_ML_TESTS=1 pytest tests/ml/test_detect_track_real.py -v
```

**Validates:**
- Detection and tracking run successfully
- `tracks_per_minute < 50`
- `short_track_fraction < 0.3`
- `id_switch_rate < 0.1`

### 2. Faces Embed Test
```bash
RUN_ML_TESTS=1 pytest tests/ml/test_faces_embed.py -v
```

**Validates:**
- Crop generation and embedding extraction
- `quality_mean ≥ 0.75`
- `embedding_dimension == 512`
- `embedding_norm ≈ 1.0`

### 3. Cluster Test
```bash
RUN_ML_TESTS=1 pytest tests/ml/test_cluster.py -v
```

**Validates:**
- Track-level pooling and clustering
- `singleton_fraction < 0.5`
- `largest_cluster_fraction < 0.6`

### 4. End-to-End Integration Test
```bash
RUN_ML_TESTS=1 pytest tests/integration/test_full_pipeline.py -v
```

**Validates:**
- Complete detect → track → embed → cluster → cleanup flow
- All artifacts present and valid
- Metrics within acceptance ranges

---

## Promotion

When your feature is ready for production:

### 1. Pre-Promotion Checklist

- [ ] All tests passing (`pytest FEATURES/<name>/tests/ -v`)
- [ ] Lint clean (`black`, `ruff`, `mypy`)
- [ ] Docs complete (`FEATURES/<name>/docs/`)
- [ ] Config-driven (no hardcoded magic numbers)
- [ ] `TODO.md` status updated
- [ ] Integration tests passing (if applicable)
- [ ] Row added to `ACCEPTANCE_MATRIX.md`

### 2. Run Promotion Script

```bash
python tools/promote-feature.py <feature-name> --dest <target-path>
```

**Example:**
```bash
python tools/promote-feature.py my-feature --dest apps/api/services/
```

**What happens:**
1. Code moves from `FEATURES/my-feature/src/` → `apps/api/services/my_module.py`
2. Tests move to `tests/api/services/test_my_module.py`
3. Docs merge into `docs/` (or link from existing docs)
4. `TODO.md` status → `PROMOTED`
5. CI runs post-promotion checks
6. Agents auto-update root docs (README, PRD, Solution Architecture, Directory Structure)

### 3. Post-Promotion CI Checks

CI verifies:
- ✅ No production imports from `FEATURES/**` (re-check)
- ✅ Full integration test passes on sample clip
- ✅ `ACCEPTANCE_MATRIX.md` row exists and marked ✅ Accepted
- ✅ Config docs updated (`docs/reference/config/pipeline_configs.md`)

---

## Acceptance Criteria

Before marking a feature **Accepted** in `ACCEPTANCE_MATRIX.md`:

### Functional
- ✅ Feature works as intended (manual QA on validation clips)
- ✅ All edge cases handled (null inputs, missing files, etc.)
- ✅ Error messages are clear and actionable

### Performance
- ✅ Runtime within budget (see `ACCEPTANCE_MATRIX.md` for targets)
- ✅ No memory leaks or unbounded growth
- ✅ Thermal/threading limits respected (CPU-safe defaults)

### Quality
- ✅ Metrics within target ranges (see `ACCEPTANCE_MATRIX.md`)
- ✅ No regressions on existing tests
- ✅ Code coverage ≥ 80% (if applicable)

### Documentation
- ✅ Config keys documented in `docs/reference/config/pipeline_configs.md`
- ✅ Pipeline stage documented in `docs/pipeline/<stage>.md` (if applicable)
- ✅ Troubleshooting section added to `docs/ops/troubleshooting_faces_pipeline.md`
- ✅ README links updated (if applicable)

---

## Feature Expiry

Features older than **30 days** without promotion are flagged by CI:

```
⚠️ WARNING: FEATURES/stale-feature/ is 35 days old (TTL: 30 days)
Action required: Promote or archive
```

**Options:**
1. **Promote:** Run `tools/promote-feature.py stale-feature`
2. **Archive:** Move to `archive/FEATURES/stale-feature/` (for reference only)
3. **Delete:** Remove entirely if no longer needed

---

## Common Workflows

### Quick Feature → Promotion
```bash
# 1. Create feature
python tools/new-feature.py my-feature

# 2. Implement
vim FEATURES/my-feature/src/my_module.py

# 3. Test
pytest FEATURES/my-feature/tests/ -v

# 4. Document
vim FEATURES/my-feature/docs/README.md

# 5. Promote
python tools/promote-feature.py my-feature --dest apps/api/services/
```

### Integration Testing During Development
```bash
# Run your feature against detect/track pipeline
RUN_ML_TESTS=1 pytest tests/ml/test_detect_track_real.py -v

# Check metrics
cat data/manifests/<test-ep-id>/track_metrics.json | jq .
```

### Debugging Failed Promotion
If `tools/promote-feature.py` fails:

1. Check CI output for specific failures
2. Verify tests pass: `pytest FEATURES/<name>/tests/ -v`
3. Verify lint: `black --check FEATURES/<name>/ && ruff check FEATURES/<name>/`
4. Verify `ACCEPTANCE_MATRIX.md` row exists
5. Re-run promotion with `--verbose` flag

---

## References

- **[Directory Structure](docs/architecture/directory_structure.md)** — Full promotion workflow
- **[ACCEPTANCE_MATRIX.md](ACCEPTANCE_MATRIX.md)** — Quality gates and acceptance criteria
- **[Pipeline Overview](docs/pipeline/overview.md)** — How features integrate into pipeline

---

**Maintained by:** Screenalytics Engineering

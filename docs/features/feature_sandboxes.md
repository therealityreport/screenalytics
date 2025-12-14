# Feature Sandboxes — Screenalytics

Version: 2.0
Last Updated: 2025-11-18

---

## What is a Feature Sandbox?

`FEATURES/<name>/` is a temporary area for in-progress modules with a **30-day TTL** from creation.

---

## Structure

```
FEATURES/<feature_name>/
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
mkdir -p FEATURES/<feature_name>/{src,tests,docs}
```

Then add a `TODO.md` and start implementing:
```
FEATURES/<feature_name>/
├── src/
├── tests/
├── docs/
└── TODO.md (status: IN_PROGRESS, owner: $USER, created: today)
```

---

## Developing in a Feature Sandbox

### 1. Implement in `src/`
```python
# FEATURES/my_feature/src/my_module.py
def my_function():
    # Implementation
    pass
```

### 2. Write Tests in `tests/`
```python
# FEATURES/my_feature/tests/test_my_module.py
from FEATURES.my_feature.src.my_module import my_function

def test_my_function():
    assert my_function() == expected_result
```

### 3. Document in `docs/`
```markdown
# FEATURES/my_feature/docs/README.md

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
# TODO: my_feature

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

### 2. Promote (manual)

Promote by opening a PR that moves code out of `FEATURES/<feature_name>/` into production paths (`apps/`, `web/`, `packages/`).

**Example:**
```bash
git mv FEATURES/my_feature/src/my_module.py apps/api/services/my_module.py
git mv FEATURES/my_feature/tests/test_my_module.py tests/api/services/test_my_module.py
```

**What happens:**
1. Code moves from `FEATURES/my_feature/src/` → `apps/api/services/my_module.py`
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
⚠️ WARNING: FEATURES/stale_feature/ is 35 days old (TTL: 30 days)
Action required: Promote or archive
```

**Options:**
1. **Promote:** Open a PR that moves code/tests/docs out of `FEATURES/stale_feature/`
2. **Archive:** Move to `archive/FEATURES/stale_feature/` (for reference only)
3. **Delete:** Remove entirely if no longer needed

---

## Common Workflows

### Quick Feature → Promotion
```bash
# 1. Create feature
mkdir -p FEATURES/my_feature/{src,tests,docs}

# 2. Implement
vim FEATURES/my_feature/src/my_module.py

# 3. Test
pytest FEATURES/my_feature/tests/ -v

# 4. Document
vim FEATURES/my_feature/docs/README.md

# 5. Promote (via PR / git mv)
git mv FEATURES/my_feature/src/my_module.py apps/api/services/my_module.py
git mv FEATURES/my_feature/tests/test_my_module.py tests/api/services/test_my_module.py
```

### Integration Testing During Development
```bash
# Run your feature against detect/track pipeline
RUN_ML_TESTS=1 pytest tests/ml/test_detect_track_real.py -v

# Check metrics
cat data/manifests/<test-ep-id>/track_metrics.json | jq .
```

### Debugging Failed Promotion
If your promotion PR fails CI:

1. Check CI output for specific failures
2. Verify tests pass: `pytest FEATURES/<name>/tests/ -v`
3. Verify lint: `black --check FEATURES/<name>/ && ruff check FEATURES/<name>/`
4. Verify `ACCEPTANCE_MATRIX.md` row exists
5. Ensure production code does not import from `FEATURES/**`

---

## References

- **[Directory Structure](../architecture/directory_structure.md)** — Full promotion workflow
- **[ACCEPTANCE_MATRIX.md](../../ACCEPTANCE_MATRIX.md)** — Quality gates and acceptance criteria
- **[Pipeline Overview](../pipeline/overview.md)** — How features integrate into pipeline

---

## Active Feature Sandboxes

### Body Tracking (`FEATURES/body_tracking/`)

**Status:** Implemented
**TTL:** 2026-01-10
**Owner:** Engineering

Adds person body tracking to maintain identity when faces aren't visible.

**Components:**
- YOLO person detection
- ByteTrack temporal tracking
- OSNet Re-ID embeddings (256-d)
- Face↔body track fusion
- Screen-time comparison (face-only vs face+body)

**Usage:**
```bash
# Run full pipeline
python -m FEATURES.body_tracking --episode-id rhoslc-s06e01

# Run specific stage
python -m FEATURES.body_tracking --episode-id rhoslc-s06e01 --stage detect
python -m FEATURES.body_tracking --episode-id rhoslc-s06e01 --stage track
python -m FEATURES.body_tracking --episode-id rhoslc-s06e01 --stage fuse
python -m FEATURES.body_tracking --episode-id rhoslc-s06e01 --stage compare
```

**Artifacts** (in `data/manifests/{ep_id}/body_tracking/`):
| File | Description |
|------|-------------|
| `body_detections.jsonl` | Frame-by-frame person detections |
| `body_tracks.jsonl` | Tracked persons over time |
| `body_embeddings.npy` | Re-ID embedding vectors |
| `body_embeddings_meta.json` | Metadata for embeddings |
| `track_fusion.json` | Face↔body associations |
| `screentime_comparison.json` | Face-only vs combined metrics |
| `body_metrics.json` | Pipeline summary |

**Config:**
- `config/pipeline/body_detection.yaml` - Detection, tracking, Re-ID settings
- `config/pipeline/track_fusion.yaml` - Fusion rules and thresholds

**Tests:**
```bash
pytest FEATURES/body_tracking/tests/ -v
```

**Docs:**
- [FEATURES/body_tracking/docs/README.md](../../FEATURES/body_tracking/docs/README.md)
- [FEATURES/body_tracking/TODO.md](../../FEATURES/body_tracking/TODO.md)
- [docs/todo/feature_body_tracking_reid_fusion.md](../todo/feature_body_tracking_reid_fusion.md)

**Acceptance Matrix:** Sections 3.10-3.12

---

### Face Alignment (`FEATURES/face_alignment/`)

**Status:** In Progress
**TTL:** 2026-01-10
**Owner:** Engineering

FAN-based face alignment with 68-point landmarks for improved embedding quality.

**Components:**
- FAN 2D/3D landmark extraction
- Aligned face crop generation
- Quality scoring (future: LUVLi)
- 3D head pose (future: 3DDFA_V2)

**Usage:**
```bash
# Run alignment on episode
python -m FEATURES.face_alignment --episode-id rhoslc-s06e01
```

**Artifacts** (in `data/manifests/{ep_id}/face_alignment/`):
| File | Description |
|------|-------------|
| `aligned_faces.jsonl` | Landmarks and quality scores |
| `aligned_crops/` | Aligned face images (optional) |

**Config:**
- `config/pipeline/face_alignment.yaml`

**Acceptance Matrix:** Sections 3.7-3.9

---

**Maintained by:** Screenalytics Engineering

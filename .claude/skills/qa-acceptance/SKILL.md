---
name: qa-acceptance
description: Evaluate pipeline runs against ACCEPTANCE_MATRIX.md. Use after feature releases to verify acceptance criteria.
---

# QA Acceptance Skill

Use this skill to validate features against acceptance criteria.

## When to Use

- After implementing a new feature
- Before promoting from FEATURES/ to production
- Running acceptance checks in CI
- Generating QA reports for release
- Debugging why acceptance checks failed

## Key Skills

### `run_acceptance_checks(feature_name)`
Validate a feature against its acceptance criteria.

```python
from tools.acceptance_check import run_acceptance_checks

result = run_acceptance_checks(
    feature_name="face_alignment",
    episode_id="test-episode-001"
)

print(f"Status: {result.status}")  # PASS, WARN, FAIL
for check in result.checks:
    print(f"  {check.metric}: {check.value} ({check.status})")
```

### `summarize_regressions_and_passes()`
Generate a summary report of all acceptance checks.

```python
from tools.acceptance_check import summarize_acceptance

report = summarize_acceptance(
    features=["face_alignment", "body_tracking", "tensorrt_embedding"],
    episode_id="test-episode-001"
)

print(report.to_markdown())
```

## Acceptance Workflow

1. **Load thresholds** from `ACCEPTANCE_MATRIX.md`
2. **Run relevant tests** for the feature
3. **Compare metrics** to targets
4. **Generate report** with pass/fail/warn status
5. **Flag blockers** that must be resolved

## Feature Acceptance Criteria

### Face Alignment (3.7)

| Metric | Target | Warning |
|--------|--------|---------|
| `landmark_jitter_px` | < 2.0 | > 5.0 |
| `alignment_quality_mean` | >= 0.75 | < 0.60 |
| `pose_accuracy_degrees` | <= 5 | > 10 |

### Alignment Quality Gate (3.8)

| Metric | Target | Warning |
|--------|--------|---------|
| `faces_gated_pct` | 10-30% | > 50% |
| `false_rejection_rate` | < 5% | > 10% |

### 3D Head Pose (3.9)

| Metric | Target | Warning |
|--------|--------|---------|
| `pose_yaw_accuracy` | <= 5 MAE | > 10 |
| `pose_pitch_accuracy` | <= 5 MAE | > 10 |

### Body Tracking (3.10)

| Metric | Target | Warning |
|--------|--------|---------|
| `person_recall` | >= 90% | < 80% |
| `body_track_fragmentation` | < 0.15 | > 0.25 |

### Person Re-ID (3.11)

| Metric | Target | Warning |
|--------|--------|---------|
| `reid_mAP` | >= 0.80 | < 0.70 |
| `reid_rank1_accuracy` | >= 0.90 | < 0.80 |

### Track Fusion (3.12)

| Metric | Target | Warning |
|--------|--------|---------|
| `association_accuracy` | >= 95% | < 90% |
| `screen_time_gap_reduction` | >= 30% | < 15% |

### TensorRT Embedding (3.13)

| Metric | Target | Warning |
|--------|--------|---------|
| `speedup_vs_pytorch` | >= 5x | < 3x |
| `embedding_cosine_drift` | >= 0.999 | < 0.995 |

### Face Mesh (3.14)

| Metric | Target | Warning |
|--------|--------|---------|
| `mesh_stability_px` | < 3.0 | > 6.0 |
| `visibility_fraction_accuracy` | >= 90% | < 80% |

### CenterFace (3.15)

| Metric | Target | Warning |
|--------|--------|---------|
| `precision` | >= 0.90 | < 0.85 |
| `recall` | >= 0.85 | < 0.80 |

## Report Format

```markdown
# Acceptance Report: face_alignment

**Date:** 2025-12-11
**Episode:** test-episode-001
**Status:** PASS

## Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| landmark_jitter_px | 1.5 | < 2.0 | PASS |
| alignment_quality_mean | 0.78 | >= 0.75 | PASS |
| pose_accuracy_degrees | 4.2 | <= 5 | PASS |

## Tests

- [x] test_fan_alignment.py - PASSED
- [x] test_alignment_quality.py - PASSED
- [x] test_3ddfa.py - PASSED

## Notes

All acceptance criteria met. Ready for promotion.
```

## CI Integration

Add to GitHub Actions:

```yaml
- name: Run Acceptance Checks
  run: |
    python tools/acceptance_check.py \
      --feature face_alignment \
      --episode test-episode-001 \
      --output acceptance_report.md
```

## Key Files

| File | Purpose |
|------|---------|
| `ACCEPTANCE_MATRIX.md` | Acceptance criteria |
| `tools/acceptance_check.py` | CLI tool |
| `tests/ml/` | Feature tests |
| `tests/integration/` | Integration tests |

## Related Skills

- [pipeline-insights](../pipeline-insights/SKILL.md) - Pipeline debugging
- [pipeline-debug](../pipeline-debug/SKILL.md) - Job debugging

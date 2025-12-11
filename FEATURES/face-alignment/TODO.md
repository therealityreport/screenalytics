# TODO: face-alignment

**Status:** IN_PROGRESS
**Owner:** Engineering
**Created:** 2025-12-11
**TTL:** 2026-01-10

---

## Overview

Face alignment feature using FAN (Face Alignment Network) for 68-point landmarks,
LUVLi for quality gating, and 3DDFA_V2 for selective 3D pose estimation.

**Full Documentation:** [docs/todo/feature_face_alignment_fan_luvli_3ddfa.md](../../docs/todo/feature_face_alignment_fan_luvli_3ddfa.md)

---

## Tasks

### Phase A: FAN 2D Integration (MVP)
- [ ] Create `src/fan_aligner.py` - FAN 2D landmark extraction
- [ ] Implement alignment transform (68-point → aligned crop)
- [ ] Add config: `config/pipeline/alignment.yaml`
- [ ] Write tests: `tests/test_fan_alignment.py`

### Phase B: LUVLi Quality Gate
- [ ] Create `src/luvli_quality.py` - Uncertainty estimation
- [ ] Implement `alignment_quality` metric
- [ ] Add quality gating to embedding pipeline
- [ ] Write tests: `tests/test_alignment_quality.py`

### Phase C: 3DDFA_V2 Selective 3D
- [ ] Create `src/ddfa_v2.py` - 3D alignment
- [ ] Implement adaptive execution strategy
- [ ] Extract and store head pose
- [ ] Write tests: `tests/test_3ddfa.py`

---

## Promotion Checklist

- [ ] Tests present and passing (`pytest FEATURES/face-alignment/tests/ -v`)
- [ ] Lint clean (`black`, `ruff`, `mypy`)
- [ ] Docs complete (`FEATURES/face-alignment/docs/`)
- [ ] Config-driven (no hardcoded thresholds)
- [ ] Integration tests passing
- [ ] Row added to `ACCEPTANCE_MATRIX.md` (sections 3.7, 3.8, 3.9)

---

## Acceptance Criteria

| Metric | Target |
|--------|--------|
| `landmark_jitter_px` | < 2.0 px |
| `alignment_quality_mean` | ≥ 0.75 |
| `pose_accuracy_degrees` | ≤ 5° MAE |
| `track_fragmentation_delta` | ≥ 10% reduction |
| Runtime (1hr episode) | ≤ 5 min |

---

## Key Files

- `src/fan_aligner.py` - FAN 2D/3D landmark extraction
- `src/luvli_quality.py` - Alignment quality scoring
- `src/ddfa_v2.py` - 3DDFA_V2 integration
- `tests/test_fan_alignment.py` - Alignment tests
- `tests/test_alignment_quality.py` - Quality gate tests
- `tests/test_3ddfa.py` - 3D pose tests

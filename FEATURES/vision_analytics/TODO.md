# TODO: vision-analytics

**Status:** IN_PROGRESS
**Owner:** Engineering
**Created:** 2025-12-11
**TTL:** 2026-01-10

---

## Overview

Advanced visibility analytics using MediaPipe face mesh, visibility fraction
computation, gaze direction estimation, and CenterFace detector stub.

**Full Documentation:** [docs/todo/feature_mesh_and_advanced_visibility.md](../../docs/todo/feature_mesh_and_advanced_visibility.md)

---

## Tasks

### Phase A: Face Mesh Integration
- [ ] Create `src/face_mesh.py` - MediaPipe mesh extraction
- [ ] Implement selective execution (close-ups only)
- [ ] Add config: `config/pipeline/analytics.yaml`
- [ ] Write tests: `tests/test_face_mesh.py`

### Phase B: Visibility Fraction
- [ ] Create `src/visibility.py` - Visibility computation
- [ ] Implement region-based visibility
- [ ] Add visibility to track metrics
- [ ] Write tests: `tests/test_visibility.py`

### Phase C: Gaze Direction
- [ ] Create `src/gaze.py` - Gaze estimation
- [ ] Implement iris-based gaze
- [ ] Store gaze in track metrics

### Phase D: CenterFace Stub (Future)
- [ ] Create `src/centerface.py` - Stub interface
- [ ] Add config stub
- [ ] Document roadmap

---

## Promotion Checklist

- [ ] Tests present and passing (`pytest FEATURES/vision_analytics/tests/ -v`)
- [ ] Lint clean (`black`, `ruff`, `mypy`)
- [ ] Docs complete (`FEATURES/vision_analytics/docs/`)
- [ ] Config-driven (no hardcoded thresholds)
- [ ] Integration tests passing
- [ ] Row added to `ACCEPTANCE_MATRIX.md` (sections 3.14, 3.15)

---

## Acceptance Criteria

| Metric | Target |
|--------|--------|
| `mesh_stability_px` | < 3.0 px |
| `visibility_fraction_accuracy` | ≥ 90% |
| `runtime_per_face` | ≤ 15ms |
| Gaze categories correct | ≥ 85% |

---

## Key Files

- `src/face_mesh.py` - MediaPipe mesh extraction
- `src/visibility.py` - Visibility fraction computation
- `src/gaze.py` - Gaze direction estimation
- `src/centerface.py` - [STUB] CPU detector
- `tests/test_face_mesh.py` - Mesh tests
- `tests/test_visibility.py` - Visibility tests

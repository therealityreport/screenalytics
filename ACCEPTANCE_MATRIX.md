# ACCEPTANCE_MATRIX.md — Screenalytics

Version: 2.0
Status: Live QA Standard
Last Updated: 2025-11-18

---

## 1. Purpose

To define measurable acceptance criteria for every pipeline module in Screenalytics.
A module is considered **Accepted** only when all associated checkpoints below are satisfied and verified by CI or QA review.

---

## 2. Acceptance Columns

| Field | Description |
|-------|-------------|
| **Module** | Pipeline stage or system component |
| **Scope** | What it's responsible for |
| **Acceptance Criteria** | Quantifiable pass/fail conditions |
| **Verification Method** | Unit, Integration, Manual, or CI |
| **Status** | ✅ Accepted / ⚠ Pending / ❌ Blocked |
| **Owner** | Team or engineer responsible |

---

## 3. Pipeline Modules

### 3.1 Detection & Tracking (`detect_track`)

| Metric | Target (CPU) | Target (GPU) | Warning Threshold | Verification |
|--------|--------------|--------------|-------------------|--------------|
| **tracks_per_minute** | 10–30 | 10–30 | > 50 | Integration test |
| **short_track_fraction** | < 0.20 | < 0.15 | > 0.30 | Integration test |
| **id_switch_rate** | < 0.05 | < 0.03 | > 0.10 | Integration test |
| **Runtime (1hr episode)** | ≤ 3× realtime (~3hrs) | ≤ 10 min | > 5× realtime (CPU), > 15 min (GPU) | Benchmark |
| **Missing faces (recall)** | ≥ 85% | ≥ 90% | < 80% | Manual QA on validation clips |

**Tests:**
- `tests/ml/test_detect_track_real.py`

**Config Dependencies:**
- `config/pipeline/detection.yaml`
- `config/pipeline/tracking.yaml`
- `config/pipeline/performance_profiles.yaml`

**Docs:**
- [docs/pipeline/detect_track_faces.md](docs/pipeline/detect_track_faces.md)
- [docs/reference/artifacts_faces_tracks_identities.md](docs/reference/artifacts_faces_tracks_identities.md)

**Status:** ⚠ Pending

---

### 3.2 Faces Embedding (`faces_embed`)

| Metric | Target | Warning Threshold | Verification |
|--------|--------|-------------------|--------------|
| **faces_per_track_avg** | 20–50 | > 100 | Integration test |
| **quality_mean** | ≥ 0.75 | < 0.60 | Integration test |
| **rejection_rate** | < 0.30 | > 0.50 | Integration test |
| **embedding_dimension** | 512 | ≠ 512 | Unit test |
| **embedding_norm** | ≈ 1.0 (±0.01) | > 1.05 or < 0.95 | Unit test |
| **Runtime (1hr episode, CPU)** | ≤ 2 min | > 5 min | Benchmark |
| **Runtime (1hr episode, GPU)** | ≤ 30 sec | > 1 min | Benchmark |

**Tests:**
- `tests/ml/test_faces_embed.py`

**Config Dependencies:**
- `config/pipeline/faces_embed_sampling.yaml`

**Docs:**
- [docs/pipeline/faces_harvest.md](docs/pipeline/faces_harvest.md)

**Status:** ⚠ Pending

---

### 3.3 Identity Clustering (`cluster`)

| Metric | Target | Warning Threshold | Verification |
|--------|--------|-------------------|--------------|
| **num_clusters** | 5–15 (typical TV episode) | > 30 | Integration test |
| **singleton_fraction** | < 0.30 | > 0.50 | Integration test |
| **largest_cluster_fraction** | < 0.40 | > 0.60 | Integration test |
| **avg_cluster_size** | 2–5 | > 10 | Integration test |
| **cluster_centroid_norm** | ≈ 1.0 (±0.01) | > 1.05 or < 0.95 | Unit test |
| **Runtime (1hr episode)** | ≤ 10 sec | > 30 sec | Benchmark |

**Tests:**
- `tests/ml/test_cluster.py`

**Config Dependencies:**
- Clustering params (TBD: `config/pipeline/recognition.yaml` or inline)

**Docs:**
- [docs/pipeline/cluster_identities.md](docs/pipeline/cluster_identities.md)

**Status:** ⚠ Pending

---

### 3.4 Episode Cleanup (`episode_cleanup`)

| Metric | Target | Warning Threshold | Verification |
|--------|--------|-------------------|--------------|
| **short_track_fraction_improvement** | ≥ 30% reduction | < 10% reduction | Integration test |
| **singleton_fraction_improvement** | ≥ 20% reduction | < 5% reduction | Integration test |
| **tracks_split** | ≥ 5% of long tracks | 0 (no splits performed) | Integration test |
| **Runtime (1hr episode)** | ≤ 5 min | > 15 min | Benchmark |

**Tests:**
- `tests/ml/test_episode_cleanup.py`

**Config Dependencies:**
- Reuses all pipeline configs

**Docs:**
- [docs/pipeline/episode_cleanup.md](docs/pipeline/episode_cleanup.md)

**Status:** ⚠ Pending

---

### 3.5 Facebank Management (`facebank`)

| Metric | Target | Warning Threshold | Verification |
|--------|--------|-------------------|--------------|
| **Duplicate faces** | < 2% | > 5% | Integration test |
| **Embedding drift** | < 3% per month | > 10% | Manual QA |
| **Cross-episode match accuracy** | ≥ 85% | < 75% | Manual QA on validation episodes |

**Tests:**
- `tests/ml/test_facebank.py`

**Docs:**
- [docs/reference/facebank.md](docs/reference/facebank.md)

**Status:** ⚠ Pending

---

## 4. API & UI Modules

| Module | Scope | Acceptance Criteria | Verification | Status |
|--------|-------|---------------------|--------------|--------|
| **Jobs API** | Async job orchestration | Jobs complete successfully; progress.json updated; state transitions correct | Integration test | ⚠ Pending |
| **Facebank UI** | Identity moderation | Merge/split/move/delete operations persist; thumbnails load; presigned URLs valid | UI smoke test | ⚠ Pending |
| **Episode Workspace** | Multi-tab interface | All tabs load; detect/track/cluster jobs launch; artifacts downloadable | UI smoke test | ⚠ Pending |

---

## 5. Non-Functional Criteria

| Category | Target | Verification |
|----------|--------|--------------|
| **Performance (GPU)** | ≤ 10 min / hour episode | Benchmark CI |
| **Performance (CPU)** | ≤ 3× realtime | Benchmark CI |
| **Scalability** | Horizontal workers supported (4+ parallel) | Stress Test |
| **Reliability** | No data loss on crash; graceful degradation | Restart Simulation |
| **Security** | Signed URLs; no secrets in repo | Code Scan |
| **Docs / Config** | Updated README, PRD, Pipeline docs, Config reference | CI doc-check |
| **CI Stability** | 100% green builds on `main` | GitHub Actions |
| **Promotion Policy** | TTL ≤ 30 days; tests+docs required | CI Gate |

---

## 6. Agents & Automation

| Task | Expected Behavior | Verification | Status |
|------|-------------------|--------------|--------|
| **update-docs-on-change** | Auto-syncs README, PRD, Directory, Architecture on file changes | Codex CI log | ⚠ Pending |
| **promotion-check** | Validates TODO → PROMOTED, tests, docs | CI | ⚠ Pending |
| **feature-expiry** | Flags features older than 30 days | CI | ⚠ Pending |
| **config-validator** | Ensures YAML schemas valid | CI | ⚠ Pending |

---

## 7. Sign-off Procedure

1. Feature owner verifies all applicable criteria
2. CI passes automated checks (tests, lint, schema validation)
3. Manual QA on validation clips (if applicable)
4. PR labeled `promotion` merges into `main`
5. `ACCEPTANCE_MATRIX.md` row updated to ✅ **Accepted**
6. Agents post summary under release tag

---

## 8. Change Control

- All edits tracked via PR review
- Agents automatically append new rows when features are added in `FEATURES/`
- CI compares feature folders against matrix entries; warns if missing
- CI blocks promotion if production feature lacks ✅ entry

---

## 9. Module Status Summary

| Module | Status | Blocker | Next Action |
|--------|--------|---------|-------------|
| **detect_track** | ⚠ Pending | Integration tests not passing thresholds | Tune thresholds, add test assertions |
| **faces_embed** | ⚠ Pending | Tests not implemented | Write integration tests |
| **cluster** | ⚠ Pending | Tests not implemented | Write integration tests |
| **episode_cleanup** | ⚠ Pending | Tests not implemented | Write integration tests |
| **facebank** | ⚠ Pending | Cross-episode matching not validated | Manual QA on validation episodes |
| **jobs_api** | ⚠ Pending | Integration tests incomplete | Extend API test coverage |
| **facebank_ui** | ⚠ Pending | UI smoke tests not automated | Add Playwright/Cypress tests |
| **agents** | ⚠ Pending | Playbooks not validated | Test doc-sync automation |

---

**Next Review:** After Phase 2 completion
**Maintainers:** Screenalytics Engineering + QA

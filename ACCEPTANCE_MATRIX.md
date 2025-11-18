# ACCEPTANCE_MATRIX.md — Screenalytics

Version: 1.0
Status: Live QA Standard

---

## 1. Purpose
To define measurable acceptance criteria for every feature or system module in Screenalytics.
A feature is considered **Accepted** only when all associated checkpoints below are satisfied and verified by CI or QA review.

---

## 2. Acceptance Columns
| Field | Description |
|--------|-------------|
| **Feature / Module** | Name of pipeline stage, UI component, or agent |
| **Scope** | What it’s responsible for |
| **Acceptance Criteria** | Quantifiable pass/fail conditions |
| **Verification Method** | Unit, Integration, Manual, or CI |
| **Status** | ✅ Accepted / ⚠ Pending / ❌ Blocked |
| **Owner** | Team or engineer responsible |

---

## 3. Pipeline Modules

| Feature / Module | Scope | Acceptance Criteria | Verification Method | Status | Owner |
|-------------------|--------|----------------------|---------------------|--------|--------|
| **Detection** | Face detection | ≥90 % recall on benchmark set; min_size, conf_th configurable | Integration + sample clip | ⚠ Pending | Vision |
| **Tracking** | ByteTrack association | No ID switches on sample; FPS ≥ 10 GPU | Integration | ⚠ Pending | Vision |
| **Embedding / Identity** | ArcFace ONNX recognition | ≥ 0.9 cosine mean for self-pairs; ≤ 0.2 for others | Unit + Integration | ⚠ Pending | Vision |
| **Audio / Diarization** | Speaker segmentation | DER ≤ 0.15; segments align with ASR | Integration | ⚠ Pending | Audio |
| **A/V Fusion** | Join tracks ↔ speakers | ≥ 85 % overlap precision | Integration | ⚠ Pending | Audio |
| **Aggregation** | Compute screen_time | Matches manual benchmark ±5 % | Integration + CI | ⚠ Pending | Core |
| **QA / Overrides** | Manual reassignment + undo | Edits persist, audit trail logged | UI Manual | ⚠ Pending | UI |
| **Facebank Management** | Build/update facebank | Duplicate < 2 %; drift < 3 % | Integration | ⚠ Pending | Vision |
| **SHOWS / PEOPLE Views** | Catalog UI | CRUD works; thumbnails link globally | UI Manual | ⚠ Pending | UI |
| **Agents / MCP Servers** | Codex / Claude automation | Docs sync verified; playbooks execute | CI | ⚠ Pending | Automation |

---

## 4. Non-Functional Criteria

| Category | Target | Verification |
|-----------|---------|--------------|
| **Performance** | ≤ 10 min / hour GPU episode | Benchmark CI |
| **Scalability** | Horizontal workers supported | Stress Test |
| **Reliability** | No data loss on crash | Restart Simulation |
| **Security** | Signed URLs; no secrets in repo | Code Scan |
| **Docs / Config** | Updated README, PRD, Directory, Solution | Codex Auto Check |
| **CI Stability** | 100 % green builds on `main` | GitHub Actions |
| **Promotion Policy** | TTL ≤ 30 days; tests+docs required | CI Gate |

---

## 5. Agents & Automation

| Task | Expected Behavior | Verification |
|------|-------------------|---------------|
| **update-docs-on-change** | Auto-syncs README, PRD, Directory, Architecture | Codex CI log |
| **promotion-check** | Validates TODO → PROMOTED, tests, docs | CI |
| **feature-expiry** | Flags features older than 30 days | CI |
| **config-validator** | Ensures YAML schemas valid | CI |

---

## 6. Sign-off Procedure
1. Feature owner verifies all applicable criteria.
2. CI passes automated checks.
3. PR labeled `promotion` merges into `main`.
4. `ACCEPTANCE_MATRIX.md` row updated to ✅ **Accepted**.
5. Codex posts summary under release tag.

---

## 7. Change Control
- All edits tracked via PR review.
- Codex automatically appends new rows when features are added in `FEATURES/`.
- CI compares feature folders against matrix entries; warns if missing.

---

**Next Review:** Quarterly QA Sync
**Maintainers:** TRR Analytics + Automation Ops

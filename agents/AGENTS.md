# Agent Catalog — Screenalytics

> **Operational rules live in [/AGENTS.md](/AGENTS.md)** — this file is the agent catalog.

Last Updated: 2025-12-16

---

## Overview

This catalog documents:
1. **Feature Agents** — Domain-specific agents that own pipeline features
2. **Automation Agents** — Bots that handle docs, promotions, and housekeeping
3. **MCP Servers** — Model Context Protocol interfaces for agent tools
4. **Agent Policies** — Write and threshold restrictions

---

## 1. Feature Agents

Feature agents own specific pipeline capabilities and their associated code, configs, and artifacts.

> **Canonical status source:** [`docs/_meta/feature_status.json`](/docs/_meta/feature_status.json) — the registry below summarizes status but the JSON is the single source of truth for job integration and bucket classification.

### FaceAlignmentAgent

**Status:** `partial` (sandbox implementation exists; not integrated into main pipeline)

**Scope:** Face alignment, landmark extraction, and quality gating.

**Owns:**
- `FEATURES/face_alignment/src/` — Alignment implementation
- `config/pipeline/face_alignment.yaml` — Alignment config

**Key Tasks:**
- FAN 68-point 2D landmark extraction
- Aligned face crop generation (ArcFace normalization)
- Quality scoring (heuristic stub; LUVLi planned)
- 3D head pose estimation (planned: 3DDFA_V2)

**Phase Status:**
| Phase | Status | Notes |
|-------|--------|-------|
| A_FAN_2D | `implemented_sandbox` | Sandbox runner exists; harvest can consume output if present but does NOT run FAN |
| B_LUVLi_quality_gate | `heuristic_stub` | Placeholder only; not model-based |
| C_3DDFA_V2 | `not_started` | Planned |

**Artifacts:**
- `data/manifests/{ep_id}/face_alignment/aligned_faces.jsonl`
- `data/manifests/{ep_id}/face_alignment/aligned_crops/` (optional)

**Config Flags (rollback levers):**
- `face_alignment.enabled: false` — Disable alignment
- `config/pipeline/embedding.yaml` → `min_alignment_quality: 0.0` — Disable quality gating

**Acceptance Matrix:** Sections 3.7-3.9

---

### BodyTrackingAgent

**Status:** `in_progress` (sandbox implementation exists; not integrated into main pipeline)

**Scope:** Person detection, tracking, Re-ID, and face↔body fusion.

**Owns:**
- `FEATURES/body_tracking/src/` — Body tracking implementation
- `config/pipeline/body_detection.yaml` — Detection and Re-ID config
- `config/pipeline/track_fusion.yaml` — Fusion rules

**Key Tasks:**
- YOLO person detection (class 0 = person)
- ByteTrack temporal tracking with ID offset (100000+)
- OSNet Re-ID embedding computation (256-d)
- Face↔body IoU association
- Re-ID handoff when face disappears
- Screen-time comparison (face-only vs face+body)

**Phase Status:**
| Phase | Status | Notes |
|-------|--------|-------|
| sandbox_runner | `implemented_sandbox` | Sandbox pipeline exists |
| pipeline_integration | `future` | Not in `tools/episode_run.py` yet |

**Artifacts:**
- `data/manifests/{ep_id}/body_tracking/body_detections.jsonl`
- `data/manifests/{ep_id}/body_tracking/body_tracks.jsonl`
- `data/manifests/{ep_id}/body_tracking/body_embeddings.npy`
- `data/manifests/{ep_id}/body_tracking/track_fusion.json`
- `data/manifests/{ep_id}/body_tracking/screentime_comparison.json`

**Config Flags (rollback levers):**
- `body_tracking.enabled: false` — Disable entire feature
- `person_reid.enabled: false` — Disable Re-ID embeddings
- `track_fusion.enabled: false` — Disable face↔body fusion

**Acceptance Matrix:** Sections 3.10-3.12

---

### EmbeddingEngineAgent

**Status:** `production` (InsightFace/PyTorch+ONNXRuntime is default; TensorRT scaffold exists)

**Scope:** Face embedding computation backends.

**Owns:**
- `py_screenalytics/embeddings/` — Core embedding logic
- `FEATURES/arcface_tensorrt/` — TensorRT acceleration (scaffold only)
- `config/pipeline/embedding.yaml` — Embedding config

**Key Tasks:**
- ArcFace 512-d embedding computation
- Backend selection (ONNX vs TensorRT)
- CPU/GPU fallback handling
- Embedding normalization and versioning

**Phase Status:**
| Phase | Status | Notes |
|-------|--------|-------|
| InsightFace_ONNX | `production` | Default backend |
| TensorRT_acceleration | `scaffold_only` | Not validated for production |

**Artifacts:**
- `data/manifests/{ep_id}/embeddings.npy`
- `data/manifests/{ep_id}/faces.jsonl` (embedding field)

**Config Flags:**
- `embedding.backend: onnx` — Use ONNX backend (default)
- `embedding.device: auto` — Auto-detect CPU/GPU

---

### VisionAnalyticsAgent

**Status:** `not_started`

**Scope:** Advanced visibility analysis using head pose, face mesh, and temporal patterns.

**Owns:**
- `FEATURES/vision_analytics/` — Vision analytics implementation (planned)
- `config/pipeline/analytics.yaml` — Analytics config

**Key Tasks:**
- Face-visible vs body-only breakdown
- Head pose analysis for engagement metrics
- Temporal visibility patterns

**Phase Status:**
| Phase | Status | Notes |
|-------|--------|-------|
| all | `not_started` | Docs/config only |

---

### QAAcceptanceAgent

**Status:** `partial` (acceptance matrix exists; automated validation partial)

**Scope:** Evaluate pipeline runs against acceptance criteria.

**Owns:**
- `ACCEPTANCE_MATRIX.md` — Quality gates definition
- `.claude/skills/qa-acceptance/` — QA skill definition

**Key Tasks:**
- Validate pipeline outputs against acceptance criteria
- Flag regressions in quality metrics
- Generate acceptance reports

---

## 2. Automation Agents

Automation agents handle documentation, promotions, and housekeeping tasks.

### Documentation Sync Agent

**Purpose:** Auto-update top-level docs when file structure changes

**Trigger:** File add/delete/rename under monitored paths

**Playbook:** `agents/playbooks/update-docs-on-change.yaml`

**Actions:**
1. Detect file changes via Git hooks or CI
2. Update only these docs:
   - `docs/architecture/solution_architecture.md`
   - `docs/architecture/directory_structure.md`
   - `docs/product/prd.md`
   - `README.md`
3. Commit with message: `docs(sync): auto-update architecture after file change`

**Safety:** No other files altered automatically; human review for major changes.

---

### Promotion Check Agent

**Purpose:** Validate feature promotion readiness

**Trigger:** PR labeled `promotion`

**Checks:**
- `TODO.md` status → `PROMOTED`
- Tests present and passing
- Docs present in `FEATURES/<name>/docs/`
- No production imports from `FEATURES/**`
- `ACCEPTANCE_MATRIX.md` row exists

**Actions:** Pass → allow merge; Fail → block merge with specific errors.

---

### Feature Expiry Agent

**Purpose:** Flag features older than 30 days without promotion

**Trigger:** Nightly CI cron job

**Actions:**
- Create GitHub issue for stale features
- Tag owner from `TODO.md`
- Options: Promote, Archive, or Delete

---

### Low-Confidence Relabeling Agent (Future)

**Purpose:** Auto-assign identities for low-confidence tracks using Facebank

**Status:** `planned`

**Workflow:**
1. Query tracks with `confidence < 0.7` and `person_id = null`
2. Compute cluster centroid
3. Query pgvector for nearest Facebank seed embeddings
4. If similarity > 0.75, auto-assign `person_id`
5. Write audit trail to `assignment_history`

**Safety:** Never overwrite `locked: true` identities; human review for ambiguous matches.

---

## 3. MCP Servers

Agents interact with Screenalytics via MCP (Model Context Protocol) servers.

### `screenalytics` MCP Server

**Tools:**
- `list_low_confidence_tracks(ep_id, threshold)` — Query tracks below confidence
- `assign_identity(track_id, person_id, source)` — Assign identity to track
- `export_screentime(ep_id, format)` — Export screentime analytics
- `promote_to_facebank(person_id, seed_id)` — Promote identity to Facebank

### `storage` MCP Server

**Tools:**
- `list_artifacts(ep_id)` — List all artifacts for episode
- `get_signed_url(path, expiry)` — Generate presigned URL
- `purge_artifacts(ep_id, artifact_types)` — Delete specified artifacts

### `postgres` MCP Server

**Tools:**
- `query_safe(sql)` — Execute read-only analytics queries
- `get_episode_metadata(ep_id)` — Fetch episode metadata
- `get_facebank_seeds(person_id)` — Fetch Facebank seeds for person

---

## 4. Agent Policies

### Write Policies

**During feature development:**
- Agents **can write** to `FEATURES/<name>/docs/`
- Agents **cannot write** to root `/docs/**` or `/config/**`

**After feature promotion:**
- Agents **can update** root docs only via playbook automation
- Agents **cannot modify** pipeline configs without human approval

**CI-enforced:** `.github/workflows/doc-policy.yml` validates permissions

### Threshold Policies

**Agents cannot change:**
- Detection thresholds (`confidence_th`, `min_size`)
- Tracking thresholds (`track_thresh`, `match_thresh`)
- Clustering thresholds (`cluster_thresh`, `min_cluster_size`)
- Quality gating thresholds (`min_quality`)

**Rationale:** Threshold changes can silently affect runtime and track/cluster counts. All threshold changes require human review via PR.

### Doc Sprawl Prevention

**Agents cannot:**
- Create new top-level markdown files without approval
- Duplicate content across multiple files
- Generate docs longer than 10,000 words without review

---

## 5. Configuration

### Codex Config

Location: `config/codex.config.toml`

### Claude Policy

Location: `config/claude.policies.yaml`

---

## References

- **[/AGENTS.md](/AGENTS.md)** — Operational rules for all agents
- **[ACCEPTANCE_MATRIX.md](/ACCEPTANCE_MATRIX.md)** — Quality gates
- **[Feature Sandboxes](../docs/features/feature_sandboxes.md)** — Feature sandbox workflow

# Directory Structure — Screenalytics

**Version:** 2.0
**Status:** Active
**Last Updated:** 2025-11-18

---

## 1. Top-Level Repository Map

```
screenalytics/
├── apps/                        # Frontend + API applications (STABLE)
│   ├── api/                     # FastAPI backend
│   │   ├── main.py              # API entrypoint
│   │   ├── routers/             # Endpoint modules (episodes, jobs, identities, facebank)
│   │   ├── services/            # Business logic (jobs, track_reps, facebank)
│   │   └── models/              # Pydantic schemas
│   └── workspace-ui/            # Streamlit workspace UI (Upload/Episode Detail/Faces Review)
│       ├── Upload_Video.py      # Streamlit entrypoint
│       └── pages/               # Streamlit subpages
│
├── web/                         # Next.js prototype app (events/divisions/agents)
│
├── packages/                    # Shared libraries (STABLE)
│   ├── py-screenalytics/        # Python SDK (artifacts, config, storage)
│   │   ├── artifacts.py         # Artifact path helpers
│   │   ├── config.py            # Config loader with override support
│   │   └── storage.py           # S3/local storage abstraction
│   └── ts-sdk/                  # TypeScript SDK (API client, types)
│
├── db/                          # Database migrations, views, seeds (VERSIONED)
│   ├── migrations/              # SQL migration scripts
│   ├── views/                   # Database views
│   └── seeds/                   # Seed data for local dev
│
├── config/                      # All YAML/TOML configs (VERSIONED)
│   ├── pipeline/                # Pipeline stage configs
│   │   ├── detection.yaml       # RetinaFace settings
│   │   ├── tracking.yaml        # ByteTrack settings
│   │   ├── tracking-strict.yaml # Strict tracking variant
│   │   ├── faces_embed_sampling.yaml  # Faces harvest quality gating
│   │   ├── performance_profiles.yaml  # Device-aware performance profiles
│   │   ├── audio.yaml           # NeMo diarization + ASR settings
│   │   ├── body_detection.yaml  # Body tracking (YOLO + ByteTrack + Re-ID)
│   │   ├── track_fusion.yaml    # Face↔body fusion rules
│   │   └── screen_time_v2.yaml  # Screentime DAG + presets
│   ├── agents/                  # Agent policies
│   │   └── policies.yaml        # Agent write rules
│   ├── codex.config.toml        # Codex SDK config
│   ├── agents.sdk.yaml          # Agents SDK graph
│   └── claude.policies.yaml     # Claude-specific policies
│
├── FEATURES/                    # Experimental feature sandboxes (TTL: 30 days)
│   ├── <feature-name>/
│   │   ├── src/                 # Throwaway implementation
│   │   ├── tests/               # Focused tests
│   │   ├── docs/                # Working notes (agents write here)
│   │   └── TODO.md              # Status, owner, plan
│   ├── detection/               # Detection feature (promoted candidate)
│   ├── tracking/                # Tracking feature (promoted candidate)
│   ├── identity/                # Identity feature (promoted candidate)
│   └── ...                      # Other experimental features
│
├── agents/                      # Codex/Claude profiles, playbooks, prompts (CONTROLLED)
│   ├── AGENTS.md                # Agent behavior documentation
│   ├── playbooks/               # Automation playbooks
│   │   └── update-docs-on-change.yaml
│   ├── profiles/                # Agent profiles (relabel, doc-sync, etc.)
│   └── tasks/                   # Task definitions (JSON)
│
├── mcps/                        # MCP servers for Screanalytics (CONTROLLED)
│   ├── screanalytics/           # Screenalytics MCP server
│   ├── storage/                 # Storage MCP server
│   └── postgres/                # Postgres MCP server
│
├── docs/                        # Permanent project documentation (STABLE)
│   ├── architecture/            # System architecture docs
│   │   ├── solution_architecture.md
│   │   └── directory_structure.md  (this file)
│   ├── pipeline/                # Pipeline stage docs
│   │   ├── overview.md
│   │   ├── detect_track_faces.md
│   │   ├── faces_harvest.md
│   │   ├── cluster_identities.md
│   │   └── episode_cleanup.md
│   ├── reference/               # Reference documentation
│   │   ├── artifacts_faces_tracks_identities.md
│   │   ├── facebank.md
│   │   └── config/
│   │       └── pipeline_configs.md
│   ├── ops/                     # Operations guides
│   │   ├── performance_tuning_faces_pipeline.md
│   │   ├── troubleshooting_faces_pipeline.md
│   │   └── hardware_sizing.md
│   ├── changes/                 # Historical change logs
│   └── code-updates/            # Code update summaries
│
├── infra/                       # Docker, IaC, deployment configs (SUPPORT)
│   ├── docker/
│   │   ├── compose.yaml         # Local dev stack (Postgres, Redis, MinIO)
│   │   └── Dockerfile.*         # Container images
│   └── terraform/               # IaC for cloud deployment (TBD)
│
├── tests/                       # Integration/unit/e2e tests (REQUIRED)
│   ├── api/                     # API endpoint tests
│   ├── ml/                      # ML pipeline tests (detect, track, embed, cluster)
│   └── integration/             # End-to-end integration tests
│
├── tools/                       # Helper scripts (SUPPORT)
│   ├── episode_run.py           # CLI for detect/track/embed/cluster
│   ├── episode_cleanup.py       # CLI for episode cleanup workflow
│   ├── run_pipeline.py          # Orchestrate multi-stage runs
│   ├── analyze_screen_time.py   # Generate screentime.json/csv
│   ├── dev-up.sh                # Start local dev stack
│   └── ...                      # Other utility scripts
│
├── .github/                     # CI/CD workflows and promotion checks (ENFORCED)
│   └── workflows/
│       ├── ci.yml               # Lint + unit tests + smoke dry-run
│       ├── on-push-doc-sync.yml # Auto-sync docs on push
│       ├── codex-review.yml     # Automated PR review (Codex)
│       ├── codex-manual.yml     # Manual Codex workflow trigger
│       └── claude-review.yml    # Automated PR review (Claude)
│
└── (root files)                 # Minimal root entrypoints / CI gates
    ├── README.md                # High-level entrypoint (quickstart, links)
    ├── SETUP.md                 # Environment bootstrap and infra
    ├── ACCEPTANCE_MATRIX.md     # Quality gates and acceptance criteria (CI/test referenced)
    ├── CONTRIBUTING.md          # Contribution workflow
    ├── AGENTS.md                # Repo agent policy
    ├── LICENSE                  # Project license
    ├── .env.example             # Environment variable template
    ├── requirements.txt         # Python dependencies
    ├── pyproject.toml           # Python project metadata
    └── ...                      # Other root files
```

---

## 2. Folder Purposes

| Folder | Purpose | Stability | Import Policy |
|--------|---------|-----------|---------------|
| **apps/** | FastAPI backend + Streamlit workspace UI | STABLE | ✅ Production imports allowed |
| **web/** | Next.js prototype app | STABLE | ✅ Production imports allowed |
| **packages/** | Shared Python & TypeScript libs | STABLE | ✅ Production imports allowed |
| **db/** | Migrations, views, seeds | VERSIONED | ✅ Production imports allowed |
| **config/** | All YAML/TOML configs, policies | VERSIONED | ✅ Production reads allowed |
| **FEATURES/** | Experimental feature sandboxes | TTL: 30 days | ❌ **NO production imports** |
| **agents/** | Codex/Claude profiles, playbooks | CONTROLLED | ⚠️ Schema-locked |
| **mcps/** | MCP servers | CONTROLLED | ⚠️ Schema-locked |
| **docs/** | Permanent project documentation | STABLE | ✅ Always up-to-date |
| **infra/** | Docker, Terraform, IaC | SUPPORT | ✅ Low churn |
| **tests/** | Unit/integration/e2e tests | REQUIRED | ✅ Test imports only |
| **tools/** | Helper scripts (CLI utilities) | SUPPORT | ✅ CLI-only imports |
| **.github/** | CI/CD workflows, promotion checks | ENFORCED | 🚫 CI-only |

---

## 3. FEATURES/ Policy

### 3.1 Structure
```
FEATURES/<feature>/
├── src/              # Throwaway implementation (not imported by production)
├── tests/            # Focused tests for this feature
├── docs/             # Working notes (agents write here during development)
└── TODO.md           # Status, owner, plan, promotion checklist
```

### 3.2 Rules
- **Time-to-live:** 30 days from creation
- **Import policy:** Imports from `FEATURES/**` are **forbidden** in production code (`apps/`, `web/`, `packages/`)
- **Promotion requirements:**
  - ✅ Tests present and passing
  - ✅ Docs written (explain what it does, config keys, metrics)
  - ✅ Config-driven (no hardcoded thresholds)
  - ✅ CI green (lint, tests, acceptance checks)
  - ✅ Row in `ACCEPTANCE_MATRIX.md` marked ✅ Accepted
- **Promotion process:**
  1. Open a PR that moves code out of `FEATURES/<feature>/src/` into production paths (`apps/`, `web/`, `packages/`)
  2. Move tests into `tests/`
  3. Merge docs into `docs/` (or link into existing docs)
  4. Update `TODO.md` status → `PROMOTED` (or archive/remove the sandbox)
  5. Agents auto-update root docs (README, PRD, Solution Architecture, Directory Structure)

### 3.3 CI Enforcement
- **Feature expiry:** CI flags features older than 30 days
- **Import validation:** CI fails if production code imports from `FEATURES/**`
- **Acceptance matrix:** CI checks that promoted features have a ✅ entry in `ACCEPTANCE_MATRIX.md`

---

## 4. Promotion Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Develop in FEATURES/<name>/                                  │
│    - Implement in src/                                           │
│    - Write tests in tests/                                       │
│    - Document in docs/                                           │
│    - Update TODO.md with status                                  │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. CI Pre-Promotion Checks                                      │
│    ✅ Tests pass (pytest)                                        │
│    ✅ Lint clean (black, ruff, mypy)                             │
│    ✅ Docs present (docs/*.md)                                   │
│    ✅ Config-driven (no hardcoded magic numbers)                 │
│    ✅ No production imports from FEATURES/**                     │
│    ✅ TODO.md status != ABANDONED                                │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Promote via PR                                                │
│    - Move src/ → target path (apps/, web/, packages/)            │
│    - Move tests/ → tests/<category>/                             │
│    - Merge docs/ → docs/<category>/                              │
│    - Update TODO.md status → PROMOTED (or archive/remove)        │
│    - Add row to ACCEPTANCE_MATRIX.md                             │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. CI Post-Promotion Checks                                     │
│    ✅ Full integration test passes (e2e sample clip)             │
│    ✅ ACCEPTANCE_MATRIX.md row marked ✅ Accepted                 │
│    ✅ Config docs updated (docs/reference/config/)               │
│    ✅ No production imports from FEATURES/** (re-check)          │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Agents Auto-Update Docs                                      │
│    - README.md (pipeline summary, quickstart)                   │
│    - docs/product/prd.md (feature addition under "Core Features") │
│    - docs/architecture/solution_architecture.md (component/path updates) │
│    - docs/architecture/directory_structure.md (tree and descriptions) │
│    Playbook: agents/playbooks/update-docs-on-change.yaml        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Stable vs Experimental Zones

| Zone | Status | Write Policy | Import Policy |
|------|--------|--------------|---------------|
| `/apps`, `/web`, `/packages`, `/db`, `/config`, `/docs` | **STABLE** | Protected (PR review required) | ✅ Production imports allowed |
| `/FEATURES` | **EXPERIMENTAL** | TTL + CI gating | ❌ NO production imports |
| `/agents`, `/mcps` | **CONTROLLED** | Schema-locked (versioned) | ⚠️ MCP protocol only |
| `/infra`, `/tools` | **SUPPORT** | Low churn (utility scripts) | ✅ CLI-only |
| `/.github` | **ENFORCED** | CI-only updates | 🚫 No imports |

---

## 6. Agent Update Hooks

### 6.1 Trigger
Any file `add`, `delete`, or `rename` event under:
- `/apps/**`
- `/web/**`
- `/packages/**`
- `/db/**`
- `/config/**`
- `/FEATURES/**`
- `/agents/**`
- `/mcps/**`
- `/docs/**`
- `/infra/**`
- `/tests/**`
- `/tools/**`

### 6.2 Playbook
**Location:** `agents/playbooks/update-docs-on-change.yaml`

**Behavior:**
- For each detected file change, open or update:
  - `docs/architecture/solution_architecture.md` → Adjust affected components/paths
  - `docs/architecture/directory_structure.md` → Update tree and descriptions
  - `docs/product/prd.md` → Mark feature addition/removal under "Core Features"
  - `README.md` → Reflect new/removed directories in "Repository Layout"
- Commit changes with message:
  ```
  docs(sync): auto-update architecture and directory docs after file change
  ```

### 6.3 Safety
- **No other files** may be altered automatically
- CI ensures all four files remain consistent with live repo structure
- Claude policy alignment: Mirrors this rule in `config/claude.policies.yaml` under `auto_doc_update: true`

---

## 7. Key Concepts

### 7.1 "STABLE" Paths
- **Definition:** Production-ready code that has passed promotion gates
- **Examples:** `apps/api/`, `apps/workspace-ui/`, `web/app/`, `packages/py-screenalytics/`
- **Policy:** Changes require PR review, tests, and docs

### 7.2 "FEATURES" Sandboxes
- **Definition:** Temporary experimental code under development
- **Examples:** `FEATURES/detection/`, `FEATURES/tracking/`, `FEATURES/identity/`
- **Policy:** 30-day TTL, no production imports, promotion required for graduation

### 7.3 "CONTROLLED" Paths
- **Definition:** Schema-locked components with versioned interfaces
- **Examples:** `agents/`, `mcps/`, `config/agents/`
- **Policy:** Changes require schema validation, backward compatibility checks

### 7.4 "SUPPORT" Paths
- **Definition:** Utility scripts and infrastructure config
- **Examples:** `tools/`, `infra/`, `.github/`
- **Policy:** Low churn, CLI-only usage, no production dependencies

---

## 8. Import Rules Summary

### ✅ **ALLOWED** Production Imports
```python
from apps.api.models import Episode
from py_screenalytics import artifacts
from apps.common.cpu_limits import apply_global_cpu_limits
```

### ❌ **FORBIDDEN** Production Imports
```python
# CI will FAIL if production code imports from FEATURES/
from FEATURES.detection.src.detector import MyExperimentalDetector
```

### ⚠️ **CONDITIONAL** Imports
```python
# OK in tests/
from FEATURES.tracking.src.tracker import ExperimentalTracker

# OK in tools/
from FEATURES.identity.src.cluster import ClusterExperiment
```

---

## 9. Documentation Hierarchy

### 9.1 Root-Level Docs
- `README.md` — High-level quickstart and links
- `SETUP.md` — Environment bootstrap and infra setup
- `ACCEPTANCE_MATRIX.md` — CI/test-referenced quality gates and thresholds
- `CONTRIBUTING.md` — Contribution workflow
- `AGENTS.md` — Repo agent policy and guardrails

Deprecated root docs that have been superseded are archived under `docs/_archive/root_docs/`.

### 9.2 Deep Documentation (`docs/`)
- **Architecture:** System design, component diagrams, data model
- **Pipeline:** Stage-by-stage guides (detect, track, embed, cluster, cleanup)
- **Reference:** Artifact schemas, config keys, Facebank layout
- **Ops:** Performance tuning, troubleshooting, hardware sizing

---

## 10. CI/CD Workflows

Selected workflows (see `.github/workflows/` for the full list):

- `ci.yml` — Lint/typecheck + unit tests + smoke dry-run
- `on-push-doc-sync.yml` — Auto-sync docs on push to `main`
- `codex-review.yml` / `claude-review.yml` — Automated PR review workflows

---

## 11. References

- [Solution Architecture](solution_architecture.md) — System design and data flow
- [Pipeline Overview](../pipeline/overview.md) — Stage-by-stage pipeline details
- [Config Reference](../reference/config/pipeline_configs.md) — Key-by-key config docs
- [Feature sandboxes](../features/feature_sandboxes.md) — Feature sandbox workflow
- [ACCEPTANCE_MATRIX.md](../../ACCEPTANCE_MATRIX.md) — Quality gates

---

**Maintained by:** Screenalytics Engineering
**Next Review:** Quarterly

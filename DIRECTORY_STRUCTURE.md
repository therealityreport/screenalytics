# DIRECTORY_STRUCTURE.md — Screenalytics

Version: 1.1 (Enhanced)
Status: Planning

---

## 1. Top-Level Map
```

screenalytics/
├─ apps/
│  ├─ api/
│  └─ workspace-ui/
├─ workers/
├─ packages/
├─ db/
├─ config/
├─ FEATURES/
├─ agents/
├─ mcps/
├─ docs/
├─ infra/
├─ tests/
├─ tools/
├─ .github/
└─ root files (MANIFEST.md, README.md, PRD.md, etc.)

```

---

## 2. Folder Purposes

| Folder | Purpose | Notes |
|---------|----------|-------|
| **apps/** | Frontend + API endpoints | stable |
| **workers/** | ML & analytics pipeline stages | stable |
| **packages/** | Shared libs (Python & TypeScript) | stable |
| **db/** | migrations, views, seeds | versioned |
| **config/** | all YAML/TOML configs, codex, policies | versioned |
| **FEATURES/** | experimental feature sandboxes | 30-day TTL |
| **agents/** | Codex/Claude profiles, playbooks, prompts | controlled |
| **mcps/** | MCP servers for Screenalytics, storage, postgres | controlled |
| **docs/** | permanent project documentation | stable |
| **infra/** | docker, terraform, IaC | support |
| **tests/** | integration/unit/e2e tests | required |
| **tools/** | helper scripts (new-feature, promote-feature) | support |
| **.github/** | CI/CD workflows and promotion checks | enforced |

---

## 3. FEATURES/ Policy
```

FEATURES/<feature>/
├─ src/
├─ tests/
├─ docs/
└─ TODO.md

````
- Time-to-live: 30 days
- Imports from `FEATURES/**` forbidden in production code
- Promotion requires CI approval, docs, and tests

---

## 4. Promotion Workflow
1. Implement under `FEATURES/<feature>/`.
2. Add docs and configs in that folder.
3. Pass CI and promotion checklist.
4. Run `tools/promote-feature.py`.
5. Agents update global docs (README, PRD, Solution Architecture, Directory Structure).
6. Update `ACCEPTANCE_MATRIX.md` to ✅ for the feature, either via the promotion trigger or manually before merge.

---

## 5. CI Enforcement
- `.github/workflows/feature-promote.yml` checks:
  - `TODO.md` status → PROMOTED
  - `docs/**` updated
  - tests present
  - imports validated
- `.github/workflows/doc-policy.yml` also runs `tools/check-acceptance-matrix.py` to block promotion if a production feature lacks a ✅ entry.

---

## 6. Stable vs Experimental

| Zone | Status | Policy |
|------|---------|--------|
| `/apps`, `/workers`, `/packages`, `/db`, `/config`, `/docs` | **Stable** | protected |
| `/FEATURES` | **Experimental** | TTL + CI gating |
| `/agents`, `/mcps` | **Controlled** | schema-locked |
| `/infra`, `/tools` | **Support** | low churn |

---

## 7. Agent Update Hooks
- Any file creation, deletion, or rename triggers **Codex** and **Claude** agents.
- Agents automatically patch:
  - `SOLUTION_ARCHITECTURE.md`
  - `DIRECTORY_STRUCTURE.md`
  - `PRD.md`
  - `README.md`

Hook defined in `/agents/playbooks/update-docs-on-change.yaml`.

---

# Message to CODEX (configuration directive)

> **To:** Codex system agent
> **Subject:** Repo-wide doc synchronization rule
> **Action:** Update configuration to maintain documentation parity.

---

**Instruction**

Codex, please configure the repository automation as follows:

1. **Watch Paths**
   - Any `add`, `delete`, or `rename` event under:
     - `/apps/**`
     - `/workers/**`
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

2. **Trigger Playbook**
   - Run: `agents/playbooks/update-docs-on-change.yaml`

3. **Playbook Behavior**
   - For each detected file change, open or update:
     - `SOLUTION_ARCHITECTURE.md` → adjust affected components/paths
     - `DIRECTORY_STRUCTURE.md` → update tree and descriptions
     - `PRD.md` → mark feature addition/removal under “Core Features”
     - `README.md` → reflect new or removed directories in “Repository Layout”
   - Commit changes with message:
     ```
     docs(sync): auto-update architecture and directory docs after file change
     ```

4. **Claude Policy Alignment**
   - Mirror this rule in `config/claude.policies.yaml` under `auto_doc_update: true`.
   - Claude performs the same sync if Codex is inactive or unavailable.

5. **Safety**
   - No other files may be altered automatically.
   - CI ensures all four files remain consistent with the live repo structure.

---

**Outcome**

Every structural change instantly refreshes the project’s high-level documentation.
Codex and Claude stay aware of evolving architecture, keeping README, PRD, SolutionArchitecture, and DirectoryStructure perpetually in sync.

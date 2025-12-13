# AGENTS.md — Screenalytics Automation

Version: 2.0
Last Updated: 2025-11-18

---

## Overview

Screenalytics uses **Codex SDK** and **Claude API** agents for autonomous documentation updates, low-confidence relabeling, and pipeline automation.

**Agents operate under strict policies** to prevent silent config changes, doc sprawl, and threshold manipulation.

---

## Agent Types

### 1. Documentation Sync Agent

**Purpose:** Auto-update top-level docs when file structure changes

**Trigger:** Any `add`, `delete`, or `rename` event under:
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

**Playbook:** `agents/playbooks/update-docs-on-change.yaml`

**Actions:**
1. Detect file changes via Git hooks or CI
2. Update these docs only:
   - `docs/architecture/solution_architecture.md` — Adjust affected components/paths
   - `docs/architecture/directory_structure.md` — Update tree and descriptions
   - `docs/product/prd.md` — Mark feature addition/removal
   - `README.md` — Reflect new/removed directories
3. Commit with message: `docs(sync): auto-update architecture after file change`

**Safety:**
- **No other files** may be altered automatically
- CI ensures consistency between docs and live repo structure
- Human review required for major changes

---

### 2. Promotion Check Agent

**Purpose:** Validate feature promotion readiness

**Trigger:** PR labeled `promotion`

**Checks:**
- ✅ `TODO.md` status → `PROMOTED`
- ✅ Tests present and passing
- ✅ Docs present in `FEATURES/<name>/docs/`
- ✅ No production imports from `FEATURES/**`
- ✅ `ACCEPTANCE_MATRIX.md` row exists

**Actions:**
- Pass: Allow merge
- Fail: Block merge, comment on PR with specific failures

---

### 3. Feature Expiry Agent

**Purpose:** Flag features older than 30 days

**Trigger:** Nightly CI cron job

**Checks:**
- For each `FEATURES/<name>/`, check creation date
- If > 30 days and `TODO.md` status != `PROMOTED`, flag as stale

**Actions:**
- Create GitHub issue: "Feature `<name>` is 35 days old (TTL: 30 days)"
- Tag owner (from `TODO.md`)
- Options: Promote, Archive, or Delete

---

### 4. Low-Confidence Relabeling Agent (Future)

**Purpose:** Auto-assign identities for low-confidence tracks using Facebank

**Trigger:** Manual invocation via MCP or cron job

**Workflow:**
1. Query tracks with `confidence < 0.7` and `person_id = null`
2. For each track, compute cluster centroid
3. Query pgvector for nearest Facebank seed embeddings
4. If similarity > threshold (e.g., 0.75), auto-assign `person_id`
5. Write audit trail to `assignment_history`

**Safety:**
- Only assign if similarity > high threshold (0.75)
- Never overwrite `locked: true` identities
- Human review queue for ambiguous matches

---

## MCP Servers

Agents interact with Screenalytics via **MCP (Model Context Protocol)** servers:

### 1. `screanalytics` MCP Server

**Tools:**
- `list_low_confidence_tracks(ep_id, threshold)` — Query tracks below confidence threshold
- `assign_identity(track_id, person_id, source)` — Assign identity to track
- `export_screentime(ep_id, format)` — Export screentime analytics
- `promote_to_facebank(person_id, seed_id)` — Promote identity to Facebank

### 2. `storage` MCP Server

**Tools:**
- `list_artifacts(ep_id)` — List all artifacts for episode
- `get_signed_url(path, expiry)` — Generate presigned URL
- `purge_artifacts(ep_id, artifact_types)` — Delete specified artifacts

### 3. `postgres` MCP Server

**Tools:**
- `query_safe(sql)` — Execute read-only analytics queries
- `get_episode_metadata(ep_id)` — Fetch episode metadata
- `get_facebank_seeds(person_id)` — Fetch Facebank seeds for person

---

## Agent Policies

### Write Policies

**During feature development:**
- Agents **can write** to `FEATURES/<name>/docs/`
- Agents **cannot write** to root `/docs/**` or `/config/**`

**After feature promotion:**
- Agents **can update** root docs (README, PRD, Solution Architecture, Directory Structure) **only** via playbook automation
- Agents **cannot modify** pipeline configs (`config/pipeline/*.yaml`) without human approval

**CI-enforced:**
- `.github/workflows/doc-policy.yml` validates agent write permissions
- Commits from agents tagged with `[bot]` prefix

### Threshold Policies

**Agents cannot:**
- Change detection thresholds (`confidence_th`, `min_size`, etc.)
- Change tracking thresholds (`track_thresh`, `match_thresh`, etc.)
- Change clustering thresholds (`cluster_thresh`, `min_cluster_size`)
- Change quality gating thresholds (`min_quality`, etc.)

**Rationale:** Silently changing thresholds can blow up runtime or track/cluster counts. All threshold changes require human review via PR.

### Doc Sprawl Prevention

**Agents cannot:**
- Create new top-level markdown files without human approval
- Duplicate content across multiple files
- Generate docs longer than 10,000 words without human review

**Rationale:** Prevent doc proliferation and maintain single source of truth.

---

## Playbooks

### `agents/playbooks/update-docs-on-change.yaml`

```yaml
name: Update Docs on File Change
trigger:
  - file_add
  - file_delete
  - file_rename
scope:
  - apps/**
  - web/**
  - packages/**
  - config/**
  - FEATURES/**
  - docs/**
  - tests/**
  - tools/**

actions:
  - update_file:
      path: docs/architecture/solution_architecture.md
      method: sync_components
  - update_file:
      path: docs/architecture/directory_structure.md
      method: sync_tree
  - update_file:
      path: docs/product/prd.md
      method: sync_features
  - update_file:
      path: README.md
      method: sync_layout
  - commit:
      message: "docs(sync): auto-update architecture after file change"
```

### `agents/playbooks/promotion-check.yaml`

```yaml
name: Promotion Check
trigger:
  - pr_labeled: promotion

actions:
  - check_file_exists:
      path: FEATURES/{{feature}}/tests/
  - check_file_exists:
      path: FEATURES/{{feature}}/docs/
  - check_yaml_field:
      path: FEATURES/{{feature}}/TODO.md
      field: status
      value: PROMOTED
  - check_acceptance_matrix:
      feature: "{{feature}}"
      status: Accepted
  - block_if_fail: true
```

---

## Configuration

### Codex Config (`config/codex.config.toml`)

```toml
[agent.doc_sync]
enabled = true
playbook = "agents/playbooks/update-docs-on-change.yaml"
trigger = "file_change"
allowed_files = [
  "docs/architecture/solution_architecture.md",
  "docs/architecture/directory_structure.md",
  "docs/product/prd.md",
  "README.md"
]

[agent.promotion_check]
enabled = true
playbook = "agents/playbooks/promotion-check.yaml"
trigger = "pr_label:promotion"
```

### Claude Policy (`config/claude.policies.yaml`)

```yaml
auto_doc_update: true
allowed_write_paths:
  - docs/architecture/solution_architecture.md
  - docs/architecture/directory_structure.md
  - docs/product/prd.md
  - README.md
  - FEATURES/*/docs/**

forbidden_write_paths:
  - config/pipeline/*.yaml
  - config/agents/*.yaml
  - .github/workflows/**

max_doc_length: 10000  # words
```

---

## Testing Agents

### 1. Test Doc Sync Locally
```bash
# Make a file change
touch apps/api/new_module.py

# Trigger agent manually (via Codex CLI)
codex exec --config config/codex.config.toml --playbook agents/playbooks/update-docs-on-change.yaml

# Verify docs updated
git diff docs/architecture/solution_architecture.md docs/architecture/directory_structure.md
```

### 2. Test Promotion Check in CI
```bash
# Create PR with promotion label
gh pr create --title "Promote my-feature" --label promotion

# CI runs promotion-check playbook
# Verify CI passes or fails with specific errors
```

---

## Monitoring & Audit

### Agent Activity Logs
- All agent actions logged to `data/agents/activity.log`
- Format: `[timestamp] [agent] [action] [file] [outcome]`

### Audit Trail
- All automated commits tagged with `[bot]` prefix
- Git history shows exactly what agent changed
- Human can revert via `git revert <commit>`

### Alerts
- If agent fails to update docs: GitHub issue created
- If agent modifies forbidden file: CI fails, blocks merge

---

## Future Enhancements

1. **Auto-cluster validation:** Agent runs clustering on new episodes, flags anomalies (singleton_fraction > 0.6, etc.)
2. **Performance regression detection:** Agent benchmarks detect/track runtime, alerts if > threshold
3. **Cross-episode identity matching:** Agent auto-assigns person_id via Facebank similarity
4. **Stale feature cleanup:** Agent archives features > 30 days old if no promotion activity

---

## References

- **[Directory Structure](../docs/architecture/directory_structure.md)** — Promotion workflow
- **[Feature sandboxes](../docs/features/feature_sandboxes.md)** — Feature sandbox workflow
- **[ACCEPTANCE_MATRIX.md](../ACCEPTANCE_MATRIX.md)** — Quality gates

---

**Maintained by:** Screenalytics Engineering

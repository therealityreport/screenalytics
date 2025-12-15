# Branching Strategy: Single Directory Workflow

## Overview

Screenalytics uses a simplified branching model with **one stable `main` branch** and **one active task branch** at a time. All development happens in a **single working directory** without git worktrees.

---

## Core Principles

### Single Working Directory

All development happens inside `/Volumes/HardDrive/SCREENALYTICS`.

**Do NOT use git worktrees in this workflow.** The single-directory approach keeps things simple:
- One location for all work
- No confusion about which directory has which changes
- Easier to track state

### One Branch Per Task

Each task gets exactly one branch. You keep working on that branch until it's ready for PR and merge to main.

---

## Branch Naming Convention

**Required pattern**: `screentime-improvements/<task-slug>`

### Naming Rules

| Rule | Example |
|------|---------|
| Lowercase only | `screentime-improvements/run-isolation` |
| Hyphen-separated | `screentime-improvements/face-alignment-qa` |
| Descriptive slug | `screentime-improvements/web-modal-dialogs` |
| No spaces/underscores | ~~`screentime-improvements/my_feature`~~ |

### Examples

```
screentime-improvements/run-isolation-screentime
screentime-improvements/web-modal-dialogs
screentime-improvements/workflow-docs-and-repo-structure
screentime-improvements/face-alignment-phase-b
```

---

## Task Lifecycle

### Stage 1: Start Task

```bash
# Switch to main and pull latest
git switch main
git pull --ff-only origin main

# Verify clean state
git status -sb

# Create the task branch
git switch -c screentime-improvements/<task-slug>

# Confirm you're on the new branch
git status -sb
```

### Stage 2: Work Loop (repeat as needed)

```bash
# Make changes, then stage and commit
git add <files>
git commit -m "<type>: <description>"

# Push regularly to keep remote updated
git push -u origin screentime-improvements/<task-slug>

# Run tests as appropriate
python -m py_compile <python_files>
pytest tests/ -v
```

**Commit frequently.** Small, focused commits are easier to review and debug.

### Stage 3: PR Stage

```bash
# Ensure all changes are committed and pushed
git status -sb
git push

# Verify branch exists on GitHub
gh pr list

# Open PR
gh pr create --base main --title "<title>" --body "<description>"
```

### Stage 4: Merge Stage

```bash
# After PR is approved and merged on GitHub:
git switch main
git pull --ff-only origin main

# Delete local branch
git branch -d screentime-improvements/<task-slug>

# Delete remote branch (if not auto-deleted by GitHub)
git push origin --delete screentime-improvements/<task-slug>
```

---

## Required Checkpoints

### Before Switching Branches

You **must** commit your work OR explicitly stash it:

```bash
# Option A: Commit (preferred)
git add .
git commit -m "wip: <what you were doing>"

# Option B: Stash (temporary only)
git stash push -m "switching to other work"
```

**Prefer commits over stash.** Stashed work can be lost or forgotten.

### Before Opening a PR

- [ ] All changes committed (`git status` shows clean)
- [ ] Branch pushed to origin (`git push`)
- [ ] Branch exists on GitHub (`gh pr list` or check web UI)
- [ ] Tests pass locally
- [ ] No merge conflicts with main

### Always Keep Main Clean

- Never commit directly to main
- Never force-push to main
- Always use PRs for merging

---

## Complete Command Reference

### Starting a New Task

```bash
git switch main
git pull --ff-only origin main
git status -sb
git switch -c screentime-improvements/<task-slug>
git status -sb
```

### Regular Work Cycle

```bash
# After making changes
git add <files>
git commit -m "<type>: <description>"
git push -u origin screentime-improvements/<task-slug>
```

### Opening a PR

```bash
git status -sb
git push
gh pr create --base main --title "<title>" --body "$(cat <<'EOF'
## Summary
- <bullet points>

## Test Plan
- [ ] <checklist>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

### After PR Merge

```bash
git switch main
git pull --ff-only origin main
git branch -d screentime-improvements/<task-slug>
```

---

## Handling Urgent Work

If you need to handle something urgent while a task is in progress:

1. **Commit current work** (don't leave uncommitted changes):
   ```bash
   git add .
   git commit -m "wip: pausing for urgent work"
   git push
   ```

2. **Switch to main and create urgent branch**:
   ```bash
   git switch main
   git pull --ff-only origin main
   git switch -c screentime-improvements/urgent-fix
   ```

3. **Complete and merge urgent work via PR**

4. **Return to original task**:
   ```bash
   git switch screentime-improvements/<original-task>
   git rebase origin/main  # incorporate any changes from main
   ```

---

## Prohibited Patterns

| Don't Do This | Why |
|---------------|-----|
| Create git worktrees | Adds complexity, causes confusion |
| Multiple long-lived branches | Hard to track, causes merge conflicts |
| Leave uncommitted changes when switching | Changes can be lost |
| Force-push to main | Destroys history |
| Merge without CI passing | Breaks main for everyone |
| Keep branches after merge | Clutters branch list |
| Branches older than 2 weeks | Stale branches cause merge hell |

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│ SCREENALYTICS WORKFLOW                                      │
├─────────────────────────────────────────────────────────────┤
│ Working Directory: /Volumes/HardDrive/SCREENALYTICS         │
│ Branch Pattern:    screentime-improvements/<task-slug>      │
├─────────────────────────────────────────────────────────────┤
│ START:  git switch main && git pull && git switch -c ...    │
│ WORK:   edit → git add → git commit → git push              │
│ PR:     gh pr create --base main                            │
│ DONE:   git switch main && git pull && git branch -d ...    │
└─────────────────────────────────────────────────────────────┘
```

# Branching Strategy: Main + 1 Feature Branch

## Overview

Screenalytics uses a simplified branching model with **one stable `main` branch** and **one active feature branch** at a time. This prevents branch sprawl and merge conflicts.

---

## Branch Types

### `main` (protected)
- Always deployable
- All CI tests must pass before merge
- Direct commits prohibited except for critical hotfixes

### `feature/<name>` (working branch)
- One active feature branch at a time
- Created from latest `main`
- Merged back to `main` when complete
- Deleted after merge

---

## Workflow

### Starting New Work

```bash
# 1. Ensure main is up to date
git checkout main
git pull origin main

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Work on feature, commit regularly
git add .
git commit -m "feat: description of change"

# 4. Push to remote
git push -u origin feature/my-feature
```

### Completing Work

```bash
# 1. Ensure main hasn't diverged
git fetch origin
git rebase origin/main  # or merge if preferred

# 2. Push final changes
git push origin feature/my-feature

# 3. Create PR and wait for CI
gh pr create --base main

# 4. After merge, delete feature branch
git checkout main
git pull origin main
git branch -d feature/my-feature
git push origin --delete feature/my-feature
```

---

## Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feature/<name>` | `feature/voice-clustering` |
| Bug fix | `fix/<name>` | `fix/embedding-cache` |
| Infrastructure | `infra/<name>` | `infra/docker-compose` |
| Experiment | `exp/<name>` | `exp/whisper-v3` |

---

## Pre-Flight Checklist (Before Starting New Feature)

Before creating a new feature branch, verify:

- [ ] All CI checks passing on main
- [ ] No pending PRs that should be merged first
- [ ] No uncommitted local changes
- [ ] Local main is synced with remote

```bash
# Quick pre-flight check
git fetch origin
git status
gh pr list --state open
gh run list --limit 5
```

---

## Handling Parallel Work

If you need to work on something else while a feature is in progress:

1. **Stash current work**: `git stash`
2. **Create hotfix from main**: `git checkout main && git checkout -b fix/urgent-fix`
3. **Complete and merge hotfix**
4. **Return to feature**: `git checkout feature/my-feature && git stash pop`
5. **Rebase on updated main**: `git rebase origin/main`

---

## Prohibited Patterns

- Multiple long-lived feature branches
- Branches older than 2 weeks without activity
- Force-pushing to main
- Merging without CI passing
- Keeping branches after merge

---

## Recovery: Branch Cleanup

If branches have accumulated, follow the cleanup process in [2025-12-main-promotion-plan.md](branching/2025-12-main-promotion-plan.md).

```bash
# List all branches with their status
for branch in $(git branch -r | grep -v HEAD | sed 's/origin\///'); do
  ahead=$(git rev-list --count main..origin/$branch 2>/dev/null || echo "?")
  behind=$(git rev-list --count origin/$branch..main 2>/dev/null || echo "?")
  echo "$branch: +$ahead -$behind"
done
```

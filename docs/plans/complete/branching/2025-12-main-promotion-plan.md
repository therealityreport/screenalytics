# Main Promotion Plan - December 2025

## Executive Summary

**Finding: `main` is already the most complete branch.**

- The legacy branch was only 1 commit ahead (a stripped Voices Review that would be destructive)
- `main` is 18 commits ahead of the legacy branch with all audio pipeline work, CI fixes, and the screen-time engine
- **Infra branches are OLD snapshots** that would DELETE 10,000-15,000 lines if merged

**Status: Cleanup complete. `main` is the canonical trunk.**

---

## Cleanup Completed (2025-12-03)

### Branches Deleted

| Branch | Local | Remote | Reason |
|--------|-------|--------|--------|
| `feature/audio-pipeline` | ✅ | ✅ | Fully merged |
| `nov-29-audio-pipeline` | ✅ | ✅ | Fully merged |
| `feature/screen-time-engine-refactor` | ✅ | ✅ | Cherry-picked to main |
| `feature/pyannote-official-workflow` | ✅ | N/A | Content on main |
| `legacy branch (retired)` | ✅ | ✅ | Would overwrite Voices Review |
| `nov-29-voices-review` | ✅ | ✅ | WIP superseded |
| `rescue/local-sync-2025-11-29` | ✅ | ✅ | Old snapshot |
| `review/codex-full-snapshot` | ✅ | ✅ | Just a review snapshot |

### Worktrees Cleaned

- `/private/tmp/faces-review` - removed
- `/private/tmp/api-contracts-and-events` - pruned
- `/private/tmp/upload-parity` - pruned

---

## Remaining Branches

### Safe to Delete (pending manual confirmation)

| Branch | Status | Notes |
|--------|--------|-------|
| `infra/ec2-bootstrap-main` | **OLD SNAPSHOT** | Would DELETE 14,979 lines. main already has all infra/* files. |
| `infra/ec2-bootstrap` | **OLD SNAPSHOT** | Would DELETE 10,373 lines. Subset of above. |

**WARNING:** Do NOT merge these via PR. They diverged before the audio pipeline work and would revert critical code. Delete them outright.

```bash
# When ready to delete:
git branch -D infra/ec2-bootstrap-main
git branch -D infra/ec2-bootstrap
git push origin --delete infra/ec2-bootstrap-main
# (infra/ec2-bootstrap may not be on remote)
```

### Keep for Reference (prototype work)

| Branch | Unique Work | Status |
|--------|-------------|--------|
| `feature/web-app/api-contracts-and-episode-events` | API error standardization for Next.js | Keep for future web app |
| `feature/web-app/upload-parity` | Upload improvements for Next.js | Keep for future web app |
| `feature/web-app/faces-review` | Faces Review in Next.js | Keep for future web app |

These branches contain prototype Next.js work. Keep them as reference for when the real web app is built.

---

## Current State

`main` is the canonical trunk containing:
- Full audio pipeline with pyannote voiceprint identification
- Complete 2786-line Voices Review UI with all features
- Word-level Smart Split feature
- Time-range cast assignment
- Screen-time engine (`py_screenalytics/pipeline/`)
- All CI fixes (ML tests, FFmpeg dev libs)
- Complete infra/* configs (systemd, nginx, docker)

---

## Next Steps

1. User to manually delete infra branches when ready
2. Start new feature work from `main` using single-branch workflow
3. Reference web-app/* branches when building Next.js app

See [BRANCHING_STRATEGY.md](../../../reference/branching/BRANCHING_STRATEGY.md) for the new workflow.

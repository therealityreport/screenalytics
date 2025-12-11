# Branch cleanup report (2025-12-11)

Repo: https://github.com/therealityreport/screenalytics
Generated: 2025-12-11T22:19:00Z

## Summary

- Total branches (including main): 10
- Buckets: DELETE_NOW=0, KEEP=10, MANUAL_REVIEW=0
- Deleted now (this run): 0

## Policy (safety rules)

1. Never delete `main` or any protected branch.
2. Never delete any branch with an open PR.
3. Auto-delete only branches with **ahead=0 vs `origin/main`** (no unique commits).
4. If **ahead>0**, do not auto-delete; classify as KEEP or MANUAL_REVIEW.

How ahead/behind is determined: local git counts `git rev-list --left-right --count origin/main...origin/<branch>` (ahead/behind).
How PR linkage is determined: `gh pr list --state all` and matching by `headRefName`.

## Inventory

| Branch | Last commit date (UTC) | SHA | Ahead/Behind vs main | PR | Protected | Bucket | Action |
|---|---:|---|---:|---|---:|---|---|
| chore/branch-hygiene | 2025-12-11 | 87fbd5cfd8c0 | 1/0 | OPEN #16 (https://github.com/therealityreport/screenalytics/pull/16) | no | KEEP | — |
| infra/claude-code-action | 2025-12-11 | 06b7030e7a68 | 1/0 | OPEN #15 (https://github.com/therealityreport/screenalytics/pull/15) | no | KEEP | — |
| main | 2025-12-11 | 9b18b5ece512 | 0/0 | none | no | KEEP | — |
| feature/web-app/faces-review | 2025-12-09 | bd6d1fa016b0 | 7/8 | none | no | KEEP | — |
| feature/web-app/episode-detail | 2025-12-09 | 4a424a64235f | 4/8 | OPEN #12 (https://github.com/therealityreport/screenalytics/pull/12) | no | KEEP | — |
| feature/web-app/episodes-list | 2025-12-09 | 1d0c06a7cbdd | 3/8 | none | no | KEEP | — |
| feature/web-app/upload-episode-hardening | 2025-12-09 | 897dc1cf109b | 2/8 | none | no | KEEP | — |
| feature/web-app/upload-parity | 2025-11-29 | 7cf521fb8c20 | 1/45 | none | no | KEEP | — |
| feature/web-app/api-contracts-and-episode-events | 2025-11-29 | 7c10a9740e9e | 1/45 | none | no | KEEP | — |
| feature/screenalytics-next-foundation | 2025-11-29 | 60e7cd33fbbd | 2/44 | none | no | KEEP | — |

## Manual review list

- (none)

## Observed remote deletions

The following remote branches were observed as deleted during `git fetch --prune` in this workspace session (they no longer exist on the remote):

- awesome-feynman (absent on remote)
- modest-shaw (absent on remote)
- screentime-improvements (absent on remote)
- feature/2025-12-engine-api-jobs (absent on remote)
- feature/2025-12-small-face-detection-fallback (absent on remote)

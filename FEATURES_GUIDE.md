# FEATURES_GUIDE.md — Screenalytics

Version: 1.0

## What is a Feature Sandbox?
`FEATURES/<name>/` is a temporary area for an in-progress module.

## Structure
- `src/` — throwaway implementation
- `tests/` — focused tests
- `docs/` — working notes (agents write here)
- `TODO.md` — status, owner, plan

## Rules
- TTL 30 days
- No imports from `FEATURES/**` in production paths
- Promotion requires tests, docs, config updates, and CI green

## Promotion
- Run `tools/promote-feature.py <name> --dest <final_path>`
- Add Acceptance row and mark ✅
- Agents auto-sync README/PRD/Architecture/Directory docs

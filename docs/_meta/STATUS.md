# Documentation Status Taxonomy

This repo uses a small set of status labels so docs stay **operationally useful** and less noisy.

## Statuses

- `complete` — Accurate and current; safe to treat as source of truth.
- `in_progress` — Actively being worked on; may be incomplete or evolving.
- `draft` — Early thinking; not ready for operational use.
- `outdated` — Known to be incorrect or misleading vs current code/behavior; should not be followed as-is.
- `superseded` — Replaced by a newer doc; keep only as a pointer for historical context.
- `archive` — Historical record; intentionally kept out of the main doc surface area.
- `unknown` — Not yet reviewed; treat as untrusted until triaged.

## Conventions

- Outdated/superseded docs should include a short banner at the top:
  - `Status: outdated; see <replacement>`
  - `Status: superseded; see <replacement>`
- When content is still valuable but not canonical, prefer moving it under `docs/plans/<status>/` or `docs/_archive/`.

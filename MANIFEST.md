# Screenalytics — Project Manifest

## 1. Origin
Successor to **Screenalyzer v2**.
Built to eliminate legacy path drift, silent errors, and configuration sprawl.
Mission: precise, modular, cloud-native analytics for cast presence, speech, and engagement in reality television.

---

## 2. Vision Statement
> “Every frame tells a story — Screenalytics quantifies it.”

Screenalytics measures and contextualizes on-screen presence using unified vision, speech, and metadata pipelines.
It powers insights, merch, and fan engagement across *The Reality Report* ecosystem.

---

## 3. Core Principles
1. **Central truth:** one artifact resolver, one schema per stage.
2. **Modular compute:** detection → tracking → identity → audio → fusion → export.
3. **Human-in-the-loop:** every auto decision overrideable and auditable.
4. **Agents over scripts:** automation via Codex + MCP, not ad-hoc jobs.
5. **Docs before code:** every module ships with usage notes and acceptance tests.

---

## 4. Architecture Snapshot
- **API:** FastAPI (typed endpoints, OpenAPI)
- **UI:** Next.js workspace (Shows / People / Episodes)
- **Workers:** Python pipelines (RetinaFace, ByteTrack, ArcFace, Pyannote, Whisper)
- **Storage:** S3-compatible object store
- **Database:** Postgres + pgvector
- **Agents:** Codex + MCP servers (screenalytics / storage / postgres)
- **Automation:** Zapier / n8n for exports + alerts

---

## 5. Immediate Deliverables
| Priority | Deliverable | Description |
|-----------|--------------|--------------|
| P0 | `README.md` | Public intro + quickstart |
| P0 | `PRD.md` | Product requirements |
| P1 | `SOLUTION_ARCHITECTURE.md` | System flow + data model |
| P1 | `DIRECTORY_STRUCTURE.md` | Folder map + promotion rules |
| P1 | `MASTER_TODO.md` | Global milestone tracker |
| P1 | `/config/` | Initial YAML + TOML templates |
| P2 | `/FEATURES/` | Empty staging zones |
| P2 | `/docs/` | Base doc set |
| P2 | `/agents/` & `/mcps/` | Agent + MCP scaffolds |

---

## 6. Naming Conventions
- **Repo name:** `screenalytics`
- **Package name:** `py-screenalytics`
- **CLI command (future):** `screenalytics run <episode>`
- **Database schema:** `screenalytics_prod`
- **Object storage prefix:** `screenalytics/<show>/<episode>/`

---

## 7. Governance
- Branch model: `main` (stable) + `feature/*` (active).
- Every PR references a checklist in `MASTER_TODO.md`.
- Promotion rules from `FEATURES/README.md`.
- CI gates enforce tests + config updates.

---

## 8. Milestone 0 — Repo Initialization
- [ ] Create MANIFEST.md (this file)
- [ ] Create README.md (public face)
- [ ] Create PRD.md
- [ ] Create SOLUTION_ARCHITECTURE.md
- [ ] Create DIRECTORY_STRUCTURE.md
- [ ] Commit baseline configs and .gitignore
- [ ] Initialize `db/migrations/0001_init_core.sql`
- [ ] Open first tracking issue “Initialize Repo Skeleton”

---

## 9. Tagline Ideas
- *“See every second.”*
- *“Screenalytics — Reality quantified.”*
- *“From frame to fame.”*

---

**Created:** _(today’s date)_
**Author:** The Reality Report Core Team
**Version:** 0.0.1 (Pre-init)

# Directory Structure — Screenalytics


**Quick Reference** | [Full Documentation →](../../architecture/directory_structure.md)

---

## Repository Layout

```
screenalytics/
├── apps/                        # Frontend + API (STABLE)
│   ├── api/                     # FastAPI backend
│   └── workspace-ui/            # Streamlit workspace UI
├── web/                         # Next.js prototype (event/division admin)
├── packages/                    # Shared Python/TS libs (STABLE)
├── db/                          # Migrations, views, seeds (VERSIONED)
├── config/                      # All YAML/TOML configs (VERSIONED)
│   └── pipeline/                # detection.yaml, tracking.yaml, etc.
├── FEATURES/                    # Experimental sandboxes (TTL: 30 days)
├── agents/                      # Codex/Claude automation (CONTROLLED)
├── mcps/                        # MCP servers (CONTROLLED)
├── docs/                        # Permanent documentation (STABLE)
│   ├── architecture/
│   ├── pipeline/
│   ├── reference/
│   └── ops/
├── infra/                       # Docker, IaC (SUPPORT)
├── tests/                       # Unit/integration/e2e (REQUIRED)
└── tools/                       # CLI utilities (SUPPORT)
```

---

## Key Concepts

### STABLE Paths
Production-ready code that has passed promotion gates. Changes require PR review, tests, and docs.
- `apps/`, `web/`, `packages/`, `db/`, `config/`, `docs/`

### FEATURES Sandboxes
Temporary experimental code (30-day TTL). **No production imports allowed.**
- Structure: `FEATURES/<name>/` with `src/`, `tests/`, `docs/`, `TODO.md`
- Promotion: `tools/promote-feature.py <name>`

### Import Policy
✅ **ALLOWED:** Production imports from `apps/`, `web/`, `packages/`
❌ **FORBIDDEN:** Production imports from `FEATURES/**` (CI enforced)

---

## Promotion Workflow

1. Develop in `FEATURES/<name>/`
2. Pass CI (tests + docs + lint)
3. Run `tools/promote-feature.py <name> --dest <path>`
4. Code moves to target path, docs updated automatically
5. CI verifies ACCEPTANCE_MATRIX.md entry exists

---

## Documentation

**For complete details, see:**

- **[Full Directory Structure](../../architecture/directory_structure.md)** — Detailed repo map, policies, promotion workflow
- **[FEATURES_GUIDE.md](FEATURES_GUIDE.md)** — Feature sandbox workflow
- **[ACCEPTANCE_MATRIX.md](ACCEPTANCE_MATRIX.md)** — Quality gates

---

**Maintained by:** Screenalytics Engineering

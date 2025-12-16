# SCREENALYTICS

Reality TV face tracking and screen time analytics platform.

## Project Overview

Screenalytics processes reality TV episodes through an ML pipeline to identify cast members, track their appearances, and calculate screen time metrics. The system supports manual curation through a Streamlit workspace UI.

**Pipeline Flow**: `detect → track → embed → cluster → analytics`

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `apps/api/` | FastAPI backend - routers, services, storage |
| `apps/workspace-ui/` | Streamlit UI - Faces Review, Smart Suggestions |
| `tools/` | CLI tools (`episode_run.py`) |
| `py_screenalytics/` | Core ML pipeline modules |
| `config/pipeline/` | YAML configs for thresholds/profiles |
| `data/` | Local manifests, faces, embeddings |
| `tests/` | pytest tests (ml/, api/, ui/) |

## Tech Stack

- **Runtime**: Python 3.11
- **API**: FastAPI with Pydantic models
- **UI**: Streamlit multi-page app
- **ML Models**:
  - RetinaFace (face detection)
  - ByteTrack (face tracking across frames)
  - ArcFace (512-d face embeddings)
- **Storage**: Local filesystem + S3/MinIO
- **Async Jobs**: Celery + Redis

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Episode** | Processing unit, e.g., `rhobh-s05e02` |
| **Track** | Detected face sequence across frames (track_id) |
| **Cluster/Identity** | Grouped tracks representing same person (cluster_id) |
| **Cast Member** | Linked identity with name/metadata (cast_id) |
| **Facebank** | Reference embeddings for known cast |
| **Singleton** | Cluster with only one track |

## Key Metrics

| Metric | What It Measures | Good Value |
|--------|-----------------|------------|
| Identity Similarity | Cluster cohesion (how similar faces are within cluster) | ≥0.75 |
| Cast Similarity | Match strength to facebank reference | ≥0.70 |
| Track Similarity | Frame-to-frame consistency within a track | ≥0.80 |
| Ambiguity | Gap between 1st and 2nd best match | ≥0.15 |
| Temporal Consistency | Appearance pattern over episode timeline | varies |

## Coding Conventions

### Python
- Type hints on public functions
- Pydantic/dataclasses for data structures
- No bare `except:` - catch specific exceptions
- Log at INFO/WARN/ERROR with context keys

### Error Handling
- Log errors with context, then surface via UI/API
- Never silently swallow exceptions
- Use `ApiResult` pattern for structured API responses
- Fail fast on missing models/configs

### Streamlit UI
- Call `init_page()` first (sets page config)
- Wrap heavy operations in `with st.spinner(...)`
- Namespace session state keys: `f"{ep_id}::key_name"`
- Unique widget keys to avoid duplicate ID errors
- Two-step confirmation for destructive actions

### API
- Pydantic models for request/response
- Explicit HTTP error codes
- New fields should be optional for backwards compat

### Testing
- `python -m py_compile <file>` for syntax check
- `pytest tests/ml/` for ML logic
- `pytest tests/api/` for API endpoints

## Safety Invariants

1. **Never overwrite manifests** without backup/versioning
2. **Preserve curated identities** unless explicitly reclustering
3. **No secrets in code/logs** - use env vars via config
4. **Atomic facebank updates** - emit refresh jobs after changes
5. **Unique widget keys** in Streamlit

## Working with Claude

### Pipeline Changes
1. Start in [tools/episode_run.py](tools/episode_run.py) - CLI entry point
2. Check [apps/api/services/](apps/api/services/) for business logic
3. Review [config/pipeline/*.yaml](config/pipeline/) for thresholds

### UI Changes
1. Check [apps/workspace-ui/pages/](apps/workspace-ui/pages/) for page files
2. Use patterns from [ui_helpers.py](apps/workspace-ui/ui_helpers.py)
3. Metrics rendering: [similarity_badges.py](apps/workspace-ui/similarity_badges.py)

### Debugging
- Look for `[PHASE]`, `[JOB]`, `[GUARDRAIL]` log markers
- Check manifests in `data/manifests/{ep_id}/`
- Review `track_metrics.json` for quality thresholds

### Common Tasks

| Task | Start Here |
|------|------------|
| Fix clustering | `apps/api/services/grouping.py` |
| Fix UI metrics | `apps/workspace-ui/similarity_badges.py` |
| Fix pipeline stage | `tools/episode_run.py`, then service file |
| Add API endpoint | `apps/api/routers/episodes.py` |
| Fix storage/S3 | `apps/api/services/storage.py` |

## Branch Strategy

See [docs/reference/branching/BRANCHING_STRATEGY.md](docs/reference/branching/BRANCHING_STRATEGY.md) for full details.

**Single Directory Workflow:**
- All work happens in `/Volumes/HardDrive/SCREENALYTICS`
- Do NOT use git worktrees
- `main` - stable, always deployable
- Branch pattern: `screentime-improvements/<task-slug>`
- One branch per task; commit frequently; push regularly
- Keep changes scoped; prefer small focused edits

## Running Tests

```bash
# Syntax check
python -m py_compile apps/api/routers/episodes.py

# ML tests
pytest tests/ml/ -v

# API tests
pytest tests/api/ -v

# All tests
pytest tests/ -v
```

## Related Documentation

- [AGENTS.md](AGENTS.md) - Agent behavior rules and review priorities
- [docs/CLAUDE-SETUP.md](docs/CLAUDE-SETUP.md) - Claude Code integration guide
- [config/pipeline/](config/pipeline/) - Pipeline configuration reference

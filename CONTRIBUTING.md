- Use `FEATURES/<name>` for all new work until promotion.
- Include tests and documentation before promoting any module.
- Follow `SETUP.md` to bootstrap your environment and run services locally.

## Dev loop (API + UI)

- Run `scripts/dev_auto.sh` from the repo root to start uvicorn with reload and the Streamlit UI (`apps/workspace-ui/streamlit_app.py`) with runOnSave (see `.streamlit/config.toml`, port 8505).
- Tail `api_server.log` to confirm API reloads when you edit files under `apps/`, `tools/`, or `config/` (data/.venv are excluded); streamlit prints “Rerunning...” in the terminal when pages refresh after a save.
- Stop with `Ctrl+C` when you are done; the script cleans up the background API process before exiting.

## Testing

### Overview

SCREENALYTICS has comprehensive test coverage across three categories:

1. **Unit Tests** (`tests/unit/`) - Fast, isolated tests for config resolution, profile validation, and core utilities
2. **ML Integration Tests** (`tests/ml/`) - End-to-end pipeline tests with metric validation against acceptance thresholds
3. **API Smoke Tests** (`tests/api/`) - HTTP-based workflow tests for the FastAPI endpoints

All ML integration tests assert against thresholds defined in [ACCEPTANCE_MATRIX.md](ACCEPTANCE_MATRIX.md).

### Running Tests Locally

#### Prerequisites

Install test dependencies:

```bash
pip install pytest pytest-timeout requests opencv-python-headless numpy pyyaml
```

#### Unit Tests (Fast)

Run configuration and profile resolution tests:

```bash
pytest tests/unit/ -v
```

These tests validate:
- Profile resolution order (explicit > env > profile > config > default)
- Pydantic validation for profile/device enums
- CLI command building with `--profile` flag
- Documentation cross-references

#### ML Integration Tests (Slow)

ML tests are **gated by the `RUN_ML_TESTS=1` environment variable** to prevent accidental long-running test execution.

Run all ML integration tests:

```bash
RUN_ML_TESTS=1 pytest tests/ml/ -v
```

Run specific test modules:

```bash
# Detect/track metrics validation (tracks_per_minute, short_track_fraction, id_switch_rate)
RUN_ML_TESTS=1 pytest tests/ml/test_detect_track_metrics.py -v

# Face embedding quality validation (unit-norm embeddings, quality gating)
RUN_ML_TESTS=1 pytest tests/ml/test_faces_embed_metrics.py -v

# Clustering validation (singleton_fraction, largest_cluster_fraction)
RUN_ML_TESTS=1 pytest tests/ml/test_cluster_metrics.py -v

# Episode cleanup workflow (before/after validation, dangling reference checks)
RUN_ML_TESTS=1 pytest tests/ml/test_episode_cleanup_metrics.py -v
```

**What These Tests Validate:**

| Test Module | Key Assertions | Thresholds |
|-------------|----------------|------------|
| `test_detect_track_metrics.py` | tracks_per_minute, short_track_fraction, id_switch_rate | ≤ 50, ≤ 0.30, ≤ 0.10 |
| `test_faces_embed_metrics.py` | Embedding unit-norm, quality gating active, max_crops_per_track | ±0.05 from 1.0, rejection_rate > 0 |
| `test_cluster_metrics.py` | Clustering with synthetic embeddings, threshold sensitivity | singleton_fraction ≤ 0.50, largest_cluster_fraction ≤ 0.60 |
| `test_episode_cleanup_metrics.py` | Full pipeline + cleanup, before/after metrics, no dangling track_id refs | Metrics improve or stay acceptable |

#### API Smoke Tests

Run end-to-end API workflow tests (starts API server, submits jobs via HTTP):

```bash
RUN_ML_TESTS=1 pytest tests/api/test_jobs_smoke.py -v
```

These tests:
- Start a local uvicorn server on port 8765
- Submit `detect_track_async` → `faces_embed_async` → `cluster_async` jobs
- Poll for completion and validate metrics are exposed in responses
- Test error handling (404 for missing jobs, 422 for invalid profiles)

#### Full Test Suite

Run everything (unit + ML + API):

```bash
RUN_ML_TESTS=1 pytest tests/ -v
```

### Continuous Integration

ML integration tests run in GitHub Actions on every push to `main` and on pull requests.

**Workflow:** [`.github/workflows/ml-tests.yml`](.github/workflows/ml-tests.yml)

**Matrix Strategy:**
- Python 3.10, 3.11, 3.12
- Ubuntu latest with ffmpeg and OpenCV dependencies

**Metric Gating:**
- Tests assert against [ACCEPTANCE_MATRIX.md](ACCEPTANCE_MATRIX.md) thresholds
- If any metric exceeds its threshold, the test **fails** and CI fails
- Test artifacts (video fixtures, manifests, track_metrics.json) are uploaded on failure

**Example Gating Logic:**
```python
# From test_detect_track_metrics.py
THRESHOLDS = {
    "cpu": {
        "tracks_per_minute": 50,  # Warning threshold
        "short_track_fraction": 0.30,
        "id_switch_rate": 0.10,
    }
}

assert metrics["tracks_per_minute"] <= THRESHOLDS["cpu"]["tracks_per_minute"]
assert metrics["short_track_fraction"] <= THRESHOLDS["cpu"]["short_track_fraction"]
assert metrics["id_switch_rate"] <= THRESHOLDS["cpu"]["id_switch_rate"]
```

**Viewing CI Results:**
- Check the "ML Integration Tests" job in GitHub Actions
- Failed tests will show which metric threshold was breached
- Download test artifacts to inspect `track_metrics.json` and fixture data

### Updating Acceptance Thresholds

If legitimate changes require adjusting metric thresholds:

1. Update [ACCEPTANCE_MATRIX.md](ACCEPTANCE_MATRIX.md) with new thresholds and justification
2. Update corresponding `THRESHOLDS` dict in test files:
   - `tests/ml/test_detect_track_metrics.py`
   - `tests/ml/test_faces_embed_metrics.py`
   - `tests/ml/test_cluster_metrics.py`
3. Document the change in your PR description

**Do not lower thresholds without strong justification** — metric regressions often indicate real quality issues.

### Test Architecture

**Synthetic Fixtures:**
- ML tests create synthetic videos using OpenCV with programmatic face-like regions
- Cluster tests use synthetic 512-dim embeddings with known structure (e.g., 5 identities × 4 tracks)
- Fixtures are reproducible (seeded random number generators)

**Full Pipeline Coverage:**
- `test_episode_cleanup_metrics.py` runs the full detect→track→faces→cluster→cleanup workflow
- Before/after validation ensures cleanup improves metrics without breaking references
- Dangling reference checks ensure all `track_id` refs in `faces.jsonl` and `identities.json` point to valid tracks

**API Testing:**
- `test_jobs_smoke.py` uses the FastAPI test client pattern with background uvicorn server
- Jobs are submitted via HTTP POST and polled via GET until terminal state
- Validates that metrics are correctly exposed in API responses

### Troubleshooting

**Tests Skipped:**
- If you see "set RUN_ML_TESTS=1 to run ML integration tests", ensure the environment variable is set
- Unit tests and profile resolution tests run regardless of `RUN_ML_TESTS`

**Timeout Errors:**
- ML tests have generous timeouts (180-900s depending on test)
- If tests timeout on slow hardware, consider using `--profile fast_cpu` in fixtures

**Import Errors:**
- Ensure `opencv-python-headless` (not `opencv-python`) is installed to avoid GUI dependencies
- Ensure `torch` is installed for model loading (CPU mode is forced in tests)

**Metric Threshold Failures:**
- Check which specific metric failed: `tracks_per_minute`, `short_track_fraction`, etc.
- Inspect `test_data/manifests/*/track_metrics.json` to see actual values
- Compare against thresholds in [ACCEPTANCE_MATRIX.md](ACCEPTANCE_MATRIX.md)

### Adding New Tests

When adding new pipeline stages or features:

1. **Add unit tests** to `tests/unit/` for configuration and validation logic
2. **Add integration tests** to `tests/ml/` with:
   - Synthetic fixture creation
   - Full pipeline execution
   - Metric assertions against ACCEPTANCE_MATRIX.md thresholds
3. **Update API smoke tests** if adding new endpoints
4. **Update this document** with new test commands and expected behavior

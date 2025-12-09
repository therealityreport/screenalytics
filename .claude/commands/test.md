Run tests for a specific area of the codebase.

## Usage

`/test [area]`

## Areas

| Area | Command | Description |
|------|---------|-------------|
| `ml` | `pytest tests/ml/ -v` | ML pipeline tests |
| `api` | `pytest tests/api/ -v` | API endpoint tests |
| `ui` | `pytest tests/ui/ -v` | UI component tests |
| `all` | `pytest tests/ -v` | All tests |
| `syntax` | `python -m py_compile` | Syntax check only |

## Examples

```bash
# Run ML tests
/test ml

# Run API tests
/test api

# Run all tests
/test all

# Syntax check specific files
/test syntax apps/api/routers/episodes.py apps/workspace-ui/pages/3_Faces_Review.py
```

## Quick Syntax Check

For rapid validation without full test suite:

```bash
python -m py_compile apps/api/routers/episodes.py
python -m py_compile apps/workspace-ui/pages/3_Faces_Review.py
python -m py_compile apps/workspace-ui/pages/3_Smart_Suggestions.py
```

## Test Markers

Some tests require specific infrastructure:

```bash
# Skip tests requiring S3
pytest tests/ -v -m "not requires_s3"

# Only fast tests
pytest tests/ -v -m "fast"
```

## Coverage

To run with coverage:

```bash
pytest tests/ --cov=apps --cov=py_screenalytics --cov-report=html
```

## Related

- `pipeline-debug` skill for runtime debugging
- `cluster-quality` skill for metrics analysis

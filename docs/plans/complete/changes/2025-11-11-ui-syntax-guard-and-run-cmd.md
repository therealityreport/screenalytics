# 2025-11-11 — UI syntax guard & run command fix

## Summary
- Removed a stray `*** End of File` marker from `apps/workspace-ui/streamlit_app.py` that broke Python parsing.
- Added `tests/ui/test_streamlit_syntax.py` to compile-check the Streamlit script each test run.
- README now defaults to `python -m uvicorn apps.api.main:app --reload` for the API, keeping `uv run ...` as an optional alternative.

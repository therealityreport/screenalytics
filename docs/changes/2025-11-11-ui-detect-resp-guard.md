# 2025-11-11 — UI detect_resp guard

## Summary
- Initialized `detect_resp`/`job_error` in the Streamlit uploader and only reference them when the "Run detect/track" checkbox is enabled, avoiding NameError paths.
- Added `tests/ui/test_streamlit_undefined_names.py` as an extra compile guard.
- API behavior unchanged; this is purely a UI robustness fix.

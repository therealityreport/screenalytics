# 2025-11-11 — API health endpoint & UI preflight

## Summary
- Added `GET /healthz` to the FastAPI app so external tools can quickly verify readiness alongside `/health`.
- Streamlit uploader now surfaces the API base URL in the sidebar, runs a `/healthz` preflight before enabling uploads, and propagates exact request URLs + response bodies on failure.
- README documents common connection errors and their fixes.

## Notes
- Run: `python -m uvicorn apps.api.main:app --reload` and `streamlit run apps/workspace-ui/streamlit_app.py`.
- If the UI reports "Health check failed", confirm `curl $SCREENALYTICS_API_URL/healthz` succeeds or update `.env`.

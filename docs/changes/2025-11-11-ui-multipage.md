# 2025-11-11 — Streamlit multipage UI

## Highlights
- Split the workspace UI into Streamlit's multipage layout: Upload & Run (home) plus Episodes, Episode Detail, Faces Review, Screentime, and Health pages with shared sidebar status.
- Added shared helpers for API calls, query-param based episode selection, and local path links so each page stays thin while reusing status/health logic.
- New UI panels let operators hydrate an episode from S3, re-run detect/track, queue faces/cluster/screentime jobs, and inspect local manifests/analytics without leaving the browser.

## Why it matters
As the upload helper grew past a single workflow, navigation became unwieldy. The multipage pattern mirrors how ops teams actually work: browse → inspect → rerun → review analytics, all while keeping the API contracts unchanged.

## How to use it
1. Launch Streamlit (`streamlit run apps/workspace-ui/streamlit_app.py`).
2. Use the sidebar to jump between pages; selections persist via the `ep_id` query param.
3. From Episode Detail, hydrate from S3, run detect/track, then head to Screentime or Faces Review for downstream analysis.

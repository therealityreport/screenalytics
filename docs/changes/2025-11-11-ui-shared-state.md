# 2025-11-11 — Streamlit helpers + shared ep_id state

## Highlights
- Added `apps/workspace-ui/ui_helpers.py` to centralize `/healthz` probing, API config, ep_id session/query syncing, and JSON helpers so every page shares the same context.
- All Streamlit pages (Upload, Episodes, Episode Detail, Faces Review, Screentime, Health) now call `helpers.init_page(...)`, read `helpers.get_ep_id()`, and make REST calls via `helpers.api_get/api_post`.
- Query parameters stay in sync (`?ep_id=...`), enabling deep links and consistent navigation across pages.

## Why it matters
Previously, each page reimplemented API calls and local state, leading to drift and broken deep links. The shared helper keeps the sidebar status consistent and guarantees every page sees the same episode selection without retyping IDs.

## How to use it
- Launch the UI (`streamlit run apps/workspace-ui/streamlit_app.py`).
- Uploading an episode automatically sets the global `ep_id`; navigating to Episodes or Episode Detail picks it up instantly.
- Bookmark `http://localhost:8501/?ep_id=<your-episode>` to jump straight into a specific episode context.

# 2025-11-11 — API health endpoint, UI preflight, and one-command dev

## Summary
- Added `GET /healthz` to the FastAPI app so the UI and scripts have a stable readiness probe.
- Streamlit uploader now shows the configured API base URL, runs a `/healthz` preflight before enabling the form, and surfaces full request URLs and response bodies on error.
- `scripts/dev.sh` (and `make dev`) spin up the API, wait for `/healthz`, and launch Streamlit in a single command.
- README “Upload via UI” now highlights the new runner, common connection issues, and path/working-directory gotchas.

## Usage
```bash
bash scripts/dev.sh
docker compose up  # (in another shell if you need MinIO/Postgres)
```

Troubleshooting hints now cover API connectivity and Streamlit path issues.

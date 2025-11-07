# Agents & MCPs Progress

## Current status
- Common `schemas.py` and `auth.py` modules created for reuse.
- `screenalytics`, `storage`, and `postgres` MCP servers expose CLI-friendly tools.
- Smoke tests cover CLI invocation to ensure JSON output before wiring into Codex playbooks.

## Next steps
- Implement actual DB queries/storage signing logic.
- Add feature playbooks that call these MCP tools during promotion.
- Document error handling and authentication upgrades.

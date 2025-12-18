#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_screenalytics.artifacts import ensure_dirs, get_path  # noqa: E402
from py_screenalytics import run_layout  # noqa: E402


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        return


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the body tracking stage for a run_id.")
    parser.add_argument("--ep-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--progress-file", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ep_id = str(args.ep_id).strip().lower()
    run_id = run_layout.normalize_run_id(str(args.run_id))
    progress_path = Path(str(args.progress_file))

    ensure_dirs(ep_id)
    video_path = get_path(ep_id, "video")
    if not video_path.exists():
        _write_progress(
            progress_path,
            {
                "ep_id": ep_id,
                "run_id": run_id,
                "phase": "body_tracking",
                "step": "done",
                "status": "error",
                "updated_at": _utcnow_iso(),
                "error": f"Video not found: {video_path}",
            },
        )
        return 2

    import_status: dict[str, Any] | None = None
    try:
        from py_screenalytics.env_diagnostics import (
            DEFAULT_ENV_PACKAGES,
            collect_env_diagnostics,
            write_env_diagnostics_json,
        )

        env_diag = collect_env_diagnostics(DEFAULT_ENV_PACKAGES)
        write_env_diagnostics_json(run_layout.run_root(ep_id, run_id) / "env_diagnostics.json", env_diag)
        import_status = env_diag.get("import_status") if isinstance(env_diag, dict) else None
    except Exception:
        import_status = None

    from tools import episode_run

    payload = episode_run._maybe_run_body_tracking(
        ep_id=ep_id,
        run_id=run_id,
        effective_run_id=run_id,
        video_path=video_path,
        import_status=import_status if isinstance(import_status, dict) else None,
    )

    success = isinstance(payload, dict) and str(payload.get("status") or "").strip().lower() == "success"
    _write_progress(
        progress_path,
        {
            "ep_id": ep_id,
            "run_id": run_id,
            "phase": "body_tracking",
            "step": "done",
            "status": "completed" if success else "error",
            "updated_at": _utcnow_iso(),
            "payload": payload,
        },
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())


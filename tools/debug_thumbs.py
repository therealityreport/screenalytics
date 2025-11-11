from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable

TRUTHY = {"1", "true", "yes", "on"}


class JsonlLogger:
    """Line-buffered logger for crop diagnostics."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # line buffering for tail -f
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)

    def __call__(self, payload: Dict[str, Any]) -> None:
        self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:
            # Closing failures aren't actionable; the log is best-effort.
            pass


class NullLogger:
    def __call__(self, payload: Dict[str, Any]) -> None:  # pragma: no cover - noop
        return None

    def close(self) -> None:  # pragma: no cover - noop
        return None


def debug_thumbs_enabled(flag: str | None = None) -> bool:
    if flag is None:
        flag = os.environ.get("DEBUG_THUMBS")
    return bool(flag) and flag.strip().lower() in TRUTHY


def init_debug_logger(ep_id: str, base_dir: str | Path, *, enabled: bool | None = None):
    """Return a JsonlLogger pointing at <base_dir>/crops_debug.jsonl when enabled."""
    if enabled is None:
        enabled = debug_thumbs_enabled()
    if not enabled:
        return NullLogger()
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    return JsonlLogger(base_path / "crops_debug.jsonl")


def summarize_log(path: str | Path) -> Dict[str, Any]:
    """Return aggregate counts for a crops_debug.jsonl file."""
    counts = Counter()
    total = 0
    file_path = Path(path)
    if not file_path.exists():
        return {"path": str(file_path), "total": 0, "errors": {}}
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            reason = payload.get("save_err") or payload.get("err_before_save")
            if reason:
                counts[str(reason)] += 1
    return {"path": str(file_path), "total": total, "errors": dict(counts)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize crops_debug.jsonl diagnostics")
    parser.add_argument("logfile", type=str, help="Path to crops_debug.jsonl")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    summary = summarize_log(args.logfile)
    errors = summary.get("errors", {})
    print(f"Log: {summary['path']}")
    print(f"Entries: {summary['total']}")
    for key, count in sorted(errors.items(), key=lambda item: (-item[1], item[0])):
        print(f"{key:>18}: {count}")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()

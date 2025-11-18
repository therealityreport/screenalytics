"""Runtime bridge to packages/py-screenalytics/artifacts.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PKG_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "py-screenalytics"
    / "artifacts.py"
)

if not _PKG_FILE.exists():
    raise FileNotFoundError(f"Artifact resolver not found at {_PKG_FILE}")

_spec = importlib.util.spec_from_file_location(
    "screenalytics_artifacts_impl", _PKG_FILE
)
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise ImportError("Unable to load screenalytics artifact resolver")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)  # type: ignore[arg-type]

# Re-export public helpers.
get_path = getattr(_module, "get_path")
ensure_dirs = getattr(_module, "ensure_dirs")

__all__ = ["get_path", "ensure_dirs"]

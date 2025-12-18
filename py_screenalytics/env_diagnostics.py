from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ENV_PACKAGES: tuple[str, ...] = (
    "supervision",
    "torchreid",
    "torch",
    "onnxruntime",
    "ultralytics",
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_venv_active() -> bool:
    base_prefix = getattr(sys, "base_prefix", None)
    if isinstance(base_prefix, str) and base_prefix and base_prefix != sys.prefix:
        return True
    return bool(os.environ.get("VIRTUAL_ENV"))


def _pip_version_best_effort() -> str | None:
    try:
        import pip  # type: ignore

        version = getattr(pip, "__version__", None)
        if isinstance(version, str) and version.strip():
            return version.strip()
    except Exception:
        pass

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return None

    text = (completed.stdout or "").strip()
    if not text:
        return None
    # Example: "pip 24.2 from ... (python 3.11)"
    parts = text.split()
    if len(parts) >= 2 and parts[0].lower() == "pip":
        return parts[1].strip() or None
    return None


def diagnose_import(module_name: str) -> dict[str, Any]:
    """Return deterministic import status for a module.

    Schema:
        {"status": "ok"|"missing"|"import_error", "version": str|None, "error": str|None}
    """
    name = (module_name or "").strip()
    if not name:
        return {"status": "import_error", "version": None, "error": "empty module name"}

    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        # If the missing name is the package itself, treat as missing. Otherwise it's a subdependency import error.
        missing_name = getattr(exc, "name", None)
        if missing_name == name:
            return {"status": "missing", "version": None, "error": str(exc)}
        return {"status": "import_error", "version": None, "error": str(exc)}
    except Exception as exc:
        return {"status": "import_error", "version": None, "error": f"{type(exc).__name__}: {exc}"}

    version = getattr(module, "__version__", None)
    if not isinstance(version, str) or not version.strip():
        version = None
    return {"status": "ok", "version": version, "error": None}


def collect_env_diagnostics(packages: Iterable[str] | None = None) -> dict[str, Any]:
    packages_list = list(packages) if packages is not None else list(DEFAULT_ENV_PACKAGES)
    import_status: dict[str, Any] = {}
    package_versions: dict[str, str | None] = {}
    for pkg in packages_list:
        status = diagnose_import(pkg)
        import_status[pkg] = status
        package_versions[pkg] = status.get("version")

    return {
        "generated_at": _utcnow_iso(),
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "sys_base_prefix": getattr(sys, "base_prefix", None),
        "venv_active": _is_venv_active(),
        "python_version": platform.python_version(),
        "pip_version": _pip_version_best_effort(),
        "package_versions": package_versions,
        "import_status": import_status,
    }


def write_env_diagnostics_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


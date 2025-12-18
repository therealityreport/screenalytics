from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata as importlib_metadata
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
        {
            "status": "ok"|"missing"|"import_error",
            "version": str|None,
            "error": str|None,
            "distribution": list[{"name": str, "version": str|None, "location": str|None}]|None,
        }
    """
    name = (module_name or "").strip()
    if not name:
        return {"status": "import_error", "version": None, "error": "empty module name", "distribution": None}

    top_level = name.split(".", 1)[0]
    dist_info: list[dict[str, Any]] | None = None
    try:
        packages = importlib_metadata.packages_distributions()
        dist_names = packages.get(top_level, []) if isinstance(packages, dict) else []
    except Exception:
        dist_names = []

    if isinstance(dist_names, list):
        for dist_name in dist_names:
            if not isinstance(dist_name, str) or not dist_name.strip():
                continue
            try:
                dist = importlib_metadata.distribution(dist_name.strip())
            except importlib_metadata.PackageNotFoundError:
                continue
            except Exception:
                continue
            dist_info = dist_info or []
            try:
                location = str(dist.locate_file(""))
            except Exception:
                location = None
            dist_info.append(
                {
                    "name": dist.metadata.get("Name") or dist_name.strip(),
                    "version": getattr(dist, "version", None),
                    "location": location,
                }
            )

    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        # If the missing name is the package itself, treat as missing. Otherwise it's a subdependency import error.
        missing_name = getattr(exc, "name", None)
        if missing_name == name:
            return {"status": "missing", "version": None, "error": str(exc), "distribution": dist_info}
        return {"status": "import_error", "version": None, "error": str(exc), "distribution": dist_info}
    except Exception as exc:
        return {
            "status": "import_error",
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
            "distribution": dist_info,
        }

    version = getattr(module, "__version__", None)
    if not isinstance(version, str) or not version.strip():
        version = None
    return {"status": "ok", "version": version, "error": None, "distribution": dist_info}


def collect_env_diagnostics(packages: Iterable[str] | None = None) -> dict[str, Any]:
    packages_list = list(packages) if packages is not None else list(DEFAULT_ENV_PACKAGES)
    import_status: dict[str, Any] = {}
    package_versions: dict[str, str | None] = {}
    for pkg in packages_list:
        status = diagnose_import(pkg)
        import_status[pkg] = status
        package_versions[pkg] = status.get("version")

    # TorchReID runtime requirements (body Re-ID). This intentionally checks the
    # submodule import path that must exist in vetted installs.
    for name in ("torchreid.utils",):
        status = diagnose_import(name)
        import_status[name] = status
        package_versions[name] = status.get("version")

    torchreid_state = import_status.get("torchreid")
    torchreid_utils_state = import_status.get("torchreid.utils")
    runtime_ok = False
    runtime_error: str | None = None
    utils_import_ok = isinstance(torchreid_utils_state, dict) and torchreid_utils_state.get("status") == "ok"
    if utils_import_ok:
        try:
            module = importlib.import_module("torchreid.utils")
            if getattr(module, "FeatureExtractor", None) is None:
                runtime_error = "torchreid.utils missing FeatureExtractor"
            else:
                runtime_ok = True
        except Exception as exc:  # pragma: no cover - best-effort runtime smoke test
            runtime_error = f"{type(exc).__name__}: {exc}"
    else:
        if isinstance(torchreid_utils_state, dict) and torchreid_utils_state.get("error"):
            runtime_error = str(torchreid_utils_state.get("error"))
        elif isinstance(torchreid_state, dict) and torchreid_state.get("error"):
            runtime_error = str(torchreid_state.get("error"))
        else:
            runtime_error = "torchreid.utils import failed"

    if isinstance(torchreid_state, dict):
        torchreid_state.setdefault("required_imports", {})
        if isinstance(torchreid_state["required_imports"], dict):
            torchreid_state["required_imports"]["torchreid.utils"] = torchreid_utils_state
        torchreid_state["runtime_ok"] = bool(runtime_ok)
        torchreid_state["runtime_error"] = runtime_error
        torchreid_state["torchreid_utils_import_ok"] = bool(utils_import_ok)

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

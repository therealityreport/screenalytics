from __future__ import annotations

import importlib
from typing import Any


def get_torchreid_feature_extractor() -> tuple[Any, str | None]:
    """Return (FeatureExtractor, torchreid_version) with compat import paths.

    Supports both:
    - torchreid.utils.FeatureExtractor
    - torchreid.reid.utils.FeatureExtractor
    """
    try:
        torchreid = importlib.import_module("torchreid")
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) == "torchreid":
            raise ImportError(
                "torchreid_missing: torchreid package required for body Re-ID (pip install -r requirements-ml.txt)"
            ) from exc
        raise ImportError(f"torchreid_import_error: {exc}") from exc
    except Exception as exc:
        raise ImportError(f"torchreid_import_error: {type(exc).__name__}: {exc}") from exc

    version = getattr(torchreid, "__version__", None)
    torchreid_version = version.strip() if isinstance(version, str) and version.strip() else None

    errors: list[str] = []
    for module_name in ("torchreid.utils", "torchreid.reid.utils"):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        except Exception as exc:
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue

        extractor = getattr(module, "FeatureExtractor", None)
        if extractor is not None:
            return extractor, torchreid_version
        errors.append(f"{module_name}: missing FeatureExtractor")

    details = "; ".join(errors) if errors else "unknown"
    raise ImportError(
        "torchreid_import_error: FeatureExtractor not found (tried torchreid.utils and torchreid.reid.utils); "
        f"{details}"
    )


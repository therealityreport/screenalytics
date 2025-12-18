from __future__ import annotations

import importlib
from typing import Any


def get_torchreid_feature_extractor() -> tuple[Any, str | None]:
    """Return (FeatureExtractor, torchreid_version) with compat import paths.

    Requires:
    - torchreid.utils.FeatureExtractor

    Note: The PyPI `torchreid` distribution (0.2.x) is known to be incompatible
    with the deep-person-reid module layout used by Screenalytics. Install the
    vetted deep-person-reid source distribution instead.
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

    try:
        module = importlib.import_module("torchreid.utils")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "torchreid_runtime_error: missing torchreid.utils (install deep-person-reid; "
            "pip torchreid==0.2.x is incompatible)"
        ) from exc
    except Exception as exc:
        raise ImportError(f"torchreid_runtime_error: {type(exc).__name__}: {exc}") from exc

    extractor = getattr(module, "FeatureExtractor", None)
    if extractor is None:
        raise ImportError("torchreid_runtime_error: torchreid.utils missing FeatureExtractor")
    return extractor, torchreid_version

"""Workspace Upload page wrapper for sidebar navigation."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure workspace modules are importable when this page is loaded directly
PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

# Import the canonical upload page (top-level Upload_Video.py).
# All Streamlit rendering happens inside that module.
import Upload_Video  # noqa: F401  # type: ignore


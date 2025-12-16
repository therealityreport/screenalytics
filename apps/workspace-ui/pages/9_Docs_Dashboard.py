from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402


def _repo_root() -> Path:
    # apps/workspace-ui/pages/<page>.py -> apps/workspace-ui/pages -> apps/workspace-ui -> apps -> repo root
    return PAGE_PATH.parents[3]


def _feature_present_in_ui(feature_id: str, docs: list[dict[str, Any]]) -> bool:
    for doc in docs:
        if feature_id not in (doc.get("features") or []):
            continue
        surfaces = doc.get("ui_surfaces_expected") or []
        if any(isinstance(surface, str) and surface.startswith("workspace-ui:") for surface in surfaces):
            return True
    return False


def _feature_present_in_code(feature: dict[str, Any]) -> tuple[bool, list[str]]:
    repo_root = _repo_root()
    paths_expected = feature.get("paths_expected") or []
    if not isinstance(paths_expected, list):
        return False, ["<invalid paths_expected>"]

    missing: list[str] = []
    for expected in paths_expected:
        if not isinstance(expected, str) or not expected.strip():
            continue
        if not (repo_root / expected).exists():
            missing.append(expected)
    return len(missing) == 0, missing


cfg = helpers.init_page("Docs Dashboard")
helpers.render_page_header("workspace-ui:9_Docs_Dashboard", "Docs Dashboard")
st.caption("Read-only view of docs + feature coverage (driven by `docs/_meta/docs_catalog.json`).")

catalog, error = helpers.load_docs_catalog()
if error:
    st.error(error)
    st.info("Merge the docs catalog PR (adds `docs/_meta/docs_catalog.json`) to enable this dashboard.")
else:
    assert catalog is not None
    docs = [d for d in catalog.get("docs", []) if isinstance(d, dict)]
    features = catalog.get("features") or {}
    if not isinstance(features, dict):
        features = {}

    st.subheader("Docs")
    status_options = sorted({str(d.get("status")) for d in docs if d.get("status")})
    type_options = sorted({str(d.get("type")) for d in docs if d.get("type")})

    filter_cols = st.columns([2, 2, 4], vertical_alignment="bottom")
    with filter_cols[0]:
        statuses = st.multiselect("Status", options=status_options, default=status_options)
    with filter_cols[1]:
        types = st.multiselect("Type", options=type_options, default=type_options)
    with filter_cols[2]:
        query = st.text_input("Search", value="", placeholder="title / path / tags / features")

    query_norm = query.strip().lower()
    filtered_docs: list[dict[str, Any]] = []
    for doc in docs:
        if statuses and str(doc.get("status")) not in statuses:
            continue
        if types and str(doc.get("type")) not in types:
            continue
        if query_norm:
            haystack = " ".join(
                [
                    str(doc.get("title") or ""),
                    str(doc.get("path") or ""),
                    " ".join(t for t in (doc.get("tags") or []) if isinstance(t, str)),
                    " ".join(f for f in (doc.get("features") or []) if isinstance(f, str)),
                ]
            ).lower()
            if query_norm not in haystack:
                continue
        filtered_docs.append(doc)

    m_cols = st.columns(4)
    m_cols[0].metric("Total docs", len(docs))
    m_cols[1].metric("Filtered", len(filtered_docs))
    todo_count = sum(1 for d in docs if d.get("status") in {"in_progress", "draft", "outdated"})
    m_cols[2].metric("To-do docs", todo_count)
    m_cols[3].metric("Features", len(features))

    rows = [
        {
            "title": d.get("title", ""),
            "status": d.get("status", ""),
            "type": d.get("type", ""),
            "last_updated": d.get("last_updated", ""),
            "path": d.get("path", ""),
            "features": ", ".join(f for f in (d.get("features") or []) if isinstance(f, str)),
            "ui_surfaces_expected": ", ".join(s for s in (d.get("ui_surfaces_expected") or []) if isinstance(s, str)),
        }
        for d in filtered_docs
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Feature Coverage")
    feature_rows: list[dict[str, Any]] = []
    for feature_id, feature in features.items():
        if not isinstance(feature_id, str) or not isinstance(feature, dict):
            continue

        present_in_code, missing_paths = _feature_present_in_code(feature)
        present_in_ui = _feature_present_in_ui(feature_id, docs)
        phases = feature.get("phases") or {}
        phase_summary = ", ".join([f"{k}:{v}" for k, v in phases.items()]) if isinstance(phases, dict) else ""

        feature_rows.append(
            {
                "feature": feature.get("title") or feature_id,
                "id": feature_id,
                "status": feature.get("status") or "unknown",
                "present_in_code": present_in_code,
                "present_in_ui": present_in_ui,
                "missing_paths": ", ".join(missing_paths),
                "phases": phase_summary,
            }
        )
    feature_rows.sort(key=lambda r: str(r.get("feature", "")))
    st.dataframe(feature_rows, use_container_width=True, hide_index=True)

    st.caption("This page is read-only; edits must be made in the repo (`docs/_meta/docs_catalog.json`).")


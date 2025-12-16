#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALLOWED_DOC_STATUSES = {
    "complete",
    "in_progress",
    "draft",
    "outdated",
    "superseded",
    "archive",
    "unknown",
}


@dataclass(frozen=True)
class CatalogDoc:
    doc_id: str
    title: str
    path: str
    status: str
    last_updated: str
    doc_type: str
    tags: list[str]
    features: list[str]
    models: list[str]
    jobs: list[str]
    ui_surfaces_expected: list[str]


def _as_str_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f'Field "{field_name}" must be a list[str].')
    return value


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Catalog file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _parse_doc(doc: dict[str, Any]) -> CatalogDoc:
    for field in (
        "id",
        "title",
        "path",
        "status",
        "last_updated",
        "type",
        "tags",
        "features",
        "models",
        "jobs",
        "ui_surfaces_expected",
    ):
        if field not in doc:
            raise ValueError(f'Missing required doc field "{field}".')

    doc_id = doc["id"]
    title = doc["title"]
    path = doc["path"]
    status = doc["status"]
    last_updated = doc["last_updated"]
    doc_type = doc["type"]

    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError('Field "id" must be a non-empty string.')
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f'Doc "{doc_id}": field "title" must be a non-empty string.')
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f'Doc "{doc_id}": field "path" must be a non-empty string.')
    if not isinstance(status, str) or status not in ALLOWED_DOC_STATUSES:
        raise ValueError(
            f'Doc "{doc_id}": field "status" must be one of {sorted(ALLOWED_DOC_STATUSES)}.'
        )
    if not isinstance(last_updated, str) or not last_updated.strip():
        raise ValueError(
            f'Doc "{doc_id}": field "last_updated" must be a non-empty string (or "unknown").'
        )
    if not isinstance(doc_type, str) or not doc_type.strip():
        raise ValueError(f'Doc "{doc_id}": field "type" must be a non-empty string.')

    tags = _as_str_list(doc["tags"], field_name="tags")
    features = _as_str_list(doc["features"], field_name="features")
    models = _as_str_list(doc["models"], field_name="models")
    jobs = _as_str_list(doc["jobs"], field_name="jobs")
    ui_surfaces_expected = _as_str_list(
        doc["ui_surfaces_expected"], field_name="ui_surfaces_expected"
    )

    return CatalogDoc(
        doc_id=doc_id.strip(),
        title=title.strip(),
        path=path.strip(),
        status=status,
        last_updated=last_updated.strip(),
        doc_type=doc_type.strip(),
        tags=[tag.strip() for tag in tags if tag.strip()],
        features=[feature.strip() for feature in features if feature.strip()],
        models=[model.strip() for model in models if model.strip()],
        jobs=[job.strip() for job in jobs if job.strip()],
        ui_surfaces_expected=[surface.strip() for surface in ui_surfaces_expected if surface.strip()],
    )


def validate_catalog(*, repo_root: Path, catalog_path: Path) -> int:
    data = _load_json(catalog_path)

    if not isinstance(data, dict):
        raise ValueError("Catalog JSON must be a top-level object.")

    for field in ("version", "generated_at", "features", "docs"):
        if field not in data:
            raise ValueError(f'Missing top-level field "{field}".')

    if not isinstance(data["docs"], list) or any(not isinstance(item, dict) for item in data["docs"]):
        raise ValueError('Top-level field "docs" must be a list[object].')
    if not isinstance(data["features"], dict) or any(
        not isinstance(key, str) or not isinstance(value, dict) for key, value in data["features"].items()
    ):
        raise ValueError('Top-level field "features" must be an object keyed by feature id.')

    docs: list[CatalogDoc] = [_parse_doc(item) for item in data["docs"]]

    seen_ids: set[str] = set()
    for doc in docs:
        if doc.doc_id in seen_ids:
            raise ValueError(f'Duplicate doc id: "{doc.doc_id}".')
        seen_ids.add(doc.doc_id)

        resolved = repo_root / doc.path
        if not resolved.exists():
            raise ValueError(f'Doc "{doc.doc_id}": path does not exist: {doc.path}')
        if not resolved.is_file():
            raise ValueError(f'Doc "{doc.doc_id}": path is not a file: {doc.path}')

        for feature in doc.features:
            if feature not in data["features"]:
                raise ValueError(
                    f'Doc "{doc.doc_id}": unknown feature "{feature}" (missing from catalog.features).'
                )

    for feature_id, feature in data["features"].items():
        for field in ("title", "status", "paths_expected", "phases", "pending"):
            if field not in feature:
                raise ValueError(f'Feature "{feature_id}": missing field "{field}".')
        if not isinstance(feature["title"], str) or not feature["title"].strip():
            raise ValueError(f'Feature "{feature_id}": field "title" must be a non-empty string.')
        if not isinstance(feature["status"], str) or not feature["status"].strip():
            raise ValueError(f'Feature "{feature_id}": field "status" must be a non-empty string.')
        if not isinstance(feature["phases"], dict):
            raise ValueError(f'Feature "{feature_id}": field "phases" must be an object.')
        _as_str_list(feature["paths_expected"], field_name=f'features.{feature_id}.paths_expected')
        _as_str_list(feature["pending"], field_name=f'features.{feature_id}.pending')

    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate docs/_meta/docs_catalog.json.")
    parser.add_argument(
        "--catalog",
        default="docs/_meta/docs_catalog.json",
        help='Path to catalog JSON (default: "docs/_meta/docs_catalog.json").',
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    catalog_path = (repo_root / args.catalog).resolve()

    try:
        return validate_catalog(repo_root=repo_root, catalog_path=catalog_path)
    except Exception as exc:
        print(f"[docs_catalog] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


"""One-time migration: move legacy run exports to the canonical S3 layout.

Legacy export layout (no longer written for parsed ep_ids):
    runs/{ep_id}/{run_id}/exports/...

Canonical export layout:
    runs/{show}/s{ss}/e{ee}/{run_id}/exports/...

Usage:
    python -m tools.migrate_legacy_run_exports --ep-id rhoslc-s06e11 --run-id <run_id> --apply
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CopyPlan:
    source_key: str
    dest_key: str


def _infer_content_type(key: str) -> str | None:
    lower = (key or "").lower()
    if lower.endswith(".pdf"):
        return "application/pdf"
    if lower.endswith(".zip"):
        return "application/zip"
    if lower.endswith(".json"):
        return "application/json"
    if lower.endswith(".jsonl") or lower.endswith(".ndjson"):
        return "application/x-ndjson"
    return None


def _build_plan(*, legacy_exports_prefix: str, canonical_exports_prefix: str, legacy_keys: list[str]) -> list[CopyPlan]:
    plans: list[CopyPlan] = []
    for src in legacy_keys:
        if not src.startswith(legacy_exports_prefix):
            continue
        rel = src[len(legacy_exports_prefix) :]
        if not rel:
            continue
        plans.append(CopyPlan(source_key=src, dest_key=f"{canonical_exports_prefix}{rel}"))
    return plans


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy run exports to canonical S3 key layout.")
    parser.add_argument("--ep-id", required=True, help="Episode id, e.g. rhoslc-s06e11")
    parser.add_argument("--run-id", required=True, help="Run id (UUID hex)")
    parser.add_argument("--max-items", type=int, default=200, help="Max export objects to scan under legacy prefix")
    parser.add_argument("--apply", action="store_true", help="Perform copy (default: dry-run)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination keys if present")
    parser.add_argument(
        "--delete-legacy",
        action="store_true",
        help="Delete legacy exports prefix after successful copy (dangerous; requires --apply)",
    )
    args = parser.parse_args(argv)

    from apps.api.services.storage import StorageService
    from py_screenalytics import run_layout

    storage = StorageService()
    if not storage.s3_enabled():
        print("ERROR: Storage backend is not S3/MinIO (or client not configured); nothing to migrate.", file=sys.stderr)
        return 2

    layout = run_layout.get_run_s3_layout(args.ep_id, args.run_id)
    if layout.canonical_prefix is None:
        print(
            f"ERROR: ep_id={args.ep_id!r} is not parseable, canonical prefix unavailable; refusing migration.",
            file=sys.stderr,
        )
        return 2

    legacy_exports_prefix = f"{layout.legacy_prefix}exports/"
    canonical_exports_prefix = f"{layout.canonical_prefix}exports/"
    if legacy_exports_prefix == canonical_exports_prefix:
        print("Legacy and canonical exports prefixes are the same; nothing to migrate.")
        return 0

    legacy_keys = storage.list_objects(legacy_exports_prefix, max_items=max(0, int(args.max_items)))
    if not legacy_keys:
        print(f"No legacy exports found under s3://{storage.bucket}/{legacy_exports_prefix}")
        return 0

    plan = _build_plan(
        legacy_exports_prefix=legacy_exports_prefix,
        canonical_exports_prefix=canonical_exports_prefix,
        legacy_keys=legacy_keys,
    )
    if not plan:
        print("No export objects found to migrate (unexpected key format).")
        return 0

    print(f"S3 bucket: {storage.bucket}")
    print(f"Legacy exports prefix:    {legacy_exports_prefix}")
    print(f"Canonical exports prefix: {canonical_exports_prefix}")
    print(f"Objects to consider: {len(plan)} (dry-run={not args.apply})")

    copied = 0
    skipped = 0
    failed = 0
    for item in plan:
        if not args.overwrite and storage.object_exists(item.dest_key):
            skipped += 1
            print(f"SKIP exists: {item.dest_key}")
            continue

        if not args.apply:
            print(f"WOULD COPY: {item.source_key} -> {item.dest_key}")
            continue

        payload = storage.download_bytes(item.source_key)
        if payload is None:
            failed += 1
            print(f"FAIL download: {item.source_key}", file=sys.stderr)
            continue

        content_type = _infer_content_type(item.dest_key)
        ok = storage.upload_bytes(payload, item.dest_key, content_type=content_type)
        if not ok:
            failed += 1
            print(f"FAIL upload: {item.dest_key}", file=sys.stderr)
            continue

        copied += 1
        print(f"COPIED: {item.source_key} -> {item.dest_key}")

    print(f"Done. copied={copied} skipped={skipped} failed={failed}")

    if failed == 0 and args.apply and args.delete_legacy:
        deleted = storage.delete_prefix(legacy_exports_prefix)
        print(f"Deleted legacy exports prefix objects: {deleted}")
    elif args.delete_legacy and not args.apply:
        print("NOTE: --delete-legacy ignored without --apply", file=sys.stderr)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


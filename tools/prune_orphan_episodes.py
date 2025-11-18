#!/usr/bin/env python
"""Remove orphaned episode assets from S3 and local storage.

This utility scans the v2 S3 layout for uploaded episodes that no longer
exist in the EpisodeStore (the ones that surface in the UI with a ⚠ badge).
It can operate in a dry-run mode to preview deletions or actually remove
both remote and local artifacts.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List

from apps.api.services.episodes import EpisodeStore
from apps.api.services.storage import (
    StorageService,
    delete_local_tree,
    delete_s3_prefix,
    episode_context_from_id,
    v2_artifact_prefixes,
)
from py_screenalytics.artifacts import get_path


LOGGER = logging.getLogger("prune_orphan_episodes")
DATA_ROOT = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()


def _local_targets(ep_id: str) -> List[Path]:
    """Return unique local directories tied to an episode."""

    dirs = [
        get_path(ep_id, "video").parent,
        get_path(ep_id, "detections").parent,
        get_path(ep_id, "tracks").parent,
        get_path(ep_id, "frames_root"),
        DATA_ROOT / "analytics" / ep_id,
    ]
    seen: set[Path] = set()
    unique: List[Path] = []
    for path in dirs:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _delete_local_assets(ep_id: str, apply_changes: bool) -> int:
    deleted = 0
    for path in _local_targets(ep_id):
        if not path.exists():
            continue
        if not apply_changes:
            LOGGER.info("[dry-run] would delete local %s", path)
            deleted += 1
            continue
        delete_local_tree(path)
        LOGGER.info("Deleted local %s", path)
        deleted += 1
    return deleted


def _delete_s3_assets(
    ep_id: str,
    storage: StorageService,
    *,
    delete_artifacts: bool,
    delete_raw: bool,
    apply_changes: bool,
) -> int:
    try:
        ep_ctx = episode_context_from_id(ep_id)
    except ValueError as exc:
        LOGGER.warning("Skipping S3 cleanup for %s: %s", ep_id, exc)
        return 0

    prefixes = v2_artifact_prefixes(ep_ctx)
    keys: List[str] = []
    if delete_artifacts:
        keys.extend(
            filter(
                None,
                [
                    prefixes.get("frames"),
                    prefixes.get("crops"),
                    prefixes.get("manifests"),
                    prefixes.get("analytics"),
                    prefixes.get("thumbs_tracks"),
                    prefixes.get("thumbs_identities"),
                ],
            )
        )
    if delete_raw:
        keys.extend(filter(None, [prefixes.get("raw_v2"), prefixes.get("raw_v1")]))

    deleted_objects = 0
    for prefix in keys:
        if not apply_changes:
            LOGGER.info(
                "[dry-run] would delete S3 prefix s3://%s/%s", storage.bucket, prefix
            )
            continue
        deleted_objects += delete_s3_prefix(storage.bucket, prefix, storage=storage)
        LOGGER.info("Deleted S3 prefix s3://%s/%s", storage.bucket, prefix)
    return deleted_objects


def _collect_candidates(
    store: EpisodeStore,
    storage: StorageService,
    *,
    explicit: Iterable[str] | None,
    limit: int,
) -> List[str]:
    if explicit:
        candidates = sorted({ep.strip() for ep in explicit if ep and ep.strip()})
        return [ep for ep in candidates]

    items = storage.list_episode_videos_s3(limit=limit)
    ep_ids = []
    for obj in items:
        ep_id = obj.get("ep_id") if isinstance(obj, dict) else None
        if not isinstance(ep_id, str) or not ep_id:
            continue
        ep_id = ep_id.strip()
        if not ep_id or store.exists(ep_id):
            continue
        ep_ids.append(ep_id)
    return sorted(set(ep_ids))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ep-id",
        action="append",
        help="Specific ep_id to prune (can be repeated). When omitted all orphaned uploads are targeted.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum S3 objects to scan when auto-discovering (default: 5000)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform deletions instead of printing a dry-run plan",
    )
    parser.add_argument(
        "--no-local", action="store_true", help="Skip local filesystem cleanup"
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Skip deleting S3 artifact prefixes (frames/crops/manifests)",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Skip deleting the raw video prefixes from S3",
    )
    parser.add_argument(
        "--include-tracked",
        action="store_true",
        help="Also prune episodes that still exist in the EpisodeStore (default skips tracked episodes)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    store = EpisodeStore()
    storage = StorageService()

    candidates = _collect_candidates(
        store, storage, explicit=args.ep_id, limit=args.limit
    )
    if not candidates:
        LOGGER.info("No orphan episodes found.")
        return 0

    LOGGER.info(
        "%s %d candidate episode(s): %s",
        "Deleting" if args.apply else "Dry-run for",
        len(candidates),
        ", ".join(candidates),
    )

    total_local = 0
    total_remote = 0
    for ep_id in candidates:
        tracked = store.exists(ep_id)
        if tracked and not args.include_tracked:
            LOGGER.info("Skipping %s (still tracked in EpisodeStore)", ep_id)
            continue
        LOGGER.info("==> %s", ep_id)
        if not args.no_local:
            total_local += _delete_local_assets(ep_id, args.apply)
        else:
            LOGGER.info("Skipping local cleanup for %s", ep_id)
        if args.no_artifacts and args.no_raw:
            LOGGER.info("Skipping S3 cleanup for %s", ep_id)
        else:
            total_remote += _delete_s3_assets(
                ep_id,
                storage,
                delete_artifacts=not args.no_artifacts,
                delete_raw=not args.no_raw,
                apply_changes=args.apply,
            )

    LOGGER.info(
        "%s complete. Local targets: %d, S3 objects deleted: %d",
        "Deletion" if args.apply else "Dry-run",
        total_local,
        total_remote,
    )
    if not args.apply:
        LOGGER.info("Re-run with --apply once you're satisfied with the plan above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

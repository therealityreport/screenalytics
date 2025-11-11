#!/usr/bin/env python
"""Copy episode videos from v1 layout (raw/videos/{ep_id}/episode.mp4) to v2 layout."""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from apps.api.services.episodes import EpisodeStore
from apps.api.services.storage import StorageService


def _parse_ep_id(ep_id: str) -> Optional[tuple[str, int, int]]:
    parts = ep_id.split("-s")
    if len(parts) != 2 or "e" not in parts[1]:
        return None
    show = parts[0]
    season_part, episode_part = parts[1].split("e", 1)
    try:
        season = int(season_part)
        episode = int(episode_part)
    except ValueError:
        return None
    return show, season, episode


def migrate(dry_run: bool = True) -> None:
    store = EpisodeStore()
    storage = StorageService()
    items = storage.list_episode_videos_s3()
    client = storage._client
    if client is None:
        print("Storage client unavailable; ensure STORAGE_BACKEND is s3/minio", file=sys.stderr)
        return

    for item in items:
        if item.get("key_version") != "v1":
            continue
        ep_id = item["ep_id"]
        record = store.get(ep_id)
        if record:
            show = record.show_ref
            season = record.season_number
            episode = record.episode_number
        else:
            parsed = _parse_ep_id(ep_id)
            if not parsed:
                print(f"Skipping {ep_id}: unable to derive show/season/episode")
                continue
            show, season, episode = parsed
        src_key = item["key"]
        dest_key = storage.video_object_key_v2(show, season, episode)
        if storage.object_exists(dest_key):
            print(f"Skip {ep_id}: dest already exists ({dest_key})")
            continue
        print(f"Copy {src_key} → {dest_key}")
        if dry_run:
            continue
        client.copy_object(Bucket=storage.bucket, CopySource={"Bucket": storage.bucket, "Key": src_key}, Key=dest_key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate S3 episode videos from v1 to v2 layout")
    parser.add_argument("--apply", action="store_true", help="Perform copy instead of dry-run")
    args = parser.parse_args()
    migrate(dry_run=not args.apply)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

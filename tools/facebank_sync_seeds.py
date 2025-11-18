"""Backfill facebank seeds to S3 artifacts."""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Iterable, List

from apps.api.services.facebank import FacebankService
from apps.api.services.storage import StorageService


def _candidate_paths(base_root: Path, value: str) -> List[Path]:
    path = Path(value)
    candidates = [path]
    if not path.is_absolute():
        candidates.append((base_root / value).resolve())
    return candidates


def _locate_image(fs: FacebankService, uri: str) -> Path | None:
    for candidate in _candidate_paths(Path.cwd(), uri):
        if candidate.exists():
            return candidate
    for candidate in _candidate_paths(fs.data_root, uri):
        if candidate.exists():
            return candidate
    return None


def backfill_facebank_seeds(
    show_id: str,
    *,
    cast_id: str | None = None,
    dry_run: bool = False,
) -> Dict[str, int]:
    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
    fs = FacebankService(data_root)
    storage = StorageService()
    stats = {"updated": 0, "skipped": 0, "failed": 0}

    show_dir = fs.facebank_dir / show_id
    if not show_dir.exists():
        raise SystemExit(f"Show '{show_id}' has no facebank data at {show_dir}")

    cast_dirs: Iterable[Path]
    if cast_id:
        cast_dirs = [show_dir / cast_id]
    else:
        cast_dirs = sorted(path for path in show_dir.iterdir() if path.is_dir())

    for cast_path in cast_dirs:
        cid = cast_path.name
        data = fs._load_facebank(show_id, cid)
        seeds = data.get("seeds", [])
        changed = False
        for seed in seeds:
            existing_key = seed.get("display_s3_key") or seed.get("image_s3_key")
            if existing_key:
                stats["skipped"] += 1
                continue
            image_uri = seed.get("image_uri")
            if not image_uri:
                stats["failed"] += 1
                print(
                    f"[WARN] {show_id}/{cid} seed {seed.get('fb_id')} missing image path",
                    file=sys.stderr,
                )
                continue
            image_path = _locate_image(fs, image_uri)
            if not image_path:
                stats["failed"] += 1
                print(
                    f"[WARN] {show_id}/{cid} seed {seed.get('fb_id')} image not found: {image_uri}",
                    file=sys.stderr,
                )
                continue
            if dry_run:
                stats["updated"] += 1
                print(
                    f"[DRY-RUN] would upload {image_path} for {show_id}/{cid}/{seed['fb_id']}"
                )
                continue
            key = storage.upload_facebank_seed(
                show_id,
                cid,
                seed.get("fb_id", "seed"),
                image_path,
            )
            if not key:
                stats["failed"] += 1
                print(
                    f"[WARN] upload failed for {show_id}/{cid}/{image_path}",
                    file=sys.stderr,
                )
                continue
            seed["image_s3_key"] = key
            seed["display_s3_key"] = key
            stats["updated"] += 1
            changed = True
            print(f"[OK] Uploaded {image_path} → s3://.../{key}")
        if changed and not dry_run:
            fs._save_facebank(show_id, cid, data)
    return stats


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill facebank seeds to S3")
    parser.add_argument("show_id", help="Show identifier (e.g. RHOBH)")
    parser.add_argument("--cast-id", help="Optional cast id to limit backfill")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only report actions without uploading"
    )
    args = parser.parse_args(argv)

    stats = backfill_facebank_seeds(
        args.show_id, cast_id=args.cast_id, dry_run=args.dry_run
    )
    print(
        "Summary: updated={updated} skipped={skipped} failed={failed}".format(**stats)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

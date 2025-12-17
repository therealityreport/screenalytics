#!/usr/bin/env python3
"""Cleanup local storage after verifying S3 backup.

This script identifies local data that exists in S3 and can safely be deleted
to free up local disk space while keeping S3 as the primary storage.

Usage:
    # Dry run - show what would be deleted
    python tools/cleanup_local_storage.py --dry-run

    # Actually delete (with confirmation)
    python tools/cleanup_local_storage.py --delete

    # Delete without confirmation
    python tools/cleanup_local_storage.py --delete --yes
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def get_s3_client():
    """Get boto3 S3 client."""
    import boto3
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    return boto3.client("s3", region_name=region)


def list_s3_objects(bucket: str, prefix: str) -> set[str]:
    """List all S3 keys under a prefix."""
    s3 = get_s3_client()
    keys = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.add(obj["Key"])
    return keys


def get_local_manifest_files() -> list[tuple[Path, str]]:
    """Get all local manifest files with their expected S3 keys."""
    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", PROJECT_ROOT / "data"))
    manifests_dir = data_root / "manifests"

    if not manifests_dir.exists():
        return []

    files = []
    for ep_dir in manifests_dir.iterdir():
        if not ep_dir.is_dir():
            continue
        ep_id = ep_dir.name

        # Legacy manifests (non-run-scoped)
        for artifact in ep_dir.glob("*.json*"):
            if artifact.is_file() and "runs" not in str(artifact):
                # Legacy S3 key format
                s3_key = f"artifacts/manifests/{ep_id}/{artifact.name}"
                files.append((artifact, s3_key))

        # Run-scoped artifacts
        runs_dir = ep_dir / "runs"
        if runs_dir.exists():
            for run_dir in runs_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                run_id = run_dir.name

                for artifact in run_dir.rglob("*"):
                    if artifact.is_file():
                        # Run-scoped S3 key format
                        rel_path = artifact.relative_to(run_dir)
                        from py_screenalytics.run_layout import run_artifact_s3_key
                        s3_key = run_artifact_s3_key(ep_id, run_id, str(rel_path))
                        files.append((artifact, s3_key))

    return files


def main():
    parser = argparse.ArgumentParser(description="Cleanup local storage backed up to S3")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    parser.add_argument("--delete", action="store_true", help="Actually delete files")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    parser.add_argument("--bucket", default=None, help="S3 bucket (default from env)")
    args = parser.parse_args()

    if not args.dry_run and not args.delete:
        print("Must specify --dry-run or --delete")
        sys.exit(1)

    bucket = args.bucket or os.environ.get("SCREENALYTICS_S3_BUCKET") or os.environ.get("AWS_S3_BUCKET")
    if not bucket:
        print("ERROR: S3 bucket not configured. Set SCREENALYTICS_S3_BUCKET in .env")
        sys.exit(1)

    print(f"Bucket: {bucket}")
    print()

    # Get local files
    print("Scanning local manifest files...")
    local_files = get_local_manifest_files()
    print(f"Found {len(local_files)} local files")

    if not local_files:
        print("No local files to check")
        return

    # Get S3 keys for comparison
    print("\nScanning S3 for existing backups...")
    print("  Checking artifacts/ prefix...")
    artifacts_keys = list_s3_objects(bucket, "artifacts/")
    print(f"  Found {len(artifacts_keys)} objects in artifacts/")

    print("  Checking runs/ prefix...")
    runs_keys = list_s3_objects(bucket, "runs/")
    print(f"  Found {len(runs_keys)} objects in runs/")

    all_s3_keys = artifacts_keys | runs_keys

    # Find files that can be deleted
    can_delete = []
    not_in_s3 = []

    for local_path, s3_key in local_files:
        if s3_key in all_s3_keys:
            can_delete.append((local_path, s3_key))
        else:
            not_in_s3.append((local_path, s3_key))

    # Calculate sizes
    delete_size = sum(p.stat().st_size for p, _ in can_delete if p.exists())
    keep_size = sum(p.stat().st_size for p, _ in not_in_s3 if p.exists())

    print()
    print("=" * 60)
    print(f"Files backed up in S3 (can delete): {len(can_delete)} ({delete_size / (1024*1024):.1f} MB)")
    print(f"Files NOT in S3 (keeping):          {len(not_in_s3)} ({keep_size / (1024*1024):.1f} MB)")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would delete:")
        for local_path, s3_key in can_delete[:20]:
            print(f"  {local_path}")
        if len(can_delete) > 20:
            print(f"  ... and {len(can_delete) - 20} more files")

        if not_in_s3:
            print("\n[DRY RUN] Would KEEP (not in S3):")
            for local_path, s3_key in not_in_s3[:10]:
                print(f"  {local_path}")
                print(f"    (expected S3 key: {s3_key})")
            if len(not_in_s3) > 10:
                print(f"  ... and {len(not_in_s3) - 10} more files")
        return

    # Delete mode
    if not can_delete:
        print("\nNo files to delete (none are backed up in S3)")
        return

    if not args.yes:
        confirm = input(f"\nDelete {len(can_delete)} files ({delete_size / (1024*1024):.1f} MB)? [y/N] ")
        if confirm.lower() != "y":
            print("Aborted")
            return

    print(f"\nDeleting {len(can_delete)} files...")
    deleted = 0
    errors = 0

    for local_path, s3_key in can_delete:
        try:
            if local_path.exists():
                local_path.unlink()
                deleted += 1
        except Exception as e:
            print(f"  ERROR deleting {local_path}: {e}")
            errors += 1

    # Clean up empty directories
    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", PROJECT_ROOT / "data"))
    manifests_dir = data_root / "manifests"

    for ep_dir in manifests_dir.iterdir():
        if not ep_dir.is_dir():
            continue
        # Clean empty run directories
        runs_dir = ep_dir / "runs"
        if runs_dir.exists():
            for run_dir in list(runs_dir.iterdir()):
                if run_dir.is_dir():
                    # Remove empty subdirs first
                    for subdir in list(run_dir.iterdir()):
                        if subdir.is_dir() and not any(subdir.iterdir()):
                            try:
                                subdir.rmdir()
                            except Exception:
                                pass
                    # Remove run dir if empty
                    if not any(run_dir.iterdir()):
                        try:
                            run_dir.rmdir()
                        except Exception:
                            pass

    print(f"\nDone! Deleted {deleted} files, {errors} errors")
    print(f"Freed {delete_size / (1024*1024):.1f} MB of local storage")


if __name__ == "__main__":
    main()

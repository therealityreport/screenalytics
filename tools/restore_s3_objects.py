#!/usr/bin/env python3
"""Restore deleted S3 objects by removing delete markers.

Usage:
    python tools/restore_s3_objects.py --prefix artifacts/crops/rhoslc/s06/e02/
"""

import argparse
import subprocess
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def list_delete_markers_paginated(bucket: str, prefix: str):
    """List all delete markers for a prefix using pagination."""
    markers = []
    next_key_marker = None
    next_version_marker = None
    page = 0

    while True:
        page += 1
        cmd = [
            'aws', 's3api', 'list-object-versions',
            '--bucket', bucket,
            '--prefix', prefix,
            '--max-items', '1000',
        ]

        if next_key_marker:
            cmd.extend(['--key-marker', next_key_marker])
        if next_version_marker:
            cmd.extend(['--version-id-marker', next_version_marker])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}", file=sys.stderr)
            break

        data = json.loads(result.stdout)

        # Get delete markers that are IsLatest=True
        for dm in data.get('DeleteMarkers', []):
            if dm.get('IsLatest'):
                markers.append((dm['Key'], dm['VersionId']))

        print(f"  Page {page}: found {len(markers)} delete markers so far...")

        # Check for more pages
        if data.get('IsTruncated'):
            next_key_marker = data.get('NextKeyMarker')
            next_version_marker = data.get('NextVersionIdMarker')
        else:
            break

    return markers


def remove_delete_marker(bucket: str, key: str, version_id: str) -> bool:
    """Remove a single delete marker."""
    result = subprocess.run([
        'aws', 's3api', 'delete-object',
        '--bucket', bucket,
        '--key', key,
        '--version-id', version_id
    ], capture_output=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Restore deleted S3 objects')
    parser.add_argument('--bucket', default='screenalytics', help='S3 bucket name')
    parser.add_argument('--prefix', required=True, help='Object prefix to restore')
    parser.add_argument('--workers', type=int, default=50, help='Number of parallel workers')
    parser.add_argument('--dry-run', action='store_true', help='Just list, do not restore')
    args = parser.parse_args()

    print(f"Listing delete markers for s3://{args.bucket}/{args.prefix}...")
    markers = list_delete_markers_paginated(args.bucket, args.prefix)
    print(f"\nFound {len(markers)} delete markers to remove")

    if args.dry_run:
        print("Dry run - not removing any markers")
        for key, vid in markers[:20]:
            print(f"  {key}")
        if len(markers) > 20:
            print(f"  ... and {len(markers) - 20} more")
        return

    if not markers:
        print("Nothing to restore!")
        return

    print(f"\nRestoring with {args.workers} parallel workers...")

    restored = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(remove_delete_marker, args.bucket, key, vid): (key, vid)
            for key, vid in markers
        }

        for future in as_completed(futures):
            if future.result():
                restored += 1
            else:
                failed += 1

            if (restored + failed) % 500 == 0:
                print(f"  Progress: {restored} restored, {failed} failed...")

    print(f"\nDone! Restored: {restored}, Failed: {failed}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""CLI tool to bulk import cast members from JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

DEFAULT_API_BASE = "http://localhost:8000"


def import_cast_from_file(
    show_id: str,
    json_file: Path,
    api_base: str = DEFAULT_API_BASE,
    force_new: bool = False,
) -> dict:
    """Import cast members from JSON file.

    Args:
        show_id: Show identifier (e.g., 'rhobh')
        json_file: Path to JSON file with cast members
        api_base: API base URL
        force_new: Always create new members (skip merge)

    Returns:
        Import result dict with audit summary
    """
    # Load JSON
    data = json.loads(json_file.read_text(encoding="utf-8"))

    # Prepare payload
    if "members" in data:
        # Already has wrapper
        payload = data
        payload["force_new"] = force_new
    else:
        # Assume it's just a list of members
        payload = {"members": data, "force_new": force_new}

    # POST to API
    url = f"{api_base}/shows/{show_id}/cast/import"
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()

    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Bulk import cast members from JSON")
    parser.add_argument("show_id", help="Show ID (e.g., rhobh)")
    parser.add_argument("json_file", type=Path, help="Path to JSON file")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="API base URL")
    parser.add_argument(
        "--force-new",
        action="store_true",
        help="Always create new members (skip merge by name)",
    )

    args = parser.parse_args()

    if not args.json_file.exists():
        print(f"Error: File not found: {args.json_file}", file=sys.stderr)
        sys.exit(1)

    try:
        result = import_cast_from_file(
            args.show_id,
            args.json_file,
            api_base=args.api_base,
            force_new=args.force_new,
        )

        print(f"✓ Import complete for {result['show_id']}")
        print(f"  Total: {result['total']}")
        print(f"  Created: {result['created_count']}")
        print(f"  Updated: {result['updated_count']}")
        print(f"  Skipped: {result['skipped_count']}")

        if result["created"]:
            print("\nCreated:")
            for item in result["created"]:
                print(f"  - {item['name']} ({item['cast_id']})")

        if result["updated"]:
            print("\nUpdated:")
            for item in result["updated"]:
                print(f"  - {item['name']} ({item['cast_id']})")

        if result["skipped"]:
            print("\nSkipped:")
            for item in result["skipped"]:
                print(f"  - {item.get('name', 'N/A')}: {item['reason']}")

    except requests.RequestException as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

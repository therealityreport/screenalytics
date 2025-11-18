#!/usr/bin/env python3
"""Manual smoke test for TRR metadata database connection and queries.

This script verifies that:
1. TRR_DB_URL is configured
2. Connection to the Postgres database succeeds
3. Queries against core.cast return expected data

Usage:
    # Set TRR_DB_URL in your environment, then run:
    TRR_DB_URL=postgresql://user:pass@host:5432/trr_metadata python tools/test_trr_db.py

    # Or export it first:
    export TRR_DB_URL=postgresql://user:pass@localhost:5432/trr_metadata
    python tools/test_trr_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

# Add project root to path so we can import from apps.api
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.api.services.trr_metadata_db import (
    get_cast_for_show,
    get_episodes_for_show,
    get_seasons_for_show,
    get_show_by_slug,
)


def main():
    """Test TRR metadata database connection and queries."""
    # Hard-code a slug for testing; change this when running locally
    test_show_slug = "RHOBH"

    print("=" * 60)
    print("Testing TRR metadata DB connection")
    print("=" * 60)

    # Test 1: Get show by slug
    print(f"\n1️⃣  Testing get_show_by_slug('{test_show_slug}')...")
    try:
        show = get_show_by_slug(test_show_slug)
        if show:
            print(f"✅ Show found: {show.get('title', 'N/A')}")
            print(f"   Show ID: {show.get('show_id')}")
            print(f"   Network: {show.get('network', 'N/A')}")
        else:
            print(f"⚠️  No show found for slug={test_show_slug!r}")
            print("   This might be expected if the show hasn't been added yet.")
    except Exception as exc:
        print(f"❌ Failed to query show: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Test 2: Get seasons if show was found
    if show:
        print(f"\n2️⃣  Testing get_seasons_for_show(show_id={show['show_id']!r})...")
        try:
            seasons = get_seasons_for_show(show["show_id"])
            print(f"✅ Retrieved {len(seasons)} seasons")
            if seasons:
                print(f"   Seasons: {[s.get('season_number') for s in seasons]}")
        except Exception as exc:
            print(f"❌ Failed to query seasons: {exc}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

        # Test 3: Get episodes if show was found
        print(f"\n3️⃣  Testing get_episodes_for_show(show_id={show['show_id']!r})...")
        try:
            episodes = get_episodes_for_show(show["show_id"])
            print(f"✅ Retrieved {len(episodes)} episodes")
            if episodes and len(episodes) <= 5:
                print("\n   Sample episodes:")
                pprint(episodes[:5])
            elif episodes:
                print(f"\n   First 3 episodes:")
                pprint(episodes[:3])
        except Exception as exc:
            print(f"❌ Failed to query episodes: {exc}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    # Test 4: Get cast
    print(f"\n4️⃣  Testing get_cast_for_show('{test_show_slug}')...")
    try:
        cast = get_cast_for_show(test_show_slug)
        print(f"✅ Retrieved {len(cast)} cast members")
        if cast:
            print("\n   Cast members:")
            pprint(cast)
    except Exception as exc:
        print(f"❌ Failed to query cast: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

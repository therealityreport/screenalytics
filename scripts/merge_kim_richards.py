#!/usr/bin/env python3
"""Merge duplicate Kim Richards entries in RHOBH people.json"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from apps.api.services.people import PeopleService


def main():
    show_id = "RHOBH"
    people_service = PeopleService()

    # Get all people
    people = people_service.list_people(show_id)

    # Find the Kim Richards entries
    kim_entries = []
    for person in people:
        name = (person.get("name") or "").strip()
        if name.lower() in ["kim richards", "kim"]:
            kim_entries.append(person)
            print(
                f"Found: {person['person_id']} - {person.get('name')} - cast_id:{person.get('cast_id')} - clusters:{len(person.get('cluster_ids', []))}"
            )

    if len(kim_entries) < 2:
        print(f"Found {len(kim_entries)} Kim entries, nothing to merge")
        return

    # Find the canonical one (with cast_id)
    canonical = None
    to_merge = []

    for entry in kim_entries:
        if entry.get("cast_id"):
            if canonical:
                print(
                    f"Warning: Multiple entries with cast_id! Using {canonical['person_id']}"
                )
            else:
                canonical = entry
                print(f"Canonical entry (has cast_id): {canonical['person_id']}")
        else:
            to_merge.append(entry)

    # If no canonical entry, use the first one
    if not canonical:
        canonical = kim_entries[0]
        to_merge = kim_entries[1:]
        print(
            f"No cast_id found, using first entry as canonical: {canonical['person_id']}"
        )

    print(f"\nMerging {len(to_merge)} entries into {canonical['person_id']}:")

    for source in to_merge:
        print(
            f"  Merging {source['person_id']} ({source.get('name')}) -> {canonical['person_id']}"
        )
        result = people_service.merge_people(
            show_id, source["person_id"], canonical["person_id"]
        )
        if result:
            print(
                f"    ✓ Success - {len(result.get('cluster_ids', []))} total clusters"
            )
        else:
            print(f"    ✗ Failed")

    # Show final state
    final = people_service.get_person(show_id, canonical["person_id"])
    if final:
        print(f"\nFinal merged person:")
        print(f"  ID: {final['person_id']}")
        print(f"  Name: {final.get('name')}")
        print(f"  Aliases: {final.get('aliases', [])}")
        print(f"  Cast ID: {final.get('cast_id')}")
        print(f"  Clusters: {len(final.get('cluster_ids', []))}")
        print(
            f"  Cluster IDs: {json.dumps(sorted(set(final.get('cluster_ids', []))), indent=2)}"
        )


if __name__ == "__main__":
    main()

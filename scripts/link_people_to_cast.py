#!/usr/bin/env python3
"""Link existing people to cast members by name matching."""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from apps.api.services.people import PeopleService
import requests


def main():
    show_id = "RHOBH"
    people_service = PeopleService()

    # Fetch cast members from API
    try:
        response = requests.get(f"http://localhost:8000/shows/{show_id}/cast")
        response.raise_for_status()
        cast_data = response.json()
        cast_members = cast_data.get("cast", [])
    except Exception as e:
        print(f"Error fetching cast members: {e}")
        return

    print(f"Found {len(cast_members)} cast members")

    # Get all people
    people = people_service.list_people(show_id)
    print(f"Found {len(people)} people records")

    # Link people to cast members by name
    for person in people:
        person_id = person["person_id"]
        person_name = person.get("name")
        current_cast_id = person.get("cast_id")

        # Skip if already linked
        if current_cast_id:
            print(
                f"✓ {person_id} ({person_name}) already linked to cast_id {current_cast_id}"
            )
            continue

        # Skip unnamed people
        if not person_name:
            continue

        # Find matching cast member by name
        normalized_name = people_service.normalize_name(person_name)
        matched_cast = None

        for cast in cast_members:
            cast_name = cast.get("name")
            if not cast_name:
                continue

            # Check primary name
            if people_service.normalize_name(cast_name) == normalized_name:
                matched_cast = cast
                break

            # Check aliases
            for alias in cast.get("aliases", []):
                if people_service.normalize_name(alias) == normalized_name:
                    matched_cast = cast
                    break

            if matched_cast:
                break

        if matched_cast:
            cast_id = matched_cast["cast_id"]
            cast_name = matched_cast["name"]

            # Update person with cast_id
            people_service.update_person(show_id, person_id, cast_id=cast_id)
            print(
                f"✓ Linked {person_id} ({person_name}) to cast member {cast_name} ({cast_id})"
            )
        else:
            print(f"  No cast member found for {person_id} ({person_name})")

    print("\nDone!")


if __name__ == "__main__":
    main()

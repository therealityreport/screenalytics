"""Tests for show_id normalization in PeopleService and GroupingService."""

from __future__ import annotations


import numpy as np

from apps.api.services.people import PeopleService
from apps.api.services.grouping import GroupingService


def test_people_service_normalizes_show_id(tmp_path):
    """Test that PeopleService normalizes show_id to uppercase."""
    data_root = tmp_path / "data"
    service = PeopleService(data_root)

    # Create person with lowercase show_id
    person1 = service.create_person("rhobh", name="Test Person 1")
    assert person1["show_id"] == "rhobh"  # Stored as provided

    # Create person with uppercase show_id
    person2 = service.create_person("RHOBH", name="Test Person 2")
    assert person2["show_id"] == "RHOBH"  # Stored as provided

    # Both should be in the same file (RHOBH directory)
    people_upper = service.list_people("RHOBH")
    people_lower = service.list_people("rhobh")

    # Both queries should return the same people (2 total)
    assert len(people_upper) == 2
    assert len(people_lower) == 2
    assert people_upper[0]["person_id"] == people_lower[0]["person_id"]
    assert people_upper[1]["person_id"] == people_lower[1]["person_id"]

    # Verify only one directory was created (uppercase)
    shows_dir = data_root / "shows"
    created_dirs = [d.name for d in shows_dir.iterdir() if d.is_dir()]
    assert len(created_dirs) == 1
    assert created_dirs[0] == "RHOBH"  # Only uppercase directory exists


def test_grouping_service_matches_normalized_people(tmp_path):
    """Test that GroupingService can match people created with mixed case."""
    data_root = tmp_path / "data"
    people_service = PeopleService(data_root)
    grouping_service = GroupingService(data_root)

    # Create a person with lowercase show_id (simulating API call)
    prototype = np.random.rand(512).astype(np.float32)
    prototype = prototype / np.linalg.norm(prototype)  # L2 normalize

    person = people_service.create_person(
        "rhobh",  # lowercase
        name="Kim Richards",
        prototype=prototype.tolist(),
    )
    person_id = person["person_id"]

    # GroupingService uses uppercase from _parse_ep_id
    # It should still find the person created with lowercase
    match = people_service.find_matching_person(
        "RHOBH",  # uppercase
        prototype,
        max_distance=0.1,
    )

    assert match is not None, "Should find person regardless of show_id casing"
    matched_person_id, distance = match
    assert matched_person_id == person_id
    assert distance < 0.001  # Should be nearly identical


def test_people_service_get_person_case_insensitive(tmp_path):
    """Test that get_person works with mixed case show_ids."""
    data_root = tmp_path / "data"
    service = PeopleService(data_root)

    # Create with one casing
    person = service.create_person("demo", name="Test Person")
    person_id = person["person_id"]

    # Retrieve with different casing
    person_upper = service.get_person("DEMO", person_id)
    person_lower = service.get_person("demo", person_id)
    person_mixed = service.get_person("DeMo", person_id)

    assert person_upper is not None
    assert person_lower is not None
    assert person_mixed is not None
    assert person_upper["person_id"] == person_id
    assert person_lower["person_id"] == person_id
    assert person_mixed["person_id"] == person_id


def test_people_service_update_person_case_insensitive(tmp_path):
    """Test that update_person works with mixed case show_ids."""
    data_root = tmp_path / "data"
    service = PeopleService(data_root)

    # Create with lowercase
    person = service.create_person("rhobh", name="Original Name")
    person_id = person["person_id"]

    # Update with uppercase
    updated = service.update_person("RHOBH", person_id, name="Updated Name")
    assert updated is not None
    assert updated["name"] == "Updated Name"

    # Verify with lowercase
    person_check = service.get_person("rhobh", person_id)
    assert person_check["name"] == "Updated Name"


def test_people_service_delete_person_case_insensitive(tmp_path):
    """Test that delete_person works with mixed case show_ids."""
    data_root = tmp_path / "data"
    service = PeopleService(data_root)

    # Create with mixed case
    person = service.create_person("RhObH", name="To Delete")
    person_id = person["person_id"]

    # Verify it exists with different casing
    assert service.get_person("rhobh", person_id) is not None

    # Delete with uppercase
    success = service.delete_person("RHOBH", person_id)
    assert success is True

    # Verify deleted with lowercase
    assert service.get_person("rhobh", person_id) is None


def test_normalize_show_id_static_method():
    """Test the normalize_show_id static method directly."""
    assert PeopleService.normalize_show_id("rhobh") == "RHOBH"
    assert PeopleService.normalize_show_id("RHOBH") == "RHOBH"
    assert PeopleService.normalize_show_id("RhObH") == "RHOBH"
    assert PeopleService.normalize_show_id("demo") == "DEMO"
    assert PeopleService.normalize_show_id("DEMO") == "DEMO"

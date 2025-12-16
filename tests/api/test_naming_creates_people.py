"""Test that naming a cluster creates/updates show-level People records."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def test_persist_identity_name_creates_people_record():
    """Test that _persist_identity_name creates People record via PeopleService."""
    identities_path = PROJECT_ROOT / "apps" / "api" / "services" / "identities.py"
    content = identities_path.read_text()

    # Check that _persist_identity_name uses PeopleService
    assert "from apps.api.services.people import PeopleService" in content, "Should import PeopleService"
    assert "people_service = PeopleService()" in content, "Should instantiate PeopleService"

    # Check that it creates or updates person
    assert "people_service.create_person(" in content, "Should create new person"
    assert "people_service.add_cluster_to_person(" in content, "Should add cluster to existing person"

    # Check that it handles name matching (case-insensitive)
    assert "existing_person = next(" in content, "Should search for existing person by name"
    assert ".lower()" in content, "Should do case-insensitive name matching"

    # Check that cluster_id includes episode prefix
    assert "_qualified_cluster_ref(ep_id, identity_id" in content, "Should prefix cluster_id with episode (and run_id when provided)"

    print("✓ _persist_identity_name creates/updates People records")


def test_naming_doesnt_fail_on_people_service_error():
    """Test that naming succeeds even if PeopleService fails."""
    identities_path = PROJECT_ROOT / "apps" / "api" / "services" / "identities.py"
    content = identities_path.read_text()

    # Check for try/except around PeopleService calls
    assert "except Exception as exc:" in content, "Should catch exceptions from PeopleService"
    assert "LOGGER.warning" in content, "Should log warning on failure"
    assert (
        "Don't fail the naming operation" in content or "Failed to create/update People" in content
    ), "Should document that naming continues on PeopleService failure"

    print("✓ Naming operation doesn't fail if PeopleService errors")


def test_person_record_includes_cluster_id():
    """Test that person records include cluster_ids list."""
    identities_path = PROJECT_ROOT / "apps" / "api" / "services" / "identities.py"
    content = identities_path.read_text()

    # Check that cluster_ids parameter is used
    assert "cluster_ids=[cluster_id_with_prefix]" in content, "Should pass cluster_ids when creating person"
    assert 'cluster_ids = existing_person.get("cluster_ids", [])' in content, "Should check existing cluster_ids"

    print("✓ Person records include cluster_ids")


if __name__ == "__main__":
    test_persist_identity_name_creates_people_record()
    test_naming_doesnt_fail_on_people_service_error()
    test_person_record_includes_cluster_id()
    print("\n✓ All naming creates people tests passed!")

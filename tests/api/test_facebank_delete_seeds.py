"""Test facebank seed deletion functionality."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def test_delete_seeds_endpoint_signature():
    """Test that delete_seeds endpoint exists and has correct signature."""
    from apps.api.routers.facebank import router, DeleteSeedsRequest

    # Find the delete endpoint
    delete_route = None
    for route in router.routes:
        if hasattr(route, "path") and "/cast/{cast_id}/seeds" in route.path:
            if hasattr(route, "methods") and "DELETE" in route.methods:
                delete_route = route
                break

    assert delete_route is not None, "DELETE /cast/{cast_id}/seeds endpoint not found"

    # Verify DeleteSeedsRequest model
    assert hasattr(
        DeleteSeedsRequest, "__fields__"
    ), "DeleteSeedsRequest should be a Pydantic model"
    assert (
        "seed_ids" in DeleteSeedsRequest.__fields__
    ), "DeleteSeedsRequest should have seed_ids field"

    print("✓ Delete seeds endpoint exists with correct signature")


def test_delete_seeds_request_model():
    """Test DeleteSeedsRequest model validation."""
    from apps.api.routers.facebank import DeleteSeedsRequest

    # Valid request
    request = DeleteSeedsRequest(seed_ids=["seed_001", "seed_002"])
    assert len(request.seed_ids) == 2
    assert request.seed_ids[0] == "seed_001"

    # Empty list is valid
    empty_request = DeleteSeedsRequest(seed_ids=[])
    assert len(empty_request.seed_ids) == 0

    print("✓ DeleteSeedsRequest model validates correctly")


def test_delete_seed_ui_function_exists():
    """Test that UI has delete seed function."""
    import importlib.util

    cast_page_path = PROJECT_ROOT / "apps" / "workspace-ui" / "pages" / "4_Cast.py"
    spec = importlib.util.spec_from_file_location("cast_page", cast_page_path)

    if spec and spec.loader:
        # Read the file content to check for the function
        content = cast_page_path.read_text()

        assert (
            "def _delete_seed(" in content
        ), "_delete_seed function should exist in Cast.py"
        assert "_api_delete(" in content, "Should call _api_delete for deletion"
        assert (
            '"🗑️ Delete"' in content or "'🗑️ Delete'" in content
        ), "Should have delete button in UI"
        assert "confirm_delete_seed_" in content, "Should have confirmation logic"

        print("✓ UI has delete seed function and confirmation flow")


def test_facebank_service_delete_seeds():
    """Test that facebank service has delete_seeds method."""
    from apps.api.services.facebank import FacebankService

    service = FacebankService()

    # Verify method exists
    assert hasattr(
        service, "delete_seeds"
    ), "FacebankService should have delete_seeds method"
    assert callable(service.delete_seeds), "delete_seeds should be callable"

    print("✓ FacebankService has delete_seeds method")


if __name__ == "__main__":
    test_delete_seeds_endpoint_signature()
    test_delete_seeds_request_model()
    test_delete_seed_ui_function_exists()
    test_facebank_service_delete_seeds()
    print("\n✓ All facebank seed deletion tests passed!")

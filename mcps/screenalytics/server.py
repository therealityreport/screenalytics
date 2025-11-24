# MCP skeleton: expose tools via a simple dispatcher until the Apps SDK is added.
import json
import sys
from mcps.common.auth import check
from mcps.common.schemas import EpisodeId, TrackId, PersonId, ScreenTimeRow


def list_low_confidence(ep_id: str, thresh: float = 0.62):
    check()  # TODO: query DB once DB layer exists
    return {"items": []}


def assign_identity(track_id: str, person_id: str, reason: str = "agent"):
    check(write=True)  # TODO: write to DB
    TrackId(track_id=track_id)
    PersonId(person_id=person_id)
    return {"track_id": track_id, "person_id": person_id, "reason": reason, "ok": True}


def export_screen_time(ep_id: str, fmt: str = "json"):
    check()
    EpisodeId(ep_id=ep_id)
    rows = [
        ScreenTimeRow(
            ep_id=ep_id,
            person_id="demo",
            visual_s=10,
            speaking_s=5,
            both_s=4,
            confidence=0.9,
        ).model_dump()
    ]
    return {"rows": rows, "format": fmt}


if __name__ == "__main__":
    fn = globals().get(sys.argv[1]) if len(sys.argv) > 1 else None
    args = sys.argv[2:]
    out = fn(*args) if fn else {"error": "unknown tool"}
    print(json.dumps(out))

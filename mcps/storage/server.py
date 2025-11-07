import json
import sys
import time

from mcps.common.auth import check


def sign_url(kind: str, owner_id: str, method: str = "GET"):
    check()
    return {
        "url": f"https://storage.local/{kind}/{owner_id}",
        "expires_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def list(prefix: str):
    check()
    return {"items": []}


def purge(prefix: str):
    check(write=True)
    return {"ok": True, "prefix": prefix}


if __name__ == "__main__":
    fn = globals().get(sys.argv[1]) if len(sys.argv) > 1 else None
    out = fn(*sys.argv[2:]) if fn else {"error": "unknown"}
    print(json.dumps(out))

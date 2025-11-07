import json
import os
import sys

try:
    import psycopg  # type: ignore
except ImportError:  # pragma: no cover
    psycopg = None

from mcps.common.auth import check


def _fake_conn():
    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, *args, **kwargs):
            self._rows = []

        def fetchall(self):
            return getattr(self, "_rows", [])

    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def cursor(self):
            return FakeCursor()

    return FakeConn()


def _conn():
    if os.getenv("SCREENALYTICS_FAKE_DB") == "1":
        return _fake_conn()
    if psycopg is None:
        raise RuntimeError("psycopg not installed")
    return psycopg.connect(os.getenv("DB_URL"))


def presence_by_person(person_id: str, show_id: str | None = None):
    check()
    q = """select ep_id,
                  sum(visual_s) visual_s,
                  sum(speaking_s) speaking_s,
                  sum(both_s) both_s
           from screen_time
           where person_id = %s
           group by ep_id
           order by ep_id"""
    with _conn() as con, con.cursor() as cur:
        cur.execute(q, (person_id,))
        rows = [
            {
                "ep_id": r[0],
                "visual_s": float(r[1] or 0),
                "speaking_s": float(r[2] or 0),
                "both_s": float(r[3] or 0),
            }
            for r in cur.fetchall()
        ]
    return {"rows": rows}


def episodes_by_show(show_id: str):
    check()
    q = "select ep_id from episode e join season s on e.season_id=s.season_id where s.show_id=%s order by ep_id"
    with _conn() as con, con.cursor() as cur:
        cur.execute(q, (show_id,))
        rows = [r[0] for r in cur.fetchall()]
    return {"ep_ids": rows}


if __name__ == "__main__":
    fn = globals().get(sys.argv[1]) if len(sys.argv) > 1 else None
    out = fn(*sys.argv[2:]) if fn else {"error": "unknown"}
    print(json.dumps(out))

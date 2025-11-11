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


def presence_by_person(person_ref: str, show_ref: str | None = None):
    check()
    q = """
        select st.ep_id::text,
               coalesce(sum(st.visual_s), 0),
               coalesce(sum(st.speaking_s), 0),
               coalesce(sum(st.both_s), 0)
        from screen_time st
        join person p on st.person_id = p.person_id
        join episode e on st.ep_id = e.ep_id
        join season s on e.season_id = s.season_id
        join show sh on s.show_id = sh.show_id
        where (
            p.person_id::text = %(person)s
            or p.canonical_name = %(person)s
            or p.display_name = %(person)s
            or %(person)s = any(p.aliases)
        )
        and (
            %(show)s::text is null
            or sh.slug = %(show)s
            or sh.show_id::text = %(show)s
        )
        group by st.ep_id, s.number, e.number
        order by s.number, e.number
    """
    with _conn() as con, con.cursor() as cur:
        cur.execute(q, {"person": person_ref, "show": show_ref})
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


def episodes_by_show(show_ref: str):
    check()
    q = """
        select e.ep_id::text
        from episode e
        join season s on e.season_id = s.season_id
        join show sh on s.show_id = sh.show_id
        where sh.slug = %s
           or sh.show_id::text = %s
        order by s.number, e.number
    """
    with _conn() as con, con.cursor() as cur:
        cur.execute(q, (show_ref, show_ref))
        rows = [r[0] for r in cur.fetchall()]
    return {"ep_ids": rows}


if __name__ == "__main__":
    fn = globals().get(sys.argv[1]) if len(sys.argv) > 1 else None
    out = fn(*sys.argv[2:]) if fn else {"error": "unknown"}
    print(json.dumps(out))

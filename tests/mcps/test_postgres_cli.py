import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_postgres_presence_cli(monkeypatch):
    monkeypatch.setenv("DB_URL", "postgresql://demo")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))
    script = Path("mcps/postgres/server.py")
    proc = subprocess.run(
        [sys.executable, str(script), "presence_by_person", "person-1"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    assert "rows" in data

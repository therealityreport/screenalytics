import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_export_screen_time_cli(monkeypatch):
    monkeypatch.setenv("DB_URL", "postgresql://demo")
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))
    script = Path("mcps/screenalytics/server.py")
    result = subprocess.run(
        [sys.executable, str(script), "export_screen_time", "ep_demo", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    assert data["format"] == "json"
    assert data["rows"]

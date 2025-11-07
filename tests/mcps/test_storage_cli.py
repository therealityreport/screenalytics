import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_storage_sign_url(monkeypatch):
    monkeypatch.setenv("DB_URL", "postgresql://demo")
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))
    script = Path("mcps/storage/server.py")
    result = subprocess.run(
        [sys.executable, str(script), "sign_url", "chips", "track-1"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    assert data["url"].startswith("https://storage.local")
    assert "expires_at" in data

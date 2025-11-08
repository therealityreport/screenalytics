# SETUP.md — Screenalytics Environment Guide

---

## 1️⃣ Prerequisites
- Python 3.11+
- Node 20+ and pnpm
- Docker + Docker Compose
- ffmpeg installed
- (Optional GPU) CUDA 12+ for PyTorch
- GitHub CLI or SSH key configured

---

## 2️⃣ Clone & bootstrap
```bash
git clone https://github.com/<your-org>/screenalytics.git
cd screenalytics
```

### Virtual environment

> **macOS (Apple Silicon):** install `pyenv`, add the following to `~/.zshrc`, then install + select Python 3.11.9 before creating the venv.
> ```bash
> brew install pyenv
> if command -v pyenv >/dev/null; then
>   eval "$(pyenv init --path)"
> fi
> if [[ $- == *i* ]]; then
>   eval "$(pyenv init -)"
> fi
> pyenv install 3.11.9
> pyenv local 3.11.9
> ```

Now create the environment with the pinned interpreter:

```bash
python -m venv .venv
source .venv/bin/activate # or .\.venv\Scripts\activate on Windows
pip install -U pip wheel setuptools
pip install -r requirements.txt
```

### Node deps
```bash
pnpm install --filter workspace-ui
```

### Install Codex CLI
```bash
npm i -g @openai/codex   # or: brew install --cask codex
codex                    # sign in with your ChatGPT plan
```

### Environment file
Copy `.env.example` → `.env` and fill in credentials:
```
DB_URL=postgresql://user:pass@localhost:5432/screenalytics
REDIS_URL=redis://localhost:6379/0
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=minio
S3_SECRET_KEY=miniosecret
S3_BUCKET=screenalytics
OPENAI_API_KEY=sk-xxxx
```

---

## 3️⃣ Local services (Docker)
```bash
docker compose -f infra/docker/compose.yaml up -d
```
Starts Postgres, Redis, and MinIO.

---

## 4️⃣ Run core components
| Component | Command |
| ----------- | -------------------------------- |
| **API** | `uv run apps/api/main.py` |
| **Workers** | `uv run workers/orchestrator.py` |
| **UI** | `pnpm --filter workspace-ui dev` |
| **Tests** | `pytest -q` |

---

## 5️⃣ Optional: initialize database
```bash
psql "$DB_URL" -f db/migrations/0001_init_core.sql
```

---

## 6️⃣ Verify install
Visit [http://localhost:3000](http://localhost:3000) → Upload tab should appear.  
API health check: [http://localhost:8000/health](http://localhost:8000/health) returns `{"status":"ok"}`.

---

## 7️⃣ Agents & automation
* Codex config: `config/codex.config.toml`
* Claude policy: `config/claude.policies.yaml`
* Agents auto-update docs (README, PRD, SolutionArchitecture, DirectoryStructure) when files change.

To run Codex locally:
```bash
codex exec --config config/codex.config.toml --task agents/tasks/aggregate-screen-time.json
```

---

## 8️⃣ Promotion workflow
1. Create feature via `python tools/new-feature.py <name>`
2. Work inside `FEATURES/<name>/`
3. Pass CI (tests + docs)
4. Promote: `python tools/promote-feature.py <name>`
5. CI + Agents update docs automatically

---

## 9️⃣ Troubleshooting
| Symptom | Fix |
| --------------------- | ------------------------------------------------ |
| `ModuleNotFoundError` | Activate `.venv` |
| `pgvector` not found | Run migrations, ensure Postgres ≥15 |
| UI blank | Run `pnpm build` inside `apps/workspace-ui` |
| Codex writes blocked | Ensure correct profile (`promote` for root docs) |

---

**Screenalytics – “Every frame tells a story.”**

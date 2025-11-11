# 2025-11-11 — Core vs ML dependencies & stubbed pipeline

## Highlights
- Split dependency profiles so the API/UI/test surface installs quickly via `requirements-core.txt`, while the optional ML stack (RetinaFace, ByteTrack, Whisper) is isolated in `requirements-ml.txt`.
- Streamlit + FastAPI upload UX only needs the core profile; stub mode mirrors uploads locally and emits deterministic detections/tracks without OpenCV/GPU deps.

## Why it matters
Macs frequently fail while compiling PyAV/torch wheels. Keeping the upload flow lightweight lets QA and PMs verify uploads/detect-track stubs without containerized ML infrastructure.

## How to use it
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
uvicorn apps.api.main:app --reload
streamlit run apps/workspace-ui/streamlit_app.py
```

Install `requirements-ml.txt` only when running the full detection/track toolchain.

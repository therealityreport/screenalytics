# TODO: embedding-engines

**Status:** IN_PROGRESS
**Owner:** Engineering
**Created:** 2025-12-11
**TTL:** 2026-01-10

---

## Overview

Optimized embedding engines feature using TensorRT for GPU acceleration,
pluggable backend architecture, and ONNXRuntime C++ service planning.

**Full Documentation:** [docs/todo/feature_arcface_tensorrt_onnxruntime.md](../../docs/todo/feature_arcface_tensorrt_onnxruntime.md)

---

## Tasks

### Phase A: TensorRT ArcFace
- [ ] Create `src/tensorrt_arcface.py` - TRT inference
- [ ] Implement engine building from ONNX
- [ ] Implement S3/MinIO engine storage
- [ ] Add local build fallback
- [ ] Add config: `config/pipeline/embedding.yaml`
- [ ] Write tests: `tests/test_tensorrt_embedding.py`

### Phase B: Embedding Drift Validation
- [ ] Create `src/arcface_pytorch.py` - Reference implementation
- [ ] Implement embedding comparison
- [ ] Create regression test suite
- [ ] Write tests: `tests/test_embedding_regression.py`

### Phase C: Pluggable Backend Architecture
- [ ] Create `src/engine_registry.py` - Backend registry
- [ ] Update pipeline to use registry
- [ ] Add automatic fallback

### Phase D: ONNXRuntime C++ Service (Future)
- [ ] Create architecture documentation
- [ ] Define migration path
- [ ] Stub ONNX backend in registry

---

## Promotion Checklist

- [ ] Tests present and passing (`pytest FEATURES/embedding-engines/tests/ -v`)
- [ ] Lint clean (`black`, `ruff`, `mypy`)
- [ ] Docs complete (`FEATURES/embedding-engines/docs/`)
- [ ] Config-driven (no hardcoded thresholds)
- [ ] Integration tests passing
- [ ] Row added to `ACCEPTANCE_MATRIX.md` (section 3.13)

---

## Acceptance Criteria

| Metric | Target |
|--------|--------|
| `speedup_vs_pytorch` | ≥ 5x @ batch=32 |
| `embedding_cosine_drift` | ≥ 0.999 |
| `vram_usage` | ≤ 2GB |
| `fp16_accuracy_delta` | < 0.1% |

---

## Key Files

- `src/tensorrt_arcface.py` - TensorRT ArcFace
- `src/arcface_pytorch.py` - Reference implementation
- `src/engine_registry.py` - Pluggable backend
- `tests/test_tensorrt_embedding.py` - TRT tests
- `tests/test_embedding_regression.py` - Drift tests

# TODO: embedding-engines

**Status:** PARTIAL (implementation complete; acceptance pending)
**Owner:** Engineering
**Created:** 2025-12-11
**Updated:** 2025-12-23
**TTL:** 2026-01-10

---

## Overview

Optimized embedding engines feature using TensorRT for GPU acceleration,
pluggable backend architecture, and ONNXRuntime C++ service planning.

**Full Documentation:** [docs/plans/in_progress/feature_arcface_tensorrt_onnxruntime.md](../../docs/plans/in_progress/feature_arcface_tensorrt_onnxruntime.md)

**Implementation:** `FEATURES/arcface_tensorrt/src/` (production code)

---

## Tasks

### Phase A: TensorRT ArcFace ✅
- [x] A1: TRT inference wrapper → `FEATURES/arcface_tensorrt/src/tensorrt_inference.py`
- [x] A2: Engine building from ONNX → `FEATURES/arcface_tensorrt/src/tensorrt_builder.py`
- [x] A3: S3/MinIO engine storage → `FEATURES/arcface_tensorrt/src/tensorrt_builder.py`
- [x] A4: Local build fallback → `build_or_load_engine()`
- [x] A5: Config → `config/pipeline/embedding.yaml`, `config/pipeline/arcface_tensorrt.yaml`
- [x] A6: Tests → `tests/unit/test_embedding_backend_tensorrt_fallback.py`

### Phase B: Embedding Drift Validation ✅
- [x] B1: Reference ArcFace via InsightFace (in-process)
- [x] B2: Embedding comparison → `FEATURES/arcface_tensorrt/src/embedding_compare.py`
- [x] B3: Parity utilities + unit coverage
- [x] B4: Drift check config → `config/pipeline/embedding.yaml`
- [x] B5: Tests → `tests/ml/test_arcface_embeddings.py`

### Phase C: Pluggable Backend Architecture ✅
- [~] C1: `engine_registry.py` (OPTIONAL - `FEATURES/embedding_engines/` is docs-only today)
- [x] C2: Backend switchable via config (`embedding.backend: pytorch|tensorrt`)
- [x] C3: Automatic fallback (TRT → PyTorch)

### Phase D: ONNXRuntime C++ Service (Future - NOT STARTED)
- [ ] D1: Architecture documentation
- [ ] D2: Define migration path
- [ ] D3: Stub ONNX backend in registry

**Note:** Phase D is intentionally deferred as future work.

---

## Promotion Checklist

- [x] Tests present and passing
- [x] Docs complete
- [x] Config-driven (no hardcoded thresholds)
- [ ] Acceptance criteria validated on GPU hardware
- [ ] Row added to `ACCEPTANCE_MATRIX.md` (section 3.13)

---

## Acceptance Criteria (Pending Validation)

| Metric | Target | Status |
|--------|--------|--------|
| `speedup_vs_pytorch` | ≥ 5x @ batch=32 | Pending GPU test |
| `embedding_cosine_drift` | ≥ 0.999 | Pending GPU test |
| `vram_usage` | ≤ 2GB | Pending GPU test |
| `fp16_accuracy_delta` | < 0.1% | Pending GPU test |

---

## Key Files (Actual Implementation)

| Purpose | File |
|---------|------|
| TensorRT inference | `FEATURES/arcface_tensorrt/src/tensorrt_inference.py` |
| Engine building | `FEATURES/arcface_tensorrt/src/tensorrt_builder.py` |
| Embedding comparison | `FEATURES/arcface_tensorrt/src/embedding_compare.py` |
| Config | `config/pipeline/embedding.yaml` |
| TensorRT config | `config/pipeline/arcface_tensorrt.yaml` |
| Fallback tests | `tests/unit/test_embedding_backend_tensorrt_fallback.py` |
| Embedding tests | `tests/ml/test_arcface_embeddings.py` |

# TODO: arcface-tensorrt

**Status:** IN_PROGRESS
**Owner:** Engineering
**Created:** 2025-12-11
**TTL:** 2026-01-10

---

## Overview

TensorRT-accelerated ArcFace embeddings for high-throughput face recognition.
Targets 5x+ speedup over PyTorch with < 0.5% accuracy loss.

**Full Documentation:** [docs/README.md](docs/README.md)

---

## Tasks

### Phase A: Engine Building ✅ SCAFFOLD COMPLETE
- [x] Create `src/tensorrt_builder.py` - ONNX to TensorRT conversion
- [x] Implement SM architecture detection
- [x] Add local caching with versioned filenames
- [x] Add S3/MinIO storage support (stubbed)
- [x] Add config: `config/pipeline/arcface_tensorrt.yaml`
- [ ] **PENDING**: Test with real ONNX model
- [ ] **PENDING**: Verify S3 upload/download

### Phase B: Inference Wrapper ✅ SCAFFOLD COMPLETE
- [x] Create `src/tensorrt_inference.py` - TensorRT inference
- [x] Implement ArcFace preprocessing (matching PyTorch)
- [x] Add batch inference support
- [x] Add warmup and timing utilities
- [ ] **PENDING**: Test on real GPU

### Phase C: Comparison & Validation ✅ SCAFFOLD COMPLETE
- [x] Create `src/embedding_compare.py` - Backend comparison
- [x] Implement cosine similarity and L2 distance metrics
- [x] Add synthetic face generation for testing
- [x] Create comparison CLI
- [ ] **PENDING**: Run full comparison on eval set

### Phase D: Integration (FUTURE)
- [ ] Wire into main pipeline as optional backend
- [ ] Add config flag: `embedding.backend: tensorrt`
- [ ] Performance benchmarks in CI
- [ ] S3 engine distribution for team

---

## Promotion Checklist

- [ ] Tests present and passing (`pytest FEATURES/arcface-tensorrt/tests/ -v`)
- [ ] Lint clean (`black`, `ruff`, `mypy`)
- [ ] Docs complete (`FEATURES/arcface-tensorrt/docs/`)
- [ ] Config-driven (no hardcoded paths)
- [ ] Comparison passes acceptance threshold
- [ ] Row added to `ACCEPTANCE_MATRIX.md` (section 3.14)

---

## Acceptance Criteria

| Metric | Target |
|--------|--------|
| `cosine_sim_mean` vs PyTorch | ≥ 0.995 |
| `cosine_sim_min` vs PyTorch | ≥ 0.990 |
| `speedup` @ batch=32 | ≥ 5.0x |
| `latency` @ batch=1 | ≤ 10ms |
| VRAM usage | ≤ 2GB |

---

## Key Files

- `src/tensorrt_builder.py` - Engine building and caching
- `src/tensorrt_inference.py` - TensorRT inference wrapper
- `src/embedding_compare.py` - PyTorch vs TensorRT comparison
- `src/__main__.py` - CLI interface
- `tests/test_tensorrt_embedding.py` - Tests
- `config/pipeline/arcface_tensorrt.yaml` - Configuration

---

## Usage

```bash
# Build TensorRT engine
python -m FEATURES.arcface_tensorrt --mode build --onnx-path models/arcface.onnx

# Compare embeddings
python -m FEATURES.arcface_tensorrt --mode compare --n-samples 100

# Run benchmark
python -m FEATURES.arcface_tensorrt --mode benchmark
```

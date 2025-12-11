---
name: embedding-engine
description: Unified embedding interface across PyTorch, TensorRT, and ONNXRuntime. Use when optimizing embedding throughput or debugging embedding drift.
---

# Embedding Engine Skill

Use this skill to optimize embedding performance and debug embedding drift.

## When to Use

- Embedding pipeline running slowly
- Need to switch between PyTorch and TensorRT
- Debugging embedding drift between backends
- Setting up TensorRT engine storage
- Running embedding regression tests

## Sub-agents

| Sub-agent | Purpose |
|-----------|---------|
| **PyTorchEmbeddingSubagent** | Reference ArcFace (training/validation) |
| **TensorRTEmbeddingSubagent** | GPU-optimized TRT inference |
| **ONNXEmbeddingSubagent** | Future ONNXRuntime C++ service |

## Key Skills

### `embed_faces_batched()`
Run embedding with configured backend.

```python
from FEATURES.embedding_engines.src.engine_registry import EngineRegistry

embedder = EngineRegistry.get("tensorrt", engine_path="path/to/engine.trt")
embeddings = embedder.encode(face_crops)  # (N, 512)
```

### `compare_embedding_backends()`
Analyze drift between backends.

```python
from FEATURES.embedding_engines.src.validation import compare_embeddings

pytorch_emb = pytorch_embedder.encode(faces)
tensorrt_emb = tensorrt_embedder.encode(faces)

report = compare_embeddings(pytorch_emb, tensorrt_emb)
print(f"Mean cosine sim: {report.mean_cosine_sim}")
print(f"Min cosine sim: {report.min_cosine_sim}")
```

### `run_embedding_regression_tests()`
Verify TensorRT matches PyTorch within tolerance.

```python
from FEATURES.embedding_engines.src.validation import run_embedding_regression_tests

success = run_embedding_regression_tests(
    eval_faces=test_faces,
    pytorch_embedder=pytorch,
    tensorrt_embedder=tensorrt,
    max_drift=0.001
)
```

## Config Reference

**File:** `config/pipeline/embedding.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `backend` | `tensorrt` | Backend: `pytorch`, `tensorrt`, `onnx` |
| `tensorrt.precision` | `fp16` | TRT precision: `fp16`, `fp32`, `int8` |
| `tensorrt.batch_size` | 32 | Faces per batch |
| `tensorrt.max_batch_size` | 64 | Max batch for engine |
| `storage.type` | `s3` | Engine storage: `s3`, `local` |
| `storage.bucket` | `screenalytics-models` | S3 bucket |
| `fallback.build_locally` | true | Build engine if not in S3 |
| `fallback.fallback_to_pytorch` | true | PyTorch if TRT fails |
| `validation.max_drift_cosine` | 0.001 | Max tolerated drift |

## Engine Storage

TensorRT engines are GPU-architecture specific. Stored in S3:

```
s3://screenalytics-models/engines/
├── arcface_r100_v1-sm75.trt   # Ampere (RTX 30xx)
├── arcface_r100_v1-sm80.trt   # A100
├── arcface_r100_v1-sm86.trt   # Ada (RTX 40xx)
└── arcface_r100_v1-sm89.trt   # Hopper (H100)
```

**Naming convention:** `{model_name}-{version}-sm{arch}.trt`

## Common Issues

### "Engine not found"

**Cause:** No engine for current GPU architecture

**Check:** `nvidia-smi` for GPU model, compute capability

**Fix:** Build locally (automatic with `build_locally: true`) or upload to S3:
```bash
# Build and upload
python tools/build_trt_engine.py --upload
```

### Embedding drift too high

**Cause:** FP16 quantization or TRT optimization changes

**Check:** Run regression tests:
```bash
pytest tests/ml/test_embedding_regression.py -v
```

**Fix:** Use FP32 precision:
```yaml
tensorrt:
  precision: fp32  # default is fp16
```

### TensorRT slower than expected

**Cause:** Not batching, wrong batch size, CPU fallback

**Check:** Throughput benchmark:
```python
from FEATURES.embedding_engines.src.benchmark import benchmark_embedder
result = benchmark_embedder(embedder, batch_size=32)
```

**Fix:** Increase batch size, ensure GPU backend:
```yaml
tensorrt:
  batch_size: 64  # default is 32
```

### Out of GPU memory

**Cause:** Engine workspace too large

**Check:** `nvidia-smi` during inference

**Fix:** Reduce workspace:
```yaml
tensorrt:
  workspace_gb: 1.0  # default is 2.0
```

## Benchmark Reference

| Backend | Batch | Throughput | Latency | VRAM |
|---------|-------|------------|---------|------|
| PyTorch | 32 | ~50 fps | ~640ms | 2GB |
| TensorRT FP16 | 32 | ~250 fps | ~128ms | 1GB |
| TensorRT FP32 | 32 | ~180 fps | ~178ms | 1.5GB |

## Diagnostic Output

```json
{
  "backend": "tensorrt",
  "engine_path": "~/.cache/screenalytics/engines/arcface_r100_v1-sm86.trt",
  "precision": "fp16",
  "batch_size": 32,
  "embedding_dim": 512,
  "throughput_fps": 245.3,
  "latency_ms": 130.5,
  "vram_mb": 1024,
  "validation": {
    "drift_vs_pytorch": 0.9995,
    "regression_test": "passed"
  }
}
```

## Key Files

| File | Purpose |
|------|---------|
| `FEATURES/embedding-engines/src/tensorrt_arcface.py` | TRT inference |
| `FEATURES/embedding-engines/src/arcface_pytorch.py` | PyTorch reference |
| `FEATURES/embedding-engines/src/engine_registry.py` | Backend registry |
| `config/pipeline/embedding.yaml` | Configuration |
| `tests/ml/test_tensorrt_embedding.py` | TRT tests |
| `tests/ml/test_embedding_regression.py` | Drift tests |

## Related Skills

- [pipeline-insights](../pipeline-insights/SKILL.md) - General pipeline debugging
- [face-alignment](../face-alignment/SKILL.md) - Alignment before embedding

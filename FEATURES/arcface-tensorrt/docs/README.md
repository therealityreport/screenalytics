# ArcFace TensorRT Feature

TensorRT-accelerated ArcFace embeddings for high-throughput face recognition.

## Overview

This feature provides:
- ONNX to TensorRT engine conversion
- GPU-accelerated embedding inference
- PyTorch reference comparison for validation
- S3/MinIO engine storage (for team distribution)

## Quick Start

```bash
# Build TensorRT engine from ONNX
python -m FEATURES.arcface_tensorrt --mode build --onnx-path models/arcface_r100.onnx

# Compare TensorRT vs PyTorch embeddings
python -m FEATURES.arcface_tensorrt --mode compare --n-samples 100

# Run benchmark
python -m FEATURES.arcface_tensorrt --mode benchmark

# Get engine info
python -m FEATURES.arcface_tensorrt --mode info --engine-path data/engines/arcface_r100-fp16-sm86.plan
```

## Architecture

```
ONNX Model
    │
    ▼
┌────────────────────┐
│  TensorRT Builder  │  ← Converts ONNX to TensorRT engine
└────────────────────┘
    │
    ├──→ Local Cache (data/engines/)
    │
    └──→ S3/MinIO (engines/arcface/)
            │
            ▼
┌────────────────────┐
│  TensorRT Inferrer │  ← Loads engine, runs inference
└────────────────────┘
    │
    ▼
  Embeddings (512-d)
```

## Engine Naming

Engine files are named with model, precision, and GPU architecture:

```
{model_name}-{precision}-{sm_arch}.plan

Examples:
- arcface_r100-fp16-sm75.plan  (Turing)
- arcface_r100-fp16-sm86.plan  (Ampere)
- arcface_r100-fp16-sm89.plan  (Ada Lovelace)
```

## Configuration

See [config/pipeline/arcface_tensorrt.yaml](../../../config/pipeline/arcface_tensorrt.yaml):

```yaml
tensorrt:
  model_name: arcface_r100
  precision: fp16            # fp32, fp16, int8
  max_batch_size: 32
  engine_local_dir: data/engines

comparison:
  min_cosine_similarity: 0.995
```

## Comparison Metrics

When comparing TensorRT vs PyTorch embeddings:

| Metric | Target | Warning |
|--------|--------|---------|
| Cosine Similarity (mean) | ≥ 0.995 | < 0.990 |
| Cosine Similarity (min) | ≥ 0.990 | < 0.980 |
| L2 Distance (mean) | < 0.05 | > 0.10 |

## Benchmark Results

Expected performance (varies by GPU):

| Batch Size | PyTorch (ms) | TensorRT (ms) | Speedup |
|------------|--------------|---------------|---------|
| 1 | ~25 | ~5 | ~5x |
| 8 | ~50 | ~8 | ~6x |
| 32 | ~150 | ~20 | ~7.5x |

## Usage in Code

```python
from FEATURES.arcface_tensorrt.src import (
    TensorRTArcFace,
    run_tensorrt_embeddings,
)

# Direct usage
embedder = TensorRTArcFace(engine_path="data/engines/arcface_r100-fp16-sm86.plan")
embeddings = embedder.embed(face_images)  # (N, 112, 112, 3) BGR

# Or with auto-build
embedder = TensorRTArcFace()  # Auto-builds engine if not found
embeddings = embedder.embed(face_images)
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| tensorrt | 8.6+ | TensorRT runtime |
| pycuda | 2022.2+ | CUDA memory management |
| numpy | 1.24+ | Array operations |

## Rollback

To disable TensorRT and use PyTorch:

```yaml
# config/pipeline/arcface_tensorrt.yaml
arcface_tensorrt:
  enabled: false
```

Or in main pipeline config:
```yaml
embedding:
  backend: pytorch  # Instead of tensorrt
```

## Testing

```bash
# Run unit tests (no TensorRT required)
pytest FEATURES/arcface-tensorrt/tests/ -v

# Run full tests including TensorRT (requires GPU)
pytest FEATURES/arcface-tensorrt/tests/ -v -m slow
```

## Troubleshooting

### Engine build fails
- Ensure ONNX model exists at specified path
- Check TensorRT version compatibility
- Verify CUDA driver matches TensorRT requirements

### Embedding mismatch
- Check preprocessing matches PyTorch (BGR, normalized)
- Verify engine was built from correct ONNX model
- Try FP32 precision for debugging

### Out of memory
- Reduce max_batch_size in config
- Reduce workspace_size_mb
- Check other GPU processes

## Related Documentation

- [TODO.md](../TODO.md) - Task tracking
- [ACCEPTANCE_MATRIX.md](../../../ACCEPTANCE_MATRIX.md) - Section 3.14

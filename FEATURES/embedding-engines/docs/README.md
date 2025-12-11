# Embedding Engines Feature

## Overview

This feature provides optimized face embedding backends:
- **TensorRT ArcFace** - GPU-accelerated FP16 inference (5x+ speedup)
- **PyTorch Reference** - Baseline for validation
- **ONNXRuntime C++** - Future high-load microservice

## Why This Feature?

Current PyTorch ArcFace is slow for large-scale processing:
- ~50 faces/sec throughput (GPU)
- Memory inefficient eager execution
- Blocking pipeline completion on long episodes

## Components

### TensorRT ArcFace (`src/tensorrt_arcface.py`)
GPU-optimized inference with FP16 precision and batching.

### PyTorch Reference (`src/arcface_pytorch.py`)
Reference implementation for training and validation.

### Engine Registry (`src/engine_registry.py`)
Pluggable backend selection via config.

## Configuration

See `config/pipeline/embedding.yaml`:

```yaml
embedding:
  backend: tensorrt

  tensorrt:
    precision: fp16
    batch_size: 32

  storage:
    type: s3
    bucket: screenalytics-models
    prefix: engines/
```

## Engine Storage

Engines stored in S3/MinIO with versioning:
```
engines/arcface_r100_v1-sm75.trt   # Ampere (RTX 30xx)
engines/arcface_r100_v1-sm86.trt   # Ada (RTX 40xx)
engines/arcface_r100_v1-sm89.trt   # Hopper
```

## Performance

| Backend | Batch | Throughput | Latency |
|---------|-------|------------|---------|
| PyTorch | 32 | ~50 fps | ~640ms |
| TensorRT FP16 | 32 | ~250 fps | ~128ms |

## Related Documentation

- [Full TODO](../../../docs/todo/feature_arcface_tensorrt_onnxruntime.md)
- [Feature Overview](../../../docs/features/vision_alignment_and_body_tracking.md)
- [ACCEPTANCE_MATRIX.md](../../../ACCEPTANCE_MATRIX.md) - Section 3.13

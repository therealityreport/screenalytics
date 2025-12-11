# TensorRT Engine Management

TensorRT engines are GPU-specific optimized inference artifacts that must be rebuilt for different hardware.

## Engine Identity

Each TensorRT engine is uniquely identified by:

| Component | Description | Example |
|-----------|-------------|---------|
| `model_name` | Base model name | `arcface_r100` |
| `model_version` | Model version/hash | `v1.0` or `a1b2c3d4` |
| `precision` | Inference precision | `fp16`, `fp32`, `int8` |
| `gpu_arch` | GPU compute capability | `sm_86` (RTX 3090) |
| `trt_version` | TensorRT version | `8.6.1` |

## Engine Naming Convention

```
{model_name}_{model_version}_{precision}_{gpu_arch}_{trt_version}.trt
```

Examples:
- `arcface_r100_v1.0_fp16_sm_86_8.6.1.trt`
- `arcface_r100_a1b2c3d4_fp32_sm_75_8.5.3.trt`

## GPU Architecture Reference

| GPU | Architecture | Compute Capability |
|-----|--------------|-------------------|
| RTX 4090/4080 | Ada Lovelace | `sm_89` |
| RTX 3090/3080 | Ampere | `sm_86` |
| A100 | Ampere | `sm_80` |
| RTX 2080 Ti | Turing | `sm_75` |
| V100 | Volta | `sm_70` |
| GTX 1080 Ti | Pascal | `sm_61` |

## Remote Engine Caching

Engines can be cached in S3/MinIO to avoid rebuilding on each deployment.

### Storage Structure

```
s3://screenalytics-artifacts/
└── engines/
    └── arcface/
        ├── arcface_r100_v1.0_fp16_sm_86_8.6.1.trt
        ├── arcface_r100_v1.0_fp16_sm_89_8.6.1.trt
        └── arcface_r100_v1.0_fp32_sm_80_8.6.1.trt
```

### Build Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TensorRT Engine Build Flow                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Compute engine identity (model + GPU + TRT version)          │
│                     │                                            │
│                     ▼                                            │
│  2. Check local cache (~/.cache/screenalytics/engines/)          │
│         │                                                        │
│         ├─── Found ──→ Load and return                           │
│         │                                                        │
│         └─── Not found                                           │
│                     │                                            │
│                     ▼                                            │
│  3. Check remote storage (S3/MinIO)                              │
│         │                                                        │
│         ├─── Found ──→ Download, cache locally, return           │
│         │                                                        │
│         └─── Not found                                           │
│                     │                                            │
│                     ▼                                            │
│  4. Build engine from ONNX                                       │
│         │                                                        │
│         ├─── Cache locally                                       │
│         │                                                        │
│         └─── Upload to remote (if credentials available)         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

```bash
# Remote storage (optional)
export ARTIFACTS_ENDPOINT="http://minio:9000"
export ARTIFACTS_ACCESS_KEY="..."
export ARTIFACTS_SECRET_KEY="..."

# Local cache directory (optional)
export TENSORRT_CACHE_DIR="/path/to/cache"
```

### Usage

```python
from FEATURES.arcface_tensorrt.src.tensorrt_builder import build_or_load_engine

# Automatically handles local/remote caching
engine, source = build_or_load_engine(
    onnx_path="models/arcface_r100.onnx",
    precision="fp16",
    batch_size=32,
)

print(f"Engine loaded from: {source}")
# "local_cache" | "remote" | "built"
```

## Rebuilding Engines

Engines should be rebuilt when:
1. TensorRT version changes
2. GPU architecture changes
3. Model weights change
4. Precision requirements change

```bash
# Force rebuild
python -m FEATURES.arcface_tensorrt.build --force-rebuild

# Clear local cache
rm -rf ~/.cache/screenalytics/engines/
```

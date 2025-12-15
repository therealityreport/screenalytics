# TODO: ArcFace TensorRT + ONNXRuntime Embedding Engines

Version: 1.0
Status: PARTIAL (TensorRT implemented; registry/service planned)
Owner: Engineering
Created: 2025-12-11
TTL: 2026-01-10

**Feature Sandbox:** `FEATURES/arcface_tensorrt/` (implementation), `FEATURES/embedding_engines/` (planning/docs)

---

## Problem Statement

PyTorch ArcFace inference is slow for large-scale processing:

- **Throughput bottleneck** - Current: ~50 faces/sec (GPU), need 5-10x for batch processing
- **Memory inefficient** - PyTorch eager mode has overhead vs optimized engines
- **No batching optimization** - Current implementation doesn't fully utilize GPU parallelism

**Impact:** Long embedding times on episodes with many faces, blocking pipeline completion.

**Goal:** 5x+ speedup via TensorRT, with ONNXRuntime as future C++ option.

---

## Dependencies

| Library | Version | License | Purpose |
|---------|---------|---------|---------|
| tensorrt | 8.6+ | NVIDIA EULA | TensorRT runtime |
| torch2trt | 0.4+ | MIT | PyTorch to TRT conversion |
| onnx | 1.14+ | Apache-2.0 | ONNX model format |
| onnxruntime | 1.17+ | MIT | ONNX inference (future) |
| insightface | 0.7+ | MIT | Reference ArcFace |

**Installation:**
```bash
# TensorRT (requires CUDA)
pip install tensorrt>=8.6.0

# For ONNX path
pip install onnx>=1.14.0 onnxruntime-gpu>=1.17.0

# Reference implementation
pip install insightface>=0.7.0
```

---

## Implementation Tasks

### Phase A: TensorRT ArcFace

**Goal:** Build and deploy TensorRT engine for ArcFace embedding.

- [x] **A1.** Implement TensorRT inference wrapper: `FEATURES/arcface_tensorrt/src/tensorrt_inference.py`
  ```python
  class TensorRTArcFace:
      """
      ArcFace embedding using TensorRT.

      Supports FP16/FP32 precision, batched inference.
      """
      def __init__(
          self,
          engine_path: str,
          precision: str = "fp16",
          batch_size: int = 32,
          device: int = 0
      ):
          self.engine = self._load_engine(engine_path)
          self.context = self.engine.create_execution_context()
          ...

      def encode(self, faces: np.ndarray) -> np.ndarray:
          """
          Input: (N, 112, 112, 3) BGR faces
          Output: (N, 512) L2-normalized embeddings
          """
  ```

- [x] **A2.** Implement engine building + caching: `FEATURES/arcface_tensorrt/src/tensorrt_builder.py`
  ```python
  def build_trt_engine(
      onnx_path: str,
      output_path: str,
      precision: str = "fp16",
      max_batch_size: int = 64,
      workspace_gb: float = 2.0
  ) -> str:
      """
      Convert ONNX model to TensorRT engine.

      Engine is GPU-architecture specific (sm_XX).
      """
      import tensorrt as trt

      builder = trt.Builder(TRT_LOGGER)
      network = builder.create_network(...)
      parser = trt.OnnxParser(network, TRT_LOGGER)

      # Parse ONNX
      parser.parse_from_file(onnx_path)

      # Configure precision
      config = builder.create_builder_config()
      if precision == "fp16":
          config.set_flag(trt.BuilderFlag.FP16)

      # Build and serialize
      engine = builder.build_serialized_network(network, config)
      ...
  ```

- [x] **A3.** Implement optional S3 engine download/upload (boto3): `FEATURES/arcface_tensorrt/src/tensorrt_builder.py`
  ```python
  class EngineStorage:
      """
      Store and retrieve TensorRT engines from S3/MinIO.

      Versioning: {model_name}-{model_version}-sm{arch}.trt
      Example: arcface_r100_v1-sm75.trt
      """
      def __init__(self, bucket: str, prefix: str = "engines/"):
          self.s3_client = boto3.client("s3")
          ...

      def get_engine(self, model_name: str, sm_arch: int) -> str:
          """
          Download engine to local cache if not present.
          Falls back to building locally if not in S3.
          """

      def upload_engine(self, local_path: str, model_name: str, sm_arch: int):
          """Upload built engine to S3."""
  ```

- [x] **A4.** Add local build/load fallback: `build_or_load_engine()` in `FEATURES/arcface_tensorrt/src/tensorrt_builder.py`
  ```python
  def get_or_build_engine(config: EngineConfig) -> str:
      """
      Get TensorRT engine, building locally if needed.

      1. Check local cache
      2. Try download from S3
      3. Build from ONNX if not available
      """
      local_path = f"~/.cache/screenalytics/engines/{engine_name}"

      if os.path.exists(local_path):
          return local_path

      try:
          return storage.get_engine(model_name, sm_arch)
      except EngineNotFound:
          logger.info("Building TRT engine locally (first run)")
          return build_trt_engine(onnx_path, local_path, config)
  ```

- [x] **A5.** Create config: `config/pipeline/embedding.yaml`
  ```yaml
  embedding:
    backend: pytorch                 # pytorch | tensorrt
    tensorrt_config: config/pipeline/arcface_tensorrt.yaml

  tensorrt:
    precision: fp16                  # fp16 | fp32 | int8
    batch_size: 32
    max_batch_size: 64
    workspace_gb: 2.0

  storage:
    type: s3                         # s3 | local
    bucket: screenalytics-models
    prefix: engines/
    cache_dir: ~/.cache/screenalytics/engines

  fallback:
    build_locally: true
    fallback_to_pytorch: true

  validation:
    enabled: true
    max_drift_cosine: 0.001
  ```

- [x] **A6.** Write tests: `FEATURES/arcface_tensorrt/tests/test_tensorrt_embedding.py`
  - Test engine loading and inference
  - Test batch inference
  - Test precision modes (fp16, fp32)
  - Benchmark vs PyTorch baseline

**Acceptance Criteria (Phase A):**
- [ ] Speedup: ≥5x at batch_size=32 vs PyTorch
- [ ] VRAM usage: ≤2GB for TRT engine
- [ ] Engine build time: ≤5 minutes

---

### Phase B: Embedding Drift Validation

**Goal:** Ensure TensorRT embeddings match PyTorch reference.

- [x] **B1.** Use `insightface` as the reference ArcFace runtime (in-process)
  - Pipeline runtime: `tools/episode_run.py` (default backend)
  - Model fetch: `scripts/fetch_models.py`
  ```python
  class PyTorchArcFace:
      """
      Reference ArcFace implementation using InsightFace.

      Used for:
      - Training/fine-tuning baseline
      - Embedding drift validation
      - Fallback when TRT unavailable
      """
      def __init__(self, model_name: str = "arcface_r100_v1"):
          from insightface.model_zoo import get_model
          self.model = get_model(model_name)
          ...
  ```

- [x] **B2.** Implement embedding comparison: `FEATURES/arcface_tensorrt/src/embedding_compare.py`
  ```python
  def compare_embeddings(
      pytorch_embeddings: np.ndarray,
      tensorrt_embeddings: np.ndarray
  ) -> EmbeddingDriftReport:
      """
      Compare embeddings from two backends.

      Returns cosine similarity statistics.
      """
      cosine_sims = np.sum(pytorch_embeddings * tensorrt_embeddings, axis=1)

      return EmbeddingDriftReport(
          mean_cosine_sim=cosine_sims.mean(),
          min_cosine_sim=cosine_sims.min(),
          max_drift_face_idx=cosine_sims.argmin(),
          drift_histogram=np.histogram(cosine_sims, bins=20)
      )
  ```

- [x] **B3.** Provide parity utilities + unit coverage (GPU parity still needs real hardware)
  ```python
  def run_embedding_regression_tests(
      eval_faces: np.ndarray,
      pytorch_embedder: PyTorchArcFace,
      tensorrt_embedder: TensorRTArcFace,
      max_drift: float = 0.001
  ) -> bool:
      """
      Validate TensorRT embeddings match PyTorch.

      Fails if any face has cosine_sim < (1 - max_drift).
      """
      pytorch_emb = pytorch_embedder.encode(eval_faces)
      tensorrt_emb = tensorrt_embedder.encode(eval_faces)

      report = compare_embeddings(pytorch_emb, tensorrt_emb)

      assert report.min_cosine_sim >= (1 - max_drift), \
          f"Embedding drift too high: {report.min_cosine_sim}"

      return True
  ```

- [x] **B4.** Add drift check config
  ```yaml
  # config/pipeline/embedding.yaml
  validation:
    enabled: true
    max_drift_cosine: 0.001     # Max tolerated drift
    eval_set_size: 100          # Faces to test
    run_on_startup: true        # Validate when loading engine
  ```

- [x] **B5.** Write tests (synthetic/unit + ML-gated)
  - `FEATURES/arcface_tensorrt/tests/test_tensorrt_embedding.py`
  - `tests/ml/test_arcface_embeddings.py` (ML-gated; validates embedding invariants)
  - Test PyTorch vs TensorRT on eval set
  - Test different batch sizes produce same results
  - Test FP16 vs FP32 drift

**Acceptance Criteria (Phase B):**
- [ ] Embedding drift: cosine_sim ≥ 0.999 (max drift 0.001)
- [ ] FP16 accuracy delta: <0.1% on recognition tasks
- [ ] Regression tests pass on CI

---

### Phase C: Pluggable Backend Architecture

**Goal:** Unified interface for multiple embedding backends.

- [ ] **C1.** Create `FEATURES/embedding_engines/src/engine_registry.py` (optional; `FEATURES/embedding_engines/` is docs-only today)
  ```python
  class EmbeddingEngine(Protocol):
      """Protocol for embedding backends."""

      def encode(self, faces: np.ndarray) -> np.ndarray:
          """Encode faces to embeddings."""
          ...

      @property
      def embedding_dim(self) -> int:
          """Return embedding dimensionality."""
          ...


  class EngineRegistry:
      """
      Registry for embedding backends.

      Allows runtime backend selection via config.
      """
      _engines: Dict[str, Type[EmbeddingEngine]] = {}

      @classmethod
      def register(cls, name: str):
          def decorator(engine_cls):
              cls._engines[name] = engine_cls
              return engine_cls
          return decorator

      @classmethod
      def get(cls, name: str, **kwargs) -> EmbeddingEngine:
          return cls._engines[name](**kwargs)


  @EngineRegistry.register("pytorch")
  class PyTorchArcFace(EmbeddingEngine):
      ...

  @EngineRegistry.register("tensorrt")
  class TensorRTArcFace(EmbeddingEngine):
      ...

  @EngineRegistry.register("onnx")
  class ONNXArcFace(EmbeddingEngine):
      ...
  ```

- [x] **C2.** Backend is switchable via config today
  - `config/pipeline/embedding.yaml` selects `embedding.backend: pytorch|tensorrt`
  - `tools/episode_run.py` uses `get_embedding_backend()` to select + initialize the backend
  ```python
  # In tools/episode_run.py
  def _init_embedder(config: EmbedConfig) -> EmbeddingEngine:
      return EngineRegistry.get(
          config.backend,
          **config.backend_kwargs
      )
  ```

- [x] **C3.** Automatic fallback exists (TRT → PyTorch) via embedding config + backend loader
  ```python
  class FallbackEmbedder(EmbeddingEngine):
      """
      Try primary backend, fall back to secondary on failure.
      """
      def __init__(self, primary: str, fallback: str, **kwargs):
          try:
              self.engine = EngineRegistry.get(primary, **kwargs)
          except EngineInitError:
              logger.warning(f"{primary} unavailable, using {fallback}")
              self.engine = EngineRegistry.get(fallback, **kwargs)
  ```

**Acceptance Criteria (Phase C):**
- [ ] Backend switchable via config without code changes
- [ ] Automatic fallback works when TRT unavailable
- [ ] All backends produce compatible embeddings

---

### Phase D: ONNXRuntime C++ Service (Future)

**Goal:** Architecture for high-load C++ embedding service.

**Note:** This phase is documentation/planning only for now.

- [ ] **D1.** Create architecture doc
  ```markdown
  # ONNXRuntime C++ Embedding Service

  ## Overview
  Standalone C++ microservice for high-throughput embedding.

  ## Communication
  - gRPC for structured requests
  - Shared memory for large batches (optional)

  ## Interface
  ```protobuf
  service EmbeddingService {
    rpc Embed(EmbedRequest) returns (EmbedResponse);
    rpc EmbedBatch(EmbedBatchRequest) returns (EmbedBatchResponse);
    rpc Health(HealthRequest) returns (HealthResponse);
  }
  ```

  ## Deployment
  - Docker container with CUDA support
  - Kubernetes deployment with HPA
  - Model loading from S3/MinIO
  ```

- [ ] **D2.** Define migration path
  - Phase 1: TensorRT in Python (current)
  - Phase 2: ONNXRuntime in Python (optional)
  - Phase 3: ONNXRuntime C++ service (high load)

- [ ] **D3.** Stub ONNX backend in registry
  ```python
  @EngineRegistry.register("onnx")
  class ONNXArcFace(EmbeddingEngine):
      """
      ONNX Runtime backend.

      Can use CUDA EP, TensorRT EP, or CPU EP.
      Future: Will call C++ service instead of local inference.
      """
      def __init__(self, model_path: str, providers: List[str] = None):
          import onnxruntime as ort

          if providers is None:
              providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

          self.session = ort.InferenceSession(model_path, providers=providers)
  ```

**Acceptance Criteria (Phase D):**
- [ ] Architecture doc complete
- [ ] Migration path documented
- [ ] ONNX backend stub registered

---

## Integration Checklist

### Code Integration

- [x] Backend selection + fallback integrated in `tools/episode_run.py`
  - Backend loader: `get_embedding_backend()` (selects `pytorch` vs `tensorrt`)
  - Config: `config/pipeline/embedding.yaml` + `config/pipeline/arcface_tensorrt.yaml`
  - Validation knobs: `embedding.yaml` → `validation.*` (runtime behavior may be environment-dependent)

- [ ] Update `py_screenalytics/pipeline/constants.py`:
  - Add engine storage paths
  - Add default config values

### Config Integration

- [x] `config/pipeline/embedding.yaml` (backend selection + gating + validation)
- [x] `config/pipeline/arcface_tensorrt.yaml` (TensorRT builder/runtime configuration)
- [ ] Update `EpisodeRunConfig` with embedding backend fields (only if exposing via API payload/schema)

### CI/CD Integration

- [ ] Add TRT engine build job
- [ ] Upload engines to S3 after build
- [ ] Add embedding regression tests to CI

### Storage Setup

- [ ] Create S3 bucket/prefix for engines
- [ ] Set up engine versioning
- [ ] Document engine naming convention

---

## Benchmarks

### Target Performance

| Backend | Batch Size | Throughput | Latency | VRAM |
|---------|------------|------------|---------|------|
| PyTorch | 32 | ~50 fps | ~640ms | 2GB |
| TensorRT FP16 | 32 | ~250 fps | ~128ms | 1GB |
| TensorRT FP32 | 32 | ~180 fps | ~178ms | 1.5GB |
| ONNX TRT EP | 32 | ~230 fps | ~140ms | 1.2GB |

### Benchmark Script

```python
def benchmark_embedder(
    embedder: EmbeddingEngine,
    num_faces: int = 1000,
    batch_size: int = 32,
    warmup_batches: int = 5
) -> BenchmarkResult:
    """
    Benchmark embedding throughput.
    """
    faces = np.random.rand(num_faces, 112, 112, 3).astype(np.float32)

    # Warmup
    for i in range(warmup_batches):
        embedder.encode(faces[:batch_size])

    # Timed run
    start = time.perf_counter()
    for i in range(0, num_faces, batch_size):
        batch = faces[i:i+batch_size]
        embedder.encode(batch)
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        throughput=num_faces / elapsed,
        latency_ms=(elapsed / (num_faces / batch_size)) * 1000,
        batch_size=batch_size
    )
```

---

## Milestones

| Phase | Target Date | Deliverable |
|-------|-------------|-------------|
| Phase A | +2 weeks | TensorRT engine, S3 storage |
| Phase B | +3 weeks | Drift validation, regression tests |
| Phase C | +4 weeks | Pluggable backend architecture |
| Phase D | +5 weeks | ONNX C++ architecture doc |
| Promotion | +6 weeks | Move to `py_screenalytics/embedding/` |

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TRT version incompatibility | Medium | Medium | Pin TRT version, rebuild engines |
| CUDA version mismatch | Medium | Medium | Document CUDA requirements |
| FP16 accuracy issues | Low | Medium | Validate on domain data, use FP32 fallback |
| Engine build takes too long | Low | Low | Pre-build in CI, cache engines |

---

## References

- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [InsightFace ArcFace](https://github.com/deepinsight/insightface)
- [ONNXRuntime C++ API](https://onnxruntime.ai/docs/api/c/)
- [tensorrtx ArcFace](https://github.com/wang-xinyu/tensorrtx/tree/master/arcface)

---

## Related Documents

- [Feature Overview](../features/vision_alignment_and_body_tracking.md)
- [ACCEPTANCE_MATRIX.md](../../ACCEPTANCE_MATRIX.md) - Section 3.14
- [Skill: embedding-engine](../../.claude/skills/embedding-engine/SKILL.md)

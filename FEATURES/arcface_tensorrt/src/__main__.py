#!/usr/bin/env python
"""
CLI entrypoint for ArcFace TensorRT sandbox.

Usage:
    # Build TensorRT engine from ONNX
    python -m FEATURES.arcface_tensorrt --mode build

    # Compare TensorRT vs PyTorch embeddings
    python -m FEATURES.arcface_tensorrt --mode compare --n-samples 100

    # Run benchmark
    python -m FEATURES.arcface_tensorrt --mode benchmark

    # Get engine info
    python -m FEATURES.arcface_tensorrt --mode info --engine-path path/to/engine.plan
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


logger = logging.getLogger(__name__)


def cmd_build(args):
    """Build TensorRT engine from ONNX."""
    from .tensorrt_builder import TensorRTConfig, build_or_load_engine

    config = TensorRTConfig(
        model_name=args.model_name,
        precision=args.precision,
        max_batch_size=args.max_batch_size,
    )

    if args.onnx_path:
        config.onnx_path = Path(args.onnx_path)

    if args.output_dir:
        config.engine_local_dir = Path(args.output_dir)

    engine_path, was_built = build_or_load_engine(config, force_rebuild=args.force)

    if engine_path:
        action = "Built" if was_built else "Loaded cached"
        print(f"\n{action} engine: {engine_path}")

        # Show engine info
        from .tensorrt_builder import get_engine_info
        info = get_engine_info(engine_path)
        print(f"  Size: {info.get('size_mb', 0):.1f} MB")
        print(f"  Bindings: {info.get('num_bindings', 0)}")
        return 0
    else:
        print("\nFailed to build engine")
        return 1


def cmd_compare(args):
    """Compare TensorRT vs PyTorch embeddings."""
    from .embedding_compare import compare_backends

    engine_path = Path(args.engine_path) if args.engine_path else None

    result = compare_backends(
        n_samples=args.n_samples,
        tensorrt_engine_path=engine_path,
        min_cosine_sim=args.min_cosine_sim,
        batch_size=args.batch_size,
    )

    print("\n" + "=" * 60)
    print("EMBEDDING COMPARISON RESULTS")
    print("=" * 60)
    print(f"\nSamples: {result.n_samples}")
    print(f"\nCosine Similarity:")
    print(f"  Mean: {result.cosine_sim_mean:.6f}")
    print(f"  Std:  {result.cosine_sim_std:.6f}")
    print(f"  Min:  {result.cosine_sim_min:.6f}")
    print(f"  Max:  {result.cosine_sim_max:.6f}")
    print(f"\nL2 Distance:")
    print(f"  Mean: {result.l2_dist_mean:.6f}")
    print(f"  Max:  {result.l2_dist_max:.6f}")
    print(f"\nTiming:")
    print(f"  PyTorch:   {result.pytorch_time_ms:.2f} ms")
    print(f"  TensorRT:  {result.tensorrt_time_ms:.2f} ms")
    print(f"  Speedup:   {result.speedup:.2f}x")
    print(f"\nValidation: {'PASSED' if result.passed else 'FAILED'}")

    if result.failure_reason:
        print(f"  Reason: {result.failure_reason}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return 0 if result.passed else 1


def cmd_benchmark(args):
    """Run TensorRT benchmark."""
    from .embedding_compare import run_benchmark

    engine_path = Path(args.engine_path) if args.engine_path else None

    if engine_path is None:
        from .tensorrt_builder import build_or_load_engine
        engine_path, _ = build_or_load_engine()
        if engine_path is None:
            print("No engine available. Build one first with --mode build")
            return 1

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    results = run_benchmark(
        engine_path,
        n_iterations=args.iterations,
        batch_sizes=batch_sizes,
    )

    print("\n" + "=" * 60)
    print("TENSORRT BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\nEngine: {engine_path}")
    print(f"Iterations per batch: {args.iterations}")
    print("\n{:>10} {:>12} {:>12} {:>12}".format(
        "Batch", "Mean (ms)", "Std (ms)", "Images/sec"
    ))
    print("-" * 50)

    for bs in sorted(results.keys()):
        mean_ms, std_ms = results[bs]
        images_per_sec = bs * 1000 / mean_ms if mean_ms > 0 else 0
        print(f"{bs:>10} {mean_ms:>12.2f} {std_ms:>12.2f} {images_per_sec:>12.1f}")

    return 0


def cmd_info(args):
    """Show TensorRT engine info."""
    from .tensorrt_builder import get_engine_info

    engine_path = Path(args.engine_path)

    if not engine_path.exists():
        print(f"Engine not found: {engine_path}")
        return 1

    info = get_engine_info(engine_path)

    if "error" in info:
        print(f"Error: {info['error']}")
        return 1

    print("\n" + "=" * 60)
    print("TENSORRT ENGINE INFO")
    print("=" * 60)
    print(f"\nPath: {info['path']}")
    print(f"Size: {info['size_mb']:.1f} MB")
    print(f"Bindings: {info['num_bindings']}")

    for binding in info.get("bindings", []):
        io_type = "Input" if binding["is_input"] else "Output"
        print(f"\n  {io_type}: {binding['name']}")
        print(f"    Shape: {binding['shape']}")
        print(f"    DType: {binding['dtype']}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="ArcFace TensorRT Embedding Sandbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["build", "compare", "benchmark", "info"],
        required=True,
        help="Operation mode",
    )

    # Build options
    parser.add_argument(
        "--model-name",
        default="arcface_r100",
        help="Model name (default: arcface_r100)",
    )

    parser.add_argument(
        "--onnx-path",
        type=str,
        help="Path to ONNX model file",
    )

    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="TensorRT precision (default: fp16)",
    )

    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Maximum batch size (default: 32)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for engine",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if cached",
    )

    # Compare options
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples for comparison (default: 100)",
    )

    parser.add_argument(
        "--min-cosine-sim",
        type=float,
        default=0.995,
        help="Minimum acceptable cosine similarity (default: 0.995)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )

    # Benchmark options
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Iterations per batch size (default: 100)",
    )

    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8,16,32",
        help="Comma-separated batch sizes to test",
    )

    # Common options
    parser.add_argument(
        "--engine-path",
        type=str,
        help="Path to TensorRT engine file",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Dispatch to command handler
    if args.mode == "build":
        return cmd_build(args)
    elif args.mode == "compare":
        return cmd_compare(args)
    elif args.mode == "benchmark":
        return cmd_benchmark(args)
    elif args.mode == "info":
        return cmd_info(args)


if __name__ == "__main__":
    sys.exit(main())

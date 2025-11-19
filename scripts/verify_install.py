#!/usr/bin/env python3
"""
Installation verification script for Screenalytics.

Checks for platform-specific requirements and provides actionable error messages.
D3: Explicit CoreML runtime validation on Apple Silicon.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

# Color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (ARM-based Mac)."""
    return sys.platform == "darwin" and platform.machine().lower().startswith(("arm", "aarch64"))


def check_coreml_runtime() -> tuple[bool, str]:
    """
    Check if CoreML ONNX runtime is available.

    Returns:
        (success, message): True if available or not needed, False if required but missing
    """
    try:
        import onnxruntime as ort  # type: ignore

        providers = ort.get_available_providers()
        has_coreml = any(provider.lower().startswith("coreml") for provider in providers)

        if has_coreml:
            return True, f"{GREEN}✓{RESET} CoreML ONNX runtime is available"

        if is_apple_silicon():
            # Critical: CoreML is required on Apple Silicon for good performance
            return False, (
                f"{RED}✗{RESET} {BOLD}CoreML ONNX runtime is missing on Apple Silicon{RESET}\n\n"
                f"  Without CoreML, face detection will be extremely slow and may cause\n"
                f"  thermal throttling, leading to a poor user experience.\n\n"
                f"  {BOLD}Solution:{RESET} Install onnxruntime-coreml:\n"
                f"    pip uninstall -y onnxruntime\n"
                f"    pip install onnxruntime-coreml\n\n"
                f"  Note: onnxruntime and onnxruntime-coreml cannot coexist.\n"
                f"        The coreml variant includes CPU fallback support."
            )
        else:
            # Not critical: CPU/CUDA on non-Apple platforms is acceptable
            return True, f"{YELLOW}⚠{RESET} CoreML not available (CPU/CUDA will be used)"

    except ImportError:
        return False, (
            f"{RED}✗{RESET} {BOLD}onnxruntime is not installed{RESET}\n\n"
            f"  {BOLD}Solution:{RESET} Install onnxruntime:\n"
            f"    pip install onnxruntime-coreml  # For Apple Silicon\n"
            f"    pip install onnxruntime          # For other platforms"
        )


def check_insightface() -> tuple[bool, str]:
    """Check if InsightFace is available."""
    try:
        import insightface  # type: ignore

        return True, f"{GREEN}✓{RESET} InsightFace is installed"
    except ImportError:
        return False, (
            f"{RED}✗{RESET} {BOLD}InsightFace is not installed{RESET}\n\n"
            f"  {BOLD}Solution:{RESET} Install insightface:\n"
            f"    pip install insightface"
        )


def check_opencv() -> tuple[bool, str]:
    """Check if OpenCV is available."""
    try:
        import cv2  # type: ignore

        return True, f"{GREEN}✓{RESET} OpenCV is installed (version {cv2.__version__})"
    except ImportError:
        return False, (
            f"{RED}✗{RESET} {BOLD}OpenCV is not installed{RESET}\n\n"
            f"  {BOLD}Solution:{RESET} Install opencv-python:\n"
            f"    pip install opencv-python"
        )


def check_numpy() -> tuple[bool, str]:
    """Check if NumPy is available."""
    try:
        import numpy as np  # type: ignore

        return True, f"{GREEN}✓{RESET} NumPy is installed (version {np.__version__})"
    except ImportError:
        return False, (
            f"{RED}✗{RESET} {BOLD}NumPy is not installed{RESET}\n\n"
            f"  {BOLD}Solution:{RESET} Install numpy:\n"
            f"    pip install numpy"
        )


def check_models() -> tuple[bool, str]:
    """Check if required model files exist."""
    try:
        from py_screenalytics.artifacts import get_path  # type: ignore

        # Check for InsightFace model home
        insightface_home = Path.home() / ".insightface"
        if not insightface_home.exists():
            return False, (
                f"{RED}✗{RESET} {BOLD}InsightFace models not downloaded{RESET}\n\n"
                f"  {BOLD}Solution:{RESET} Run the model fetcher:\n"
                f"    python scripts/fetch_models.py"
            )

        # Check for buffalo_l model pack
        buffalo_models = insightface_home / "models" / "buffalo_l"
        if not buffalo_models.exists():
            return False, (
                f"{RED}✗{RESET} {BOLD}buffalo_l model pack not found{RESET}\n\n"
                f"  {BOLD}Solution:{RESET} Run the model fetcher:\n"
                f"    python scripts/fetch_models.py"
            )

        return True, f"{GREEN}✓{RESET} Model files are present"
    except Exception as e:
        return False, f"{YELLOW}⚠{RESET} Could not verify models: {e}"


def main() -> int:
    """Run all verification checks."""
    print(f"\n{BOLD}Screenalytics Installation Verification{RESET}\n")
    print(f"Platform: {platform.system()} {platform.machine()}")
    if is_apple_silicon():
        print(f"Detected: {BOLD}Apple Silicon{RESET} (CoreML acceleration recommended)\n")
    else:
        print()

    checks = [
        ("NumPy", check_numpy),
        ("OpenCV", check_opencv),
        ("ONNX Runtime (CoreML)", check_coreml_runtime),
        ("InsightFace", check_insightface),
        ("Model Files", check_models),
    ]

    results: list[tuple[str, bool, str]] = []
    all_passed = True

    for name, check_fn in checks:
        success, message = check_fn()
        results.append((name, success, message))
        if not success:
            all_passed = False

    # Print results
    for name, success, message in results:
        print(message)

    print()
    if all_passed:
        print(f"{GREEN}{BOLD}✓ All checks passed!{RESET}")
        print(f"\nYou can now run Screenalytics with:")
        print(f"  python tools/episode_run.py <episode_id> <video_path>")
        return 0
    else:
        print(f"{RED}{BOLD}✗ Some checks failed{RESET}")
        print(f"\nPlease address the issues above before running Screenalytics.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

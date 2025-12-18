from __future__ import annotations


def resolve_torch_device_request(device: str | None) -> tuple[str, str, str | None]:
    """Resolve a torch device string from a pipeline device request.

    Torch does not accept "coreml" as a device string. For convenience, treat:
      - requested "coreml" → torch "mps" (else "cpu")
      - requested "metal"/"apple" → torch "mps" (else "cpu")

    Returns:
        (requested_normalized, resolved_device, fallback_reason)
    """
    requested_raw = (device or "auto").strip().lower()

    def _has_cuda() -> bool:
        try:
            import torch  # type: ignore

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _has_mps() -> bool:
        try:
            import torch  # type: ignore

            mps = getattr(torch.backends, "mps", None)
            return bool(mps is not None and mps.is_available())
        except Exception:
            return False

    if requested_raw in {"0", "cuda", "gpu"}:
        requested = "cuda"
        if _has_cuda():  # pragma: no cover - depends on env
            return requested, "cuda", None
        return requested, "cpu", "CUDA requested but not available; using cpu"

    if requested_raw in {"mps", "metal", "apple"}:
        requested = "mps"
        if _has_mps():  # pragma: no cover - mac only
            return requested, "mps", None
        return requested, "cpu", "MPS requested but not available; using cpu"

    if requested_raw == "coreml":
        requested = "mps"
        if _has_mps():  # pragma: no cover - mac only
            return requested, "mps", "CoreML requested; torch models run on mps"
        return requested, "cpu", "CoreML requested; mps not available; torch models run on cpu"

    if requested_raw == "cpu":
        return "cpu", "cpu", None

    # auto (or unknown): prefer CUDA, then MPS, then CPU.
    if _has_cuda():  # pragma: no cover - depends on env
        return "cuda", "cuda", "auto: selected cuda"
    if _has_mps():  # pragma: no cover - mac only
        return "mps", "mps", "auto: selected mps"
    return "cpu", "cpu", "auto: selected cpu"


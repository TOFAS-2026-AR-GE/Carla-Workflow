"""Inference device selection shared by the perception models."""


def resolve_device(requested_device: str) -> str:
    """Return a usable Ultralytics device without assuming CUDA exists."""
    requested = str(requested_device).strip().lower() or "auto"

    if requested == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        # Ultralytics normally installs torch. Keeping this fallback makes the
        # error message from model loading clearer on incomplete setups.
        return "cpu"

    cuda_available = bool(torch.cuda.is_available())

    if requested == "auto":
        return "0" if cuda_available else "cpu"

    wants_cuda = requested.isdigit() or requested.startswith("cuda")
    if wants_cuda and not cuda_available:
        print(
            f"[WARN] CUDA cihazi {requested_device!r} kullanilamiyor; "
            "inference CPU ile devam edecek."
        )
        return "cpu"

    return str(requested_device).strip()


def move_model_to_cpu(model) -> None:
    """Move a PyTorch-backed model to CPU when the backend supports it."""
    try:
        model.to("cpu")
    except (AttributeError, RuntimeError, TypeError):
        # ONNX backends select their provider during predict instead.
        pass

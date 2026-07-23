"""Algılama modellerinin CPU veya ekran kartı seçimini yapar."""


def is_cuda_device(device):
    """Ultralytics ve PyTorch cihaz yazımlarının CUDA olup olmadığını söyler."""
    normalized = str(device).strip().lower()
    return normalized.isdigit() or normalized.startswith("cuda")


def configure_torch_inference():
    """CUDA bulunduğunda sabit boyutlu inference için hızlı PyTorch ayarlarını açar."""
    try:
        import torch
    except ImportError:
        return False

    if not torch.cuda.is_available():
        return False

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except (AttributeError, RuntimeError):
        pass
    return True


def recover_cuda_memory():
    """Geçici CUDA bellek baskısından sonra aynı cihazda bir kez daha denemeyi sağlar."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_device(requested_device):
    """CUDA yoksa güvenli biçimde CPU kullanımına döner."""
    requested = str(requested_device).strip().lower() or "auto"

    if requested == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        # Eksik kurulumda model yükleyicinin anlaşılır hata vermesi için
        # burada doğrudan CPU seçilir.
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


def move_model_to_cpu(model):
    """Model destekliyorsa modeli CPU üzerine taşır."""
    try:
        model.to("cpu")
    except (AttributeError, RuntimeError, TypeError):
        # ONNX modelleri çalışacağı aygıtı tahmin çağrısında seçer.
        pass

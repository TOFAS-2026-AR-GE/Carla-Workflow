"""Algılama modellerinin CPU veya ekran kartı seçimini yapar."""

import gc


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
        gc.collect()
        torch.cuda.empty_cache()


def cuda_free_memory_mb():
    """CUDA aygıtında diğer süreçler sonrası kalan yaklaşık belleği döndürür."""
    try:
        import torch
    except ImportError:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
    except (RuntimeError, AttributeError):
        return 0.0
    return float(free_bytes) / (1024.0 * 1024.0)


def resolve_device(requested_device, minimum_free_memory_mb=0.0):
    """CUDA yoksa veya yeterli boş VRAM kalmadıysa güvenli biçimde CPU seçer."""
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

    wants_cuda = requested.isdigit() or requested.startswith("cuda")
    if wants_cuda and not cuda_available:
        print(
            f"[WARN] CUDA cihazi {requested_device!r} kullanilamiyor; "
            "inference CPU ile devam edecek."
        )
        return "cpu"

    if requested == "auto":
        requested = "0" if cuda_available else "cpu"
        wants_cuda = cuda_available

    if wants_cuda and minimum_free_memory_mb > 0.0:
        free_memory_mb = cuda_free_memory_mb()
        if 0.0 < free_memory_mb < float(minimum_free_memory_mb):
            print(
                f"[WARN] CUDA bos bellek {free_memory_mb:.0f} MB; "
                f"gereken guvenli pay {float(minimum_free_memory_mb):.0f} MB. "
                "Model CPU ile acilacak."
            )
            return "cpu"

    return requested


def move_model_to_cpu(model):
    """Model destekliyorsa modeli CPU üzerine taşır."""
    try:
        model.to("cpu")
    except (AttributeError, RuntimeError, TypeError):
        # ONNX modelleri çalışacağı aygıtı tahmin çağrısında seçer.
        pass


def release_ultralytics_cuda(model):
    """Ultralytics predictor önbelleğini ve CUDA tensörlerini tamamen bırakır."""
    try:
        model.predictor = None
    except AttributeError:
        pass
    move_model_to_cpu(model)
    recover_cuda_memory()

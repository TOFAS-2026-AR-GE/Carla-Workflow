"""Bilgisayar kapasitesine göre güvenli algılama çalışma profili seçer."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceProfile:
    """Başlangıçta kullanılacak sade performans ayarları."""

    name: str
    cuda_available: bool
    total_vram_mb: float
    free_vram_mb: float
    perception_every_n_frames: int
    maximum_perception_period: int
    vehicle_image_size: int
    camera_inference_batch_size: int
    camera_wait_timeout_ms: float
    navigation_render_every_n_frames: int


def choose_performance_profile(
    cuda_available,
    total_vram_mb=0.0,
    free_vram_mb=0.0,
):
    """Toplam ve boş VRAM'e göre dengeli bir başlangıç profili döndürür."""
    total_vram_mb = max(0.0, float(total_vram_mb))
    free_vram_mb = max(0.0, float(free_vram_mb))

    if not cuda_available:
        return PerformanceProfile(
            name="cpu",
            cuda_available=False,
            total_vram_mb=total_vram_mb,
            free_vram_mb=free_vram_mb,
            perception_every_n_frames=2,
            maximum_perception_period=3,
            vehicle_image_size=416,
            camera_inference_batch_size=1,
            camera_wait_timeout_ms=6.0,
            navigation_render_every_n_frames=3,
        )

    if total_vram_mb >= 20_000.0 and free_vram_mb >= 8_000.0:
        return PerformanceProfile(
            name="cuda-ultra",
            cuda_available=True,
            total_vram_mb=total_vram_mb,
            free_vram_mb=free_vram_mb,
            perception_every_n_frames=1,
            maximum_perception_period=2,
            vehicle_image_size=640,
            camera_inference_batch_size=7,
            camera_wait_timeout_ms=6.0,
            navigation_render_every_n_frames=2,
        )

    if total_vram_mb >= 8_000.0 and free_vram_mb >= 3_000.0:
        return PerformanceProfile(
            name="cuda-high",
            cuda_available=True,
            total_vram_mb=total_vram_mb,
            free_vram_mb=free_vram_mb,
            perception_every_n_frames=1,
            maximum_perception_period=2,
            vehicle_image_size=640,
            camera_inference_batch_size=4,
            camera_wait_timeout_ms=8.0,
            navigation_render_every_n_frames=2,
        )

    if total_vram_mb >= 5_000.0 and free_vram_mb >= 1_800.0:
        return PerformanceProfile(
            name="cuda-balanced",
            cuda_available=True,
            total_vram_mb=total_vram_mb,
            free_vram_mb=free_vram_mb,
            perception_every_n_frames=1,
            maximum_perception_period=2,
            vehicle_image_size=576,
            camera_inference_batch_size=2,
            camera_wait_timeout_ms=8.0,
            navigation_render_every_n_frames=2,
        )

    return PerformanceProfile(
        name="cuda-low-vram",
        cuda_available=True,
        total_vram_mb=total_vram_mb,
        free_vram_mb=free_vram_mb,
        perception_every_n_frames=2,
        maximum_perception_period=3,
        vehicle_image_size=512,
        camera_inference_batch_size=1,
        camera_wait_timeout_ms=6.0,
        navigation_render_every_n_frames=2,
    )


def detect_performance_profile():
    """PyTorch üzerinden mevcut bilgisayar profilini otomatik belirler."""
    try:
        import torch
    except ImportError:
        return choose_performance_profile(False)

    if not torch.cuda.is_available():
        return choose_performance_profile(False)

    try:
        properties = torch.cuda.get_device_properties(0)
        total_vram_mb = float(properties.total_memory) / (1024.0 * 1024.0)
    except (AttributeError, RuntimeError):
        total_vram_mb = 0.0

    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        free_vram_mb = float(free_bytes) / (1024.0 * 1024.0)
    except (AttributeError, RuntimeError):
        free_vram_mb = total_vram_mb

    return choose_performance_profile(
        True,
        total_vram_mb=total_vram_mb,
        free_vram_mb=free_vram_mb,
    )

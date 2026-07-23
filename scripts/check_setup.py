"""Gerekli Python paketleri ile model dosyalarının varlığını kontrol eder."""

import sys
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_PACKAGES = (
    "carla",
    "cv2",
    "numpy",
    "torch",
    "yaml",
    "ultralytics",
    "dotenv",
)


def required_components(settings):
    """Açık özelliklere göre ek paketleri ve gerekli dosyaları döndürür."""
    packages = list(BASE_PACKAGES)
    model_paths = [Path(settings.vehicle_model)]
    if settings.enable_sign_detection:
        packages.append("onnxruntime")
        model_paths.extend(
            (
                Path(settings.sign_detector_model),
                Path(settings.sign_classifier_model),
                Path(settings.sign_class_names),
            )
        )
    if settings.enable_lane_detection:
        packages.append("torchvision")
        model_paths.append(Path(settings.lane_model))
    return packages, model_paths


def package_is_available(package_finder, package):
    """Bozuk modül tanımlarını da eksik kabul eden güvenli paket kontrolü."""
    try:
        return package_finder(package) is not None
    except (AttributeError, ImportError, ValueError):
        return False


def main(
    settings=None,
    package_finder=None,
    module_importer=None,
):
    """Kurulumu raporlar; zorunlu bir bileşen eksikse başarısız olur."""
    package_finder = package_finder or find_spec
    module_importer = module_importer or import_module

    base_availability = {
        package: package_is_available(package_finder, package)
        for package in BASE_PACKAGES
    }
    if settings is None:
        # Settings'in kendisi python-dotenv gerektirir. Temel kurulum eksikse
        # traceback üretmek yerine bulunan bütün temel eksikleri raporla.
        if not base_availability["dotenv"]:
            for package, available in base_availability.items():
                print(f"{'OK' if available else 'EKSIK':5} {package}")
            return 1

        from carla_app.config import Settings

        settings = Settings()

    packages, model_paths = required_components(settings)
    availability = {}
    for package in packages:
        available = base_availability.get(package)
        if available is None:
            available = package_is_available(package_finder, package)
        availability[package] = available
        print(f"{'OK' if available else 'EKSIK':5} {package}")

    missing_packages = {
        package for package, available in availability.items() if not available
    }

    torch = None
    if availability.get("torch"):
        try:
            torch = module_importer("torch")
        except (AttributeError, ImportError, OSError, RuntimeError) as error:
            missing_packages.add("torch")
            print(f"HATA  torch içe aktarılamadı: {error}")

    if torch is not None:
        if torch.cuda.is_available():
            print(
                "OK    CUDA "
                f"torch={torch.__version__} "
                f"device={torch.cuda.get_device_name(0)}"
            )
        else:
            print(
                "UYARI CUDA kullanilamiyor; YOLO CPU'da belirgin bicimde "
                "daha yavas calisir."
            )

    if settings.enable_sign_detection and availability.get("onnxruntime"):
        try:
            onnxruntime = module_importer("onnxruntime")
            providers = set(onnxruntime.get_available_providers())
        except (AttributeError, ImportError, OSError, RuntimeError) as error:
            missing_packages.add("onnxruntime")
            print(f"HATA  onnxruntime içe aktarılamadı: {error}")
        else:
            cuda_provider = "CUDAExecutionProvider" in providers
            status = "OK" if cuda_provider else "UYARI"
            print(
                f"{status:5} ONNX CUDA provider "
                f"{'aktif' if cuda_provider else 'bulunamadi'}"
            )

    if torch is not None and torch.cuda.is_available():
        print(
            "OK    Model cihazlari "
            f"vehicle={settings.vehicle_device} "
            f"sign={settings.sign_device} "
            f"lane={settings.lane_device} "
            f"fp16={settings.enable_fp16_inference}"
        )

    missing_models = []
    for path in model_paths:
        available = path.is_file()
        print(f"{'OK' if available else 'EKSIK':5} {path}")
        if not available:
            missing_models.append(path)

    if missing_packages or missing_models:
        print(
            "HATA  Kurulum tamamlanmadı: "
            f"{len(missing_packages)} paket, "
            f"{len(missing_models)} dosya eksik veya bozuk."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

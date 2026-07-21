"""Gerekli Python paketleri ile model dosyalarının varlığını kontrol eder."""

import sys
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Betik doğrudan çalıştırıldığında proje yolu yerel importtan önce eklenmelidir.
from carla_app.config import Settings  # noqa: E402

PACKAGES = [
    "carla",
    "cv2",
    "numpy",
    "torch",
    "yaml",
    "ultralytics",
    "dotenv",
]
settings = Settings()
model_paths = [settings.vehicle_model]
if settings.enable_sign_detection:
    PACKAGES.append("onnxruntime")
    model_paths.extend(
        (
            settings.sign_detector_model,
            settings.sign_classifier_model,
            settings.sign_class_names,
        )
    )
if settings.enable_lane_detection:
    PACKAGES.append("torchvision")
    model_paths.append(settings.lane_model)

for package in PACKAGES:
    status = "OK" if find_spec(package) else "EKSIK"
    print(f"{status:5} {package}")

try:
    torch = import_module("torch")

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
except ImportError:
    pass

for path in model_paths:
    status = "OK" if path.is_file() else "EKSIK"
    print(f"{status:5} {path}")

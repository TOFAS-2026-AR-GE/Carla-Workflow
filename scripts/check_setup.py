from importlib.util import find_spec

from carla_app.config import Settings

PACKAGES = ["carla", "cv2", "numpy", "yaml", "ultralytics", "dotenv"]
settings = Settings.load()
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

for package in PACKAGES:
    status = "OK" if find_spec(package) else "EKSIK"
    print(f"{status:5} {package}")

for path in model_paths:
    status = "OK" if path.is_file() else "EKSIK"
    print(f"{status:5} {path}")

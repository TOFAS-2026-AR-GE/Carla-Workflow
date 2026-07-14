from importlib.util import find_spec

from carla_app.config import Settings


PACKAGES = ["carla", "cv2", "numpy", "yaml", "ultralytics", "onnxruntime"]


for package in PACKAGES:
    status = "OK" if find_spec(package) else "EKSIK"
    print(f"{status:5} {package}")

settings = Settings.load()
for path in (
    settings.vehicle_model,
    settings.sign_detector_model,
    settings.sign_classifier_model,
    settings.sign_class_names,
):
    status = "OK" if path.is_file() else "EKSIK"
    print(f"{status:5} {path}")

"""Ortam değişkenlerini tek yerde okuyup uygulama ayarlarına dönüştürür."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def _path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _boolean(name, default):
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"{name} true/false olmali; gelen deger: {value!r}")


class Settings:
    """`.env` içindeki bütün uygulama ayarlarını normal alanlarda tutar."""

    def __init__(self):
        load_dotenv(ROOT / ".env")
        self.host = os.getenv("HOST", "127.0.0.1")
        self.port = int(os.getenv("PORT", "2000"))
        self.timeout = float(os.getenv("TIMEOUT", "60.0"))
        self.vehicle_name = os.getenv("VEHICLE_NAME", "vehicle.tesla.model3")

        self.ego_role_name = os.getenv("EGO_ROLE_NAME", "ego_vehicle").strip()
        if not self.ego_role_name:
            self.ego_role_name = "ego_vehicle"

        self.scenario_file = _path(
            os.getenv("SCENARIO_FILE", "scenarios/traffic.yaml")
        )
        self.fixed_delta_seconds = float(
            os.getenv("FIXED_DELTA_SECONDS", "0.05")
        )
        self.save_every_n_frames = max(
            1,
            int(os.getenv("SAVE_EVERY_N_FRAMES", "5")),
        )
        self.perception_every_n_frames = max(
            1,
            int(os.getenv("PERCEPTION_EVERY_N_FRAMES", "2")),
        )
        self.output_folder = _path(os.getenv("OUTPUT_FOLDER", "data/runs"))

        self.camera_width = int(os.getenv("CAMERA_WIDTH", "800"))
        self.camera_height = int(os.getenv("CAMERA_HEIGHT", "600"))
        self.camera_fov = float(os.getenv("CAMERA_FOV", "90"))

        self.vehicle_model = _path(
            os.getenv("VEHICLE_MODEL", "models/vehicle/carla_yolov8n_best.pt")
        )
        self.sign_detector_model = _path(
            os.getenv("SIGN_DETECTOR_MODEL", "models/signs/detector.onnx")
        )
        self.sign_classifier_model = _path(
            os.getenv("SIGN_CLASSIFIER_MODEL", "models/signs/classifier.onnx")
        )
        self.sign_class_names = _path(
            os.getenv("SIGN_CLASS_NAMES", "models/signs/class_names.json")
        )

        self.vehicle_device = os.getenv("VEHICLE_DEVICE", "cpu").strip()
        if not self.vehicle_device:
            self.vehicle_device = "cpu"
        self.sign_device = os.getenv("SIGN_DEVICE", "cpu").strip()
        if not self.sign_device:
            self.sign_device = "cpu"

        self.vehicle_confidence = float(
            os.getenv("VEHICLE_CONFIDENCE", "0.05")
        )
        self.sign_detector_confidence = float(
            os.getenv("SIGN_DETECTOR_CONFIDENCE", "0.25")
        )
        self.sign_detector_iou = float(
            os.getenv("SIGN_DETECTOR_IOU", "0.50")
        )
        self.sign_classifier_confidence = float(
            os.getenv("SIGN_CLASSIFIER_CONFIDENCE", "0.50")
        )
        self.vehicle_image_size = int(os.getenv("VEHICLE_IMAGE_SIZE", "640"))
        self.sign_detector_image_size = int(
            os.getenv("SIGN_DETECTOR_IMAGE_SIZE", "512")
        )
        self.sign_classifier_image_size = int(
            os.getenv("SIGN_CLASSIFIER_IMAGE_SIZE", "96")
        )

        self.enable_sign_detection = _boolean("ENABLE_SIGN_DETECTION", False)
        self.enable_data_recording = _boolean("ENABLE_DATA_RECORDING", False)
        self.status_period_seconds = max(
            0.2,
            float(os.getenv("STATUS_PERIOD_SECONDS", "2.0")),
        )
        self.max_runtime_seconds = max(
            0.0,
            float(os.getenv("MAX_RUNTIME_SECONDS", "0")),
        )
        self.maximum_speed_kmh = max(
            10.0,
            float(os.getenv("MAXIMUM_SPEED_KMH", "60.0")),
        )

    def check_models(self):
        required_models = [self.vehicle_model]
        if self.enable_sign_detection:
            required_models.extend(
                (
                    self.sign_detector_model,
                    self.sign_classifier_model,
                    self.sign_class_names,
                )
            )

        missing = []
        for path in required_models:
            if not path.is_file():
                missing.append(path)

        if missing:
            lines = []
            for path in missing:
                lines.append(f"- {path}")
            names = "\n".join(lines)
            raise FileNotFoundError(
                "Model dosyalari eksik:\n"
                f"{names}\n\n"
                "Kopyalamak icin: python scripts/copy_models.py ESKI_PROJE_YOLU"
            )

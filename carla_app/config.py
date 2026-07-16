import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"{name} true/false olmali; gelen deger: {value!r}")


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    timeout: float
    vehicle_name: str
    scenario_file: Path
    fixed_delta_seconds: float
    save_every_n_frames: int
    perception_every_n_frames: int
    output_folder: Path
    camera_width: int
    camera_height: int
    camera_fov: float
    vehicle_model: Path
    sign_detector_model: Path
    sign_classifier_model: Path
    sign_class_names: Path
    vehicle_device: str
    sign_device: str
    vehicle_confidence: float
    sign_detector_confidence: float
    sign_detector_iou: float
    sign_classifier_confidence: float
    vehicle_image_size: int
    sign_detector_image_size: int
    sign_classifier_image_size: int
    enable_sign_detection: bool
    enable_data_recording: bool
    status_period_seconds: float
    max_runtime_seconds: float

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv(ROOT / ".env")
        get = os.getenv

        return cls(
            host=get("HOST", "127.0.0.1"),
            port=int(get("PORT", "2000")),
            timeout=float(get("TIMEOUT", "60.0")),
            vehicle_name=get("VEHICLE_NAME", "vehicle.tesla.model3"),
            scenario_file=_path(get("SCENARIO_FILE", "scenarios/traffic.yaml")),
            fixed_delta_seconds=float(get("FIXED_DELTA_SECONDS", "0.05")),
            save_every_n_frames=max(1, int(get("SAVE_EVERY_N_FRAMES", "5"))),
            perception_every_n_frames=max(
                1, int(get("PERCEPTION_EVERY_N_FRAMES", "2"))
            ),
            output_folder=_path(get("OUTPUT_FOLDER", "data/runs")),
            camera_width=int(get("CAMERA_WIDTH", "800")),
            camera_height=int(get("CAMERA_HEIGHT", "600")),
            camera_fov=float(get("CAMERA_FOV", "90")),
            vehicle_model=_path(
                get("VEHICLE_MODEL", "models/vehicle/carla_yolov8n_best.pt")
            ),
            sign_detector_model=_path(
                get("SIGN_DETECTOR_MODEL", "models/signs/detector.onnx")
            ),
            sign_classifier_model=_path(
                get("SIGN_CLASSIFIER_MODEL", "models/signs/classifier.onnx")
            ),
            sign_class_names=_path(
                get("SIGN_CLASS_NAMES", "models/signs/class_names.json")
            ),
            vehicle_device=get("VEHICLE_DEVICE", "cpu").strip() or "cpu",
            sign_device=get("SIGN_DEVICE", "cpu").strip() or "cpu",
            vehicle_confidence=float(get("VEHICLE_CONFIDENCE", "0.05")),
            sign_detector_confidence=float(get("SIGN_DETECTOR_CONFIDENCE", "0.25")),
            sign_detector_iou=float(get("SIGN_DETECTOR_IOU", "0.50")),
            sign_classifier_confidence=float(get("SIGN_CLASSIFIER_CONFIDENCE", "0.50")),
            vehicle_image_size=int(get("VEHICLE_IMAGE_SIZE", "640")),
            sign_detector_image_size=int(get("SIGN_DETECTOR_IMAGE_SIZE", "512")),
            sign_classifier_image_size=int(get("SIGN_CLASSIFIER_IMAGE_SIZE", "96")),
            enable_sign_detection=_boolean("ENABLE_SIGN_DETECTION", False),
            enable_data_recording=_boolean("ENABLE_DATA_RECORDING", False),
            status_period_seconds=max(
                0.2,
                float(get("STATUS_PERIOD_SECONDS", "2.0")),
            ),
            max_runtime_seconds=max(
                0.0,
                float(get("MAX_RUNTIME_SECONDS", "0")),
            ),
        )

    def check_models(self) -> None:
        required_models = [self.vehicle_model]
        if self.enable_sign_detection:
            required_models.extend(
                (
                    self.sign_detector_model,
                    self.sign_classifier_model,
                    self.sign_class_names,
                )
            )

        missing = [path for path in required_models if not path.is_file()]
        if missing:
            names = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(
                "Model dosyalari eksik:\n"
                f"{names}\n\n"
                "Kopyalamak icin: python scripts/copy_models.py ESKI_PROJE_YOLU"
            )

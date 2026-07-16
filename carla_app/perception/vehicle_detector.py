"""YOLO based vehicle detection."""

import numpy as np
from ultralytics import YOLO

from carla_app.perception.device import move_model_to_cpu, resolve_device


class VehicleDetector:
    """Detect road vehicles and return a small, backend-independent result."""

    VEHICLE_NAMES = {
        "vehicle",
        "car",
        "automobile",
        "sedan",
        "suv",
        "jeep",
        "van",
        "minivan",
        "pickup",
        "truck",
        "lorry",
        "bus",
        "coach",
        "motorcycle",
        "motorbike",
        "motobike",  # Name used by the bundled CARLA model.
        "bicycle",
        "bike",
        "tram",
        "ambulance",
        "police",
        "taxi",
    }

    def __init__(self, model_path, confidence, image_size, device):
        self.model = YOLO(str(model_path), task="detect")
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.device = resolve_device(device)

        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("VEHICLE_CONFIDENCE 0 ile 1 arasinda olmali.")
        if self.image_size <= 0:
            raise ValueError("VEHICLE_IMAGE_SIZE sifirdan buyuk olmali.")

        self.model_names = self._normalize_names(self.model.names)
        self.vehicle_class_ids = self._find_vehicle_class_ids(self.model_names)

        if not self.vehicle_class_ids:
            raise ValueError(
                "Vehicle modelinde arac sinifi bulunamadi. "
                f"Model siniflari: {self.model_names}"
            )

        selected_names = {
            class_id: self.model_names[class_id] for class_id in self.vehicle_class_ids
        }
        print(
            f"[OK] Vehicle modeli: device={self.device}, "
            f"classes={selected_names}, conf={self.confidence:.2f}"
        )

    def detect(self, rgb_image):
        """Return vehicle detections for one RGB camera frame."""
        self._validate_image(rgb_image)

        # CARLA processor returns RGB; Ultralytics expects an OpenCV-style BGR
        # ndarray and converts it to RGB internally.
        bgr_image = np.ascontiguousarray(rgb_image[:, :, :3][:, :, ::-1])
        results = self._predict(bgr_image)

        if not results or results[0].boxes is None:
            return []

        result = results[0]
        result_names = self._normalize_names(result.names)
        image_height, image_width = bgr_image.shape[:2]
        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            if class_id not in self.vehicle_class_ids:
                continue

            coordinates = box.xyxy[0].detach().cpu().tolist()
            x1, y1, x2, y2 = [int(round(value)) for value in coordinates]
            x1 = max(0, min(x1, image_width - 1))
            x2 = max(0, min(x2, image_width - 1))
            y1 = max(0, min(y1, image_height - 1))
            y2 = max(0, min(y2, image_height - 1))

            if x2 - x1 < 3 or y2 - y1 < 3:
                continue

            detections.append(
                {
                    "type": "vehicle",
                    "class_id": class_id,
                    "class_name": result_names.get(
                        class_id, self.model_names[class_id]
                    ),
                    "confidence": float(box.conf[0].item()),
                    "bbox": (x1, y1, x2, y2),
                }
            )

        return detections

    def _predict(self, bgr_image):
        arguments = {
            "source": bgr_image,
            "conf": self.confidence,
            "iou": 0.45,
            "imgsz": self.image_size,
            "device": self.device,
            "classes": self.vehicle_class_ids,
            "verbose": False,
            "max_det": 100,
        }

        try:
            return self.model.predict(**arguments)
        except (RuntimeError, ValueError) as error:
            if self.device == "cpu":
                raise

            print(
                f"[WARN] Vehicle GPU inference basarisiz ({error}); "
                "CPU ile yeniden deneniyor."
            )
            self.device = "cpu"
            move_model_to_cpu(self.model)
            arguments["device"] = "cpu"
            return self.model.predict(**arguments)

    @staticmethod
    def _validate_image(rgb_image):
        if rgb_image is None:
            raise ValueError("VehicleDetector bos kamera goruntusu aldi.")
        if not isinstance(rgb_image, np.ndarray):
            raise TypeError("VehicleDetector numpy.ndarray bekliyor.")
        if rgb_image.ndim != 3 or rgb_image.shape[2] < 3:
            raise ValueError(
                "Kamera goruntusu HxWx3 formatinda olmali; "
                f"gelen shape: {rgb_image.shape}"
            )

    @classmethod
    def _find_vehicle_class_ids(cls, names):
        selected = []

        for class_id, class_name in names.items():
            normalized = (
                str(class_name).strip().lower().replace("-", " ").replace("_", " ")
            )
            words = set(normalized.split())

            if normalized in cls.VEHICLE_NAMES or words & cls.VEHICLE_NAMES:
                selected.append(int(class_id))

        return sorted(set(selected))

    @staticmethod
    def _normalize_names(names):
        if isinstance(names, dict):
            return {
                int(class_id): str(class_name) for class_id, class_name in names.items()
            }
        if isinstance(names, (list, tuple)):
            return {index: str(class_name) for index, class_name in enumerate(names)}
        return {}

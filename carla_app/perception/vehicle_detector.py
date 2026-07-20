"""Single-pass YOLO road-object detection for vehicles and traffic lights."""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from carla_app.perception.device import move_model_to_cpu, resolve_device


class VehicleDetector:
    """Run one YOLO inference and split vehicles from traffic-light states."""

    VEHICLE_NAMES = {
        "vehicle", "car", "automobile", "sedan", "suv", "jeep", "van",
        "minivan", "pickup", "truck", "lorry", "bus", "coach",
        "motorcycle", "motorbike", "motobike", "bicycle", "bike", "tram",
        "ambulance", "police", "taxi",
    }
    TRAFFIC_LIGHT_TOKENS = {"traffic light", "traffic signal", "signal light"}

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
        self.traffic_light_class_ids = self._find_traffic_light_class_ids(
            self.model_names
        )
        self.selected_class_ids = sorted(
            set(self.vehicle_class_ids) | set(self.traffic_light_class_ids)
        )

        if not self.vehicle_class_ids:
            raise ValueError(
                "Vehicle modelinde arac sinifi bulunamadi. "
                f"Model siniflari: {self.model_names}"
            )

        selected_names = {
            class_id: self.model_names[class_id]
            for class_id in self.selected_class_ids
        }
        print(
            f"[OK] Road-object modeli: device={self.device}, "
            f"classes={selected_names}, conf={self.confidence:.2f}"
        )
        if not self.traffic_light_class_ids:
            print(
                "[WARN] YOLO modelinde traffic_light_red/yellow/green "
                "sinifi bulunamadi; trafik isigi kontrolu vision olmadan calismaz."
            )

    def detect(self, rgb_image):
        """Backward-compatible vehicle-only result."""
        return self.detect_road_objects(rgb_image)["vehicles"]

    def detect_road_objects(self, rgb_image):
        """Return vehicles and traffic lights from the same inference."""
        self._validate_image(rgb_image)
        bgr_image = np.ascontiguousarray(rgb_image[:, :, :3][:, :, ::-1])
        results = self._predict(bgr_image)
        output = {"vehicles": [], "traffic_lights": []}

        if not results or results[0].boxes is None:
            return output

        result = results[0]
        result_names = self._normalize_names(result.names)
        image_height, image_width = bgr_image.shape[:2]

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            if class_id not in self.selected_class_ids:
                continue

            coordinates = box.xyxy[0].detach().cpu().tolist()
            x1, y1, x2, y2 = [int(round(value)) for value in coordinates]
            x1 = max(0, min(x1, image_width - 1))
            x2 = max(0, min(x2, image_width - 1))
            y1 = max(0, min(y1, image_height - 1))
            y2 = max(0, min(y2, image_height - 1))
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue

            class_name = result_names.get(
                class_id,
                self.model_names.get(class_id, str(class_id)),
            )
            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(box.conf[0].item()),
                "bbox": (x1, y1, x2, y2),
            }
            if class_id in self.vehicle_class_ids:
                detection["type"] = "vehicle"
                output["vehicles"].append(detection)
            elif class_id in self.traffic_light_class_ids:
                detection["type"] = "traffic_light"
                output["traffic_lights"].append(detection)

        return output

    def _predict(self, bgr_image):
        arguments = {
            "source": bgr_image,
            "conf": self.confidence,
            "iou": 0.45,
            "imgsz": self.image_size,
            "device": self.device,
            "classes": getattr(
                self,
                "selected_class_ids",
                getattr(self, "vehicle_class_ids", None),
            ),
            "verbose": False,
            "max_det": 120,
        }
        try:
            return self.model.predict(**arguments)
        except (RuntimeError, ValueError) as error:
            if self.device == "cpu":
                raise
            print(
                f"[WARN] Road-object GPU inference basarisiz ({error}); "
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
            normalized = cls._normalized_words(class_name)
            words = set(normalized.split())
            if normalized in cls.VEHICLE_NAMES or words & cls.VEHICLE_NAMES:
                selected.append(int(class_id))
        return sorted(set(selected))

    @classmethod
    def _find_traffic_light_class_ids(cls, names):
        selected = []
        for class_id, class_name in names.items():
            normalized = cls._normalized_words(class_name)
            contains_object = any(token in normalized for token in cls.TRAFFIC_LIGHT_TOKENS)
            contains_state = any(
                state in normalized
                for state in ("red", "green", "yellow", "orange", "amber")
            )
            if contains_object and contains_state:
                selected.append(int(class_id))
        return sorted(set(selected))

    @staticmethod
    def _normalized_words(class_name):
        return (
            str(class_name)
            .strip()
            .lower()
            .replace("-", " ")
            .replace("_", " ")
        )

    @staticmethod
    def _normalize_names(names):
        if isinstance(names, dict):
            return {int(class_id): str(name) for class_id, name in names.items()}
        if isinstance(names, (list, tuple)):
            return {index: str(name) for index, name in enumerate(names)}
        return {}

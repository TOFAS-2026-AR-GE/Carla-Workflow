"""Araç ve isteğe bağlı trafik levhası algılamasını birlikte yönetir."""

import time

from carla_app.perception.sign_detector import TrafficSignDetector
from carla_app.perception.vehicle_detector import VehicleDetector


class PerceptionSystem:
    """Bir model hata verdiğinde diğer modelin sonucunu korur."""

    def __init__(self, settings):
        self.vehicle_detector = VehicleDetector(
            settings.vehicle_model,
            settings.vehicle_confidence,
            settings.vehicle_image_size,
            settings.vehicle_device,
        )

        self.sign_detector = None
        if settings.enable_sign_detection:
            self.sign_detector = TrafficSignDetector(
                settings.sign_detector_model,
                settings.sign_classifier_model,
                settings.sign_class_names,
                settings.sign_detector_confidence,
                settings.sign_detector_iou,
                settings.sign_classifier_confidence,
                settings.sign_detector_image_size,
                settings.sign_classifier_image_size,
                settings.sign_device,
            )
        else:
            print("[INFO] Trafik levhasi algilama kapali.")

        self._last_errors = {}

    def detect(self, frame_id, rgb_image):
        start = time.perf_counter()
        errors = {}

        vehicles = self._detect_safely(
            name="vehicle",
            detector=self.vehicle_detector,
            image=rgb_image,
            errors=errors,
        )
        signs = self._detect_safely(
            name="sign",
            detector=self.sign_detector,
            image=rgb_image,
            errors=errors,
        )

        return {
            "frame_id": int(frame_id),
            "image": rgb_image,
            "vehicles": vehicles,
            "signs": signs,
            "errors": errors,
            "elapsed_ms": (time.perf_counter() - start) * 1000.0,
        }

    def _detect_safely(self, name, detector, image, errors):
        if detector is None:
            return []

        try:
            detections = detector.detect(image)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            errors[name] = message
            if self._last_errors.get(name) != message:
                print(f"[ERROR] {name} detector: {message}")
            self._last_errors[name] = message
            return []

        if name in self._last_errors:
            print(f"[OK] {name} detector yeniden calisiyor.")
            self._last_errors.pop(name, None)

        return detections

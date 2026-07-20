"""Perception pipeline orchestration."""

from __future__ import annotations

import time

from carla_app.perception.sign_detector import TrafficSignDetector
from carla_app.perception.vehicle_detector import VehicleDetector


class PerceptionSystem:
    """Araç/ışık modelini sık, tabela modelini daha seyrek çalıştırır."""

    def __init__(self, settings):
        self.road_object_detector = VehicleDetector(
            settings.vehicle_model,
            settings.vehicle_confidence,
            settings.vehicle_image_size,
            settings.vehicle_device,
        )
        self.vehicle_detector = self.road_object_detector

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
                settings.sign_max_candidates,
            )
        else:
            print("[INFO] Trafik levhasi algilama kapali.")

        self.sign_every_n_frames = max(
            1,
            int(settings.sign_every_n_frames),
        )
        self._last_sign_detection_frame = None
        self._last_errors = {}

        if self.sign_detector is not None:
            print(
                "[INFO] Trafik levhasi calisma araligi: "
                f"her {self.sign_every_n_frames} perception karesi"
            )

    def detect(self, frame_id, rgb_image):
        start = time.perf_counter()
        errors = {}

        road_objects = self._detect_road_objects_safely(
            rgb_image,
            errors,
        )

        signs = []
        if self._should_run_sign_detection(frame_id):
            signs = self._detect_safely(
                name="sign",
                detector=self.sign_detector,
                image=rgb_image,
                errors=errors,
            )
            self._last_sign_detection_frame = int(frame_id)

        return {
            "frame_id": int(frame_id),
            "image": rgb_image,
            "vehicles": road_objects["vehicles"],
            "traffic_lights": road_objects["traffic_lights"],
            "signs": signs,
            "errors": errors,
            "elapsed_ms": (time.perf_counter() - start) * 1000.0,
        }

    def _should_run_sign_detection(self, frame_id):
        if self.sign_detector is None:
            return False

        last_frame = getattr(
            self,
            "_last_sign_detection_frame",
            None,
        )
        if last_frame is None:
            return True

        period = max(
            1,
            int(getattr(self, "sign_every_n_frames", 1)),
        )
        elapsed_frames = int(frame_id) - int(last_frame)
        return elapsed_frames >= period

    def _detect_road_objects_safely(self, image, errors):
        try:
            detector = getattr(self, "road_object_detector", None)
            if detector is None:
                detector = getattr(self, "vehicle_detector", None)
            if detector is None:
                raise RuntimeError(
                    "Road-object detector is not configured."
                )

            if hasattr(detector, "detect_road_objects"):
                detections = detector.detect_road_objects(image)
            else:
                detections = {
                    "vehicles": detector.detect(image),
                    "traffic_lights": [],
                }
        except Exception as error:
            self._remember_error("road_object", error, errors)
            return {"vehicles": [], "traffic_lights": []}

        self._clear_error("road_object")
        if not isinstance(detections, dict):
            return {"vehicles": [], "traffic_lights": []}

        return {
            "vehicles": detections.get("vehicles", []) or [],
            "traffic_lights": detections.get(
                "traffic_lights",
                [],
            ) or [],
        }

    def _detect_safely(self, name, detector, image, errors):
        if detector is None:
            return []

        try:
            detections = detector.detect(image)
        except Exception as error:
            self._remember_error(name, error, errors)
            return []

        self._clear_error(name)
        return detections

    def _remember_error(self, name, error, errors):
        message = f"{type(error).__name__}: {error}"
        errors[name] = message

        if self._last_errors.get(name) != message:
            print(f"[ERROR] {name} detector: {message}")

        self._last_errors[name] = message

    def _clear_error(self, name):
        if name in self._last_errors:
            print(f"[OK] {name} detector yeniden calisiyor.")
            self._last_errors.pop(name, None)

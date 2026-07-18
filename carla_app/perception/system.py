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

        objects = self._detect_objects_safely(
            name="vehicle",
            detector=self.vehicle_detector,
            image=rgb_image,
            errors=errors,
        )
        vehicles, model_signs = self._split_objects(objects)
        signs = self._detect_safely(
            name="sign",
            detector=self.sign_detector,
            image=rgb_image,
            errors=errors,
        )
        signs.extend(model_signs)

        return {
            "frame_id": int(frame_id),
            "image": rgb_image,
            "vehicles": vehicles,
            "signs": signs,
            "objects": objects,
            "errors": errors,
            "elapsed_ms": (time.perf_counter() - start) * 1000.0,
        }

    def detect_cameras(self, camera_packet, primary_camera_name):
        """BEV modunda mevcut bütün kamera görüntülerini birlikte işler."""
        start = time.perf_counter()
        images_by_name = {}
        frame_ids = {}

        for camera_name, entry in camera_packet.items():
            images_by_name[camera_name] = entry["data"]
            frame_ids[camera_name] = int(entry["frame_id"])

        errors = {}
        objects_by_name = self.detect_objects_safely(images_by_name, errors)
        camera_results = {}

        for camera_name, image in images_by_name.items():
            signs = []
            objects = objects_by_name.get(camera_name, [])
            vehicles, model_signs = self._split_objects(objects)
            camera_errors = dict(errors)
            if camera_name == primary_camera_name:
                signs = self._detect_safely(
                    name="sign",
                    detector=self.sign_detector,
                    image=image,
                    errors=camera_errors,
                )
            signs.extend(model_signs)

            camera_results[camera_name] = {
                "camera_name": camera_name,
                "frame_id": frame_ids[camera_name],
                "image": image,
                "vehicles": vehicles,
                "signs": signs,
                "objects": objects,
                "errors": camera_errors,
            }

        primary_result = camera_results.get(primary_camera_name)
        if primary_result is None:
            return {
                "frame_id": 0,
                "image": None,
                "vehicles": [],
                "signs": [],
                "objects": [],
                "errors": {"camera": "Birincil kamera pakette yok."},
                "camera_results": camera_results,
                "elapsed_ms": (time.perf_counter() - start) * 1000.0,
            }

        return {
            "frame_id": primary_result["frame_id"],
            "image": primary_result["image"],
            "vehicles": primary_result["vehicles"],
            "signs": primary_result["signs"],
            "objects": primary_result["objects"],
            "errors": primary_result["errors"],
            "camera_results": camera_results,
            "elapsed_ms": (time.perf_counter() - start) * 1000.0,
        }

    def detect_vehicles_safely(self, images_by_name, errors):
        """Çoklu kamera model hatasını ana uygulamaya yaymadan yakalar."""
        try:
            detections = self.vehicle_detector.detect_many(images_by_name)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            errors["vehicle"] = message
            if self._last_errors.get("vehicle") != message:
                print(f"[ERROR] vehicle detector: {message}")
            self._last_errors["vehicle"] = message
            return {}

        if "vehicle" in self._last_errors:
            print("[OK] vehicle detector yeniden calisiyor.")
            self._last_errors.pop("vehicle", None)
        return detections

    def detect_objects_safely(self, images_by_name, errors):
        """Model destekliyorsa bütün trafik sınıflarını tek toplu çağrıda alır."""
        detector = self.vehicle_detector
        method = getattr(detector, "detect_many_objects", None)
        if method is None:
            method = detector.detect_many
        try:
            detections = method(images_by_name)
        except Exception as error:
            return self._handle_detection_error("vehicle", error, errors, {})
        self._clear_detection_error("vehicle")
        return detections

    def _detect_objects_safely(self, name, detector, image, errors):
        method = getattr(detector, "detect_objects", None)
        if method is None:
            method = detector.detect
        try:
            detections = method(image)
        except Exception as error:
            return self._handle_detection_error(name, error, errors, [])
        self._clear_detection_error(name)
        return detections

    def _split_objects(self, objects):
        vehicles = []
        signs = []
        for detection in objects:
            class_name = str(detection.get("class_name", "")).lower()
            if detection.get("type") == "vehicle" or class_name in {
                "bike",
                "motobike",
                "vehicle",
            }:
                vehicles.append(detection)
            if class_name.startswith("traffic_sign_"):
                signs.append(detection)
        return vehicles, signs

    def _handle_detection_error(self, name, error, errors, empty_result):
        message = f"{type(error).__name__}: {error}"
        errors[name] = message
        if self._last_errors.get(name) != message:
            print(f"[ERROR] {name} detector: {message}")
        self._last_errors[name] = message
        return empty_result

    def _clear_detection_error(self, name):
        if name in self._last_errors:
            print(f"[OK] {name} detector yeniden calisiyor.")
            self._last_errors.pop(name, None)

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

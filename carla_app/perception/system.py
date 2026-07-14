import time

from carla_app.perception.sign_detector import TrafficSignDetector
from carla_app.perception.vehicle_detector import VehicleDetector


class PerceptionSystem:
    def __init__(self, settings):
        self.vehicle_detector = VehicleDetector(
            settings.vehicle_model,
            settings.vehicle_confidence,
            settings.vehicle_image_size,
            settings.vehicle_device,
        )
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

    def detect(self, frame_id, rgb_image):
        start = time.perf_counter()
        vehicles = self.vehicle_detector.detect(rgb_image)
        signs = self.sign_detector.detect(rgb_image)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "frame_id": int(frame_id),
            "image": rgb_image,
            "vehicles": vehicles,
            "signs": signs,
            "elapsed_ms": elapsed_ms,
        }

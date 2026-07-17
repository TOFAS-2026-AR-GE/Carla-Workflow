import sys
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

# Birim testleri ağır model çalışma ortamına ihtiyaç duymaz. Gerçek paketin
# kurulu olduğu üretim ortamında scripts/check_setup.py ile doğrulanır.
if "ultralytics" not in sys.modules:
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = object
    sys.modules["ultralytics"] = ultralytics

from carla_app.perception.fusion import fuse_detections_with_radar
from carla_app.perception.sign_detector import TrafficSignDetector
from carla_app.perception.system import PerceptionSystem
from carla_app.perception.vehicle_detector import VehicleDetector
from carla_app.sensors.processors import image_to_rgb


class VehicleDetectorTests(unittest.TestCase):
    def test_bundled_model_class_names_select_vehicle_classes(self):
        names = {
            0: "bike",
            1: "motobike",
            2: "person",
            3: "traffic_light_green",
            9: "vehicle",
        }
        detector = VehicleDetector.__new__(VehicleDetector)
        self.assertEqual(detector._find_vehicle_class_ids(names), [0, 1, 9])

    def test_unknown_classes_are_not_silently_treated_as_vehicles(self):
        detector = VehicleDetector.__new__(VehicleDetector)
        names = {0: "person", 1: "sign"}
        self.assertEqual(detector._find_vehicle_class_ids(names), [])

    def test_predict_uses_configured_low_threshold_and_vehicle_classes(self):
        captured = {}

        class FakeModel:
            def predict(self, **arguments):
                captured.update(arguments)
                return []

        detector = VehicleDetector.__new__(VehicleDetector)
        detector.model = FakeModel()
        detector.confidence = 0.05
        detector.image_size = 640
        detector.device = "cpu"
        detector.vehicle_class_ids = [0, 1, 9]

        detector._predict(np.zeros((20, 30, 3), dtype=np.uint8))

        self.assertEqual(captured["conf"], 0.05)
        self.assertEqual(captured["classes"], [0, 1, 9])

    def test_multiple_cameras_use_one_batched_model_call(self):
        captured = {"call_count": 0}

        class FakeModel:
            def predict(self, **arguments):
                captured["call_count"] += 1
                captured.update(arguments)
                results = []
                for _image in arguments["source"]:
                    results.append(types.SimpleNamespace(boxes=None))
                return results

        detector = VehicleDetector.__new__(VehicleDetector)
        detector.model = FakeModel()
        detector.confidence = 0.05
        detector.image_size = 640
        detector.device = "cpu"
        detector.vehicle_class_ids = [0]

        image = np.zeros((20, 30, 3), dtype=np.uint8)
        detections = detector.detect_many(
            {
                "camera_front_wide": image,
                "camera_rear_center": image,
            }
        )

        self.assertEqual(captured["call_count"], 1)
        self.assertIsInstance(captured["source"], list)
        self.assertEqual(len(captured["source"]), 2)
        self.assertEqual(detections["camera_front_wide"], [])
        self.assertEqual(detections["camera_rear_center"], [])


class TrafficSignDetectorTests(unittest.TestCase):
    def test_detector_options_are_passed_openly_to_the_model(self):
        captured = {}

        class FakeModel:
            def predict(self, **arguments):
                captured.update(arguments)
                return []

        detector = TrafficSignDetector.__new__(TrafficSignDetector)
        detector.detector = FakeModel()
        detector.detector_confidence = 0.25
        detector.detector_iou = 0.50
        detector.detector_image_size = 512
        detector.device = "cpu"

        image = np.zeros((20, 30, 3), dtype=np.uint8)
        detector._predict_detector(image)

        self.assertEqual(captured["source"].shape, (20, 30, 3))
        self.assertEqual(captured["conf"], 0.25)
        self.assertEqual(captured["iou"], 0.50)
        self.assertEqual(captured["imgsz"], 512)
        self.assertEqual(captured["device"], "cpu")

    def test_classifier_options_are_passed_openly_to_the_model(self):
        captured = {}

        class FakeModel:
            def predict(self, **arguments):
                captured.update(arguments)
                return []

        detector = TrafficSignDetector.__new__(TrafficSignDetector)
        detector.classifier = FakeModel()
        detector.classifier_image_size = 96
        detector.device = "cpu"

        image = np.zeros((20, 30, 3), dtype=np.uint8)
        detector._predict_classifier(image)

        self.assertEqual(captured["source"].shape, (20, 30, 3))
        self.assertEqual(captured["imgsz"], 96)
        self.assertEqual(captured["device"], "cpu")


class PerceptionSystemTests(unittest.TestCase):
    def test_sign_failure_does_not_discard_vehicle_boxes(self):
        system = PerceptionSystem.__new__(PerceptionSystem)
        system.vehicle_detector = types.SimpleNamespace(
            detect=lambda image: [
                {
                    "bbox": (1, 2, 30, 40),
                    "confidence": 0.9,
                    "class_name": "vehicle",
                }
            ]
        )

        def fail(_image):
            raise RuntimeError("sign backend unavailable")

        system.sign_detector = types.SimpleNamespace(detect=fail)
        system._last_errors = {}
        with redirect_stdout(StringIO()):
            result = system.detect(7, np.zeros((20, 30, 3), dtype=np.uint8))

        self.assertEqual(len(result["vehicles"]), 1)
        self.assertEqual(result["signs"], [])
        self.assertIn("sign", result["errors"])

    def test_multi_camera_result_keeps_primary_control_shape(self):
        system = PerceptionSystem.__new__(PerceptionSystem)

        class FakeVehicleDetector:
            def detect_many(self, images_by_name):
                results = {}
                for camera_name in images_by_name:
                    results[camera_name] = [
                        {
                            "bbox": (1, 2, 20, 30),
                            "confidence": 0.8,
                            "class_name": "vehicle",
                        }
                    ]
                return results

        system.vehicle_detector = FakeVehicleDetector()
        system.sign_detector = None
        system._last_errors = {}
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        packet = {
            "camera_front_wide": {
                "frame_id": 10,
                "age_frames": 1,
                "data": image,
            },
            "camera_rear_center": {
                "frame_id": 9,
                "age_frames": 2,
                "data": image,
            },
        }

        result = system.detect_cameras(packet, "camera_front_wide")

        self.assertEqual(result["frame_id"], 10)
        self.assertEqual(len(result["vehicles"]), 1)
        self.assertEqual(len(result["camera_results"]), 2)
        self.assertEqual(
            result["camera_results"]["camera_rear_center"]["frame_id"],
            9,
        )


class FusionTests(unittest.TestCase):
    def test_center_box_receives_current_radar_range(self):
        detections = [
            {
                "bbox": (350, 200, 450, 400),
                "confidence": 0.8,
                "class_name": "vehicle",
            }
        ]
        radar_points = [
            {
                "depth_m": 20.0,
                "azimuth_deg": 0.0,
                "altitude_deg": 0.0,
                "relative_velocity_mps": 4.0,
            },
            {
                "depth_m": 20.2,
                "azimuth_deg": 0.2,
                "altitude_deg": 0.0,
                "relative_velocity_mps": 4.0,
            },
        ]

        result = fuse_detections_with_radar(
            detections,
            detection_frame_id=8,
            radar_points=radar_points,
            radar_frame_id=10,
            image_width=800,
            camera_fov_deg=90.0,
            fixed_delta_seconds=0.05,
        )[0]

        self.assertTrue(result["has_range"])
        self.assertAlmostEqual(result["range_m"], 20.0)
        self.assertAlmostEqual(result["relative_velocity_mps"], 4.0)


class ImageConversionTests(unittest.TestCase):
    def test_carla_bgra_is_converted_to_rgb(self):
        image = types.SimpleNamespace(
            width=1,
            height=1,
            raw_data=bytes([10, 20, 30, 255]),
        )
        rgb = image_to_rgb(image)
        self.assertEqual(rgb.tolist(), [[[30, 20, 10]]])


if __name__ == "__main__":
    unittest.main()

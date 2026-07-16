import sys
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

# Unit tests do not need the heavy inference runtime. The production setup
# check verifies that the real package is installed.
if "ultralytics" not in sys.modules:
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = object
    sys.modules["ultralytics"] = ultralytics

from carla_app.perception.fusion import fuse_detections_with_radar
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
        self.assertEqual(
            VehicleDetector._find_vehicle_class_ids(names),
            [0, 1, 9],
        )

    def test_unknown_classes_are_not_silently_treated_as_vehicles(self):
        self.assertEqual(
            VehicleDetector._find_vehicle_class_ids({0: "person", 1: "sign"}),
            [],
        )


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

import sys
import types
import unittest

import numpy as np


if "ultralytics" not in sys.modules:
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = object
    sys.modules["ultralytics"] = ultralytics


from carla_app.perception.system import PerceptionSystem


class SignScheduleTests(unittest.TestCase):
    def test_sign_detector_runs_once_every_five_frames(self):
        calls = []

        system = PerceptionSystem.__new__(
            PerceptionSystem
        )
        system.road_object_detector = (
            types.SimpleNamespace(
                detect_road_objects=lambda image: {
                    "vehicles": [],
                    "traffic_lights": [],
                }
            )
        )
        system.vehicle_detector = (
            system.road_object_detector
        )
        system.sign_detector = types.SimpleNamespace(
            detect=lambda image: calls.append(1) or [
                {
                    "class_name": "speed_limit_50",
                    "bbox": (1, 1, 10, 10),
                }
            ]
        )
        system.sign_every_n_frames = 5
        system._last_sign_detection_frame = None
        system._last_errors = {}

        image = np.zeros(
            (20, 30, 3),
            dtype=np.uint8,
        )

        results = [
            system.detect(frame_id, image)
            for frame_id in range(100, 106)
        ]

        self.assertEqual(len(calls), 2)
        self.assertEqual(len(results[0]["signs"]), 1)
        self.assertEqual(results[1]["signs"], [])
        self.assertEqual(len(results[5]["signs"]), 1)


if __name__ == "__main__":
    unittest.main()

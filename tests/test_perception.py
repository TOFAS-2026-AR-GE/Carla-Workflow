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
from carla_app.perception.lane_detector import (
    LANE_COUNT,
    ROW_COUNT,
    LaneDetector,
    decode_ufld_logits,
    prepare_ufld_input,
)
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

    def test_bundled_model_selects_all_supported_traffic_classes(self):
        detector = VehicleDetector.__new__(VehicleDetector)
        detector.vehicle_class_ids = [0, 1, 9]
        names = {
            0: "bike",
            1: "motobike",
            2: "person",
            3: "traffic_light_green",
            4: "traffic_light_orange",
            5: "traffic_light_red",
            6: "traffic_sign_30",
            7: "traffic_sign_60",
            8: "traffic_sign_90",
            9: "vehicle",
        }
        self.assertEqual(detector._find_relevant_class_ids(names), list(range(10)))

    def test_object_prediction_uses_all_relevant_classes(self):
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
        detector.relevant_class_ids = list(range(10))
        detector._predict(
            np.zeros((20, 30, 3), dtype=np.uint8),
            detector.relevant_class_ids,
        )
        self.assertEqual(captured["classes"], list(range(10)))

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


class LaneDetectorTests(unittest.TestCase):
    def test_rgb_preprocessing_uses_expected_shape_and_imagenet_order(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        image[:, :, 0] = 255

        tensor = prepare_ufld_input(image)

        self.assertEqual(tensor.shape, (1, 3, 288, 800))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertAlmostEqual(float(tensor[0, 0, 10, 10]), (1.0 - 0.485) / 0.229)
        self.assertAlmostEqual(float(tensor[0, 2, 10, 10]), -0.406 / 0.225)

    def test_softmax_expectation_decodes_lane_in_source_coordinates(self):
        logits = np.full((101, ROW_COUNT, LANE_COUNT), -12.0, dtype=np.float32)
        logits[100, :, :] = 12.0
        logits[49, :, 1] = 14.0

        lanes = decode_ufld_logits(logits, image_width=1600, image_height=576)

        lane = lanes[1]
        self.assertTrue(lane["detected"])
        self.assertEqual(len(lane["points"]), ROW_COUNT)
        self.assertAlmostEqual(lane["points"][0][0], 806, delta=2)
        self.assertEqual(lane["points"][0][1], 229)
        self.assertGreater(lane["confidence"], 0.8)

    def test_no_lane_class_produces_no_points(self):
        logits = np.zeros((1, 101, ROW_COUNT, LANE_COUNT), dtype=np.float32)
        logits[:, 100, :, :] = 10.0

        lanes = decode_ufld_logits(logits, image_width=800, image_height=600)

        self.assertTrue(all(not lane["detected"] for lane in lanes))
        self.assertTrue(all(lane["points"] == [] for lane in lanes))

    def test_ambiguous_grid_scores_are_not_reported_as_detected(self):
        logits = np.zeros((101, ROW_COUNT, LANE_COUNT), dtype=np.float32)
        logits[100, :, :] = -0.1

        lanes = decode_ufld_logits(logits, image_width=800, image_height=600)

        self.assertTrue(all(not lane["detected"] for lane in lanes))
        self.assertTrue(all(lane["confidence"] < 0.30 for lane in lanes))

    def test_checkpoint_wrapper_and_dataparallel_prefix_are_supported(self):
        state = {"module.pool.weight": object()}
        result = LaneDetector._state_dict_from_checkpoint(
            {"epoch": 12, "model_state_dict": state}
        )
        self.assertEqual(list(result), ["pool.weight"])


class PerceptionSystemTests(unittest.TestCase):
    def test_unified_model_keeps_traffic_signs_without_legacy_detector(self):
        system = PerceptionSystem.__new__(PerceptionSystem)
        system.vehicle_detector = types.SimpleNamespace(
            detect_objects=lambda image: [
                {
                    "type": "road_object",
                    "bbox": (10, 10, 30, 30),
                    "confidence": 0.9,
                    "class_name": "traffic_sign_60",
                }
            ]
        )
        system.sign_detector = None
        system._last_errors = {}

        result = system.detect(7, np.zeros((20, 30, 3), dtype=np.uint8))

        self.assertEqual(len(result["signs"]), 1)
        self.assertEqual(result["signs"][0]["class_name"], "traffic_sign_60")
        self.assertEqual(result["lane_detection"]["reason"], "disabled")

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

    def test_lane_failure_does_not_discard_other_perception_results(self):
        system = PerceptionSystem.__new__(PerceptionSystem)
        system.vehicle_detector = types.SimpleNamespace(
            detect_objects=lambda image: [
                {
                    "type": "vehicle",
                    "bbox": (1, 2, 30, 40),
                    "confidence": 0.9,
                    "class_name": "vehicle",
                }
            ]
        )
        system.sign_detector = None
        system.lane_detector = types.SimpleNamespace(
            detect=lambda image: (_ for _ in ()).throw(RuntimeError("lane failed"))
        )
        system._last_errors = {}

        with redirect_stdout(StringIO()):
            result = system.detect(8, np.zeros((20, 30, 3), dtype=np.uint8))

        self.assertEqual(len(result["vehicles"]), 1)
        self.assertEqual(result["lane_detection"]["reason"], "error")
        self.assertIn("lane", result["errors"])

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

    def test_multi_camera_lane_model_runs_only_on_primary_camera(self):
        calls = []
        system = PerceptionSystem.__new__(PerceptionSystem)
        system.vehicle_detector = types.SimpleNamespace(
            detect_many=lambda images: {name: [] for name in images}
        )
        system.sign_detector = None
        system.lane_detector = types.SimpleNamespace(
            detect=lambda image: calls.append(image)
            or {
                "available": True,
                "lanes": [],
                "detected_count": 0,
                "elapsed_ms": 1.0,
            }
        )
        system._last_errors = {}
        front = np.zeros((40, 60, 3), dtype=np.uint8)
        rear = np.ones((40, 60, 3), dtype=np.uint8)
        packet = {
            "camera_front_wide": {"frame_id": 10, "data": front},
            "camera_rear_center": {"frame_id": 10, "data": rear},
        }

        result = system.detect_cameras(packet, "camera_front_wide")

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], front)
        self.assertTrue(result["lane_detection"]["available"])
        rear_lane = result["camera_results"]["camera_rear_center"][
            "lane_detection"
        ]
        self.assertEqual(rear_lane["reason"], "not_primary_camera")


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

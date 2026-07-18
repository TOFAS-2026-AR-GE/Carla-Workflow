"""Trafik ışığı kararının kamera kaçırmalarına dayanıklılık testleri."""

import unittest
from types import SimpleNamespace

import numpy as np

from carla_app.config import DrivingParameters
from carla_app.controller.vehicle.behavior_planner import BehaviorPlanner
from carla_app.perception.road_context import RoadContextTracker


def vehicle_state():
    path = [SimpleNamespace(x=float(index), y=0.0) for index in range(40)]
    return {
        "location": SimpleNamespace(x=0.0, y=0.0),
        "yaw": 0.0,
        "speed_mps": 8.0,
        "reference_path": path,
    }


def perception(frame_id, class_name=None):
    objects = []
    if class_name is not None:
        objects.append(
            {
                "bbox": (360.0, 120.0, 440.0, 280.0),
                "class_name": class_name,
                "confidence": 0.90,
                "estimated_distance_m": 30.0,
            }
        )
    return {
        "frame_id": frame_id,
        "image": np.zeros((600, 800, 3), dtype=np.uint8),
        "objects": objects,
    }


class TrafficLightResilienceTests(unittest.TestCase):
    def test_confirmed_red_survives_short_detection_dropouts(self):
        parameters = DrivingParameters(0.05)
        tracker = RoadContextTracker(parameters=parameters)
        state = vehicle_state()

        context = None
        for frame_id in (1, 2, 3):
            context = tracker.update(
                frame_id,
                perception(frame_id, "traffic_light_red"),
                state=state,
            )
        self.assertEqual(context["lead_traffic_light"]["color"], "red")

        for frame_id in range(4, 16):
            context = tracker.update(
                frame_id,
                perception(frame_id),
                state=state,
            )

        self.assertFalse(context["sensor_fault"])
        self.assertIsNotNone(context["lead_traffic_light"])
        self.assertEqual(context["lead_traffic_light"]["color"], "red")

        plan = BehaviorPlanner(0.05, parameters).plan(
            state,
            curve_target_speed_mps=20.0,
            speed_plan={"speed_reason": "straight"},
            road_context=context,
        )
        self.assertEqual(plan["mode"], "APPROACH_RED_LIGHT")

    def test_reused_result_frame_does_not_confirm_red(self):
        tracker = RoadContextTracker(parameters=DrivingParameters(0.05))
        state = vehicle_state()
        result = perception(1, "traffic_light_red")

        context = tracker.update(1, result, state=state)
        context = tracker.update(2, result, state=state)
        context = tracker.update(3, result, state=state)

        self.assertIsNone(context["lead_traffic_light"])

    def test_confirmed_red_releases_only_after_confirmed_green(self):
        tracker = RoadContextTracker(parameters=DrivingParameters(0.05))
        state = vehicle_state()
        for frame_id in (1, 2, 3):
            tracker.update(
                frame_id,
                perception(frame_id, "traffic_light_red"),
                state=state,
            )
        for frame_id in range(4, 9):
            tracker.update(frame_id, perception(frame_id), state=state)

        context = tracker.update(
            9,
            perception(9, "traffic_light_green"),
            state=state,
        )
        self.assertEqual(context["lead_traffic_light"]["color"], "red")
        context = tracker.update(
            10,
            perception(10, "traffic_light_green"),
            state=state,
        )
        self.assertEqual(context["lead_traffic_light"]["color"], "red")
        context = tracker.update(
            11,
            perception(11, "traffic_light_green"),
            state=state,
        )
        self.assertEqual(context["lead_traffic_light"]["color"], "green")

    def test_expired_light_track_cannot_be_revived_by_one_detection(self):
        parameters = DrivingParameters(0.05)
        tracker = RoadContextTracker(parameters=parameters)
        state = vehicle_state()
        for frame_id in (1, 2, 3):
            tracker.update(
                frame_id,
                perception(frame_id, "traffic_light_red"),
                state=state,
            )

        old_result = perception(3, "traffic_light_red")
        context = tracker.update(30, old_result, state=state)
        self.assertIsNone(context["lead_traffic_light"])

        context = tracker.update(
            31,
            perception(31, "traffic_light_red"),
            state=state,
        )
        self.assertIsNone(context["lead_traffic_light"])


if __name__ == "__main__":
    unittest.main()

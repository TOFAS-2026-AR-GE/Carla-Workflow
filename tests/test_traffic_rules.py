import sys
import types
import unittest

import numpy as np

if "dotenv" not in sys.modules:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *arguments, **keywords: None
    sys.modules["dotenv"] = dotenv

from carla_app.config import DrivingParameters
from carla_app.controller.vehicle.behavior_planner import BehaviorPlanner
from carla_app.controller.vehicle.longitudinal_pid_controller import (
    LongitudinalPIDController,
)
from carla_app.perception.road_context import RoadContextTracker


def road_state(speed_mps=0.0, is_junction=False):
    return {
        "speed_mps": float(speed_mps),
        "speed_kmh": float(speed_mps) * 3.6,
        "is_junction": bool(is_junction),
        "location": types.SimpleNamespace(x=0.0, y=0.0),
        "yaw": 0.0,
        "reference_path": [types.SimpleNamespace(x=float(x), y=0.0) for x in range(81)],
    }


def raw_object(class_name, bbox, confidence=0.90, distance_m=20.0):
    return {
        "class_name": class_name,
        "bbox": bbox,
        "confidence": confidence,
        "estimated_distance_m": distance_m,
    }


def perception(frame_id, objects=None, errors=None):
    return {
        "frame_id": int(frame_id),
        "image": np.zeros((600, 800, 3), dtype=np.uint8),
        "objects": list(objects or []),
        "vehicles": [],
        "signs": [],
        "errors": dict(errors or {}),
    }


def light(color, distance_m, track_id=1):
    return {
        "color": color,
        "track_id": track_id,
        "estimated_distance_m": float(distance_m),
        "frame_id": 10,
        "confidence": 0.90,
        "smoothed_confidence": 0.90,
    }


class RoadContextTests(unittest.TestCase):
    def setUp(self):
        self.parameters = DrivingParameters(0.05)
        self.tracker = RoadContextTracker(parameters=self.parameters)

    def update(self, frame_id, objects=None, errors=None):
        return self.tracker.update(
            frame_id,
            perception(frame_id, objects, errors),
            state=road_state(),
        )

    def test_adapter_produces_required_detection_fields(self):
        context = self.update(
            1,
            [raw_object("person", (350, 200, 430, 580), distance_m=9.0)],
        )
        detection = context["detections"][0]
        required = {
            "class_name",
            "confidence",
            "bbox",
            "bbox_center",
            "bbox_width",
            "bbox_height",
            "estimated_distance_m",
            "frame_id",
            "valid",
        }
        self.assertTrue(required.issubset(detection))

    def test_single_red_frame_is_not_accepted(self):
        context = self.update(
            1,
            [raw_object("traffic_light_red", (360, 90, 410, 180), distance_m=25.0)],
        )
        self.assertIsNone(context["lead_traffic_light"])

    def test_red_light_is_confirmed_on_three_new_frames(self):
        box = (360, 90, 410, 180)
        for frame_id in (1, 2, 3):
            context = self.update(
                frame_id,
                [raw_object("traffic_light_red", box, distance_m=25.0)],
            )
        self.assertEqual(context["lead_traffic_light"]["color"], "red")

    def test_repeated_same_frame_does_not_confirm_light(self):
        box = (360, 90, 410, 180)
        result = perception(
            1,
            [raw_object("traffic_light_red", box, distance_m=25.0)],
        )
        for current_frame in (1, 2, 3):
            context = self.tracker.update(current_frame, result, state=road_state())
        self.assertIsNone(context["lead_traffic_light"])

    def test_one_wrong_green_frame_does_not_flip_confirmed_red(self):
        box = (360, 90, 410, 180)
        for frame_id in (1, 2, 3):
            self.update(
                frame_id,
                [raw_object("traffic_light_red", box, distance_m=25.0)],
            )
        context = self.update(
            4,
            [raw_object("traffic_light_green", box, distance_m=25.0)],
        )
        self.assertEqual(context["lead_traffic_light"]["color"], "red")

    def test_light_output_reports_green_candidate_before_confirmation(self):
        box = (360, 90, 410, 180)
        for frame_id in (1, 2, 3):
            self.update(
                frame_id,
                [raw_object("traffic_light_red", box, distance_m=25.0)],
            )

        context = self.update(
            4,
            [raw_object("traffic_light_green", box, distance_m=25.0)],
        )
        selected = context["lead_traffic_light"]

        self.assertEqual(selected["color"], "red")
        self.assertEqual(selected["observed_color"], "green")
        self.assertEqual(selected["candidate_color"], "green")
        self.assertEqual(selected["candidate_hits"], 1)

    def test_near_light_wins_over_far_light(self):
        objects = [
            raw_object("traffic_light_red", (350, 100, 400, 190), distance_m=22.0),
            raw_object("traffic_light_green", (430, 70, 455, 120), distance_m=60.0),
        ]
        for frame_id in (1, 2, 3):
            context = self.update(frame_id, objects)
        self.assertEqual(context["lead_traffic_light"]["color"], "red")
        self.assertEqual(context["lead_traffic_light"]["estimated_distance_m"], 22.0)

    def test_light_beyond_control_horizon_is_not_selected(self):
        distant = raw_object(
            "traffic_light_red",
            (370, 70, 395, 120),
            distance_m=self.parameters.traffic_light_maximum_distance_m + 5.0,
        )
        for frame_id in (1, 2, 3):
            context = self.update(frame_id, [distant])

        self.assertIsNone(context["lead_traffic_light"])

    def test_adjacent_lane_light_is_rejected(self):
        objects = [
            raw_object("traffic_light_red", (20, 100, 70, 190), distance_m=15.0),
            raw_object("traffic_light_green", (370, 100, 420, 190), distance_m=30.0),
        ]
        for frame_id in (1, 2, 3):
            context = self.update(frame_id, objects)
        self.assertEqual(context["lead_traffic_light"]["color"], "green")

    def test_temporary_detection_loss_keeps_confirmed_light(self):
        box = (360, 90, 410, 180)
        for frame_id in (1, 2, 3):
            self.update(
                frame_id,
                [raw_object("traffic_light_red", box, distance_m=25.0)],
            )
        context = self.tracker.update(5, None, state=road_state())
        self.assertEqual(context["lead_traffic_light"]["color"], "red")

    def test_speed_limit_changes_only_after_confirmation(self):
        sign = raw_object("traffic_sign_30", (560, 220, 620, 300), distance_m=18.0)
        first = self.update(1, [sign])
        self.assertIsNone(first["speed_limit_kmh"])
        self.update(2, [sign])
        third = self.update(3, [sign])
        self.assertEqual(third["speed_limit_kmh"], 30)

    def test_low_confidence_speed_sign_is_ignored(self):
        sign = raw_object(
            "traffic_sign_30",
            (560, 220, 620, 300),
            confidence=0.20,
            distance_m=18.0,
        )
        for frame_id in (1, 2, 3, 4):
            context = self.update(frame_id, [sign])
        self.assertIsNone(context["speed_limit_kmh"])

    def test_low_confidence_frames_do_not_confirm_a_speed_sign(self):
        box = (560, 220, 620, 300)
        for frame_id in (1, 2, 3):
            self.update(
                frame_id,
                [raw_object("traffic_sign_30", box, 0.20, 18.0)],
            )
        context = self.update(
            4,
            [raw_object("traffic_sign_30", box, 0.90, 18.0)],
        )
        self.assertIsNone(context["speed_limit_kmh"])

    def test_roadside_pedestrian_is_monitored_without_stop(self):
        person = raw_object("person", (700, 250, 760, 580), distance_m=8.0)
        context = self.update(1, [person])
        self.assertEqual(context["pedestrian_risk"], "MONITOR")

    def test_lane_pedestrian_requests_stop(self):
        person = raw_object("person", (370, 250, 430, 580), distance_m=9.0)
        context = self.update(1, [person])
        self.assertEqual(context["pedestrian_risk"], "PREPARE_STOP")

    def test_close_lane_pedestrian_is_emergency(self):
        person = raw_object("person", (370, 250, 430, 580), distance_m=4.0)
        context = self.update(1, [person])
        self.assertEqual(context["pedestrian_risk"], "EMERGENCY")

    def test_lidar_dropout_keeps_camera_distance(self):
        context = self.update(
            1,
            [raw_object("person", (370, 250, 430, 580), distance_m=11.0)],
        )
        detection = context["detections"][0]
        self.assertEqual(detection["distance_source"], "camera")
        self.assertEqual(detection["estimated_distance_m"], 11.0)

    def test_camera_lidar_conflict_uses_conservative_distance(self):
        distance, source, conflict = self.tracker.combine_distances(40.0, 12.0)
        self.assertTrue(conflict)
        self.assertEqual(source, "camera_lidar_conflict")
        self.assertEqual(distance, 12.0)

    def test_model_error_is_visible_in_context(self):
        context = self.update(1, errors={"vehicle": "backend failed"})
        self.assertTrue(context["sensor_fault"])
        self.assertIn("vehicle", context["errors"])

    def test_persistent_model_error_accumulates_fault_age(self):
        last_frame = self.parameters.sensor_timeout_frames + 2
        for frame_id in range(1, last_frame + 1):
            context = self.update(
                frame_id,
                errors={"vehicle": "backend failed"},
            )
        self.assertGreater(
            context["sensor_fault_age_frames"],
            self.parameters.sensor_timeout_frames,
        )


class BehaviorPlannerTests(unittest.TestCase):
    def setUp(self):
        self.parameters = DrivingParameters(0.05)
        self.planner = BehaviorPlanner(0.05, self.parameters)
        self.speed_plan = {"speed_reason": "cruise"}

    def plan(
        self,
        speed_mps,
        context=None,
        curve_speed=60.0 / 3.6,
        junction=False,
        lead=None,
    ):
        return self.planner.plan(
            road_state(speed_mps, junction),
            curve_speed,
            self.speed_plan,
            context or {},
            lead,
        )

    def test_red_light_creates_gradual_stop_target(self):
        decision = self.plan(10.0, {"lead_traffic_light": light("red", 35.0)})
        self.assertEqual(decision["mode"], "APPROACH_RED_LIGHT")
        self.assertGreater(decision["target_speed_mps"], 0.0)
        self.assertLess(decision["target_speed_mps"], 60.0 / 3.6)
        self.assertEqual(decision["control_obstacle"]["source"], "traffic_light_red")

    def test_stopped_vehicle_holds_at_red(self):
        decision = self.plan(0.0, {"lead_traffic_light": light("red", 5.0)})
        self.assertEqual(decision["mode"], "STOPPED_AT_RED")
        self.assertEqual(decision["target_speed_mps"], 0.0)

    def test_green_after_red_starts_smoothly(self):
        self.plan(0.0, {"lead_traffic_light": light("red", 5.0)})
        decision = self.plan(0.0, {"lead_traffic_light": light("green", 5.0)})
        self.assertEqual(decision["mode"], "START_ON_GREEN")
        self.assertLessEqual(
            decision["target_speed_mps"],
            self.parameters.green_start_acceleration_mps2 * 0.05,
        )

    def test_simulator_green_releases_red_when_light_leaves_camera_view(self):
        red_state = road_state(0.0)
        red_state["simulator_traffic_light"] = {
            "available": True,
            "affected": True,
            "color": "red",
        }
        first = self.planner.plan(
            red_state,
            60.0 / 3.6,
            self.speed_plan,
            {"lead_traffic_light": light("red", 5.8)},
        )
        self.assertEqual(first["mode"], "STOPPED_AT_RED")

        green_state = road_state(0.0)
        green_state["simulator_traffic_light"] = {
            "available": True,
            "affected": True,
            "color": "green",
        }
        still_waiting = self.planner.plan(
            green_state,
            60.0 / 3.6,
            self.speed_plan,
            {},
        )
        released = self.planner.plan(
            green_state,
            60.0 / 3.6,
            self.speed_plan,
            {},
        )

        self.assertEqual(still_waiting["mode"], "STOPPED_AT_RED")
        self.assertEqual(released["mode"], "START_ON_GREEN")
        self.assertIsNone(released["control_obstacle"])
        self.assertEqual(
            released["primary_detection"]["state_source"],
            "carla_vehicle_state",
        )

    def test_old_camera_red_cannot_relatch_after_simulator_green(self):
        state = road_state(0.0)
        state["simulator_traffic_light"] = {
            "available": True,
            "affected": True,
            "color": "red",
        }
        old_red = light("red", 5.8, track_id=7)
        self.planner.plan(
            state,
            60.0 / 3.6,
            self.speed_plan,
            {"lead_traffic_light": old_red},
        )

        state["simulator_traffic_light"]["color"] = "green"
        self.planner.plan(
            state,
            60.0 / 3.6,
            self.speed_plan,
            {"lead_traffic_light": old_red},
        )
        released = self.planner.plan(
            state,
            60.0 / 3.6,
            self.speed_plan,
            {"lead_traffic_light": old_red},
        )
        next_tick = self.planner.plan(
            state,
            60.0 / 3.6,
            self.speed_plan,
            {"lead_traffic_light": old_red},
        )

        self.assertEqual(released["mode"], "START_ON_GREEN")
        self.assertNotEqual(next_tick["mode"], "STOPPED_AT_RED")
        self.assertIsNone(next_tick["control_obstacle"])

    def test_confirmed_red_remains_active_after_camera_loses_the_light(self):
        state = road_state(3.0)
        decision = self.planner.plan(
            state,
            60.0 / 3.6,
            self.speed_plan,
            {"lead_traffic_light": light("red", 9.0)},
        )
        self.assertEqual(decision["mode"], "APPROACH_RED_LIGHT")

        for step in range(1, 31):
            state = road_state(2.0)
            state["location"].x = step * 0.10
            decision = self.planner.plan(
                state,
                60.0 / 3.6,
                self.speed_plan,
                {},
            )

        self.assertIn(decision["mode"], {"APPROACH_RED_LIGHT", "STOPPED_AT_RED"})
        self.assertIsNotNone(decision["control_obstacle"])
        self.assertEqual(decision["control_obstacle"]["source"], "traffic_light_red")

    def test_vehicle_stops_before_red_line_when_camera_loses_close_light(self):
        longitudinal = LongitudinalPIDController(0.05)
        speed_mps = 60.0 / 3.6
        signal_distance_m = 55.0
        decision = None
        brake = 0.0

        for tick in range(600):
            state = road_state(speed_mps)
            state["location"].x = 55.0 - signal_distance_m
            context = (
                {"lead_traffic_light": light("red", signal_distance_m)}
                if signal_distance_m > 6.0
                else {}
            )
            decision = self.planner.plan(
                state,
                60.0 / 3.6,
                self.speed_plan,
                context,
            )
            throttle, brake, _ = longitudinal.run_step(
                state,
                decision["control_obstacle"],
                decision["target_speed_mps"],
            )
            acceleration = 2.8 * throttle - 5.0 * brake
            if speed_mps > 0.02:
                acceleration -= 0.12
            speed_mps = max(0.0, speed_mps + acceleration * 0.05)
            signal_distance_m -= speed_mps * 0.05
            if speed_mps <= 0.02 and brake >= longitudinal.hold_brake:
                break

        self.assertLess(tick, 599)
        self.assertGreaterEqual(signal_distance_m, 4.8)
        self.assertIn(decision["mode"], {"APPROACH_RED_LIGHT", "STOPPED_AT_RED"})
        self.assertGreaterEqual(brake, longitudinal.hold_brake)

    def test_yellow_stops_when_comfortably_possible(self):
        decision = self.plan(8.0, {"lead_traffic_light": light("orange", 35.0)})
        self.assertEqual(decision["reason"], "yellow_safe_stop")
        self.assertIsNotNone(decision["control_obstacle"])

    def test_yellow_proceeds_inside_dilemma_zone(self):
        decision = self.plan(15.0, {"lead_traffic_light": light("orange", 10.0)})
        self.assertEqual(decision["reason"], "yellow_proceed_dilemma_zone")
        self.assertIsNone(decision["control_obstacle"])

    def test_yellow_does_not_stop_after_entering_junction(self):
        decision = self.plan(
            6.0,
            {"lead_traffic_light": light("orange", 30.0)},
            junction=True,
        )
        self.assertEqual(decision["reason"], "yellow_proceed_dilemma_zone")

    def test_confirmed_30_limit_caps_target_speed(self):
        decision = self.plan(10.0, {"speed_limit_kmh": 30})
        self.assertAlmostEqual(decision["target_speed_mps"], 30.0 / 3.6)

    def test_higher_speed_limit_increases_target_gradually(self):
        first = self.plan(8.0, {"speed_limit_kmh": 30})
        second = self.plan(8.0, {"speed_limit_kmh": 90})
        maximum_step = self.parameters.green_start_acceleration_mps2 * 0.05
        self.assertLessEqual(
            second["target_speed_mps"] - first["target_speed_mps"],
            maximum_step + 1e-9,
        )

    def test_curve_speed_remains_a_hard_upper_bound(self):
        decision = self.plan(10.0, {"speed_limit_kmh": 90}, curve_speed=8.0)
        self.assertLessEqual(decision["target_speed_mps"], 8.0)

    def test_slow_pedestrian_risk_caps_speed(self):
        context = {
            "pedestrian_risk": "SLOW",
            "pedestrian": {"estimated_distance_m": 20.0, "confidence": 0.8},
        }
        decision = self.plan(8.0, context)
        self.assertEqual(decision["mode"], "SLOW_FOR_PEDESTRIAN")
        self.assertLessEqual(decision["target_speed_mps"], 15.0 / 3.6)

    def test_lane_pedestrian_creates_stop_obstacle(self):
        context = {
            "pedestrian_risk": "PREPARE_STOP",
            "pedestrian": {"estimated_distance_m": 9.0, "confidence": 0.8},
        }
        decision = self.plan(6.0, context)
        self.assertEqual(decision["mode"], "STOP_FOR_PEDESTRIAN")
        self.assertEqual(decision["target_speed_mps"], 0.0)

    def test_close_pedestrian_has_emergency_urgency(self):
        context = {
            "pedestrian_risk": "EMERGENCY",
            "pedestrian": {"estimated_distance_m": 4.0, "confidence": 0.8},
        }
        decision = self.plan(6.0, context)
        self.assertEqual(decision["brake_urgency"], "EMERGENCY")

    def test_confirmed_lead_vehicle_sets_follow_mode(self):
        lead = {"distance_m": 18.0, "relative_speed_mps": -1.0, "confidence": 0.9}
        decision = self.plan(10.0, lead=lead)
        self.assertEqual(decision["mode"], "FOLLOW_VEHICLE")

    def test_stale_perception_requests_controlled_stop(self):
        context = {
            "sensor_fault": True,
            "perception_age_frames": self.parameters.sensor_timeout_frames + 1,
        }
        decision = self.plan(10.0, context)
        self.assertEqual(decision["mode"], "SENSOR_DEGRADED")
        self.assertEqual(decision["target_speed_mps"], 0.0)
        self.assertGreater(decision["target_stop_distance_m"], 2.0)


if __name__ == "__main__":
    unittest.main()

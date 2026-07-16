import math
import unittest
from types import SimpleNamespace

from carla_app.controller.vehicle.lead_vehicle import LeadVehicleTracker
from carla_app.controller.vehicle.longitudinal_controller import (
    LongitudinalController,
)
from carla_app.controller.vehicle.safety_supervisor import EmergencyBrakeSupervisor
from carla_app.controller.vehicle.speed_planner import CurvatureSpeedPlanner
from carla_app.controller.vehicle.stanley_controller import StanleyController


def location(x, y=0.0):
    return SimpleNamespace(x=float(x), y=float(y), z=0.0)


def straight_state(speed_mps=0.0, y=0.0):
    return {
        "location": location(0.0, y),
        "yaw": 0.0,
        "speed_mps": float(speed_mps),
        "speed_kmh": float(speed_mps) * 3.6,
        "reference_path": [location(x) for x in range(81)],
        "lane_width": 3.5,
        "road_id": 1,
        "lane_id": -1,
        "is_junction": False,
    }


def curved_state(radius_m=20.0):
    angles = [index / radius_m for index in range(41)]
    path = [
        location(
            radius_m * math.sin(angle),
            radius_m * (1.0 - math.cos(angle)),
        )
        for angle in angles
    ]
    state = straight_state(speed_mps=8.0)
    state["reference_path"] = path
    return state


class StanleyControllerTests(unittest.TestCase):
    def test_steers_left_when_vehicle_is_right_of_path(self):
        controller = StanleyController(dt=0.05)
        steer = controller.run_step(straight_state(speed_mps=8.0, y=1.0))
        self.assertLess(steer, 0.0)

    def test_steers_right_when_vehicle_is_left_of_path(self):
        controller = StanleyController(dt=0.05)
        steer = controller.run_step(straight_state(speed_mps=8.0, y=-1.0))
        self.assertGreater(steer, 0.0)

    def test_steering_change_is_rate_limited(self):
        controller = StanleyController(dt=0.05)
        first = controller.run_step(straight_state(speed_mps=10.0, y=1.5))
        second = controller.run_step(straight_state(speed_mps=10.0, y=-1.5))
        self.assertLessEqual(abs(second - first), 0.8 * 0.05 + 1e-9)

    def test_curved_path_steers_into_the_bend(self):
        controller = StanleyController(dt=0.05)
        steer = controller.run_step(curved_state())
        self.assertGreater(controller.last_info["curvature_1pm"], 0.0)
        self.assertGreater(steer, 0.0)

    def test_closed_loop_curve_tracking_stays_near_the_path(self):
        controller = StanleyController(dt=0.05)
        radius = 25.0
        path = [
            location(
                radius * math.sin(index * 0.01),
                radius * (1.0 - math.cos(index * 0.01)),
            )
            for index in range(250)
        ]
        x = 0.0
        y = 0.0
        yaw = 0.0
        speed = 6.0
        errors = []

        for _ in range(180):
            nearest_index = min(
                range(len(path)),
                key=lambda index: (path[index].x - x) ** 2 + (path[index].y - y) ** 2,
            )
            reference_path = path[
                max(0, nearest_index - 2) : min(len(path), nearest_index + 80)
            ]
            steer = controller.run_step(
                {
                    "location": location(x, y),
                    "yaw": math.degrees(yaw),
                    "speed_mps": speed,
                    "reference_path": reference_path,
                }
            )
            wheel_angle = steer * controller.maximum_wheel_angle_rad
            x += speed * math.cos(yaw) * controller.dt
            y += speed * math.sin(yaw) * controller.dt
            yaw += (
                speed / controller.wheelbase_m * math.tan(wheel_angle) * controller.dt
            )
            errors.append(abs(controller.last_info["cross_track_error_m"]))

        mean_recent_error = sum(errors[-80:]) / 80
        self.assertLess(mean_recent_error, 0.15)


class SpeedPlannerTests(unittest.TestCase):
    def test_straight_road_keeps_cruise_speed(self):
        planner = CurvatureSpeedPlanner(dt=0.05)
        speed, info = planner.run_step(straight_state(speed_mps=8.0))
        self.assertAlmostEqual(speed, 30.0 / 3.6)
        self.assertAlmostEqual(info["curvature_1pm"], 0.0)

    def test_curve_reduces_speed_with_a_rate_limit(self):
        planner = CurvatureSpeedPlanner(dt=0.05)
        speed, info = planner.run_step(curved_state(radius_m=20.0))
        self.assertAlmostEqual(info["curvature_1pm"], 1.0 / 20.0, places=3)
        self.assertLess(info["desired_speed_mps"], planner.cruise_speed_mps)
        self.assertAlmostEqual(
            planner.cruise_speed_mps - speed,
            planner.maximum_speed_decrease_mps2 * planner.dt,
        )

    def test_large_lane_error_sets_recovery_speed(self):
        planner = CurvatureSpeedPlanner(dt=0.05)
        _, info = planner.run_step(
            straight_state(speed_mps=8.0),
            lateral_info={
                "cross_track_error_m": 2.0,
                "heading_error_rad": 0.0,
            },
        )
        self.assertAlmostEqual(info["recovery_speed_mps"], 8.0 / 3.6)


class LongitudinalControllerTests(unittest.TestCase):
    def test_free_road_accelerates_toward_target(self):
        controller = LongitudinalController(dt=0.05)
        throttle, brake, info = controller.run_step(
            straight_state(speed_mps=2.0),
            lead_vehicle=None,
            target_speed=30.0 / 3.6,
        )
        self.assertEqual(info["mode"], "CRUISE")
        self.assertGreater(throttle, 0.0)
        self.assertEqual(brake, 0.0)

    def test_far_non_closing_lead_does_not_cause_braking(self):
        controller = LongitudinalController(dt=0.05)
        lead = {
            "track_id": 10,
            "distance_m": 60.0,
            "relative_speed_mps": 0.0,
        }
        controller.run_step(straight_state(speed_mps=8.0), lead, 30.0 / 3.6)
        throttle, brake, info = controller.run_step(
            straight_state(speed_mps=8.0), lead, 30.0 / 3.6
        )
        self.assertEqual(info["mode"], "LEAD_FAR")
        self.assertGreater(throttle, 0.0)
        self.assertEqual(brake, 0.0)

    def test_close_lead_activates_following_and_braking(self):
        controller = LongitudinalController(dt=0.05)
        state = straight_state(speed_mps=8.0)
        lead = {
            "track_id": 10,
            "distance_m": 9.0,
            "relative_speed_mps": -2.0,
        }
        controller.run_step(state, lead, 30.0 / 3.6)
        controller.run_step(state, lead, 30.0 / 3.6)
        throttle, brake, info = controller.run_step(state, lead, 30.0 / 3.6)
        self.assertEqual(info["mode"], "FOLLOW")
        self.assertEqual(throttle, 0.0)
        self.assertGreater(brake, 0.0)

    def test_reported_low_speed_gap_produces_throttle(self):
        """Regression for frames 5840-6000 from the reported CARLA log."""
        controller = LongitudinalController(dt=0.05)
        samples = [
            (1.0 / 3.6, 3.7, -0.26),
            (1.0 / 3.6, 3.7, -0.27),
            (1.0 / 3.6, 3.8, -0.26),
            (0.9 / 3.6, 3.8, -0.26),
            (0.9 / 3.6, 3.7, -0.25),
        ]

        outputs = []
        for speed, control_gap, relative_speed in samples:
            outputs.append(
                controller.run_step(
                    straight_state(speed_mps=speed),
                    {
                        "track_id": 11,
                        "distance_m": control_gap,
                        "relative_speed_mps": relative_speed,
                        "source": "radar_direct",
                    },
                    15.7 / 3.6,
                )
            )

        for throttle, brake, info in outputs[1:]:
            self.assertEqual(info["mode"], "FOLLOW")
            self.assertGreaterEqual(throttle, controller.low_speed_throttle_floor)
            self.assertEqual(brake, 0.0)
            self.assertGreater(info["desired_acceleration_mps2"], 0.0)

    def test_two_metres_is_held_as_the_standstill_gap(self):
        controller = LongitudinalController(dt=0.05)
        stopped_lead = {
            "track_id": 11,
            "distance_m": 2.0,
            "relative_speed_mps": 0.0,
        }
        throttle, brake, info = controller.run_step(
            straight_state(speed_mps=0.0),
            stopped_lead,
            30.0 / 3.6,
        )
        self.assertEqual(info["mode"], "HOLD")
        self.assertEqual(throttle, 0.0)
        self.assertEqual(brake, controller.hold_brake)

    def test_lead_pulling_away_restarts_after_two_ticks(self):
        controller = LongitudinalController(dt=0.05)
        state = straight_state(speed_mps=0.0)
        stopped_lead = {
            "track_id": 11,
            "distance_m": 2.0,
            "relative_speed_mps": 0.0,
        }
        controller.run_step(state, stopped_lead, 30.0 / 3.6)

        moving_lead = {
            "track_id": 11,
            "distance_m": 2.2,
            "relative_speed_mps": 0.8,
        }
        first = controller.run_step(state, moving_lead, 30.0 / 3.6)
        second = controller.run_step(state, moving_lead, 30.0 / 3.6)

        self.assertEqual(first[2]["mode"], "HOLD")
        self.assertEqual(second[2]["mode"], "RESTART")
        self.assertGreater(second[0], 0.0)
        self.assertEqual(second[1], 0.0)

    def test_temporary_sensor_dropout_does_not_release_hold(self):
        controller = LongitudinalController(dt=0.05)
        stopped_lead = {
            "track_id": 11,
            "distance_m": 2.0,
            "relative_speed_mps": 0.0,
        }
        controller.run_step(straight_state(speed_mps=0.0), stopped_lead, 30.0 / 3.6)

        throttle, brake, info = controller.run_step(
            straight_state(speed_mps=0.0), None, 30.0 / 3.6
        )

        self.assertEqual(info["mode"], "HOLD")
        self.assertEqual(throttle, 0.0)
        self.assertEqual(brake, controller.hold_brake)

    def test_noisy_stationary_lead_stops_once_near_two_metres(self):
        controller = LongitudinalController(dt=0.05)
        speed = 8.0
        distance = 25.0
        distance_noise = [0.0, 0.15, -0.12, 0.08, -0.05]
        modes = []
        last_pedal = None
        pedal_switches = 0

        for tick in range(500):
            throttle, brake, info = controller.run_step(
                straight_state(speed_mps=speed),
                {
                    "track_id": 11,
                    "distance_m": distance + distance_noise[tick % len(distance_noise)],
                    "relative_speed_mps": -speed,
                },
                30.0 / 3.6,
            )
            modes.append(info["mode"])
            pedal = "throttle" if throttle > 0.01 else "brake" if brake > 0.01 else None
            if last_pedal is not None and pedal is not None and pedal != last_pedal:
                pedal_switches += 1
            if pedal is not None:
                last_pedal = pedal

            acceleration = 2.8 * throttle - 5.0 * brake
            if speed > 0.02:
                acceleration -= 0.12
            speed = max(0.0, speed + acceleration * controller.dt)
            distance -= speed * controller.dt

            if info["mode"] == "HOLD" and speed <= 0.02:
                break

        self.assertEqual(info["mode"], "HOLD")
        self.assertNotIn("RESTART", modes)
        self.assertLessEqual(pedal_switches, 1)
        self.assertGreaterEqual(distance, 1.9)
        self.assertLessEqual(distance, 2.35)

    def test_moving_lead_converges_without_repeated_pedal_chatter(self):
        controller = LongitudinalController(dt=0.05)
        lead_speed = 1.0
        ego_speed = 0.0
        distance = 10.0
        last_pedal = None
        pedal_switches = 0

        for _ in range(400):
            throttle, brake, _ = controller.run_step(
                straight_state(speed_mps=ego_speed),
                {
                    "track_id": 11,
                    "distance_m": distance,
                    "relative_speed_mps": lead_speed - ego_speed,
                },
                30.0 / 3.6,
            )
            pedal = "throttle" if throttle > 0.01 else "brake" if brake > 0.01 else None
            if last_pedal is not None and pedal is not None and pedal != last_pedal:
                pedal_switches += 1
            if pedal is not None:
                last_pedal = pedal

            acceleration = 2.8 * throttle - 5.0 * brake
            if ego_speed > 0.02:
                acceleration -= 0.12
            ego_speed = max(0.0, ego_speed + acceleration * controller.dt)
            distance += (lead_speed - ego_speed) * controller.dt

        self.assertAlmostEqual(ego_speed, lead_speed, delta=0.10)
        self.assertGreater(distance, controller.standstill_gap_m)
        self.assertLessEqual(pedal_switches, 2)


class EmergencyBrakeTests(unittest.TestCase):
    def test_immediate_ttc_triggers_full_safety_request(self):
        supervisor = EmergencyBrakeSupervisor()
        emergency, info = supervisor.evaluate(
            {"distance_m": 5.0, "relative_speed_mps": -10.0}
        )
        self.assertTrue(emergency)
        self.assertAlmostEqual(info["ttc_s"], 0.5)

    def test_stationary_two_metre_gap_is_left_to_normal_hold_control(self):
        supervisor = EmergencyBrakeSupervisor()
        stopped_lead = {"distance_m": 2.0, "relative_speed_mps": 0.0}
        first_emergency, _ = supervisor.evaluate(stopped_lead)
        second_emergency, info = supervisor.evaluate(stopped_lead)
        self.assertFalse(first_emergency)
        self.assertFalse(second_emergency)
        self.assertEqual(info["hazard_count"], 0)

    def test_dangerous_tracked_lead_is_not_hidden_by_harmless_radar(self):
        supervisor = EmergencyBrakeSupervisor()
        emergency, info = supervisor.evaluate_candidates(
            {
                "distance_m": 5.0,
                "relative_speed_mps": -10.0,
                "source": "camera_radar_track",
            },
            {
                "distance_m": 4.0,
                "relative_speed_mps": 0.0,
                "source": "radar_direct",
            },
        )
        self.assertTrue(emergency)
        self.assertEqual(info["target_source"], "camera_radar_track")


class LeadVehicleTrackerTests(unittest.TestCase):
    def test_camera_identity_uses_closer_radar_range(self):
        tracker = LeadVehicleTracker(0.05, 800, 90.0)
        tracked_lead = {
            "track_id": 7,
            "distance_m": 6.9,
            "relative_speed_mps": -0.4,
            "lead_speed_mps": 0.2,
            "source": "camera_radar_track",
            "radar_points": None,
        }
        radar_lead = {
            "track_id": -1,
            "distance_m": 4.8,
            "relative_speed_mps": -0.6,
            "lead_speed_mps": 0.0,
            "source": "radar_direct",
            "radar_points": 5,
        }
        lead = tracker._choose_safest_lead(tracked_lead, radar_lead)
        self.assertEqual(lead["track_id"], tracked_lead["track_id"])
        self.assertEqual(lead["source"], "camera_radar_track")
        self.assertAlmostEqual(lead["distance_m"], 4.8)
        self.assertAlmostEqual(lead["relative_speed_mps"], -0.6)

    def test_radar_lead_requires_temporal_confirmation(self):
        tracker = LeadVehicleTracker(0.05, 800, 90.0)
        state = straight_state(speed_mps=8.0)
        points = [
            {
                "depth_m": 20.0 + offset,
                "azimuth_deg": offset,
                "altitude_deg": 0.0,
                "relative_velocity_mps": -3.0,
            }
            for offset in (-0.3, 0.0, 0.3)
        ]
        first = tracker.update(1, state, None, 1, points)
        second = tracker.update(2, state, None, 2, points)
        self.assertIsNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(second["source"], "radar_direct")
        self.assertAlmostEqual(second["relative_speed_mps"], -3.0)

    def test_adjacent_lane_radar_cluster_is_rejected(self):
        tracker = LeadVehicleTracker(0.05, 800, 90.0)
        state = straight_state(speed_mps=8.0)
        bearing = math.degrees(math.atan2(3.5, 20.0))
        points = [
            {
                "depth_m": 20.0,
                "azimuth_deg": bearing + offset,
                "altitude_deg": 0.0,
                "relative_velocity_mps": 0.0,
            }
            for offset in (-0.2, 0.0, 0.2)
        ]
        self.assertIsNone(tracker.update(1, state, None, 1, points))
        self.assertIsNone(tracker.update(2, state, None, 2, points))

    def test_single_close_radar_point_is_available_to_emergency_braking(self):
        tracker = LeadVehicleTracker(0.05, 800, 90.0)
        state = straight_state(speed_mps=8.0)
        points = [
            {
                "depth_m": 9.0,
                "azimuth_deg": 0.0,
                "altitude_deg": 0.0,
                "relative_velocity_mps": -8.0,
            }
        ]
        lead = tracker.update(1, state, None, 1, points)
        emergency = tracker.get_emergency_obstacle()
        self.assertIsNone(lead)
        self.assertIsNotNone(emergency)
        self.assertEqual(emergency["source"], "radar_emergency")
        self.assertAlmostEqual(emergency["relative_speed_mps"], -8.0)

    def test_adjacent_radar_point_is_not_an_emergency_obstacle(self):
        tracker = LeadVehicleTracker(0.05, 800, 90.0)
        state = straight_state(speed_mps=8.0)
        points = [
            {
                "depth_m": 9.0,
                "azimuth_deg": 13.0,
                "altitude_deg": 0.0,
                "relative_velocity_mps": -8.0,
            }
        ]
        tracker.update(1, state, None, 1, points)
        self.assertIsNone(tracker.get_emergency_obstacle())


if __name__ == "__main__":
    unittest.main()

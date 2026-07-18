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
        "vehicle_half_width_m": 0.95,
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

    def test_lane_edge_increases_centering_correction(self):
        controller = StanleyController(dt=0.05)
        state = straight_state(speed_mps=8.0)

        corrected_error = controller.calculate_lane_edge_error(0.70, state)

        self.assertGreater(corrected_error, 0.70)

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
        self.assertAlmostEqual(speed, 60.0 / 3.6)
        self.assertAlmostEqual(info["curvature_1pm"], 0.0)

    def test_cruise_speed_can_be_configured_in_kmh(self):
        planner = CurvatureSpeedPlanner(dt=0.05, cruise_speed_kmh=50.0)
        speed, _ = planner.run_step(straight_state(speed_mps=8.0))
        self.assertAlmostEqual(speed, 50.0 / 3.6)

    def test_curve_reduces_speed_with_a_rate_limit(self):
        planner = CurvatureSpeedPlanner(dt=0.05)
        speed, info = planner.run_step(curved_state(radius_m=20.0))
        self.assertAlmostEqual(info["curvature_1pm"], 1.0 / 20.0, places=3)
        self.assertLess(info["desired_speed_mps"], planner.cruise_speed_mps)
        expected = planner.cruise_speed_mps - (
            planner.maximum_speed_decrease_mps2 * planner.dt
        )
        self.assertAlmostEqual(speed, expected)

    def test_one_lane_error_sample_does_not_drop_target_speed(self):
        planner = CurvatureSpeedPlanner(dt=0.05, cruise_speed_kmh=80.0)
        speed, info = planner.run_step(
            straight_state(speed_mps=15.0),
            lateral_info={
                "cross_track_error_m": 0.70,
                "heading_error_rad": math.radians(24.0),
            },
        )

        self.assertAlmostEqual(speed, planner.cruise_speed_mps)
        self.assertIsNone(info["recovery_speed_mps"])
        self.assertAlmostEqual(info["requested_recovery_speed_mps"], 23.0 / 3.6)

    def test_high_speed_preview_sees_curve_beyond_thirty_five_metres(self):
        planner = CurvatureSpeedPlanner(dt=0.05, cruise_speed_kmh=80.0)
        path = [location(x) for x in range(41)]
        radius_m = 20.0
        for index in range(1, 41):
            angle = index / radius_m
            path.append(
                location(
                    40.0 + radius_m * math.sin(angle),
                    radius_m * (1.0 - math.cos(angle)),
                )
            )
        state = straight_state(speed_mps=80.0 / 3.6)
        state["reference_path"] = path

        speed, info = planner.run_step(state)

        self.assertGreater(info["curvature_1pm"], 0.0)
        self.assertLess(speed, planner.cruise_speed_mps)

    def test_large_lane_error_sets_recovery_speed(self):
        planner = CurvatureSpeedPlanner(dt=0.05)
        for _ in range(planner.recovery_confirmation_ticks):
            speed, info = planner.run_step(
                straight_state(speed_mps=8.0),
                lateral_info={
                    "cross_track_error_m": 2.0,
                    "heading_error_rad": 0.0,
                },
            )
        self.assertAlmostEqual(info["recovery_speed_mps"], 23.0 / 3.6)
        self.assertGreater(speed, info["recovery_speed_mps"])
        self.assertEqual(info["speed_reason"], "lane_recovery")

    def test_curve_and_lane_recovery_never_command_below_23_kmh(self):
        planner = CurvatureSpeedPlanner(dt=0.05)
        state = curved_state(radius_m=12.0)
        minimum_speed = 23.0 / 3.6

        for _ in range(250):
            speed, info = planner.run_step(
                state,
                lateral_info={
                    "cross_track_error_m": 2.0,
                    "heading_error_rad": math.radians(35.0),
                },
            )

        self.assertGreaterEqual(speed, minimum_speed)
        self.assertGreaterEqual(info["desired_speed_mps"], minimum_speed)
        self.assertGreaterEqual(info["recovery_speed_mps"], minimum_speed)
        self.assertIn("predicted_yaw_rate_radps", info)
        self.assertIn("predicted_lateral_acceleration_mps2", info)


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

    def test_free_road_converges_to_sixty_kmh(self):
        controller = LongitudinalController(dt=0.05)
        speed = 0.0

        for _ in range(500):
            throttle, brake, _ = controller.run_step(
                straight_state(speed_mps=speed),
                lead_vehicle=None,
                target_speed=60.0 / 3.6,
            )
            acceleration = 2.8 * throttle - 5.0 * brake
            if speed > 0.02:
                acceleration -= 0.12
            speed = max(0.0, speed + acceleration * controller.dt)

        self.assertAlmostEqual(speed * 3.6, 60.0, delta=1.0)

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
        """Bildirilen CARLA kaydındaki 5840-6000 karelerini tekrarlar."""
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

    def test_sustained_missing_lead_releases_hold_and_restarts(self):
        controller = LongitudinalController(dt=0.05)
        state = straight_state(speed_mps=0.0)
        stopped_lead = {
            "track_id": 11,
            "distance_m": 2.0,
            "relative_speed_mps": 0.0,
        }
        controller.run_step(state, stopped_lead, 30.0 / 3.6)

        outputs = [
            controller.run_step(state, None, 30.0 / 3.6)
            for _ in range(20)
        ]

        throttle, brake, info = outputs[-1]
        self.assertEqual(info["mode"], "CRUISE")
        self.assertGreater(throttle, 0.0)
        self.assertEqual(brake, 0.0)

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

    def test_sixty_kmh_approach_stops_near_two_metres(self):
        controller = LongitudinalController(dt=0.05)
        speed = 60.0 / 3.6
        distance = 60.0

        for _ in range(500):
            throttle, brake, info = controller.run_step(
                straight_state(speed_mps=speed),
                {
                    "track_id": 11,
                    "distance_m": distance,
                    "relative_speed_mps": -speed,
                },
                60.0 / 3.6,
            )
            acceleration = 2.8 * throttle - 5.0 * brake
            if speed > 0.02:
                acceleration -= 0.12
            speed = max(0.0, speed + acceleration * controller.dt)
            distance -= speed * controller.dt
            if info["mode"] == "HOLD" and speed <= 0.02:
                break

        self.assertEqual(info["mode"], "HOLD")
        self.assertGreaterEqual(distance, 1.9)
        self.assertLessEqual(distance, 2.20)

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
    def test_invalid_command_is_replaced_with_full_brake(self):
        supervisor = EmergencyBrakeSupervisor()
        throttle, brake, steer, info = supervisor.validate_control_command(
            throttle=math.nan,
            brake=0.0,
            steer=0.2,
            target_speed_mps=10.0,
        )
        self.assertEqual((throttle, brake, steer), (0.0, 1.0, 0.0))
        self.assertEqual(info["reason"], "non_finite_command")

    def test_pedal_conflict_keeps_brake_and_clears_throttle(self):
        supervisor = EmergencyBrakeSupervisor()
        throttle, brake, steer, info = supervisor.validate_control_command(
            throttle=0.6,
            brake=0.4,
            steer=0.2,
            target_speed_mps=10.0,
        )
        self.assertEqual(throttle, 0.0)
        self.assertEqual(brake, 0.4)
        self.assertEqual(steer, 0.2)
        self.assertEqual(info["reason"], "pedal_conflict")

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

    def test_same_raw_radar_frame_is_not_counted_twice(self):
        supervisor = EmergencyBrakeSupervisor()
        obstacle = {
            "track_id": -2,
            "distance_m": 6.0,
            "relative_speed_mps": -8.0,
            "bearing_deg": 0.0,
            "source": "radar_emergency",
            "measurement_frame_id": 10,
        }

        first_emergency, _ = supervisor.evaluate(obstacle)
        second_emergency, info = supervisor.evaluate(obstacle)

        self.assertFalse(first_emergency)
        self.assertFalse(second_emergency)
        self.assertEqual(info["hazard_count"], 1)
        self.assertFalse(info["new_observation"])

    def test_raw_radar_requires_two_new_frames_for_emergency(self):
        supervisor = EmergencyBrakeSupervisor()
        obstacle = {
            "track_id": -2,
            "distance_m": 6.0,
            "relative_speed_mps": -8.0,
            "bearing_deg": 0.0,
            "source": "radar_emergency",
            "measurement_frame_id": 10,
        }
        first_emergency, _ = supervisor.evaluate(obstacle)
        obstacle["measurement_frame_id"] = 11
        second_emergency, info = supervisor.evaluate(obstacle)

        self.assertFalse(first_emergency)
        self.assertTrue(second_emergency)
        self.assertEqual(info["hazard_count"], 2)

    def test_different_raw_points_do_not_build_one_hazard(self):
        supervisor = EmergencyBrakeSupervisor()
        first = {
            "track_id": -2,
            "distance_m": 6.0,
            "relative_speed_mps": -8.0,
            "bearing_deg": -5.0,
            "source": "radar_emergency",
            "measurement_frame_id": 10,
        }
        second = dict(first, bearing_deg=5.0, measurement_frame_id=11)
        supervisor.evaluate(first)
        emergency, info = supervisor.evaluate(second)

        self.assertFalse(emergency)
        self.assertEqual(info["hazard_count"], 1)

    def test_physically_critical_raw_point_still_brakes_immediately(self):
        supervisor = EmergencyBrakeSupervisor()
        emergency, info = supervisor.evaluate(
            {
                "distance_m": 0.5,
                "relative_speed_mps": -1.0,
                "source": "radar_emergency",
                "measurement_frame_id": 10,
            }
        )
        self.assertTrue(emergency)
        self.assertEqual(info["reason"], "critical_distance")


class LeadVehicleTrackerTests(unittest.TestCase):
    def test_front_camera_and_radar_create_one_fused_track(self):
        tracker = LeadVehicleTracker(0.05, 800, 90.0)
        state = straight_state(speed_mps=8.0)
        radar_points = [
            {
                "depth_m": 20.0 + offset,
                "azimuth_deg": offset,
                "altitude_deg": 0.0,
                "relative_velocity_mps": -2.0,
            }
            for offset in (-0.2, 0.0, 0.2)
        ]

        lead = None
        for frame_id in (1, 2, 3):
            perception = {
                "frame_id": frame_id,
                "vehicles": [
                    {
                        "bbox": (350, 180, 450, 420),
                        "confidence": 0.90,
                        "class_name": "vehicle",
                    }
                ],
            }
            lead = tracker.update(
                frame_id,
                state,
                perception,
                frame_id,
                radar_points,
            )

        self.assertIsNotNone(lead)
        self.assertEqual(lead["source"], "camera_radar_track")

    def test_logged_bumper_radar_ground_returns_are_rejected(self):
        tracker = LeadVehicleTracker(
            0.05,
            800,
            90.0,
            radar_height_m=0.40,
            radar_pitch_deg=0.0,
        )
        state = straight_state(speed_mps=0.94)
        ground_points = [
            {
                "depth_m": depth,
                "azimuth_deg": azimuth,
                "altitude_deg": -6.0,
                "relative_velocity_mps": -0.94,
            }
            for depth, azimuth in ((3.8, -0.2), (3.9, 0.0), (4.0, 0.2))
        ]

        first = tracker.update(1, state, None, 1, ground_points)
        second = tracker.update(2, state, None, 2, ground_points)
        diagnostics = tracker.get_radar_diagnostics()

        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertIsNone(tracker.get_emergency_obstacle())
        self.assertEqual(diagnostics["usable_points"], 0)
        self.assertEqual(diagnostics["ground_rejected"], 3)

    def test_vehicle_height_radar_returns_remain_available(self):
        tracker = LeadVehicleTracker(
            0.05,
            800,
            90.0,
            radar_height_m=1.0,
            radar_pitch_deg=2.0,
        )
        state = straight_state(speed_mps=8.0)
        vehicle_points = [
            {
                "depth_m": 20.0 + offset,
                "azimuth_deg": offset,
                "altitude_deg": -1.0,
                "relative_velocity_mps": -3.0,
            }
            for offset in (-0.3, 0.0, 0.3)
        ]

        tracker.update(1, state, None, 1, vehicle_points)
        lead = tracker.update(2, state, None, 2, vehicle_points)

        self.assertIsNotNone(lead)
        self.assertEqual(lead["source"], "radar_direct")
        self.assertEqual(tracker.get_radar_diagnostics()["ground_rejected"], 0)

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
        lead = tracker.choose_safest_lead(tracked_lead, radar_lead)
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

    def test_repeated_radar_frame_is_not_temporal_confirmation(self):
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
        repeated = tracker.update(2, state, None, 1, points)
        confirmed = tracker.update(3, state, None, 2, points)

        self.assertIsNone(first)
        self.assertIsNone(repeated)
        self.assertIsNotNone(confirmed)

    def test_stale_radar_frame_is_ignored(self):
        tracker = LeadVehicleTracker(0.05, 800, 90.0)
        state = straight_state(speed_mps=8.0)
        points = [
            {
                "depth_m": 6.0,
                "azimuth_deg": 0.0,
                "altitude_deg": 0.0,
                "relative_velocity_mps": -8.0,
            }
        ]

        lead = tracker.update(20, state, None, 1, points)

        self.assertIsNone(lead)
        self.assertIsNone(tracker.get_emergency_obstacle())
        self.assertFalse(tracker.get_radar_diagnostics()["fresh"])

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

    def test_roadside_point_just_inside_lane_edge_is_rejected(self):
        tracker = LeadVehicleTracker(0.05, 800, 90.0)
        state = straight_state(speed_mps=8.0)
        lateral_m = 1.55
        depth_m = math.hypot(9.0, lateral_m)
        points = [
            {
                "depth_m": depth_m,
                "azimuth_deg": math.degrees(math.atan2(lateral_m, 9.0)),
                "altitude_deg": 0.0,
                "relative_velocity_mps": -8.0,
            }
        ]

        tracker.update(1, state, None, 1, points)
        self.assertIsNone(tracker.get_emergency_obstacle())


if __name__ == "__main__":
    unittest.main()

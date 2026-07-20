import math
import sys
import types
import unittest
from types import SimpleNamespace

if "dotenv" not in sys.modules:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *arguments, **keywords: None
    sys.modules["dotenv"] = dotenv

from carla_app.controller.vehicle.lead_vehicle import LeadVehicleTracker
from carla_app.controller.vehicle.idm_speed_planner import IDMSpeedPlanner
from carla_app.controller.vehicle.longitudinal_pid_controller import (
    LongitudinalPIDController,
)
from carla_app.controller.vehicle.pure_pursuit_controller import (
    PurePursuitController,
)
from carla_app.controller.vehicle.pure_pursuit_mpc_controller import (
    PurePursuitMPCController,
)
from carla_app.controller.vehicle.safety_supervisor import EmergencyBrakeSupervisor
from carla_app.controller.vehicle.speed_planner import CurvatureSpeedPlanner


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


def curved_state(radius_m=20.0, speed_mps=8.0):
    angles = [index / radius_m for index in range(81)]
    path = [
        location(
            radius_m * math.sin(angle),
            radius_m * (1.0 - math.cos(angle)),
        )
        for angle in angles
    ]
    state = straight_state(speed_mps=speed_mps)
    state["reference_path"] = path
    return state


class PurePursuitControllerTests(unittest.TestCase):
    def test_steers_toward_route_when_vehicle_is_right_of_path(self):
        controller = PurePursuitController(dt=0.05)

        steer = controller.run_step(straight_state(speed_mps=10.0, y=1.0))

        self.assertLess(steer, 0.0)
        self.assertGreater(controller.last_info["cross_track_error_m"], 0.0)

    def test_steers_into_left_curve(self):
        controller = PurePursuitController(dt=0.05)

        steer = controller.run_step(curved_state(radius_m=25.0))

        self.assertGreater(steer, 0.0)
        self.assertEqual(controller.last_info["controller"], "pure_pursuit")

    def test_lookahead_grows_with_speed(self):
        controller = PurePursuitController(dt=0.05)

        slow = controller.calculate_lookahead(3.0)
        fast = controller.calculate_lookahead(20.0)

        self.assertGreater(fast, slow)
        self.assertLessEqual(fast, controller.maximum_lookahead_m)

    def test_steering_change_is_rate_limited(self):
        controller = PurePursuitController(dt=0.05)
        first = controller.run_step(straight_state(speed_mps=10.0, y=1.5))
        second = controller.run_step(straight_state(speed_mps=10.0, y=-1.5))

        maximum_change = (0.90 - 0.020 * 10.0) * controller.dt
        self.assertLessEqual(abs(second - first), maximum_change + 1e-9)

    def test_short_path_returns_finite_command(self):
        controller = PurePursuitController(dt=0.05)
        state = straight_state(speed_mps=5.0)
        state["reference_path"] = [location(0.0)]

        steer = controller.run_step(state)

        self.assertTrue(math.isfinite(steer))
        self.assertEqual(controller.last_info["reason"], "short_path")


class PurePursuitMPCControllerTests(unittest.TestCase):
    def test_pure_pursuit_builds_mpc_warm_start(self):
        controller = PurePursuitMPCController(dt=0.05)

        steer = controller.run_step(curved_state(radius_m=25.0))

        self.assertTrue(math.isfinite(steer))
        self.assertTrue(controller.last_info["mpc_active"])
        self.assertEqual(
            controller.last_info["controller"],
            "pure_pursuit_mpc",
        )
        self.assertTrue(
            math.isfinite(controller.last_info["pure_pursuit_seed_steer"])
        )
        self.assertLess(
            controller.last_info["mpc_solve_ms"],
            controller.parameters.mpc_time_budget_ms,
        )

    def test_low_speed_uses_only_pure_pursuit(self):
        controller = PurePursuitMPCController(dt=0.05)

        steer = controller.run_step(
            curved_state(radius_m=25.0, speed_mps=0.2)
        )

        self.assertTrue(math.isfinite(steer))
        self.assertFalse(controller.last_info["mpc_active"])
        self.assertEqual(controller.last_info["fallback_reason"], "low_speed")

    def test_warm_start_respects_first_steering_rate_limit(self):
        controller = PurePursuitMPCController(dt=0.05)
        speed_mps = 10.0
        sequence = controller.build_pure_pursuit_warm_start(
            pure_pursuit_steer=0.8,
            speed_mps=speed_mps,
            horizon=controller.parameters.mpc_horizon_steps,
            model_dt=controller.parameters.mpc_step_s,
        )
        first_normalized = sequence[0] / controller.maximum_wheel_angle_rad
        maximum_change = controller.steering_rate_limit(speed_mps) * controller.dt

        self.assertLessEqual(abs(first_normalized), maximum_change + 1e-9)


class SpeedPlannerTests(unittest.TestCase):
    def test_no_sign_uses_seventy_kmh(self):
        planner = CurvatureSpeedPlanner(dt=0.05)

        speed, info = planner.run_step(straight_state(speed_mps=10.0))

        self.assertAlmostEqual(speed, 70.0 / 3.6)
        self.assertAlmostEqual(info["road_speed_mps"], 70.0 / 3.6)

    def test_speed_sign_becomes_straight_road_target(self):
        planner = CurvatureSpeedPlanner(dt=0.05)

        for _ in range(400):
            speed, info = planner.run_step(
                straight_state(speed_mps=20.0),
                speed_limit_kmh=90,
            )

        self.assertAlmostEqual(speed, 90.0 / 3.6, places=4)
        self.assertEqual(info["speed_reason"], "speed_limit")

    def test_curve_target_never_drops_below_twenty_three_kmh(self):
        planner = CurvatureSpeedPlanner(dt=0.05)
        state = curved_state(radius_m=8.0, speed_mps=15.0)

        for _ in range(300):
            speed, info = planner.run_step(state)

        self.assertAlmostEqual(speed, 23.0 / 3.6, places=3)
        self.assertAlmostEqual(info["desired_speed_mps"], 23.0 / 3.6, places=3)
        self.assertEqual(info["speed_reason"], "curve")

    def test_lower_speed_sign_has_priority_over_curve_floor(self):
        planner = CurvatureSpeedPlanner(dt=0.05)

        _, info = planner.run_step(
            curved_state(radius_m=8.0),
            speed_limit_kmh=20,
        )

        self.assertAlmostEqual(info["desired_speed_mps"], 20.0 / 3.6)

    def test_sustained_large_lane_error_requests_twenty_three_kmh(self):
        planner = CurvatureSpeedPlanner(dt=0.05)

        for _ in range(planner.recovery_confirmation_ticks):
            _, info = planner.run_step(
                straight_state(speed_mps=12.0),
                lateral_info={
                    "cross_track_error_m": 2.0,
                    "heading_error_rad": math.radians(35.0),
                },
            )

        self.assertAlmostEqual(info["recovery_speed_mps"], 23.0 / 3.6)


class IDMSpeedPlannerTests(unittest.TestCase):
    def test_fast_closing_vehicle_increases_desired_gap(self):
        planner = IDMSpeedPlanner(dt=0.05)
        speed_mps = 50.0 / 3.6

        steady_gap = planner.calculate_desired_gap(speed_mps, 0.0)
        closing_gap = planner.calculate_desired_gap(speed_mps, -5.0)

        self.assertGreater(closing_gap, steady_gap)

    def test_confirmed_close_vehicle_reduces_pid_reference(self):
        planner = IDMSpeedPlanner(dt=0.05)
        lead = {
            "track_id": 8,
            "distance_m": 8.0,
            "relative_speed_mps": -10.0,
            "source": "camera_radar_track",
        }
        state = straight_state(speed_mps=10.0)

        planner.run_step(state, lead, 70.0 / 3.6)
        reference, info = planner.run_step(state, lead, 70.0 / 3.6)

        self.assertTrue(info["lead_confirmed"])
        self.assertTrue(info["following"])
        self.assertLess(reference, state["speed_mps"])
        self.assertLess(info["idm_acceleration_mps2"], 0.0)

    def test_far_non_closing_vehicle_keeps_free_road_behavior(self):
        planner = IDMSpeedPlanner(dt=0.05)
        lead = {
            "track_id": 9,
            "distance_m": 100.0,
            "relative_speed_mps": 0.0,
            "source": "camera_radar_track",
        }
        state = straight_state(speed_mps=10.0)

        planner.run_step(state, lead, 70.0 / 3.6)
        reference, info = planner.run_step(state, lead, 70.0 / 3.6)

        self.assertTrue(info["lead_is_far"])
        self.assertGreater(reference, state["speed_mps"])

    def test_red_light_is_used_as_immediate_virtual_vehicle(self):
        planner = IDMSpeedPlanner(dt=0.05)
        red_light = {
            "track_id": -3,
            "distance_m": 5.0,
            "relative_speed_mps": -8.0,
            "source": "traffic_light_red",
        }

        reference, info = planner.run_step(
            straight_state(speed_mps=8.0),
            red_light,
            60.0 / 3.6,
        )

        self.assertTrue(info["rule_obstacle"])
        self.assertTrue(info["lead_confirmed"])
        self.assertLess(reference, 8.0)

    def test_red_light_clears_old_normal_lead_confirmation(self):
        planner = IDMSpeedPlanner(dt=0.05)
        normal_lead = {
            "track_id": 12,
            "distance_m": 12.0,
            "relative_speed_mps": -2.0,
            "source": "radar_direct",
        }
        red_light = {
            "track_id": -3,
            "distance_m": 6.0,
            "relative_speed_mps": -5.0,
            "source": "traffic_light_red",
        }
        state = straight_state(speed_mps=5.0)

        planner.run_step(state, normal_lead, 60.0 / 3.6)
        planner.run_step(state, normal_lead, 60.0 / 3.6)
        planner.run_step(state, red_light, 60.0 / 3.6)
        _, info = planner.run_step(state, normal_lead, 60.0 / 3.6)

        self.assertFalse(info["lead_confirmed"])


class LongitudinalPIDControllerTests(unittest.TestCase):
    def test_accelerates_toward_target_without_brake(self):
        controller = LongitudinalPIDController(dt=0.05)

        throttle, brake, info = controller.run_step(
            straight_state(speed_mps=2.0),
            lead_vehicle=None,
            target_speed=70.0 / 3.6,
        )

        self.assertEqual(info["controller"], "pid")
        self.assertGreater(throttle, 0.0)
        self.assertEqual(brake, 0.0)

    def test_non_finite_feedforward_is_ignored(self):
        controller = LongitudinalPIDController(dt=0.05)

        throttle, brake, info = controller.run_step(
            straight_state(speed_mps=2.0),
            lead_vehicle=None,
            target_speed=5.0,
            feedforward_acceleration_mps2=float("nan"),
        )

        self.assertTrue(math.isfinite(throttle))
        self.assertTrue(math.isfinite(brake))
        self.assertEqual(info["feedforward_acceleration_mps2"], 0.0)

    def test_overspeed_produces_brake_without_throttle(self):
        controller = LongitudinalPIDController(dt=0.05)

        throttle, brake, _ = controller.run_step(
            straight_state(speed_mps=20.0),
            lead_vehicle=None,
            target_speed=10.0,
        )

        self.assertEqual(throttle, 0.0)
        self.assertGreater(brake, 0.0)

    def test_acceleration_change_respects_jerk_limit(self):
        controller = LongitudinalPIDController(dt=0.05)

        _, _, first = controller.run_step(
            straight_state(speed_mps=0.0),
            None,
            70.0 / 3.6,
        )
        _, _, second = controller.run_step(
            straight_state(speed_mps=0.0),
            None,
            70.0 / 3.6,
        )

        maximum_change = controller.acceleration_jerk_mps3 * controller.dt
        difference = second["acceleration_mps2"] - first["acceleration_mps2"]
        self.assertLessEqual(difference, maximum_change + 1e-9)

    def test_pid_tracks_idm_reference_without_second_gap_calculation(self):
        controller = LongitudinalPIDController(dt=0.05)
        lead = {
            "distance_m": 8.0,
            "relative_speed_mps": -10.0,
            "source": "camera_radar_track",
        }

        throttle, brake, info = controller.run_step(
            straight_state(speed_mps=10.0),
            lead,
            7.0,
        )

        self.assertEqual(info["effective_target_speed_mps"], 7.0)
        self.assertIsNone(info["desired_gap_m"])
        self.assertEqual(throttle, 0.0)
        self.assertGreater(brake, 0.0)

    def test_zero_target_holds_stopped_vehicle(self):
        controller = LongitudinalPIDController(dt=0.05)

        throttle, brake, info = controller.run_step(
            straight_state(speed_mps=0.0),
            None,
            0.0,
        )

        self.assertEqual(info["mode"], "HOLD")
        self.assertEqual(throttle, 0.0)
        self.assertEqual(brake, controller.hold_brake)

    def test_zero_reference_does_not_hold_far_from_physical_lead(self):
        controller = LongitudinalPIDController(dt=0.05)
        far_stopped_lead = {
            "distance_m": 8.0,
            "relative_speed_mps": 0.0,
            "source": "camera_radar_track",
        }

        _, _, info = controller.run_step(
            straight_state(speed_mps=0.0),
            far_stopped_lead,
            target_speed=0.0,
        )

        self.assertEqual(info["mode"], "TRACKING")
        self.assertFalse(info["hold_active"])

    def test_stopped_physical_lead_holds_at_two_metres(self):
        controller = LongitudinalPIDController(dt=0.05)
        close_stopped_lead = {
            "distance_m": 2.05,
            "relative_speed_mps": 0.0,
            "source": "camera_radar_track",
        }

        throttle, brake, info = controller.run_step(
            straight_state(speed_mps=0.0),
            close_stopped_lead,
            target_speed=0.0,
        )

        self.assertEqual(info["mode"], "HOLD")
        self.assertEqual(throttle, 0.0)
        self.assertEqual(brake, controller.hold_brake)

    def test_confirmed_green_target_releases_hold_and_uses_breakaway_throttle(self):
        controller = LongitudinalPIDController(dt=0.05)
        controller.run_step(
            straight_state(speed_mps=0.0),
            lead_vehicle=None,
            target_speed=0.0,
        )

        throttle, brake, info = controller.run_step(
            straight_state(speed_mps=0.0),
            lead_vehicle=None,
            target_speed=0.12,
        )

        self.assertEqual(info["mode"], "RESTART")
        self.assertTrue(info["hold_released"])
        self.assertGreaterEqual(throttle, controller.breakaway_throttle)
        self.assertEqual(brake, 0.0)

    def test_pid_converges_near_seventy_kmh(self):
        controller = LongitudinalPIDController(dt=0.05)
        speed = 0.0

        for _ in range(900):
            throttle, brake, _ = controller.run_step(
                straight_state(speed_mps=speed),
                None,
                70.0 / 3.6,
            )
            acceleration = 3.0 * throttle - 5.0 * brake
            if speed > 0.02:
                acceleration -= 0.12
            speed = max(0.0, speed + acceleration * controller.dt)

        self.assertAlmostEqual(speed * 3.6, 70.0, delta=1.0)


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

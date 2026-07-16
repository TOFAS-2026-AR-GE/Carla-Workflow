import math
import unittest
from types import SimpleNamespace

from carla_app.controller.vehicle.lead_vehicle import LeadVehicleTracker
from carla_app.controller.vehicle.longitudinal_controller import (
    LongitudinalController,
)
from carla_app.controller.vehicle.safety_supervisor import (
    EmergencyBrakeSupervisor,
)
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


class LongitudinalControllerTests(unittest.TestCase):
    def test_far_lead_does_not_cause_braking(self):
        controller = LongitudinalController(dt=0.05)
        state = straight_state(speed_mps=8.0)
        lead = {
            "track_id": 10,
            "distance_m": 60.0,
            "relative_speed_mps": -5.0,
        }

        throttle, brake, info = controller.run_step(state, lead, 30.0 / 3.6)
        self.assertEqual(info["mode"], "LEAD_FAR")
        self.assertGreater(throttle, 0.0)
        self.assertEqual(brake, 0.0)

    def test_known_stationary_lead_enters_approach_without_cruise_surge(self):
        controller = LongitudinalController(dt=0.05)
        state = straight_state(speed_mps=0.0)
        lead = {
            "track_id": 10,
            "distance_m": 8.0,
            "relative_speed_mps": 0.0,
        }

        controller.run_step(state, lead, 30.0 / 3.6)
        throttle, brake, info = controller.run_step(state, lead, 30.0 / 3.6)

        self.assertEqual(info["mode"], "APPROACH")
        self.assertAlmostEqual(info["approach_target_speed_mps"], 1.2)
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
        throttle, brake, info = controller.run_step(state, lead, 30.0 / 3.6)
        self.assertEqual(info["mode"], "FOLLOW")
        self.assertEqual(throttle, 0.0)
        self.assertGreater(brake, 0.0)

    def test_two_metres_is_held_as_the_standstill_gap(self):
        controller = LongitudinalController(dt=0.05)
        state = straight_state(speed_mps=0.0)
        lead = {
            "track_id": 11,
            "distance_m": 2.0,
            "relative_speed_mps": 0.0,
        }

        controller.run_step(state, lead, 30.0 / 3.6)
        throttle, brake, info = controller.run_step(state, lead, 30.0 / 3.6)
        self.assertEqual(info["mode"], "HOLD")
        self.assertAlmostEqual(info["desired_gap_m"], 2.0)
        self.assertEqual(throttle, 0.0)
        self.assertEqual(brake, controller.hold_brake)

    def test_lead_pulling_away_restarts_without_acceleration_memory_delay(self):
        controller = LongitudinalController(dt=0.05)
        state = straight_state(speed_mps=0.0)
        stopped_lead = {
            "track_id": 11,
            "distance_m": 2.0,
            "relative_speed_mps": 0.0,
        }

        controller.run_step(state, stopped_lead, 30.0 / 3.6)
        controller.run_step(state, stopped_lead, 30.0 / 3.6)

        moving_lead = {
            "track_id": 11,
            "distance_m": 2.2,
            "relative_speed_mps": 0.8,
        }
        first_throttle, first_brake, first_info = controller.run_step(
            state,
            moving_lead,
            30.0 / 3.6,
        )
        throttle, brake, info = controller.run_step(
            state,
            moving_lead,
            30.0 / 3.6,
        )

        self.assertEqual(first_info["mode"], "HOLD")
        self.assertEqual(first_throttle, 0.0)
        self.assertEqual(first_brake, controller.hold_brake)
        self.assertEqual(info["mode"], "RESTART")
        self.assertTrue(info["restart_active"])
        self.assertGreater(throttle, 0.0)
        self.assertEqual(brake, 0.0)

    def test_slow_lead_restart_is_detected_from_opening_gap(self):
        controller = LongitudinalController(dt=0.05)
        state = straight_state(speed_mps=0.0)
        stopped_lead = {
            "track_id": 11,
            "distance_m": 2.0,
            "relative_speed_mps": 0.0,
        }

        controller.run_step(state, stopped_lead, 30.0 / 3.6)
        controller.run_step(state, stopped_lead, 30.0 / 3.6)

        moving_distance = 2.0
        modes = []
        for _ in range(120):
            moving_distance += 0.2 * controller.dt
            _, _, info = controller.run_step(
                state,
                {
                    "track_id": 11,
                    "distance_m": moving_distance,
                    "relative_speed_mps": 0.2,
                },
                30.0 / 3.6,
            )
            modes.append(info["mode"])

        self.assertIn("RESTART", modes)

    def test_slow_approach_to_stopped_lead_does_not_enter_restart(self):
        controller = LongitudinalController(dt=0.05)
        state = straight_state(speed_mps=0.8)
        stopped_lead = {
            "track_id": 11,
            "distance_m": 4.0,
            "relative_speed_mps": -0.8,
        }

        controller.run_step(state, stopped_lead, 30.0 / 3.6)
        throttle, brake, info = controller.run_step(
            state,
            stopped_lead,
            30.0 / 3.6,
        )

        self.assertEqual(info["mode"], "APPROACH")
        self.assertFalse(info["restart_active"])
        self.assertEqual(throttle, 0.0)
        self.assertEqual(brake, 0.0)

    def test_moving_lead_releases_the_approach_brake_phase(self):
        controller = LongitudinalController(dt=0.05)
        state = straight_state(speed_mps=1.0)
        stopped_lead = {
            "track_id": 11,
            "distance_m": 4.0,
            "relative_speed_mps": -1.0,
        }

        controller.run_step(state, stopped_lead, 30.0 / 3.6)
        controller.run_step(state, stopped_lead, 30.0 / 3.6)
        self.assertEqual(controller.approach_phase, "BRAKE")

        moving_lead = {
            "track_id": 11,
            "distance_m": 4.2,
            "relative_speed_mps": 1.0,
        }
        for _ in range(4):
            _, _, info = controller.run_step(
                state,
                moving_lead,
                30.0 / 3.6,
            )

        self.assertEqual(info["approach_phase"], "DRIVE")

    def test_logged_noisy_ranges_do_not_trigger_false_restart_or_brake_chatter(self):
        controller = LongitudinalController(dt=0.05)
        samples = [
            (0.64, 4.8, -0.65),
            (0.64, 6.9, -0.43),
            (0.42, 5.0, -1.19),
            (0.97, 5.3, -0.81),
            (0.64, 4.6, -0.54),
            (0.31, 5.4, -0.19),
            (0.31, 4.9, -0.21),
            (0.28, 4.8, -0.17),
            (0.75, 4.6, -0.67),
            (0.06, 5.2, -0.10),
            (0.97, 7.1, -0.92),
            (1.17, 6.1, -1.06),
            (0.97, 5.3, -0.71),
            (0.14, 6.8, -0.10),
            (0.14, 4.6, -0.14),
        ]

        modes = []
        filtered_distances = []
        brakes = []
        for speed, distance, relative_speed in samples:
            throttle, brake, info = controller.run_step(
                straight_state(speed_mps=speed),
                {
                    "track_id": 11,
                    "distance_m": distance,
                    "relative_speed_mps": relative_speed,
                },
                30.0 / 3.6,
            )
            self.assertFalse(throttle > 0.0 and brake > 0.0)
            modes.append(info["mode"])
            filtered_distances.append(info["filtered_lead_distance_m"])
            brakes.append(brake)

        self.assertNotIn("RESTART", modes)
        self.assertNotIn("HOLD", modes)
        self.assertLess(max(filtered_distances) - min(filtered_distances), 0.40)
        self.assertEqual(sum(brake > 0.0 for brake in brakes), 0)

    def test_noisy_stationary_lead_ends_in_one_hold_without_stop_go_cycle(self):
        controller = LongitudinalController(dt=0.05)
        speed = 0.0
        distance = 5.0
        distance_noise = [0.0, 2.1, -0.2, 1.6, 0.1, 2.3, -0.1, 0.4]
        velocity_noise = [0.0, 0.15, -0.2, 0.1, -0.1, 0.2, -0.15, 0.0]
        last_pedal = None
        pedal_switches = 0
        modes = []

        for tick in range(400):
            measured_distance = distance + distance_noise[tick % len(distance_noise)]
            measured_relative_speed = (
                -speed + velocity_noise[tick % len(velocity_noise)]
            )
            throttle, brake, info = controller.run_step(
                straight_state(speed_mps=speed),
                {
                    "track_id": 11,
                    "distance_m": measured_distance,
                    "relative_speed_mps": measured_relative_speed,
                },
                30.0 / 3.6,
            )
            modes.append(info["mode"])

            pedal = "throttle" if throttle > 0.01 else "brake" if brake > 0.01 else None
            if pedal is not None and last_pedal is not None and pedal != last_pedal:
                pedal_switches += 1
            if pedal is not None:
                last_pedal = pedal

            acceleration = 2.8 * throttle - 5.0 * brake
            if speed > 0.02 and throttle == 0.0:
                acceleration -= 0.12
            speed = max(0.0, speed + acceleration * controller.dt)
            distance -= speed * controller.dt

            if info["mode"] == "HOLD" and speed <= 0.02:
                break

        self.assertEqual(info["mode"], "HOLD")
        self.assertLessEqual(pedal_switches, 1)
        self.assertNotIn("RESTART", modes)
        self.assertGreaterEqual(distance, 1.9)
        self.assertLessEqual(distance, 2.35)


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

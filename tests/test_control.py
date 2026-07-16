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

    def test_three_metres_is_standstill_gap_not_moving_gap(self):
        controller = LongitudinalController(dt=0.05)
        state = straight_state(speed_mps=0.0)
        lead = {
            "track_id": 11,
            "distance_m": 3.0,
            "relative_speed_mps": 0.0,
        }

        controller.run_step(state, lead, 30.0 / 3.6)
        throttle, brake, info = controller.run_step(state, lead, 30.0 / 3.6)
        self.assertEqual(info["mode"], "FOLLOW")
        self.assertAlmostEqual(info["desired_gap_m"], 3.0)
        self.assertEqual(throttle, 0.0)
        self.assertEqual(brake, 0.0)


class EmergencyBrakeTests(unittest.TestCase):
    def test_immediate_ttc_triggers_full_safety_request(self):
        supervisor = EmergencyBrakeSupervisor()
        emergency, info = supervisor.evaluate(
            {"distance_m": 5.0, "relative_speed_mps": -10.0}
        )
        self.assertTrue(emergency)
        self.assertAlmostEqual(info["ttc_s"], 0.5)


class LeadVehicleTrackerTests(unittest.TestCase):
    def test_radar_lead_requires_temporal_confirmation(self):
        tracker = LeadVehicleTracker(0.05, 800, 90.0)
        state = straight_state(speed_mps=8.0)
        points = [
            {
                "depth_m": 20.0 + offset,
                "azimuth_deg": offset,
                "altitude_deg": 0.0,
                "relative_velocity_mps": 3.0,
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
                "relative_velocity_mps": 8.0,
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
                "relative_velocity_mps": 8.0,
            }
        ]

        tracker.update(1, state, None, 1, points)

        self.assertIsNone(tracker.get_emergency_obstacle())


if __name__ == "__main__":
    unittest.main()

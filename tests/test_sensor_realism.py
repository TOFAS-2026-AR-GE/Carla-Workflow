"""Sensör tabanlı mimarinin deterministik kabul testleri."""

import math
import unittest
from types import SimpleNamespace

from carla_app.controller.vehicle.intersection_guard import (
    IntersectionGuard,
    should_hold_for_intersection,
)
from carla_app.controller.vehicle.tracking import Tracker
from carla_app.controller.vehicle.uncertainty import ControlUncertaintyManager
from carla_app.localization.ekf_localizer import ExtendedKalmanLocalizer
from carla_app.localization.odometry import OdometryAdapter
from carla_app.localization.system import compass_to_carla_yaw


class Parameters:
    intersection_clear_confirmation_frames = 2
    intersection_monitor_distance_m = 35.0
    intersection_radar_maximum_age_s = 0.20
    intersection_zone_start_m = 1.5
    intersection_zone_end_m = 18.0
    intersection_zone_half_width_m = 7.0
    intersection_prediction_horizon_s = 4.0
    control_uncertainty_sigma = 2.0
    lead_measurement_hard_age_s = 0.35
    emergency_measurement_hard_age_s = 0.20
    localization_degraded_speed_mps = 10.0 / 3.6
    localization_hard_stop_age_s = 0.60
    perception_degraded_age_s = 0.20
    perception_stop_age_s = 0.50
    perception_degraded_speed_mps = 15.0 / 3.6
    maximum_target_speed_mps = 130.0 / 3.6
    intersection_hold_distance_m = 2.0


def transform(x=0.0, y=0.0, yaw=0.0):
    return SimpleNamespace(
        location=SimpleNamespace(x=x, y=y, z=0.8),
        rotation=SimpleNamespace(yaw=yaw, pitch=0.0, roll=0.0),
    )


class EkfLocalizationTests(unittest.TestCase):
    def test_gnss_correction_reduces_position_error(self):
        localizer = ExtendedKalmanLocalizer(dt=0.05, gnss_position_std_m=0.5)
        localizer.initialize(0.0, 0.0, 0.0, frame_id=0, speed_mps=5.0)
        imu = {
            "accelerometer": {"x": 0.0, "y": 0.0, "z": 0.0},
            "gyroscope": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
        for frame_id in range(1, 21):
            localizer.predict(frame_id, imu)
        before = abs(localizer.result()["x_m"] - 4.8)
        accepted = localizer.update_gnss(4.8, 0.1, frame_id=20, std_m=0.4)
        after = abs(localizer.result()["x_m"] - 4.8)
        self.assertTrue(accepted)
        self.assertLess(after, before)

    def test_large_gnss_outlier_is_rejected(self):
        localizer = ExtendedKalmanLocalizer(dt=0.05, gnss_position_std_m=0.5)
        localizer.initialize(0.0, 0.0, 0.0, frame_id=0)
        accepted = localizer.update_gnss(100.0, 100.0, frame_id=1, std_m=0.5)
        self.assertFalse(accepted)
        self.assertEqual(localizer.result()["rejected_gnss_updates"], 1)

    def test_compass_conversion_matches_carla_yaw_convention(self):
        self.assertAlmostEqual(compass_to_carla_yaw(math.pi / 2.0), 0.0)
        self.assertAlmostEqual(compass_to_carla_yaw(0.0), math.pi / 2.0)




class OdometryAdapterTests(unittest.TestCase):
    def test_wheel_odometry_is_read_without_ground_truth_velocity(self):
        adapter = OdometryAdapter()
        result = adapter.read(
            {
                "wheel_odometry": {
                    "frame_id": 12,
                    "data": {"speed_mps": 8.5},
                }
            },
            current_frame_id=12,
        )
        self.assertEqual(result["frame_id"], 12)
        self.assertAlmostEqual(result["speed_mps"], 8.5)
        self.assertEqual(result["source"], "wheel_odometry")

class HungarianTrackerTests(unittest.TestCase):
    def test_crossing_targets_keep_two_unique_assignments(self):
        tracker = Tracker(gate_distance_m=8.0, max_misses=3)
        tracker.step(
            0.1,
            [
                {"x": -2.0, "y": 0.0, "class_name": "vehicle"},
                {"x": 2.0, "y": 0.0, "class_name": "vehicle"},
            ],
        )
        original_ids = [track.id for track in tracker.tracks]
        for step in range(1, 5):
            tracker.step(
                0.1,
                [
                    {
                        "x": -2.0 + step,
                        "y": 0.2,
                        "class_name": "vehicle",
                        "position_std_m": 0.3,
                    },
                    {
                        "x": 2.0 - step,
                        "y": -0.2,
                        "class_name": "vehicle",
                        "position_std_m": 0.3,
                    },
                ],
            )
        self.assertEqual(len(tracker.tracks), 2)
        self.assertEqual(set(original_ids), {track.id for track in tracker.tracks})
        self.assertTrue(all(track.confirmed for track in tracker.tracks))
        by_id = {track.id: track for track in tracker.tracks}
        self.assertGreater(by_id[original_ids[0]].x, 0.0)
        self.assertLess(by_id[original_ids[1]].x, 0.0)


class UncertaintyPolicyTests(unittest.TestCase):
    def test_old_lead_is_removed_from_normal_control(self):
        manager = ControlUncertaintyManager(0.05, Parameters())
        state = {
            "frame_id": 20,
            "speed_mps": 10.0,
            "localization": {"status": "NOMINAL", "sensor_age_s": 0.0},
        }
        lead = {
            "distance_m": 20.0,
            "relative_speed_mps": -1.0,
            "source": "camera_radar_track",
            "measurement_frame_id": 10,
        }
        adjusted, _, info = manager.apply(
            state,
            lead,
            None,
            {"perception_age_frames": 0},
        )
        self.assertIsNone(adjusted)
        self.assertTrue(info["lead_adjusted"] is False)

    def test_fresh_lead_keeps_measured_relative_speed(self):
        manager = ControlUncertaintyManager(0.05, Parameters())
        state = {
            "frame_id": 20,
            "speed_mps": 10.0,
            "localization": {"status": "NOMINAL", "sensor_age_s": 0.0},
        }
        lead = {
            "distance_m": 20.0,
            "relative_speed_mps": -1.0,
            "source": "camera_radar_track",
            "measurement_frame_id": 20,
        }
        adjusted, _, _ = manager.apply(
            state,
            lead,
            None,
            {"perception_age_frames": 0},
        )
        self.assertIsNotNone(adjusted)
        self.assertAlmostEqual(adjusted["relative_speed_mps"], -1.0)
        self.assertAlmostEqual(
            adjusted["relative_speed_uncertainty_mps"],
            0.0,
        )

    def test_relative_speed_margin_grows_with_measurement_age(self):
        manager = ControlUncertaintyManager(0.05, Parameters())
        state = {
            "frame_id": 20,
            "speed_mps": 10.0,
            "localization": {"status": "NOMINAL", "sensor_age_s": 0.0},
        }
        lead = {
            "distance_m": 20.0,
            "relative_speed_mps": -1.0,
            "source": "camera_radar_track",
            "measurement_frame_id": 16,
        }
        adjusted, _, _ = manager.apply(
            state,
            lead,
            None,
            {"perception_age_frames": 0},
        )
        self.assertIsNotNone(adjusted)
        self.assertLess(adjusted["relative_speed_mps"], -1.0)
        self.assertGreater(
            adjusted["relative_speed_uncertainty_mps"],
            0.0,
        )

    def test_legacy_state_without_localization_keeps_existing_tests_compatible(self):
        manager = ControlUncertaintyManager(0.05, Parameters())
        _, _, info = manager.apply(
            {"frame_id": 1, "speed_mps": 2.0},
            None,
            None,
            {"perception_age_frames": 0},
        )
        self.assertTrue(math.isinf(info["speed_cap_mps"]))
        self.assertEqual(info["reasons"], [])

    def test_degraded_localization_caps_speed(self):
        manager = ControlUncertaintyManager(0.05, Parameters())
        state = {
            "frame_id": 1,
            "speed_mps": 2.0,
            "localization": {"status": "DEGRADED", "sensor_age_s": 0.3},
        }
        _, _, info = manager.apply(
            state,
            None,
            None,
            {"perception_age_frames": 0},
        )
        self.assertAlmostEqual(info["speed_cap_mps"], 10.0 / 3.6)
        self.assertIn("localization_degraded", info["reasons"])


class IntersectionReleaseTests(unittest.TestCase):
    def test_green_light_waits_when_conflict_zone_is_unknown(self):
        hold = should_hold_for_intersection(
            {"color": "green"},
            {
                "active": True,
                "clear": False,
                "status": "UNKNOWN",
                "conflict": None,
            },
            ego_speed=0.1,
        )
        self.assertTrue(hold)

    def test_green_light_releases_only_after_clear_confirmation(self):
        hold = should_hold_for_intersection(
            {"color": "green"},
            {"active": True, "clear": True, "status": "CLEAR"},
            ego_speed=0.1,
        )
        self.assertFalse(hold)


class IntersectionGuardTests(unittest.TestCase):
    def build_guard(self):
        radars = (
            SimpleNamespace(
                name="radar_front_left",
                transform=transform(x=1.0, y=-0.8, yaw=-45.0),
            ),
            SimpleNamespace(
                name="radar_front_right",
                transform=transform(x=1.0, y=0.8, yaw=45.0),
            ),
        )
        layout = SimpleNamespace(radars=radars)
        return IntersectionGuard(layout, 0.05, Parameters())

    def state(self):
        return {
            "location": SimpleNamespace(x=0.0, y=0.0, z=0.0),
            "yaw": 0.0,
            "lane_width": 3.5,
            "is_junction": False,
        }

    def context(self):
        return {
            "lead_traffic_light": {
                "color": "green",
                "estimated_distance_m": 8.0,
            }
        }

    def test_missing_corner_radars_is_unknown_not_clear(self):
        guard = self.build_guard()
        result = guard.update(10, self.state(), {}, self.context())
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertFalse(result["clear"])

    def test_partial_corner_radar_coverage_is_unknown(self):
        guard = self.build_guard()
        snapshot = {
            "radar_front_left": {"frame_id": 10, "data": []},
        }
        result = guard.update(10, self.state(), snapshot, self.context())
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertFalse(result["clear"])
        self.assertEqual(result["fresh_radar_count"], 1)
        self.assertEqual(result["required_radar_count"], 2)
        self.assertFalse(result["coverage_complete"])
        self.assertEqual(result["clear_hits"], 0)

    def test_repeated_radar_frame_is_not_counted_twice(self):
        guard = self.build_guard()
        snapshot = {
            "radar_front_left": {"frame_id": 10, "data": []},
            "radar_front_right": {"frame_id": 10, "data": []},
        }
        first = guard.update(10, self.state(), snapshot, self.context())
        repeated = guard.update(11, self.state(), snapshot, self.context())
        self.assertEqual(first["clear_hits"], 1)
        self.assertEqual(repeated["clear_hits"], 1)
        self.assertFalse(repeated["clear"])

    def test_fresh_empty_corner_radars_require_confirmation(self):
        guard = self.build_guard()
        snapshot = {
            "radar_front_left": {"frame_id": 10, "data": []},
            "radar_front_right": {"frame_id": 10, "data": []},
        }
        first = guard.update(10, self.state(), snapshot, self.context())
        snapshot["radar_front_left"]["frame_id"] = 11
        snapshot["radar_front_right"]["frame_id"] = 11
        second = guard.update(11, self.state(), snapshot, self.context())
        self.assertEqual(first["status"], "VERIFYING_CLEAR")
        self.assertEqual(second["status"], "CLEAR")
        self.assertTrue(second["clear"])


if __name__ == "__main__":
    unittest.main()

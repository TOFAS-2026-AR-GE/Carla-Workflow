import sys
import types
import unittest

if "dotenv" not in sys.modules:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *arguments, **keywords: None
    sys.modules["dotenv"] = dotenv

if "carla" not in sys.modules:
    sys.modules["carla"] = types.ModuleType("carla")


class FakeVehicleControl:
    def __init__(self, throttle=0.0, steer=0.0, brake=0.0):
        self.throttle = throttle
        self.steer = steer
        self.brake = brake


sys.modules["carla"].VehicleControl = FakeVehicleControl

from carla_app.controller.vehicle.vehicle_controller import (  # noqa: E402
    VehicleController,
)


def location(x, y=0.0):
    return types.SimpleNamespace(x=float(x), y=float(y), z=0.0)


class VehicleControllerIntegrationTests(unittest.TestCase):
    def stopped_state(self, light_color):
        return {
            "location": location(0.0),
            "yaw": 0.0,
            "speed_mps": 0.0,
            "speed_kmh": 0.0,
            "reference_path": [location(x) for x in range(81)],
            "lane_width": 3.5,
            "road_id": 1,
            "lane_id": -1,
            "is_junction": False,
            "simulator_traffic_light": {
                "available": True,
                "affected": True,
                "color": light_color,
            },
        }

    def test_stopped_vehicle_restarts_when_simulator_confirms_green(self):
        controller = VehicleController(dt=0.05)
        stale_camera_red = {
            "color": "red",
            "track_id": 12,
            "estimated_distance_m": 5.8,
            "confidence": 0.90,
            "smoothed_confidence": 0.90,
            "state_source": "camera_tracker",
        }
        road_context = {"lead_traffic_light": stale_camera_red}

        stopped_control, stopped_info = controller.run_step(
            self.stopped_state("red"),
            lead_vehicle=None,
            road_context=road_context,
        )
        controller.run_step(
            self.stopped_state("green"),
            lead_vehicle=None,
            road_context=road_context,
        )
        green_control, green_info = controller.run_step(
            self.stopped_state("green"),
            lead_vehicle=None,
            road_context=road_context,
        )
        restart_control, restart_info = controller.run_step(
            self.stopped_state("green"),
            lead_vehicle=None,
            road_context=road_context,
        )

        self.assertEqual(stopped_info["mode"], "STOPPED_AT_RED")
        self.assertEqual(stopped_control.throttle, 0.0)
        self.assertGreater(stopped_control.brake, 0.0)
        self.assertEqual(green_info["mode"], "START_ON_GREEN")
        self.assertEqual(green_control.throttle, 0.0)
        self.assertGreater(restart_control.throttle, 0.0)
        self.assertEqual(restart_control.brake, 0.0)
        self.assertEqual(restart_info["longitudinal"]["mode"], "RESTART")

    def test_immediate_collision_risk_overrides_throttle_with_full_brake(self):
        controller = VehicleController(dt=0.05)
        state = {
            "location": location(0.0),
            "yaw": 0.0,
            "speed_mps": 8.0,
            "speed_kmh": 28.8,
            "reference_path": [location(x) for x in range(81)],
            "lane_width": 3.5,
            "road_id": 1,
            "lane_id": -1,
            "is_junction": False,
        }
        lead = {
            "track_id": 1,
            "distance_m": 5.0,
            "relative_speed_mps": -10.0,
        }

        control, info = controller.run_step(state, lead)

        self.assertEqual(info["mode"], "EMERGENCY")
        self.assertEqual(control.throttle, 0.0)
        self.assertEqual(control.brake, 1.0)

    def test_raw_radar_hazard_brakes_without_a_camera_box(self):
        controller = VehicleController(dt=0.05)
        state = {
            "location": location(0.0),
            "yaw": 0.0,
            "speed_mps": 8.0,
            "speed_kmh": 28.8,
            "reference_path": [location(x) for x in range(81)],
            "lane_width": 3.5,
            "road_id": 1,
            "lane_id": -1,
            "is_junction": False,
        }
        radar_hazard = {
            "track_id": -2,
            "distance_m": 6.0,
            "relative_speed_mps": -8.0,
            "source": "radar_emergency",
            "bearing_deg": 0.0,
            "measurement_frame_id": 10,
        }

        first_control, first_info = controller.run_step(
            state,
            lead_vehicle=None,
            emergency_obstacle=radar_hazard,
        )
        radar_hazard["measurement_frame_id"] = 11
        control, info = controller.run_step(
            state,
            lead_vehicle=None,
            emergency_obstacle=radar_hazard,
        )

        self.assertNotEqual(first_info["mode"], "EMERGENCY")
        self.assertLess(first_control.brake, 1.0)
        self.assertEqual(info["mode"], "EMERGENCY")
        self.assertEqual(control.throttle, 0.0)
        self.assertEqual(control.brake, 1.0)
        self.assertEqual(info["emergency_obstacle"], radar_hazard)

    def test_logged_close_approach_is_not_mistaken_for_pulling_away(self):
        controller = VehicleController(dt=0.05)
        state = {
            "location": location(0.0),
            "yaw": 0.0,
            "speed_mps": 7.2,
            "speed_kmh": 25.92,
            "reference_path": [location(x) for x in range(81)],
            "lane_width": 3.5,
            "road_id": 1,
            "lane_id": -1,
            "is_junction": False,
        }
        lead = {
            "track_id": 18,
            "distance_m": 4.5,
            "relative_speed_mps": -7.2,
            "source": "camera_radar_track",
        }

        control, info = controller.run_step(state, lead)

        self.assertEqual(info["mode"], "EMERGENCY")
        self.assertLess(info["safety"]["ttc_s"], 0.8)
        self.assertEqual(control.throttle, 0.0)
        self.assertEqual(control.brake, 1.0)

    def test_harmless_raw_point_does_not_hide_a_dangerous_tracked_lead(self):
        controller = VehicleController(dt=0.05)
        state = {
            "location": location(0.0),
            "yaw": 0.0,
            "speed_mps": 8.0,
            "speed_kmh": 28.8,
            "reference_path": [location(x) for x in range(81)],
            "lane_width": 3.5,
            "road_id": 1,
            "lane_id": -1,
            "is_junction": False,
        }
        lead = {
            "track_id": 4,
            "distance_m": 5.0,
            "relative_speed_mps": -10.0,
            "source": "camera_radar_track",
        }
        harmless_raw_point = {
            "track_id": -2,
            "distance_m": 7.0,
            "relative_speed_mps": 0.0,
            "source": "radar_emergency",
        }

        control, info = controller.run_step(
            state,
            lead_vehicle=lead,
            emergency_obstacle=harmless_raw_point,
        )

        self.assertEqual(info["mode"], "EMERGENCY")
        self.assertEqual(control.brake, 1.0)
        self.assertEqual(info["safety"]["target_source"], "camera_radar_track")

    def test_harmless_raw_radar_point_does_not_change_normal_following_gap(self):
        controller = VehicleController(dt=0.05)
        state = {
            "location": location(0.0),
            "yaw": 0.0,
            "speed_mps": 2.0,
            "speed_kmh": 7.2,
            "reference_path": [location(x) for x in range(81)],
            "lane_width": 3.5,
            "road_id": 1,
            "lane_id": -1,
            "is_junction": False,
        }
        tracked_lead = {
            "track_id": 4,
            "distance_m": 7.1,
            "relative_speed_mps": -0.9,
            "source": "camera_radar_track",
        }
        near_radar = {
            "track_id": -2,
            "distance_m": 3.9,
            "relative_speed_mps": -0.8,
            "source": "radar_emergency",
        }

        _, info = controller.run_step(
            state,
            lead_vehicle=tracked_lead,
            emergency_obstacle=near_radar,
        )

        self.assertEqual(info["longitudinal_lead"], tracked_lead)
        self.assertAlmostEqual(
            info["longitudinal"]["raw_lead_distance_m"],
            tracked_lead["distance_m"],
        )
        self.assertNotEqual(info["mode"], "EMERGENCY")


if __name__ == "__main__":
    unittest.main()

import sys
import types
import unittest

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
        }

        control, info = controller.run_step(
            state,
            lead_vehicle=None,
            emergency_obstacle=radar_hazard,
        )

        self.assertEqual(info["mode"], "EMERGENCY")
        self.assertEqual(control.throttle, 0.0)
        self.assertEqual(control.brake, 1.0)
        self.assertEqual(info["emergency_obstacle"], radar_hazard)

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


if __name__ == "__main__":
    unittest.main()

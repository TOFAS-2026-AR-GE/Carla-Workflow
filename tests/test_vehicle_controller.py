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


if __name__ == "__main__":
    unittest.main()

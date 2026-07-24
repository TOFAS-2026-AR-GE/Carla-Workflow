"""Oracle doğrulamasının kontrol durumunu değiştirmediğini çalıştırarak sınar."""

import unittest
from types import SimpleNamespace

from carla_app.localization.system import LocalizationSystem
from carla_app.validation.oracle import OracleValidator


class FakeVehicle:
    def __init__(self):
        self.bounding_box = SimpleNamespace(
            extent=SimpleNamespace(y=0.95),
        )

    def get_transform(self):
        return SimpleNamespace(
            location=SimpleNamespace(x=10.0, y=5.0, z=0.5),
            rotation=SimpleNamespace(yaw=15.0),
        )

    def get_velocity(self):
        return SimpleNamespace(x=2.0, y=0.0, z=0.0)

    def is_at_traffic_light(self):
        return False


class FakeRouteManager:
    def update(self, location):
        waypoint = SimpleNamespace(
            road_id=1,
            lane_id=-1,
            lane_width=3.5,
            is_junction=False,
        )
        next_location = SimpleNamespace(
            x=float(location.x) + 1.0,
            y=float(location.y),
            z=float(location.z),
        )
        return waypoint, [next_location]


class OracleIsolationTests(unittest.TestCase):
    def test_oracle_observation_does_not_mutate_localized_state(self):
        state = {
            "location": SimpleNamespace(x=9.5, y=5.0, z=0.5),
            "yaw": 14.0,
            "speed_mps": 1.8,
        }
        original_location = (
            state["location"].x,
            state["location"].y,
            state["location"].z,
        )
        original_keys = set(state)

        result = OracleValidator(enabled=True).observe(
            FakeVehicle(),
            state,
            {"lead_traffic_light": None},
        )

        self.assertFalse(result["control_connected"])
        self.assertEqual(set(state), original_keys)
        self.assertEqual(
            (state["location"].x, state["location"].y, state["location"].z),
            original_location,
        )
        self.assertNotIn("simulator_traffic_light", state)

    def test_localization_builds_state_without_simulator_light(self):
        system = LocalizationSystem.__new__(LocalizationSystem)
        system.last_result = {
            "available": True,
            "x_m": 10.0,
            "y_m": 5.0,
            "z_m": 0.5,
            "yaw_deg": 15.0,
            "speed_mps": 2.0,
            "status": "NOMINAL",
        }

        state = system.build_vehicle_state(
            frame_id=12,
            route_manager=FakeRouteManager(),
            vehicle=FakeVehicle(),
        )

        self.assertNotIn("simulator_traffic_light", state)
        self.assertEqual(state["localization"]["status"], "NOMINAL")


if __name__ == "__main__":
    unittest.main()

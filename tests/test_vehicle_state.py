"""CARLA araç durumundan okunan yardımcı alanların küçük birim testleri."""

import sys
import types
import unittest

if "carla" not in sys.modules:
    carla = types.ModuleType("carla")
    sys.modules["carla"] = carla

from carla_app.core.state import read_simulator_traffic_light


class FakeVehicle:
    def __init__(self, affected, color="Green", traffic_light=None):
        self.affected = affected
        self.color = types.SimpleNamespace(name=color)
        self.traffic_light = traffic_light

    def is_at_traffic_light(self):
        return self.affected

    def get_traffic_light_state(self):
        return self.color

    def get_traffic_light(self):
        return self.traffic_light

    def get_transform(self):
        return types.SimpleNamespace(
            location=types.SimpleNamespace(x=10.0, y=4.0, z=0.0),
            rotation=types.SimpleNamespace(yaw=0.0),
        )


class FakeTrafficLight:
    id = 73

    def get_stop_waypoints(self):
        return [
            types.SimpleNamespace(
                transform=types.SimpleNamespace(
                    location=types.SimpleNamespace(x=22.0, y=4.5, z=0.0)
                ),
                lane_width=3.5,
            ),
            # Aynı aktörün komşu şeritteki waypoint'i seçilmemeli.
            types.SimpleNamespace(
                transform=types.SimpleNamespace(
                    location=types.SimpleNamespace(x=18.0, y=10.0, z=0.0)
                ),
                lane_width=3.5,
            ),
        ]


class SimulatorTrafficLightStateTests(unittest.TestCase):
    def test_green_is_ignored_when_no_light_affects_vehicle(self):
        result = read_simulator_traffic_light(FakeVehicle(False, "Green"))

        self.assertTrue(result["available"])
        self.assertFalse(result["affected"])
        self.assertIsNone(result["color"])

    def test_affected_yellow_is_normalized_to_orange(self):
        result = read_simulator_traffic_light(FakeVehicle(True, "Yellow"))

        self.assertTrue(result["affected"])
        self.assertEqual(result["color"], "orange")

    def test_affected_light_includes_nearest_ego_lane_stop_distance(self):
        result = read_simulator_traffic_light(
            FakeVehicle(True, "Red", FakeTrafficLight())
        )

        self.assertEqual(result["actor_id"], 73)
        self.assertAlmostEqual(result["estimated_distance_m"], 12.0)


if __name__ == "__main__":
    unittest.main()

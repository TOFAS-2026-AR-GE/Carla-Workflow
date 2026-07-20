"""CARLA araç durumundan okunan yardımcı alanların küçük birim testleri."""

import sys
import types
import unittest


if "carla" not in sys.modules:
    carla = types.ModuleType("carla")
    sys.modules["carla"] = carla

from carla_app.core.state import read_simulator_traffic_light


class FakeVehicle:
    def __init__(self, affected, color="Green"):
        self.affected = affected
        self.color = types.SimpleNamespace(name=color)

    def is_at_traffic_light(self):
        return self.affected

    def get_traffic_light_state(self):
        return self.color


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


if __name__ == "__main__":
    unittest.main()

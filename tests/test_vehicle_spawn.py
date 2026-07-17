import types
import unittest
from unittest.mock import patch

from carla_app.core.vehicle import spawn_ego_vehicle


class FakeBlueprint:
    def __init__(self):
        self.attributes = {}

    def has_attribute(self, name):
        return name == "role_name"

    def set_attribute(self, name, value):
        self.attributes[name] = value


class EgoVehicleSpawnTests(unittest.TestCase):
    def test_ros_role_name_is_set_before_vehicle_is_spawned(self):
        blueprint = FakeBlueprint()
        vehicle = types.SimpleNamespace(type_id="vehicle.test")
        world = types.SimpleNamespace(
            get_blueprint_library=lambda: types.SimpleNamespace(
                find=lambda _: blueprint
            ),
            get_map=lambda: types.SimpleNamespace(
                get_spawn_points=lambda: [object()]
            ),
            try_spawn_actor=lambda used_blueprint, _: (
                vehicle
                if used_blueprint.attributes.get("role_name") == "ego_vehicle"
                else None
            ),
        )

        with patch("carla_app.core.vehicle.random.shuffle"):
            spawned = spawn_ego_vehicle(world, "vehicle.test")

        self.assertIs(spawned, vehicle)
        self.assertEqual(blueprint.attributes["role_name"], "ego_vehicle")


if __name__ == "__main__":
    unittest.main()

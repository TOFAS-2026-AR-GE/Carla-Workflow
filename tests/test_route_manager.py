import sys
import types
import unittest

if "carla" not in sys.modules:
    carla = types.ModuleType("carla")
    carla.LaneType = types.SimpleNamespace(Driving="Driving")
    sys.modules["carla"] = carla
elif not hasattr(sys.modules["carla"], "LaneType"):
    sys.modules["carla"].LaneType = types.SimpleNamespace(Driving="Driving")

from carla_app.core.route_manager import PersistentRouteManager


def location(x, y=0.0):
    return types.SimpleNamespace(x=float(x), y=float(y), z=0.0)


class FakeWaypoint:
    def __init__(self, x, y=0.0, limit=100):
        self.x = float(x)
        self.y = float(y)
        self.limit = limit
        self.road_id = 1
        self.lane_id = -1
        self.is_junction = False
        self.transform = types.SimpleNamespace(
            location=location(x, y),
            rotation=types.SimpleNamespace(yaw=0.0),
        )

    def next(self, spacing):
        if self.x + spacing > self.limit:
            return []
        return [FakeWaypoint(self.x + spacing, self.y, self.limit)]


class FakeMap:
    def __init__(self):
        self.calls = []

    def get_waypoint(self, vehicle_location, **_arguments):
        self.calls.append((vehicle_location.x, vehicle_location.y))
        return FakeWaypoint(vehicle_location.x, vehicle_location.y)


class PersistentRouteManagerTests(unittest.TestCase):
    def test_crossing_one_lane_does_not_replace_the_reference_route(self):
        carla_map = FakeMap()
        manager = PersistentRouteManager(
            carla_map,
            spacing_m=1.0,
            horizon_m=30.0,
            recovery_distance_m=8.0,
            recovery_ticks=3,
        )

        manager.update(location(0.0, 0.0))
        _, reference_path = manager.update(location(1.0, 3.7))

        self.assertEqual(len(carla_map.calls), 1)
        self.assertTrue(all(abs(point.y) < 1e-9 for point in reference_path))

    def test_route_is_reinitialized_only_after_sustained_large_deviation(self):
        carla_map = FakeMap()
        manager = PersistentRouteManager(
            carla_map,
            spacing_m=1.0,
            horizon_m=30.0,
            recovery_distance_m=8.0,
            recovery_ticks=3,
        )

        manager.update(location(0.0, 0.0))
        manager.update(location(1.0, 9.0))
        manager.update(location(2.0, 9.0))
        self.assertEqual(len(carla_map.calls), 1)

        manager.update(location(3.0, 9.0))
        self.assertEqual(len(carla_map.calls), 2)


if __name__ == "__main__":
    unittest.main()

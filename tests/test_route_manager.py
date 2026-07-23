import sys
import types
import unittest

if "carla" not in sys.modules:
    carla = types.ModuleType("carla")
    carla.LaneType = types.SimpleNamespace(Driving="Driving")
    sys.modules["carla"] = carla
elif not hasattr(sys.modules["carla"], "LaneType"):
    sys.modules["carla"].LaneType = types.SimpleNamespace(Driving="Driving")

if not hasattr(sys.modules["carla"], "Location"):
    sys.modules["carla"].Location = lambda x=0.0, y=0.0, z=0.0: (
        types.SimpleNamespace(x=float(x), y=float(y), z=float(z))
    )

from carla_app.core.route_manager import PersistentRouteManager
from carla_app.navigation.system import NavigationSystem


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
    def test_planned_route_is_not_extended_into_an_arbitrary_turn(self):
        manager = PersistentRouteManager(FakeMap(), horizon_m=30.0)
        planned = [FakeWaypoint(float(x), limit=100) for x in range(6)]

        manager.set_planned_route(planned)
        _waypoint, reference_path = manager.update(location(1.0))

        self.assertTrue(manager.planned_route_active)
        self.assertLessEqual(reference_path[-1].x, 5.0)

    def test_remaining_distance_uses_planned_route_length(self):
        manager = PersistentRouteManager(FakeMap(), horizon_m=30.0)
        manager.set_planned_route(
            [FakeWaypoint(float(x), limit=100) for x in range(11)]
        )

        remaining = manager.remaining_distance(location(4.0))

        self.assertAlmostEqual(remaining, 6.0, places=6)

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


class NavigationSystemTests(unittest.TestCase):
    def test_confirmed_destination_enables_motion_toward_selected_point(self):
        class FakeRouteManager:
            def __init__(self):
                self.route = None

            def set_planned_route(self, waypoints):
                self.route = list(waypoints)

            @staticmethod
            def remaining_distance(_vehicle_location):
                return 40.0

        class FakePlanner:
            @staticmethod
            def trace_route(start, destination):
                return [
                    FakeWaypoint(start.x, start.y),
                    FakeWaypoint(destination.x, destination.y),
                ]

        route_manager = FakeRouteManager()
        navigation = NavigationSystem(FakeMap(), route_manager)
        navigation.planner = FakePlanner()
        vehicle_location = location(1.0, 2.0)

        self.assertTrue(navigation.select_destination(25.0, -4.0))
        self.assertEqual(navigation.status, "PENDING")
        self.assertIsNone(route_manager.route)

        self.assertTrue(navigation.confirm_destination(vehicle_location))
        state = navigation.update(vehicle_location, speed_mps=0.0)

        self.assertEqual(state["status"], "DRIVING")
        self.assertTrue(state["drive_enabled"])
        self.assertGreater(state["target_speed_mps"], 0.0)
        self.assertIsNotNone(route_manager.route)


if __name__ == "__main__":
    unittest.main()

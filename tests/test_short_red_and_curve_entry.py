import math
from types import SimpleNamespace

from carla_app.controller.vehicle.speed_planner import CurvatureSpeedPlanner
from carla_app.core.scenario import Scenario
from carla_app.core.traffic import Traffic


class FakeLight:
    def __init__(self):
        self.red = None
        self.yellow = None
        self.green = None

    def set_red_time(self, value):
        self.red = value

    def set_yellow_time(self, value):
        self.yellow = value

    def set_green_time(self, value):
        self.green = value


class FakeActors:
    def __init__(self, lights):
        self.lights = lights

    def filter(self, pattern):
        assert pattern == "traffic.traffic_light*"
        return self.lights


class FakeWorld:
    def __init__(self, lights):
        self.lights = lights

    def get_actors(self):
        return FakeActors(self.lights)


def scenario():
    return SimpleNamespace(
        npc_count=0,
        traffic_light_red_time_s=4.0,
        traffic_light_yellow_time_s=1.5,
        traffic_light_green_time_s=8.0,
    )


def point(x, y):
    return SimpleNamespace(x=float(x), y=float(y))


def straight_then_curve(straight_m=45.0, radius_m=14.0):
    path = [point(index, 0.0) for index in range(int(straight_m) + 1)]
    for index in range(1, 55):
        angle = index / radius_m
        path.append(
            point(
                straight_m + radius_m * math.sin(angle),
                radius_m * (1.0 - math.cos(angle)),
            )
        )
    return path


def test_short_traffic_light_cycle_is_applied_to_all_lights():
    lights = [FakeLight(), FakeLight(), FakeLight()]
    traffic = Traffic(None, FakeWorld(lights), scenario())

    traffic.configure_traffic_lights()

    for light in lights:
        assert light.red == 4.0
        assert light.yellow == 1.5
        assert light.green == 8.0


def test_curve_is_seen_before_vehicle_reaches_entry():
    planner = CurvatureSpeedPlanner(dt=0.05, cruise_speed_kmh=60.0)
    planner.reset(60.0 / 3.6)
    state = {
        "speed_mps": 60.0 / 3.6,
        "location": point(0.0, 0.0),
        "reference_path": straight_then_curve(),
        "lane_width": 3.5,
        "vehicle_half_width_m": 0.95,
    }

    target, info = planner.run_step(
        state,
        lateral_info={
            "cross_track_error_m": 0.0,
            "heading_error_rad": 0.0,
        },
        requested_speed_mps=60.0 / 3.6,
    )

    assert info["curvature_1pm"] > 0.02
    assert info["curve_speed_mps"] < 30.0 / 3.6
    assert target < 60.0 / 3.6


def test_comfort_curve_limit_is_lower_than_old_rally_setting():
    planner = CurvatureSpeedPlanner(dt=0.05, cruise_speed_kmh=60.0)
    curve_speed = planner.calculate_curve_speed(1.0 / 20.0)

    assert curve_speed * 3.6 < 18.0
    assert planner.maximum_target_deceleration_mps2 <= 1.9

import math

from carla_app.controller.vehicle.speed_planner import CurvatureSpeedPlanner


class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)


def straight_path(length=80):
    return [Point(index, 0.0) for index in range(length)]


def circular_path(radius=12.0, count=80):
    return [
        Point(
            radius * math.sin(index / radius),
            radius * (1.0 - math.cos(index / radius)),
        )
        for index in range(count)
    ]


def test_target_reference_is_rate_limited():
    planner = CurvatureSpeedPlanner(dt=0.05, cruise_speed_kmh=60)
    state = {"speed_mps": 0.0, "reference_path": straight_path()}
    target, _ = planner.run_step(
        state,
        lateral_info={"cross_track_error_m": 0.0, "heading_error_rad": 0.0},
        requested_speed_mps=15.0,
    )
    assert 0.0 < target <= planner.maximum_target_acceleration_mps2 * planner.dt + 1e-9


def test_curve_reduces_speed():
    planner = CurvatureSpeedPlanner(dt=0.05, cruise_speed_kmh=60)
    planner.reset(15.0)
    state = {"speed_mps": 15.0, "reference_path": circular_path(radius=12.0)}
    target, info = planner.run_step(
        state,
        lateral_info={"cross_track_error_m": 0.0, "heading_error_rad": 0.0},
        requested_speed_mps=15.0,
    )
    assert info["curvature_1pm"] > 0.04
    assert info["curve_speed_mps"] < 8.0
    assert info["speed_reason"] == "curve"
    assert target < 15.0


def test_lane_recovery_is_continuous_and_confirmed():
    planner = CurvatureSpeedPlanner(dt=0.05, cruise_speed_kmh=60)
    planner.reset(12.0)
    state = {
        "speed_mps": 12.0,
        "reference_path": straight_path(),
        "lane_width": 3.5,
        "vehicle_half_width_m": 0.95,
    }
    info = None
    for _ in range(planner.recovery_confirmation_ticks):
        _, info = planner.run_step(
            state,
            lateral_info={"cross_track_error_m": 0.62, "heading_error_rad": 0.15},
            requested_speed_mps=12.0,
        )
    assert info["recovery_speed_mps"] is not None
    assert 8.0 / 3.6 < info["recovery_speed_mps"] < 12.0

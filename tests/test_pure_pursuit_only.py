import math
from types import SimpleNamespace

from carla_app.controller.vehicle.pure_pursuit_controller import (
    PurePursuitController,
)


def point(x, y=0.0):
    return SimpleNamespace(x=float(x), y=float(y), z=0.0)


def state(path, x=0.0, y=0.0, yaw=0.0, speed=8.0):
    return {
        "location": point(x, y),
        "yaw": float(yaw),
        "speed_mps": float(speed),
        "reference_path": path,
        "lane_width": 3.5,
        "vehicle_half_width_m": 0.95,
    }


def test_straight_center_has_nearly_zero_steering():
    controller = PurePursuitController(dt=0.05)
    path = [point(x) for x in range(80)]

    for _ in range(20):
        steer = controller.run_step(state(path))

    assert abs(steer) < 0.01
    assert controller.last_info["controller"] == "pure_pursuit"


def test_vehicle_right_of_center_steers_left():
    controller = PurePursuitController(dt=0.05)
    path = [point(x) for x in range(80)]

    steer = controller.run_step(state(path, y=0.8))

    assert steer < 0.0


def test_left_curve_steers_left():
    controller = PurePursuitController(dt=0.05)
    radius = 25.0
    path = [
        point(
            radius * math.sin(index / radius),
            radius * (1.0 - math.cos(index / radius)),
        )
        for index in range(80)
    ]

    steer = controller.run_step(state(path, speed=7.0))

    assert steer > 0.0
    assert controller.last_info["path_turn_rad"] > 0.0


def test_lookahead_is_longer_at_high_speed():
    controller = PurePursuitController(dt=0.05)

    slow = controller.calculate_lookahead(3.0, 0.0)
    fast = controller.calculate_lookahead(15.0, 0.0)

    assert fast > slow


def test_steering_change_is_rate_limited():
    controller = PurePursuitController(dt=0.05)
    left_path = [point(x, 0.12 * x) for x in range(80)]
    right_path = [point(x, -0.12 * x) for x in range(80)]

    first = controller.run_step(state(left_path))
    second = controller.run_step(state(right_path))

    assert abs(second - first) <= controller.maximum_steer_rate_per_s * 0.05 + 1e-9

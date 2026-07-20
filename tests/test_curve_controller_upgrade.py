import math
from types import SimpleNamespace

from carla_app.controller.vehicle.speed_planner import CurvatureSpeedPlanner
from carla_app.controller.vehicle.stanley_controller import StanleyController


def point(x, y):
    return SimpleNamespace(x=float(x), y=float(y))


def quarter_turn(radius=8.0, left=False):
    path = [point(x * 0.5, 0.0) for x in range(40)]
    center_y = -radius if left else radius
    start = math.pi / 2.0 if left else -math.pi / 2.0
    stop = 0.0
    for index in range(41):
        fraction = index / 40.0
        angle = start + fraction * (stop - start)
        path.append(
            point(
                20.0 + radius * math.cos(angle),
                center_y + radius * math.sin(angle),
            )
        )
    return path


def state(path, x=19.5, y=0.0, yaw=0.0, speed=3.3):
    return {
        "reference_path": path,
        "location": point(x, y),
        "yaw": yaw,
        "speed_mps": speed,
        "lane_width": 3.5,
        "vehicle_half_width_m": 0.95,
    }


def test_hybrid_controller_turns_towards_right_curve():
    controller = StanleyController(dt=0.05)
    path = quarter_turn(radius=8.0, left=False)
    output = 0.0
    for _ in range(8):
        output = controller.run_step(state(path))
    assert output > 0.05
    assert controller.last_info["lookahead_m"] >= 2.2


def test_hybrid_controller_turns_towards_left_curve():
    controller = StanleyController(dt=0.05)
    path = quarter_turn(radius=8.0, left=True)
    output = 0.0
    for _ in range(8):
        output = controller.run_step(state(path))
    assert output < -0.05


def test_curve_planner_slows_but_never_requests_zero():
    planner = CurvatureSpeedPlanner(dt=0.05, cruise_speed_kmh=60.0)
    path = quarter_turn(radius=8.0)
    target, info = planner.run_step(
        state(path, x=15.0, speed=8.0),
        lateral_info={
            "cross_track_error_m": 0.0,
            "heading_error_rad": 0.0,
        },
        requested_speed_mps=60.0 / 3.6,
    )
    assert info["curve_speed_mps"] < 60.0 / 3.6
    assert info["desired_speed_mps"] >= 8.0 / 3.6
    assert target > 0.0


def test_normal_curve_heading_error_does_not_force_stop():
    planner = CurvatureSpeedPlanner(dt=0.05, cruise_speed_kmh=60.0)
    path = quarter_turn(radius=8.0)
    requested, risk = planner._calculate_lane_recovery_speed(
        state=state(path),
        lateral_info={
            "cross_track_error_m": 0.15,
            "heading_error_rad": math.radians(20.0),
        },
        requested_speed_mps=30.0 / 3.6,
        curvature=1.0 / 8.0,
    )
    assert risk < 0.50
    assert requested is None

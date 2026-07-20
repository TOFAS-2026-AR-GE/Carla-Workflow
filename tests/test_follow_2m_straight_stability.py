import math

from carla_app.controller.vehicle.longitudinal_controller import LongitudinalController
from carla_app.controller.vehicle.stanley_controller import StanleyController


class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)


def state(speed_mps=8.0, y=0.0, yaw=0.0, path=None):
    return {
        "speed_mps": float(speed_mps),
        "location": Point(0.0, y),
        "yaw": float(yaw),
        "lane_width": 3.5,
        "vehicle_half_width_m": 0.95,
        "reference_path": path or [Point(i, 0.0) for i in range(80)],
    }


def test_stable_vehicle_gap_is_two_metres():
    controller = LongitudinalController(dt=0.05)
    gap = controller.calculate_desired_gap(
        ego_speed=30.0 / 3.6,
        relative_speed=0.0,
        source="camera_radar_track",
    )
    assert abs(gap - 2.0) < 1e-9


def test_closing_vehicle_keeps_dynamic_safety_term():
    controller = LongitudinalController(dt=0.05)
    gap = controller.calculate_desired_gap(
        ego_speed=30.0 / 3.6,
        relative_speed=-4.0,
        source="camera_radar_track",
    )
    assert gap > 2.0


def test_small_straight_noise_does_not_create_steering_hunt():
    controller = StanleyController(dt=0.05)
    outputs = []
    for tick in range(120):
        noisy_path = [
            Point(i, 0.025 * math.sin(0.55 * i + 0.3 * tick))
            for i in range(80)
        ]
        outputs.append(controller.run_step(state(path=noisy_path)))

    late = outputs[-40:]
    assert max(abs(value) for value in late) < 0.035
    sign_changes = sum(
        1
        for first, second in zip(late, late[1:])
        if first * second < 0.0 and abs(first - second) > 0.006
    )
    assert sign_changes <= 2


def test_real_curve_still_turns():
    controller = StanleyController(dt=0.05)
    radius = 25.0
    path = [
        Point(
            radius * math.sin(i / radius),
            radius * (1.0 - math.cos(i / radius)),
        )
        for i in range(80)
    ]
    outputs = [controller.run_step(state(speed_mps=6.0, path=path)) for _ in range(30)]
    assert max(abs(value) for value in outputs) > 0.08

import math
from types import SimpleNamespace

from carla_app.controller.vehicle.stanley_controller import StanleyController


def point(x, y=0.0):
    return SimpleNamespace(x=float(x), y=float(y), z=0.0)


def state(path, x=0.0, y=0.0, yaw=0.0, speed_mps=8.0):
    return {
        "location": point(x, y),
        "yaw": float(yaw),
        "speed_mps": float(speed_mps),
        "reference_path": path,
        "lane_width": 3.5,
        "vehicle_half_width_m": 0.95,
    }


def curved_path(radius_m=25.0, left=True, count=220):
    direction = 1.0 if left else -1.0
    result = []
    for index in range(count):
        angle = index * 0.01
        result.append(
            point(
                radius_m * math.sin(angle),
                direction * radius_m * (1.0 - math.cos(angle)),
            )
        )
    return result


def test_straight_path_small_noise_does_not_create_wobble():
    controller = StanleyController(dt=0.05)
    path = [
        point(index, 0.025 * math.sin(index * 2.3))
        for index in range(100)
    ]

    outputs = []
    for tick in range(160):
        outputs.append(
            controller.run_step(
                state(
                    path,
                    x=tick * 0.05,
                    y=0.02,
                    yaw=0.20 * math.sin(tick * 0.10),
                    speed_mps=10.0,
                )
            )
        )

    assert max(abs(value) for value in outputs[-80:]) < 0.03
    assert controller.last_info["straight_mode"] is True


def test_left_and_right_curves_have_correct_steering_sign():
    left = StanleyController(dt=0.05)
    right = StanleyController(dt=0.05)

    left_steer = 0.0
    right_steer = 0.0
    for _ in range(20):
        left_steer = left.run_step(state(curved_path(left=True), speed_mps=7.0))
        right_steer = right.run_step(state(curved_path(left=False), speed_mps=7.0))

    assert left_steer > 0.15
    assert right_steer < -0.15


def test_steering_change_is_smooth():
    controller = StanleyController(dt=0.05)
    path = [point(index, 0.0) for index in range(100)]

    outputs = []
    for y in [1.0] * 20 + [-1.0] * 20:
        outputs.append(
            controller.run_step(
                state(path, y=y, speed_mps=10.0)
            )
        )

    changes = [
        abs(current - previous)
        for previous, current in zip(outputs, outputs[1:])
    ]
    assert max(changes) <= 0.065


def test_closed_loop_curve_tracking_remains_near_lane_center():
    controller = StanleyController(dt=0.05)
    path = curved_path(radius_m=25.0, left=True, count=250)

    x = 0.0
    y = 0.0
    yaw = 0.0
    speed = 6.0
    errors = []

    for _ in range(180):
        nearest_index = min(
            range(len(path)),
            key=lambda index: (
                (path[index].x - x) ** 2
                + (path[index].y - y) ** 2
            ),
        )
        reference_path = path[
            max(0, nearest_index - 2):
            min(len(path), nearest_index + 90)
        ]
        steer = controller.run_step(
            state(
                reference_path,
                x=x,
                y=y,
                yaw=math.degrees(yaw),
                speed_mps=speed,
            )
        )
        wheel_angle = steer * controller.maximum_wheel_angle_rad
        x += speed * math.cos(yaw) * controller.dt
        y += speed * math.sin(yaw) * controller.dt
        yaw += (
            speed
            / controller.wheelbase_m
            * math.tan(wheel_angle)
            * controller.dt
        )
        errors.append(abs(controller.last_info["cross_track_error_m"]))

    mean_recent_error = sum(errors[-80:]) / 80
    assert mean_recent_error < 0.18


def test_lane_edge_correction_still_increases():
    controller = StanleyController(dt=0.05)
    path = [point(index, 0.0) for index in range(100)]
    current_state = state(path)
    corrected = controller.calculate_lane_edge_error(0.70, current_state)
    assert corrected > 0.70

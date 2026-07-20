"""Pure Pursuit kontrolcüsünü üç kinematik parkurda ölçer."""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_app.controller.vehicle.pure_pursuit_controller import (  # noqa: E402
    PurePursuitController,
)

DT = 0.05


def point(x, y):
    return SimpleNamespace(x=float(x), y=float(y))


def nearest_index(path, x, y):
    return min(
        range(len(path)),
        key=lambda index: (path[index].x - x) ** 2 + (path[index].y - y) ** 2,
    )


def distance_to_path(path, x, y, near_index):
    """Aracın örnek noktaya değil rota çizgisine uzaklığını verir."""
    best_squared = math.inf
    start_index = max(0, near_index - 3)
    end_index = min(len(path) - 1, near_index + 3)
    for index in range(start_index, end_index):
        start = path[index]
        end = path[index + 1]
        segment_x = end.x - start.x
        segment_y = end.y - start.y
        length_squared = segment_x**2 + segment_y**2
        if length_squared <= 1e-12:
            continue
        ratio = (
            (x - start.x) * segment_x + (y - start.y) * segment_y
        ) / length_squared
        ratio = max(0.0, min(1.0, ratio))
        error_x = x - (start.x + ratio * segment_x)
        error_y = y - (start.y + ratio * segment_y)
        best_squared = min(best_squared, error_x**2 + error_y**2)
    return math.sqrt(best_squared)


def simulate(path, speed_mps, initial_y=0.0, steps=200):
    controller = PurePursuitController(DT)
    x = float(path[0].x)
    y = float(initial_y)
    yaw = 0.0
    errors = []
    steering = []

    for _ in range(steps):
        index = nearest_index(path, x, y)
        state = {
            "location": point(x, y),
            "yaw": math.degrees(yaw),
            "speed_mps": speed_mps,
            "reference_path": path[max(0, index - 3) : index + 140],
        }
        steer = controller.run_step(state)
        wheel_angle = steer * controller.maximum_wheel_angle_rad
        x += speed_mps * math.cos(yaw) * DT
        y += speed_mps * math.sin(yaw) * DT
        yaw += speed_mps / controller.wheelbase_m * math.tan(wheel_angle) * DT

        index = nearest_index(path, x, y)
        errors.append(distance_to_path(path, x, y, index))
        steering.append(steer)

    evaluated_errors = errors[20:]
    evaluated_steering = steering[20:]
    return {
        "controller": "pure_pursuit",
        "mean_error_m": round(
            sum(evaluated_errors) / len(evaluated_errors),
            4,
        ),
        "maximum_error_m": round(max(evaluated_errors), 4),
        "maximum_steer": round(
            max(abs(value) for value in evaluated_steering),
            4,
        ),
    }


def benchmark():
    straight = [point(index * 0.5, 0.0) for index in range(501)]
    noisy_straight = [
        point(index * 0.5, 0.01 if index % 2 else -0.01)
        for index in range(501)
    ]
    radius_m = 30.0
    curve = [
        point(
            radius_m * math.sin(index * 0.25 / radius_m),
            radius_m * (1.0 - math.cos(index * 0.25 / radius_m)),
        )
        for index in range(1001)
    ]
    return {
        "offset_recovery": simulate(straight, 10.0, initial_y=0.8),
        "waypoint_noise_70_kmh": simulate(noisy_straight, 70.0 / 3.6),
        "curve_23_kmh": simulate(curve, 23.0 / 3.6),
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2, sort_keys=True))

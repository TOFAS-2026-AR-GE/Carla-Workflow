"""MPC ve Stanley'yi aynı kinematik parkurlarda karşılaştırır."""

import json
import math
from types import SimpleNamespace

from carla_app.config import DrivingParameters
from carla_app.controller.vehicle.mpc_controller import LateralMPCController
from carla_app.controller.vehicle.stanley_controller import StanleyController

DT = 0.05


def point(x, y):
    return SimpleNamespace(x=float(x), y=float(y))


def nearest_index(path, x, y):
    return min(
        range(len(path)),
        key=lambda index: (path[index].x - x) ** 2 + (path[index].y - y) ** 2,
    )


def distance_to_path(path, x, y, near_index):
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
        fraction = max(
            0.0,
            min(
                1.0,
                ((x - start.x) * segment_x + (y - start.y) * segment_y)
                / length_squared,
            ),
        )
        error_x = x - (start.x + fraction * segment_x)
        error_y = y - (start.y + fraction * segment_y)
        best_squared = min(best_squared, error_x**2 + error_y**2)
    return math.sqrt(best_squared)


def count_reversals(values, threshold=0.015):
    signs = [1 if value > 0.0 else -1 for value in values if abs(value) >= threshold]
    return sum(current != previous for previous, current in zip(signs, signs[1:]))


def settling_time(errors, tolerance_m=0.10):
    for index in range(len(errors)):
        if max(errors[index:]) <= tolerance_m:
            return round(index * DT, 3)
    return None


def simulate(controller, path, speed_mps, initial_y=0.0, steps=200):
    x = path[0].x
    y = float(initial_y)
    yaw = 0.0
    errors = []
    steering = []
    active_controllers = set()

    for _ in range(steps):
        path_index = nearest_index(path, x, y)
        reference_path = path[
            max(0, path_index - 3) : min(len(path), path_index + 120)
        ]
        state = {
            "location": point(x, y),
            "yaw": math.degrees(yaw),
            "speed_mps": speed_mps,
            "reference_path": reference_path,
            "lane_width": 3.5,
            "vehicle_half_width_m": 0.95,
        }
        steer = controller.run_step(state)
        active_controllers.add(controller.last_info.get("controller", "stanley"))
        wheel_angle = steer * controller.maximum_wheel_angle_rad
        x += speed_mps * math.cos(yaw) * DT
        y += speed_mps * math.sin(yaw) * DT
        yaw += speed_mps / controller.wheelbase_m * math.tan(wheel_angle) * DT

        current_index = nearest_index(path, x, y)
        errors.append(distance_to_path(path, x, y, current_index))
        steering.append(steer)

    evaluation_errors = errors[20:]
    evaluation_steering = steering[20:]
    return {
        "mean_cte_m": round(sum(evaluation_errors) / len(evaluation_errors), 4),
        "peak_cte_m": round(max(evaluation_errors), 4),
        "final_cte_m": round(errors[-1], 4),
        "peak_steer": round(max(abs(value) for value in evaluation_steering), 4),
        "steering_reversals": count_reversals(evaluation_steering),
        "settling_time_s": settling_time(errors),
        "active_controllers": sorted(active_controllers),
    }


def controllers():
    parameters = DrivingParameters(DT)
    parameters.mpc_time_budget_ms = 1000.0
    return {
        "mpc": LateralMPCController(DT, parameters),
        "stanley": StanleyController(DT),
    }


def benchmark():
    straight_noise = [
        point(index * 0.5, 0.01 if index % 2 else -0.01)
        for index in range(501)
    ]
    straight = [point(index * 0.5, 0.0) for index in range(501)]
    radius_m = 40.0
    curve = [
        point(
            radius_m * math.sin(index * 0.5 / radius_m),
            radius_m * (1.0 - math.cos(index * 0.5 / radius_m)),
        )
        for index in range(501)
    ]
    scenarios = {
        "straight_waypoint_noise": (straight_noise, 13.9, 0.0),
        "offset_recovery": (straight, 10.0, 0.8),
        "constant_radius_curve": (curve, 10.0, 0.0),
    }
    results = {}
    for scenario_name, (path, speed_mps, initial_y) in scenarios.items():
        results[scenario_name] = {
            name: simulate(controller, path, speed_mps, initial_y)
            for name, controller in controllers().items()
        }
    return results


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2, sort_keys=True))

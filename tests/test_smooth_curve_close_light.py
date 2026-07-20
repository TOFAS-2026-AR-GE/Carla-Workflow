from __future__ import annotations

import math
from types import SimpleNamespace

from carla_app.controller.vehicle.longitudinal_controller import LongitudinalController
from carla_app.controller.vehicle.stanley_controller import StanleyController


def point(x, y):
    return SimpleNamespace(x=float(x), y=float(y))


def quarter_turn(radius=9.0):
    points = []
    for index in range(31):
        angle = -0.5 * math.pi + 0.5 * math.pi * index / 30.0
        points.append(
            point(
                radius * math.cos(angle),
                radius + radius * math.sin(angle),
            )
        )
    return points


def steering_state(path):
    return {
        "reference_path": path,
        "location": point(0.0, 0.0),
        "yaw": 0.0,
        "speed_mps": 4.0,
        "lane_width": 3.5,
        "vehicle_half_width_m": 0.95,
    }


def simulate_red_stop(initial_speed_kmh, initial_distance_m):
    controller = LongitudinalController(dt=0.05)
    speed = initial_speed_kmh / 3.6
    distance = float(initial_distance_m)

    for _ in range(1200):
        throttle, brake, info = controller.run_step(
            {"speed_mps": speed},
            {
                "track_id": -100,
                "distance_m": distance,
                "relative_speed_mps": -speed,
                "lead_speed_mps": 0.0,
                "source": "traffic_light_red",
            },
            initial_speed_kmh / 3.6,
        )
        acceleration = 2.8 * throttle - 5.0 * brake
        if speed > 0.02:
            acceleration -= 0.12
        speed = max(0.0, speed + acceleration * controller.dt)
        distance -= speed * controller.dt
        if info["hold_active"] and speed <= 0.02:
            return distance
    raise AssertionError("Vehicle did not reach traffic-light hold.")


def test_steering_command_has_smooth_velocity_and_acceleration():
    controller = StanleyController(dt=0.05)
    outputs = [controller.run_step(steering_state(quarter_turn())) for _ in range(20)]
    changes = [b - a for a, b in zip(outputs, outputs[1:])]
    accelerations = [b - a for a, b in zip(changes, changes[1:])]

    assert max(abs(value) for value in changes) <= 0.055
    assert max(abs(value) for value in accelerations) <= 0.014
    assert abs(outputs[-1]) > 0.10


def test_red_light_stop_is_close_to_one_metre():
    distance = simulate_red_stop(30.0, 30.0)
    assert 0.65 <= distance <= 1.25


def test_early_stationary_stop_creeps_towards_line():
    distance = simulate_red_stop(0.0, 5.8)
    assert 0.65 <= distance <= 1.25

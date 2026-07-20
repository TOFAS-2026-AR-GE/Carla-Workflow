"""Pure Pursuit yanal kontrolünün kapalı çevrim kararlılık testleri."""

import math
import unittest
from types import SimpleNamespace

from carla_app.controller.vehicle.pure_pursuit_controller import (
    PurePursuitController,
)


def point(x, y):
    return SimpleNamespace(x=float(x), y=float(y))


def nearest_index(path, x, y):
    return min(
        range(len(path)),
        key=lambda index: (path[index].x - x) ** 2 + (path[index].y - y) ** 2,
    )


def distance_to_path(path, x, y, near_index):
    """Nokta aralığından etkilenmeyen doğru parçası mesafesini hesaplar."""
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


class LateralStabilityTests(unittest.TestCase):
    def simulate(self, controller, path, speed_mps, initial_y=0.0, steps=180):
        x = float(path[0].x)
        y = float(initial_y)
        yaw = 0.0
        errors = []
        steering = []

        for _ in range(steps):
            nearest = nearest_index(path, x, y)
            reference_path = path[
                max(0, nearest - 3) : min(len(path), nearest + 140)
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
            wheel_angle = steer * controller.maximum_wheel_angle_rad
            x += speed_mps * math.cos(yaw) * controller.dt
            y += speed_mps * math.sin(yaw) * controller.dt
            yaw += (
                speed_mps
                / controller.wheelbase_m
                * math.tan(wheel_angle)
                * controller.dt
            )
            nearest_after_step = nearest_index(path, x, y)
            errors.append(distance_to_path(path, x, y, nearest_after_step))
            steering.append(steer)

        return errors, steering

    def test_straight_offset_settles_without_sustained_slalom(self):
        controller = PurePursuitController(0.05)
        path = [point(index * 0.5, 0.0) for index in range(501)]

        errors, steering = self.simulate(
            controller,
            path,
            speed_mps=10.0,
            initial_y=0.8,
        )

        self.assertLess(max(errors[-60:]), 0.12)
        active_signs = [
            1 if value > 0.0 else -1
            for value in steering
            if abs(value) >= 0.015
        ]
        reversals = sum(
            current != previous
            for previous, current in zip(active_signs, active_signs[1:])
        )
        self.assertLessEqual(reversals, 2)

    def test_waypoint_noise_does_not_create_large_steering(self):
        controller = PurePursuitController(0.05)
        path = [
            point(index * 0.5, 0.01 if index % 2 else -0.01)
            for index in range(501)
        ]

        errors, steering = self.simulate(
            controller,
            path,
            speed_mps=70.0 / 3.6,
            steps=150,
        )

        self.assertLess(max(errors[-50:]), 0.15)
        self.assertLess(max(abs(value) for value in steering[-50:]), 0.03)

    def test_constant_radius_curve_is_tracked_at_curve_floor_speed(self):
        controller = PurePursuitController(0.05)
        radius_m = 25.0
        path = [
            point(
                radius_m * math.sin(index * 0.25 / radius_m),
                radius_m * (1.0 - math.cos(index * 0.25 / radius_m)),
            )
            for index in range(900)
        ]

        errors, _ = self.simulate(
            controller,
            path,
            speed_mps=23.0 / 3.6,
            steps=220,
        )

        self.assertLess(sum(errors[-80:]) / 80, 0.20)


if __name__ == "__main__":
    unittest.main()

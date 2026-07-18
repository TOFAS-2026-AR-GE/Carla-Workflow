"""Yanal kontrolün kapalı çevrim kararlılık regresyonları."""

import math
import unittest
from types import SimpleNamespace

from carla_app.config import DrivingParameters
from carla_app.controller.vehicle.mpc_controller import LateralMPCController


class LateralStabilityTests(unittest.TestCase):
    def test_straight_road_offset_settles_quickly_without_slalom(self):
        dt = 0.05
        parameters = DrivingParameters(dt)
        parameters.mpc_time_budget_ms = 1000.0
        controller = LateralMPCController(dt, parameters)
        path = [
            SimpleNamespace(x=index * 0.5, y=0.0)
            for index in range(401)
        ]
        x = 0.0
        y = 0.8
        yaw = 0.0
        speed_mps = 10.0
        errors = []
        steering = []

        for _ in range(120):
            nearest = min(
                range(len(path)),
                key=lambda index: (path[index].x - x) ** 2 + (path[index].y - y) ** 2,
            )
            state = {
                "location": SimpleNamespace(x=x, y=y),
                "yaw": math.degrees(yaw),
                "speed_mps": speed_mps,
                "reference_path": path[max(0, nearest - 3) : nearest + 120],
                "lane_width": 3.5,
                "vehicle_half_width_m": 0.95,
            }
            steer = controller.run_step(state)
            wheel_angle = steer * controller.maximum_wheel_angle_rad
            x += speed_mps * math.cos(yaw) * dt
            y += speed_mps * math.sin(yaw) * dt
            yaw += speed_mps / controller.wheelbase_m * math.tan(wheel_angle) * dt
            errors.append(abs(y))
            steering.append(steer)

        settling_index = next(
            index
            for index in range(len(errors))
            if max(errors[index:]) <= 0.10
        )
        active_signs = [
            1 if steer > 0.0 else -1
            for steer in steering
            if abs(steer) >= 0.015
        ]
        reversals = sum(
            current != previous
            for previous, current in zip(active_signs, active_signs[1:])
        )

        self.assertLessEqual(settling_index * dt, 0.85 + 1e-9)
        self.assertLessEqual(reversals, 1)

    def test_waypoint_noise_does_not_create_sustained_slalom(self):
        dt = 0.05
        parameters = DrivingParameters(dt)
        parameters.mpc_time_budget_ms = 1000.0
        controller = LateralMPCController(dt, parameters)
        path = [
            SimpleNamespace(
                x=float(index),
                y=0.01 if index % 2 else -0.01,
            )
            for index in range(120)
        ]

        x = 0.0
        y = 0.0
        yaw = 0.0
        speed_mps = 13.9
        steering = []
        lateral_errors = []
        active_controllers = set()

        for _ in range(100):
            state = {
                "location": SimpleNamespace(x=x, y=y),
                "yaw": math.degrees(yaw),
                "speed_mps": speed_mps,
                "reference_path": path,
                "lane_width": 3.5,
                "vehicle_half_width_m": 0.95,
            }
            steer = controller.run_step(state)
            active_controllers.add(controller.last_info["controller"])
            wheel_angle = steer * controller.maximum_wheel_angle_rad
            yaw += (
                speed_mps
                / controller.wheelbase_m
                * math.tan(wheel_angle)
                * dt
            )
            x += speed_mps * math.cos(yaw) * dt
            y += speed_mps * math.sin(yaw) * dt
            steering.append(steer)
            lateral_errors.append(y)

        active_signs = []
        for steer in steering[20:]:
            if abs(steer) >= 0.015:
                active_signs.append(1 if steer > 0.0 else -1)
        reversals = 0
        for previous, current in zip(active_signs, active_signs[1:]):
            if current != previous:
                reversals += 1

        self.assertLessEqual(
            reversals,
            2,
            (
                f"sustained steering reversals={reversals}, "
                f"tail={[round(value, 3) for value in steering[20:]]}"
            ),
        )
        self.assertLess(abs(lateral_errors[-1]), 0.08)
        self.assertLess(max(abs(value) for value in lateral_errors[40:]), 0.20)
        self.assertLess(max(abs(value) for value in steering[40:]), 0.02)
        self.assertEqual(active_controllers, {"mpc"})

    def test_reference_smoothing_preserves_real_curve_radius(self):
        controller = LateralMPCController(0.05, DrivingParameters(0.05))
        radius_m = 40.0
        path = []
        for distance_m in range(80):
            angle = distance_m / radius_m
            path.append(
                SimpleNamespace(
                    x=radius_m * math.sin(angle),
                    y=radius_m * (1.0 - math.cos(angle)),
                )
            )

        smoothed = controller.smooth_reference_path(path)
        curvature = controller.calculate_path_curvature(smoothed, 15)

        self.assertAlmostEqual(curvature, 1.0 / radius_m, delta=0.0025)


if __name__ == "__main__":
    unittest.main()

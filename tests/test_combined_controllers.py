"""IDM+PID ve Pure Pursuit+MPC zincirlerinin kapalı çevrim testleri."""

import math
import sys
import types
import unittest
from types import SimpleNamespace


if "dotenv" not in sys.modules:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *arguments, **keywords: None
    sys.modules["dotenv"] = dotenv

from carla_app.controller.vehicle.idm_speed_planner import IDMSpeedPlanner
from carla_app.controller.vehicle.longitudinal_pid_controller import (
    LongitudinalPIDController,
)
from carla_app.controller.vehicle.pure_pursuit_mpc_controller import (
    PurePursuitMPCController,
)


def point(x, y=0.0):
    return SimpleNamespace(x=float(x), y=float(y), z=0.0)


def longitudinal_state(speed_mps):
    return {
        "speed_mps": float(speed_mps),
        "location": point(0.0),
    }


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
        ratio = (
            (x - start.x) * segment_x + (y - start.y) * segment_y
        ) / length_squared
        ratio = max(0.0, min(1.0, ratio))
        error_x = x - (start.x + ratio * segment_x)
        error_y = y - (start.y + ratio * segment_y)
        best_squared = min(best_squared, error_x**2 + error_y**2)
    return math.sqrt(best_squared)


class IDMPIDClosedLoopTests(unittest.TestCase):
    def simulate_step(self, speed, throttle, brake, dt):
        acceleration = 3.0 * throttle - 5.0 * brake
        if speed > 0.02:
            acceleration -= 0.12
        return max(0.0, speed + acceleration * dt)

    def test_stops_behind_stationary_vehicle_without_collision(self):
        idm = IDMSpeedPlanner(0.05)
        pid = LongitudinalPIDController(0.05)
        speed = 60.0 / 3.6
        distance = 55.0

        for tick in range(500):
            lead = {
                "track_id": 1,
                "distance_m": distance,
                "relative_speed_mps": -speed,
                "source": "camera_radar_track",
            }
            reference, idm_info = idm.run_step(
                longitudinal_state(speed),
                lead,
                70.0 / 3.6,
            )
            throttle, brake, _ = pid.run_step(
                longitudinal_state(speed),
                lead,
                reference,
                idm_info["idm_acceleration_mps2"],
            )
            speed = self.simulate_step(speed, throttle, brake, 0.05)
            distance -= speed * 0.05
            if speed <= 0.02 and brake >= pid.hold_brake:
                break

        self.assertLess(tick, 499)
        self.assertGreaterEqual(distance, 2.0)
        self.assertLessEqual(distance, 3.0)

    def test_settles_to_moving_lead_speed_and_dynamic_gap(self):
        idm = IDMSpeedPlanner(0.05)
        pid = LongitudinalPIDController(0.05)
        lead_speed = 10.0
        ego_speed = 15.0
        distance = 40.0
        idm_info = None

        for _ in range(1000):
            lead = {
                "track_id": 2,
                "distance_m": distance,
                "relative_speed_mps": lead_speed - ego_speed,
                "source": "camera_radar_track",
            }
            reference, idm_info = idm.run_step(
                longitudinal_state(ego_speed),
                lead,
                70.0 / 3.6,
            )
            throttle, brake, _ = pid.run_step(
                longitudinal_state(ego_speed),
                lead,
                reference,
                idm_info["idm_acceleration_mps2"],
            )
            ego_speed = self.simulate_step(
                ego_speed,
                throttle,
                brake,
                0.05,
            )
            distance += (lead_speed - ego_speed) * 0.05

        self.assertAlmostEqual(ego_speed, lead_speed, delta=0.15)
        self.assertGreaterEqual(distance, idm_info["desired_gap_m"])
        self.assertLess(distance - idm_info["desired_gap_m"], 1.5)


class PurePursuitMPCClosedLoopTests(unittest.TestCase):
    def test_constant_curve_stays_close_to_route(self):
        controller = PurePursuitMPCController(0.05)
        radius_m = 25.0
        speed_mps = 23.0 / 3.6
        path = [
            point(
                radius_m * math.sin(index * 0.25 / radius_m),
                radius_m * (1.0 - math.cos(index * 0.25 / radius_m)),
            )
            for index in range(900)
        ]
        x = path[0].x
        y = path[0].y
        yaw = 0.0
        errors = []
        active_ticks = 0

        for _ in range(180):
            near = nearest_index(path, x, y)
            reference_path = path[
                max(0, near - 3) : min(len(path), near + 140)
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
            active_ticks += int(controller.last_info.get("mpc_active", False))
            wheel_angle = steer * controller.maximum_wheel_angle_rad
            x += speed_mps * math.cos(yaw) * controller.dt
            y += speed_mps * math.sin(yaw) * controller.dt
            yaw += (
                speed_mps
                / controller.wheelbase_m
                * math.tan(wheel_angle)
                * controller.dt
            )
            near_after_step = nearest_index(path, x, y)
            errors.append(distance_to_path(path, x, y, near_after_step))

        self.assertGreaterEqual(active_ticks, 170)
        self.assertLess(sum(errors[-60:]) / 60, 0.08)


if __name__ == "__main__":
    unittest.main()

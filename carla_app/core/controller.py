import math

import carla


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


class LaneFollowController:
    def __init__(self, dt=0.05):
        self.target_speed_kmh = 30.0
        self.steer_gain = 1.4
        self.max_steer = 0.7
        self.speed_kp = 0.04
        self.speed_ki = 0.001
        self.speed_kd = 0.01
        self.speed_error_sum = 0.0
        self.previous_speed_error = 0.0
        self.dt = dt

    def run_step(self, state):
        target = state["next_location"]
        if target is None:
            return carla.VehicleControl(brake=1.0)

        location = state["location"]
        yaw = math.radians(state["yaw"])
        target_angle = math.atan2(target.y - location.y, target.x - location.x)
        angle_error = math.atan2(math.sin(target_angle - yaw), math.cos(target_angle - yaw))
        steer = clamp(self.steer_gain * angle_error, -self.max_steer, self.max_steer)

        if state["is_junction"]:
            target_speed = 15.0
        elif abs(steer) > 0.45:
            target_speed = 18.0
        elif abs(steer) > 0.25:
            target_speed = 23.0
        else:
            target_speed = self.target_speed_kmh

        speed_error = target_speed - state["speed_kmh"]
        self.speed_error_sum += speed_error * self.dt
        derivative = (speed_error - self.previous_speed_error) / self.dt
        self.previous_speed_error = speed_error

        output = (
            self.speed_kp * speed_error
            + self.speed_ki * self.speed_error_sum
            + self.speed_kd * derivative
        )

        if output >= 0:
            return carla.VehicleControl(
                throttle=clamp(output, 0.0, 0.6),
                steer=steer,
            )

        return carla.VehicleControl(
            brake=clamp(abs(output), 0.0, 0.8),
            steer=steer,
        )

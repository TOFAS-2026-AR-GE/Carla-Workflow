import math

import carla


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


class LaneFollowController:
    """
    Basit şerit takip kontrolcüsü.

    Görevleri:
    1. İlerideki waypoint'e göre direksiyon üretir.
    2. Hedef hıza göre gaz/fren üretir.
    3. Dönüşlerde hedef hızı biraz düşürür.
    """

    def __init__(self):
        self.target_speed_kmh = 30.0

        self.steer_gain = 1.4
        self.max_steer = 0.7

        self.speed_kp = 0.04
        self.speed_ki = 0.001
        self.speed_kd = 0.01

        self.speed_error_sum = 0.0
        self.previous_speed_error = 0.0

        self.dt = 0.05

    def run_step(self, data):
        vehicle_location = data["location"]
        vehicle_yaw = math.radians(data["yaw"])
        current_speed = data["speed_kmh"]
        target_location = data["next_location"]

        if target_location is None:
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

        # =========================
        # 1. Şerit takip / direksiyon
        # =========================

        target_angle = math.atan2(
            target_location.y - vehicle_location.y,
            target_location.x - vehicle_location.x,
        )

        angle_error = target_angle - vehicle_yaw

        while angle_error > math.pi:
            angle_error -= 2.0 * math.pi

        while angle_error < -math.pi:
            angle_error += 2.0 * math.pi

        steer = self.steer_gain * angle_error
        steer = clamp(steer, -self.max_steer, self.max_steer)

        # =========================
        # 2. Virajda hız düşürme
        # =========================

        abs_steer = abs(steer)

        if data["is_junction"]:
            target_speed = 15.0
        elif abs_steer > 0.45:
            target_speed = 18.0
        elif abs_steer > 0.25:
            target_speed = 23.0
        else:
            target_speed = self.target_speed_kmh

        # =========================
        # 3. Hız kontrolü / PID
        # =========================

        speed_error = target_speed - current_speed

        self.speed_error_sum += speed_error * self.dt
        speed_error_change = (speed_error - self.previous_speed_error) / self.dt
        self.previous_speed_error = speed_error

        speed_output = (
            self.speed_kp * speed_error +
            self.speed_ki * self.speed_error_sum +
            self.speed_kd * speed_error_change
        )

        if speed_output >= 0:
            throttle = clamp(speed_output, 0.0, 0.6)
            brake = 0.0
        else:
            throttle = 0.0
            brake = clamp(abs(speed_output), 0.0, 0.8)

        return carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
        )
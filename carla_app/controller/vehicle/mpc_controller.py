import math

import carla


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def normalize_angle(angle):
    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


class MPCController:
    def __init__(self, dt=0.05):
        self.dt = dt

        # MPC gelecekte kaç adım düşünecek?
        self.horizon = 10

        # Tesla Model 3 için yaklaşık dingil mesafesi.
        self.wheelbase = 2.87

        # Ön tekerleklerin yaklaşık maksimum açısı.
        self.max_wheel_angle = math.radians(35.0)

        # Normal yol hedef hızı: 30 km/h.
        self.target_speed_mps = 30.0 / 3.6

        # Kavşak hedef hızı: 15 km/h.
        self.junction_speed_mps = 15.0 / 3.6

        # MPC'nin deneyeceği direksiyon komutları.
        self.steer_candidates = [
            -0.6,
            -0.4,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.4,
            0.6,
        ]

        # MPC'nin deneyeceği ivmeler.
        # Pozitif: hızlanma
        # Negatif: yavaşlama
        self.acceleration_candidates = [
            -4.0,
            -2.0,
            -1.0,
            0.0,
            1.0,
            2.0,
        ]

        self.previous_steer = 0.0
        self.previous_acceleration = 0.0

    def run_step(self, state):
        reference_path = state["reference_path"]

        if not reference_path:
            return carla.VehicleControl(brake=1.0)

        best_cost = float("inf")
        best_steer = 0.0
        best_acceleration = 0.0

        for steer in self.steer_candidates:
            for acceleration in self.acceleration_candidates:
                cost = self.calculate_cost(
                    state,
                    reference_path,
                    steer,
                    acceleration,
                )

                if cost < best_cost:
                    best_cost = cost
                    best_steer = steer
                    best_acceleration = acceleration

        self.previous_steer = best_steer
        self.previous_acceleration = best_acceleration

        return self.create_carla_control(
            best_steer,
            best_acceleration,
        )

    def calculate_cost(
        self,
        state,
        reference_path,
        steer,
        acceleration,
    ):
        location = state["location"]

        x = location.x
        y = location.y
        yaw = math.radians(state["yaw"])
        speed = state["speed_mps"]

        if state["is_junction"]:
            target_speed = self.junction_speed_mps
        else:
            target_speed = self.target_speed_mps

        total_cost = 0.0

        for step in range(self.horizon):
            x, y, yaw, speed = self.predict_state(
                x,
                y,
                yaw,
                speed,
                steer,
                acceleration,
            )

            reference_index = min(
                step,
                len(reference_path) - 1,
            )

            reference = reference_path[reference_index]

            position_error = (
                (x - reference.x) ** 2
                + (y - reference.y) ** 2
            )

            target_angle = math.atan2(
                reference.y - y,
                reference.x - x,
            )

            heading_error = normalize_angle(
                target_angle - yaw
            )

            speed_error = target_speed - speed

            total_cost += 1.0 * position_error
            total_cost += 2.0 * heading_error**2
            total_cost += 0.4 * speed_error**2
            total_cost += 0.1 * steer**2
            total_cost += 0.05 * acceleration**2

        steer_change = steer - self.previous_steer
        acceleration_change = (
            acceleration - self.previous_acceleration
        )

        total_cost += 0.8 * steer_change**2
        total_cost += 0.1 * acceleration_change**2

        return total_cost

    def predict_state(
        self,
        x,
        y,
        yaw,
        speed,
        steer,
        acceleration,
    ):
        wheel_angle = steer * self.max_wheel_angle

        next_x = x + speed * math.cos(yaw) * self.dt
        next_y = y + speed * math.sin(yaw) * self.dt

        next_yaw = yaw + (
            speed
            / self.wheelbase
            * math.tan(wheel_angle)
            * self.dt
        )

        next_speed = speed + acceleration * self.dt
        next_speed = max(0.0, next_speed)

        return (
            next_x,
            next_y,
            next_yaw,
            next_speed,
        )

    def create_carla_control(
        self,
        steer,
        acceleration,
    ):
        throttle = 0.0
        brake = 0.0

        if acceleration >= 0.0:
            throttle = acceleration / 2.0
        else:
            brake = abs(acceleration) / 4.0

        return carla.VehicleControl(
            throttle=clamp(throttle, 0.0, 0.7),
            steer=clamp(steer, -0.7, 0.7),
            brake=clamp(brake, 0.0, 1.0),
        )
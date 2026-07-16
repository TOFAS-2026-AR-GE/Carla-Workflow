import math


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def normalize_angle(angle):
    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


class LateralMPC:
    """Sadece direksiyon kontrolü yapan basit MPC."""

    def __init__(self, dt=0.05):
        self.dt = dt

        self.horizon = 10
        self.wheelbase = 2.87
        self.max_wheel_angle = math.radians(35.0)

        self.max_steer = 0.70

        # Bir tick'te direksiyon en fazla bu kadar değişebilir.
        self.max_steer_change = 0.04

        # Önceki direksiyonun çevresindeki adaylar.
        self.steer_offsets = [
            -0.12,
            -0.08,
            -0.04,
            0.0,
            0.04,
            0.08,
            0.12,
        ]

        self.previous_steer = 0.0

    def run_step(self, state):
        reference_path = state["reference_path"]

        if not reference_path:
            self.previous_steer = 0.0
            return 0.0

        best_cost = float("inf")
        best_steer = self.previous_steer

        for steer in self._build_candidates():
            cost = self._calculate_cost(
                state,
                reference_path,
                steer,
            )

            if cost < best_cost:
                best_cost = cost
                best_steer = steer

        # Hard steering-rate constraint.
        steer_change = clamp(
            best_steer - self.previous_steer,
            -self.max_steer_change,
            self.max_steer_change,
        )

        smooth_steer = clamp(
            self.previous_steer + steer_change,
            -self.max_steer,
            self.max_steer,
        )

        self.previous_steer = smooth_steer

        return smooth_steer

    def _build_candidates(self):
        candidates = []

        for offset in self.steer_offsets:
            steer = clamp(
                self.previous_steer + offset,
                -self.max_steer,
                self.max_steer,
            )

            if steer not in candidates:
                candidates.append(steer)

        return candidates

    def _calculate_cost(
        self,
        state,
        reference_path,
        steer,
    ):
        location = state["location"]

        x = float(location.x)
        y = float(location.y)
        yaw = math.radians(state["yaw"])
        speed = max(float(state["speed_mps"]), 0.0)

        total_cost = 0.0

        for step in range(self.horizon):
            x, y, yaw = self._predict_state(
                x,
                y,
                yaw,
                speed,
                steer,
            )

            reference_index = min(
                step,
                len(reference_path) - 1,
            )

            reference = reference_path[reference_index]

            reference_heading = self._reference_heading(
                reference_path,
                reference_index,
                x,
                y,
            )

            dx = x - reference.x
            dy = y - reference.y

            # Sadece şeride dik olan hata.
            # Yol boyunca önde/geride olma direksiyonu bozmaz.
            lateral_error = (
                -math.sin(reference_heading) * dx
                + math.cos(reference_heading) * dy
            )

            heading_error = normalize_angle(
                yaw - reference_heading
            )

            total_cost += 4.0 * lateral_error**2
            total_cost += 2.5 * heading_error**2
            total_cost += 0.10 * steer**2

        # Ani direksiyon değişikliğine güçlü ceza.
        steer_change = steer - self.previous_steer
        total_cost += 6.0 * steer_change**2

        return total_cost

    def _reference_heading(
        self,
        reference_path,
        index,
        x,
        y,
    ):
        if index + 1 < len(reference_path):
            current = reference_path[index]
            following = reference_path[index + 1]

            return math.atan2(
                following.y - current.y,
                following.x - current.x,
            )

        current = reference_path[index]

        return math.atan2(
            current.y - y,
            current.x - x,
        )

    def _predict_state(
        self,
        x,
        y,
        yaw,
        speed,
        steer,
    ):
        wheel_angle = steer * self.max_wheel_angle

        next_x = (
            x
            + speed * math.cos(yaw) * self.dt
        )

        next_y = (
            y
            + speed * math.sin(yaw) * self.dt
        )

        next_yaw = yaw + (
            speed
            / self.wheelbase
            * math.tan(wheel_angle)
            * self.dt
        )

        return next_x, next_y, next_yaw
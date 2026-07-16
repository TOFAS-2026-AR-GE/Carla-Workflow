import math


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def normalize_angle(angle):
    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


class CurvatureSpeedPlanner:
    """Yol eğriliğine göre hedef hız belirler."""

    def __init__(self, dt=0.05):
        self.dt = dt

        self.cruise_speed_mps = 30.0 / 3.6
        self.minimum_curve_speed_mps = 12.0 / 3.6

        # Konforlu maksimum yanal ivme.
        self.maximum_lateral_acceleration_mps2 = 2.0

        # Hedef hızın aniden değişmesini engeller.
        self.maximum_speed_increase_mps2 = 1.0
        self.maximum_speed_decrease_mps2 = 2.0

        self.previous_target_speed = (
            self.cruise_speed_mps
        )

    def run_step(self, state):
        curvature = self._estimate_curvature(
            state["reference_path"]
        )

        if curvature < 1e-3:
            curve_speed = self.cruise_speed_mps
        else:
            curve_speed = math.sqrt(
                self.maximum_lateral_acceleration_mps2
                / curvature
            )

        desired_speed = clamp(
            curve_speed,
            self.minimum_curve_speed_mps,
            self.cruise_speed_mps,
        )

        target_speed = self._limit_speed_change(
            desired_speed
        )

        self.previous_target_speed = target_speed

        return target_speed, {
            "curvature_1pm": curvature,
            "curve_speed_mps": desired_speed,
        }

    def _estimate_curvature(
        self,
        reference_path,
    ):
        if len(reference_path) < 3:
            return 0.0

        headings = []
        total_distance = 0.0

        for index in range(
            len(reference_path) - 1
        ):
            current = reference_path[index]
            following = reference_path[index + 1]

            dx = following.x - current.x
            dy = following.y - current.y

            distance = math.hypot(dx, dy)

            if distance < 1e-3:
                continue

            headings.append(
                math.atan2(dy, dx)
            )

            total_distance += distance

        if (
            len(headings) < 2
            or total_distance < 1e-3
        ):
            return 0.0

        heading_change = 0.0

        for index in range(
            len(headings) - 1
        ):
            heading_change += abs(
                normalize_angle(
                    headings[index + 1]
                    - headings[index]
                )
            )

        return heading_change / total_distance

    def _limit_speed_change(
        self,
        desired_speed,
    ):
        difference = (
            desired_speed
            - self.previous_target_speed
        )

        if difference >= 0.0:
            maximum_change = (
                self.maximum_speed_increase_mps2
                * self.dt
            )
        else:
            maximum_change = (
                self.maximum_speed_decrease_mps2
                * self.dt
            )

        difference = clamp(
            difference,
            -maximum_change,
            maximum_change,
        )

        return (
            self.previous_target_speed
            + difference
        )
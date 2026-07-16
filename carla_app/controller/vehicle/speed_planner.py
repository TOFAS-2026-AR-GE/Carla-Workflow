"""Reference speed planning from path curvature and lane-tracking error."""

import math


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


class CurvatureSpeedPlanner:
    """Choose a comfortable speed while keeping the route recoverable."""

    def __init__(self, dt=0.05):
        self.dt = float(dt)
        self.cruise_speed_mps = 30.0 / 3.6
        self.minimum_curve_speed_mps = 10.0 / 3.6
        self.maximum_lateral_acceleration_mps2 = 1.8
        self.maximum_speed_increase_mps2 = 1.0
        self.maximum_speed_decrease_mps2 = 3.0
        self.previous_target_speed_mps = self.cruise_speed_mps

    def run_step(self, state, lateral_info=None):
        curvature = self._preview_curvature(state.get("reference_path", []))
        curve_speed = self._curve_speed(curvature)
        desired_speed = clamp(
            curve_speed,
            self.minimum_curve_speed_mps,
            self.cruise_speed_mps,
        )

        recovery_speed = self._lane_recovery_speed(state, lateral_info)
        if recovery_speed is not None:
            desired_speed = min(desired_speed, recovery_speed)

        target_speed = self._rate_limit(desired_speed)
        self.previous_target_speed_mps = target_speed
        return target_speed, {
            "curvature_1pm": float(curvature),
            "curve_speed_mps": float(curve_speed),
            "desired_speed_mps": float(desired_speed),
            "recovery_speed_mps": recovery_speed,
        }

    def _curve_speed(self, curvature):
        if curvature < 1e-4:
            return self.cruise_speed_mps
        return math.sqrt(self.maximum_lateral_acceleration_mps2 / curvature)

    @staticmethod
    def _lane_recovery_speed(state, lateral_info):
        if not lateral_info:
            return None

        lane_width = max(2.5, float(state.get("lane_width", 3.5)))
        lateral_error = abs(float(lateral_info.get("cross_track_error_m", 0.0)))
        heading_error = abs(float(lateral_info.get("heading_error_rad", 0.0)))
        lane_fraction = lateral_error / lane_width

        if lane_fraction >= 0.55 or heading_error >= math.radians(30.0):
            return 8.0 / 3.6
        if lane_fraction >= 0.40 or heading_error >= math.radians(20.0):
            return 12.0 / 3.6
        if lane_fraction >= 0.30 or heading_error >= math.radians(12.0):
            return 18.0 / 3.6
        return None

    def _preview_curvature(self, path):
        if len(path) < 5:
            return 0.0

        preview = path[: min(len(path), 35)]
        curvatures = []
        # Four-point spacing makes this robust to 1 m waypoint quantization.
        for middle_index in range(2, len(preview) - 2):
            curvature = abs(
                self._three_point_curvature(
                    preview[middle_index - 2],
                    preview[middle_index],
                    preview[middle_index + 2],
                )
            )
            if curvature < 0.50:
                curvatures.append(curvature)

        if not curvatures:
            return 0.0

        # The 85th percentile reacts to a real upcoming bend without allowing
        # one bad map waypoint to set the speed for the whole preview.
        curvatures.sort()
        index = int(round(0.85 * (len(curvatures) - 1)))
        return curvatures[index]

    @staticmethod
    def _three_point_curvature(first, middle, last):
        side_a = math.hypot(middle.x - first.x, middle.y - first.y)
        side_b = math.hypot(last.x - middle.x, last.y - middle.y)
        chord = math.hypot(last.x - first.x, last.y - first.y)
        denominator = side_a * side_b * chord
        if denominator < 1e-6:
            return 0.0

        twice_area = (middle.x - first.x) * (last.y - first.y) - (
            middle.y - first.y
        ) * (last.x - first.x)
        return 2.0 * twice_area / denominator

    def _rate_limit(self, desired_speed):
        difference = desired_speed - self.previous_target_speed_mps
        acceleration_limit = (
            self.maximum_speed_increase_mps2
            if difference >= 0.0
            else self.maximum_speed_decrease_mps2
        )
        maximum_change = acceleration_limit * self.dt
        difference = clamp(difference, -maximum_change, maximum_change)
        return self.previous_target_speed_mps + difference

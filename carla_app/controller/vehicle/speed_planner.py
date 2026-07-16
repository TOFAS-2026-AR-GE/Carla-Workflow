import math


def clamp(value, minimum, maximum):
    return max(
        minimum,
        min(value, maximum),
    )


class CurvatureSpeedPlanner:
    """
    Yol egriligi ve seritten sapma miktarina
    gore hedef hiz belirler.
    """

    def __init__(self, dt=0.05):
        self.dt = float(dt)

        self.cruise_speed_mps = 30.0 / 3.6
        self.minimum_curve_speed_mps = 10.0 / 3.6

        # Konforlu yanal ivme.
        self.maximum_lateral_acceleration_mps2 = 1.8

        self.maximum_speed_increase_mps2 = 1.0
        self.maximum_speed_decrease_mps2 = 3.0

        self.previous_target_speed = self.cruise_speed_mps

    def run_step(
        self,
        state,
        lateral_info=None,
    ):
        curvature = self._estimate_preview_curvature(state["reference_path"])

        if curvature < 1e-4:
            curve_speed = self.cruise_speed_mps
        else:
            curve_speed = math.sqrt(self.maximum_lateral_acceleration_mps2 / curvature)

        desired_speed = clamp(
            curve_speed,
            self.minimum_curve_speed_mps,
            self.cruise_speed_mps,
        )

        recovery_speed = self._lane_recovery_speed(
            state,
            lateral_info,
        )

        if recovery_speed is not None:
            desired_speed = min(
                desired_speed,
                recovery_speed,
            )

        target_speed = self._limit_speed_change(desired_speed)

        self.previous_target_speed = target_speed

        return target_speed, {
            "curvature_1pm": float(curvature),
            "curve_speed_mps": float(curve_speed),
            "recovery_speed_mps": (recovery_speed),
        }

    def _lane_recovery_speed(
        self,
        state,
        lateral_info,
    ):
        if not lateral_info:
            return None

        lane_width = max(
            float(state["lane_width"]),
            2.5,
        )

        lateral_error = abs(float(lateral_info["cross_track_error_m"]))

        heading_error = abs(float(lateral_info["heading_error_rad"]))

        lane_fraction = lateral_error / lane_width

        # Arac ciddi sekilde seritten ciktiysa
        # direksiyon kontrolcusu toparlayana kadar yavasla.
        if lane_fraction >= 0.55 or heading_error >= math.radians(30.0):
            return 8.0 / 3.6

        if lane_fraction >= 0.40 or heading_error >= math.radians(20.0):
            return 12.0 / 3.6

        if lane_fraction >= 0.30 or heading_error >= math.radians(12.0):
            return 18.0 / 3.6

        return None

    def _estimate_preview_curvature(
        self,
        reference_path,
    ):
        if len(reference_path) < 3:
            return 0.0

        # Yaklasik ilk 30 metreyi kullan.
        preview_points = reference_path[
            : min(
                len(reference_path),
                31,
            )
        ]

        curvatures = []

        for index in range(
            1,
            len(preview_points) - 1,
        ):
            curvature = abs(
                self._three_point_curvature(
                    preview_points[index - 1],
                    preview_points[index],
                    preview_points[index + 1],
                )
            )

            # Harita kaynakli fiziksel olmayan
            # tek noktalik sivrilmeleri reddet.
            if curvature < 0.50:
                curvatures.append(curvature)

        if not curvatures:
            return 0.0

        # Uc ornekli median filtre.
        filtered = []

        for index in range(len(curvatures)):
            window = curvatures[
                max(0, index - 1) : min(
                    len(curvatures),
                    index + 2,
                )
            ]

            ordered = sorted(window)

            filtered.append(ordered[len(ordered) // 2])

        # Tek bir maximum outlier yerine
        # ust yuzdelik kullan.
        filtered.sort()

        percentile_index = int(round(0.85 * (len(filtered) - 1)))

        return filtered[percentile_index]

    @staticmethod
    def _three_point_curvature(
        first,
        middle,
        last,
    ):
        side_a = math.hypot(
            middle.x - first.x,
            middle.y - first.y,
        )
        side_b = math.hypot(
            last.x - middle.x,
            last.y - middle.y,
        )
        side_c = math.hypot(
            last.x - first.x,
            last.y - first.y,
        )

        denominator = side_a * side_b * side_c

        if denominator < 1e-6:
            return 0.0

        twice_area = (middle.x - first.x) * (last.y - first.y) - (
            middle.y - first.y
        ) * (last.x - first.x)

        return 2.0 * twice_area / denominator

    def _limit_speed_change(
        self,
        desired_speed,
    ):
        difference = desired_speed - self.previous_target_speed

        if difference >= 0.0:
            maximum_change = self.maximum_speed_increase_mps2 * self.dt
        else:
            maximum_change = self.maximum_speed_decrease_mps2 * self.dt

        difference = clamp(
            difference,
            -maximum_change,
            maximum_change,
        )

        return self.previous_target_speed + difference

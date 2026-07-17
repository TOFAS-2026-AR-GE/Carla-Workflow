"""Yol virajina ve serit hatasina gore guvenli hedef hizini hesaplar."""

import math


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


class CurvatureSpeedPlanner:
    """Aracin rotayi koruyabilecegi rahat ve guvenli hizi secer."""

    def __init__(self, dt=0.05, cruise_speed_kmh=60.0):
        self.dt = float(dt)
        self.cruise_speed_mps = max(10.0, float(cruise_speed_kmh)) / 3.6
        self.minimum_curve_speed_mps = min(
            15.0 / 3.6,
            self.cruise_speed_mps,
        )
        self.maximum_lateral_acceleration_mps2 = 2.0
        self.maximum_speed_increase_mps2 = 1.5
        self.maximum_speed_decrease_mps2 = 3.0
        self.recovery_confirmation_ticks = max(2, int(round(0.15 / self.dt)))
        self.recovery_evidence_ticks = 0
        self.previous_target_speed_mps = self.cruise_speed_mps

    def run_step(self, state, lateral_info=None):
        curvature = self.calculate_preview_curvature(
            state.get("reference_path", []),
            state.get("speed_mps", 0.0),
        )
        curve_speed = self.calculate_curve_speed(curvature)
        desired_speed = clamp(
            curve_speed,
            self.minimum_curve_speed_mps,
            self.cruise_speed_mps,
        )

        requested_recovery_speed = self.calculate_lane_recovery_speed(
            state,
            lateral_info,
        )
        if requested_recovery_speed is None:
            self.recovery_evidence_ticks = max(0, self.recovery_evidence_ticks - 1)
        else:
            self.recovery_evidence_ticks += 1

        recovery_speed = None
        if self.recovery_evidence_ticks >= self.recovery_confirmation_ticks:
            recovery_speed = requested_recovery_speed
        if recovery_speed is not None:
            desired_speed = min(desired_speed, recovery_speed)

        target_speed = self.limit_speed_change(desired_speed)
        self.previous_target_speed_mps = target_speed
        speed_reason = "cruise"
        if curve_speed < self.cruise_speed_mps:
            speed_reason = "curve"
        if recovery_speed is not None and recovery_speed <= desired_speed:
            speed_reason = "lane_recovery"
        return target_speed, {
            "curvature_1pm": float(curvature),
            "curve_speed_mps": float(curve_speed),
            "desired_speed_mps": float(desired_speed),
            "requested_recovery_speed_mps": requested_recovery_speed,
            "recovery_speed_mps": recovery_speed,
            "recovery_evidence_ticks": self.recovery_evidence_ticks,
            "speed_reason": speed_reason,
        }

    def calculate_curve_speed(self, curvature):
        if curvature < 1e-4:
            return self.cruise_speed_mps
        return math.sqrt(self.maximum_lateral_acceleration_mps2 / curvature)

    def calculate_lane_recovery_speed(self, state, lateral_info):
        if not lateral_info:
            return None

        lane_width = max(2.5, float(state.get("lane_width", 3.5)))
        vehicle_half_width = max(
            0.70,
            float(state.get("vehicle_half_width_m", 0.95)),
        )
        lateral_error = abs(float(lateral_info.get("cross_track_error_m", 0.0)))
        heading_error = abs(float(lateral_info.get("heading_error_rad", 0.0)))
        usable_center_offset = max(
            0.35,
            0.5 * lane_width - vehicle_half_width - 0.20,
        )
        edge_ratio = lateral_error / usable_center_offset

        if edge_ratio >= 1.00 or heading_error >= math.radians(30.0):
            return 8.0 / 3.6
        if edge_ratio >= 0.75 or heading_error >= math.radians(20.0):
            return 12.0 / 3.6
        if edge_ratio >= 0.50 or heading_error >= math.radians(12.0):
            return 18.0 / 3.6
        return None

    def calculate_preview_curvature(self, path, speed_mps=0.0):
        if len(path) < 5:
            return 0.0

        # Hiz arttikca daha uzagi tara. 80 km/sa hizda 35 metrelik eski
        # pencere viraji gec gosterebiliyordu; uc saniyelik yol daha guvenli.
        preview_distance_m = clamp(float(speed_mps) * 3.0, 35.0, 75.0)
        preview = [path[0]]
        travelled_m = 0.0
        for point in path[1:]:
            previous = preview[-1]
            travelled_m += math.hypot(point.x - previous.x, point.y - previous.y)
            preview.append(point)
            if travelled_m >= preview_distance_m:
                break
        curvatures = []
        # Dort noktalik aralik, 1 metrelik waypoint basamaklarinin
        # tek basina yanlis viraj uretmesini azaltir.
        for middle_index in range(2, len(preview) - 2):
            curvature = abs(
                self.three_point_curvature(
                    preview[middle_index - 2],
                    preview[middle_index],
                    preview[middle_index + 2],
                )
            )
            if curvature < 0.50:
                curvatures.append(curvature)

        if not curvatures:
            return 0.0

        # Yuzde 85 degeri gercek viraji yakalar fakat tek bir bozuk harita
        # noktasinin tum hedef hizi belirlemesine izin vermez.
        curvatures.sort()
        index = int(round(0.85 * (len(curvatures) - 1)))
        return curvatures[index]

    def three_point_curvature(self, first, middle, last):
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

    def limit_speed_change(self, desired_speed):
        difference = desired_speed - self.previous_target_speed_mps
        if difference >= 0.0:
            maximum_change = self.maximum_speed_increase_mps2 * self.dt
            return self.previous_target_speed_mps + min(difference, maximum_change)

        maximum_change = self.maximum_speed_decrease_mps2 * self.dt
        return self.previous_target_speed_mps + max(difference, -maximum_change)

    def limit_speed_increase(self, desired_speed):
        """Eski cagri adini kullanan kodlar icin geriye uyumluluk."""
        return self.limit_speed_change(desired_speed)

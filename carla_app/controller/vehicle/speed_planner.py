"""Yol virajına ve şerit hatasına göre güvenli hedef hızı hesaplar.

Girdi olarak referans yolu, araç hızını ve direksiyon hatasını alır. Çıktı
olarak boylamsal kontrolcünün izleyeceği metre/saniye cinsinden hızı verir.
"""

import math


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


class CurvatureSpeedPlanner:
    """Düz yol, hız tabelası ve virajdan açıklanabilir hedef hız seçer."""

    def __init__(self, dt=0.05, cruise_speed_kmh=70.0):
        self.dt = float(dt)
        self.cruise_speed_mps = max(10.0, float(cruise_speed_kmh)) / 3.6
        self.maximum_lateral_acceleration_mps2 = 1.8
        self.comfortable_curve_deceleration_mps2 = 1.8
        self.curve_entry_margin_m = 8.0
        self.maximum_speed_increase_mps2 = 1.2
        self.maximum_speed_decrease_mps2 = 2.2
        self.imu_filter_ratio = 0.20
        self.filtered_imu_lateral_acceleration_mps2 = None
        self.imu_stability_ticks = 0
        self.imu_stability_confirmation_ticks = max(
            2,
            int(round(0.15 / self.dt)),
        )
        self.recovery_confirmation_ticks = max(2, int(round(0.15 / self.dt)))
        self.recovery_release_confirmation_ticks = max(
            2,
            int(round(0.40 / self.dt)),
        )
        self.recovery_evidence_ticks = 0
        self.recovery_release_ticks = 0
        self.active_recovery_speed_mps = None
        self.previous_target_speed_mps = self.cruise_speed_mps

    def run_step(self, state, lateral_info=None, speed_limit_kmh=None):
        """Viraj ve şerit hatasından bu çevrimin hedef hızını seçer."""
        road_speed_mps = self.select_road_speed(speed_limit_kmh)
        curve_profile = self.calculate_curve_profile(
            state.get("reference_path", []),
            state.get("speed_mps", 0.0),
            road_speed_mps,
        )
        curvature = curve_profile["curvature_1pm"]
        curve_speed = curve_profile["target_speed_mps"]
        desired_speed = clamp(
            curve_speed,
            0.0,
            road_speed_mps,
        )
        imu_stability_speed = self.calculate_imu_stability_speed(state)
        if imu_stability_speed is not None:
            desired_speed = min(desired_speed, imu_stability_speed)

        requested_recovery_speed = self.calculate_lane_recovery_speed(
            state,
            lateral_info,
        )
        if requested_recovery_speed is None:
            if self.active_recovery_speed_mps is None:
                self.recovery_evidence_ticks = max(
                    0,
                    self.recovery_evidence_ticks - 1,
                )
            else:
                self.recovery_release_ticks += 1
                if (
                    self.recovery_release_ticks
                    >= self.recovery_release_confirmation_ticks
                ):
                    self.active_recovery_speed_mps = None
                    self.recovery_evidence_ticks = 0
                    self.recovery_release_ticks = 0
        else:
            self.recovery_release_ticks = 0
            if self.active_recovery_speed_mps is None:
                self.recovery_evidence_ticks += 1
                if (
                    self.recovery_evidence_ticks
                    >= self.recovery_confirmation_ticks
                ):
                    self.active_recovery_speed_mps = requested_recovery_speed
            else:
                # Hata büyürse hemen daha güvenli hıza geç; iyileşme ise çıkış
                # doğrulanana kadar mevcut kısıtı korusun.
                self.active_recovery_speed_mps = min(
                    self.active_recovery_speed_mps,
                    requested_recovery_speed,
                )

        recovery_speed = self.active_recovery_speed_mps
        if recovery_speed is not None:
            desired_speed = min(desired_speed, recovery_speed)

        previous_target_speed = self.previous_target_speed_mps
        target_speed = self.limit_speed_change(desired_speed)
        # Araç virajın içindeyse fiziksel yanal ivme sınırı geciktirilemez.
        # Uzaktaki virajlarda ise mesafeye bağlı fren zarfı ve bu slew limiti
        # birlikte çalışarak hedef hızın aniden düşmesini önler.
        if curve_profile["distance_m"] <= self.curve_entry_margin_m:
            target_speed = min(
                target_speed,
                curve_profile["local_curve_speed_mps"],
            )
        self.previous_target_speed_mps = target_speed
        speed_reason = "cruise"
        if speed_limit_kmh is not None:
            speed_reason = "speed_limit"
        if curve_speed < road_speed_mps - 0.05:
            speed_reason = "curve"
        if recovery_speed is not None and recovery_speed <= desired_speed:
            speed_reason = "lane_recovery"
        if (
            imu_stability_speed is not None
            and imu_stability_speed <= desired_speed + 1e-9
        ):
            speed_reason = "imu_stability"
        curve_distance = curve_profile["distance_m"]
        predicted_curve_entry_speed = math.sqrt(
            max(
                0.0,
                target_speed**2
                - 2.0
                * self.comfortable_curve_deceleration_mps2
                * max(0.0, curve_distance - self.curve_entry_margin_m),
            )
        )
        predicted_curve_entry_speed = min(
            predicted_curve_entry_speed,
            target_speed,
        )
        predicted_yaw_rate = predicted_curve_entry_speed * curvature
        predicted_lateral_acceleration = (
            predicted_curve_entry_speed**2 * curvature
        )
        planned_longitudinal_acceleration = (
            target_speed - previous_target_speed
        ) / self.dt
        return target_speed, {
            "curvature_1pm": float(curvature),
            "curve_speed_mps": float(curve_speed),
            "local_curve_speed_mps": float(
                curve_profile["local_curve_speed_mps"]
            ),
            "curve_distance_m": float(curve_distance),
            "predicted_curve_entry_speed_mps": float(
                predicted_curve_entry_speed
            ),
            "road_speed_mps": float(road_speed_mps),
            "speed_limit_kmh": speed_limit_kmh,
            "desired_speed_mps": float(desired_speed),
            "requested_recovery_speed_mps": requested_recovery_speed,
            "recovery_speed_mps": recovery_speed,
            "recovery_evidence_ticks": self.recovery_evidence_ticks,
            "recovery_release_ticks": self.recovery_release_ticks,
            "imu_lateral_acceleration_mps2": (
                self.filtered_imu_lateral_acceleration_mps2
            ),
            "imu_stability_speed_mps": imu_stability_speed,
            "imu_stability_ticks": self.imu_stability_ticks,
            "speed_reason": speed_reason,
            "predicted_yaw_rate_radps": float(predicted_yaw_rate),
            "predicted_lateral_acceleration_mps2": float(
                predicted_lateral_acceleration
            ),
            "planned_longitudinal_acceleration_mps2": float(
                planned_longitudinal_acceleration
            ),
        }

    def calculate_imu_stability_speed(self, state):
        """IMU yanal ivmesi yüksek kalırsa mevcut hızı yumuşakça sınırlar."""
        measurements = []
        try:
            lateral_acceleration = abs(
                float(state.get("imu_lateral_acceleration_mps2"))
            )
            if math.isfinite(lateral_acceleration):
                measurements.append(lateral_acceleration)
        except (TypeError, ValueError):
            pass

        try:
            yaw_rate = abs(float(state.get("imu_yaw_rate_radps")))
            speed_mps = max(0.0, float(state.get("speed_mps", 0.0)))
            yaw_lateral_acceleration = speed_mps * yaw_rate
            if math.isfinite(yaw_lateral_acceleration):
                measurements.append(yaw_lateral_acceleration)
        except (TypeError, ValueError):
            pass

        if not measurements:
            self.imu_stability_ticks = max(0, self.imu_stability_ticks - 1)
            return None

        measured = min(30.0, max(measurements))
        if self.filtered_imu_lateral_acceleration_mps2 is None:
            self.filtered_imu_lateral_acceleration_mps2 = measured
        else:
            self.filtered_imu_lateral_acceleration_mps2 += (
                self.imu_filter_ratio
                * (
                    measured
                    - self.filtered_imu_lateral_acceleration_mps2
                )
            )

        activation_threshold = (
            self.maximum_lateral_acceleration_mps2 * 1.15
        )
        if (
            self.filtered_imu_lateral_acceleration_mps2
            > activation_threshold
        ):
            self.imu_stability_ticks += 1
        else:
            self.imu_stability_ticks = max(0, self.imu_stability_ticks - 1)

        if (
            self.imu_stability_ticks
            < self.imu_stability_confirmation_ticks
        ):
            return None

        speed_mps = max(0.0, float(state.get("speed_mps", 0.0)))
        ratio = math.sqrt(
            self.maximum_lateral_acceleration_mps2
            / max(
                self.maximum_lateral_acceleration_mps2,
                self.filtered_imu_lateral_acceleration_mps2,
            )
        )
        return max(3.0, speed_mps * ratio)

    def calculate_curve_profile(self, path, speed_mps, road_speed_mps):
        """Rota boyunca viraj hızlarını geriye doğru fren zarfına dönüştürür."""
        samples = self.sample_preview_curvatures(path, speed_mps)
        if not samples:
            return {
                "target_speed_mps": float(road_speed_mps),
                "local_curve_speed_mps": float(road_speed_mps),
                "curvature_1pm": 0.0,
                "distance_m": 0.0,
            }

        best_target = float(road_speed_mps)
        best_local_speed = float(road_speed_mps)
        best_curvature = 0.0
        best_distance = 0.0
        for distance_m, curvature in samples:
            local_speed = self.calculate_curve_speed(
                curvature,
                road_speed_mps,
            )
            braking_distance = max(
                0.0,
                distance_m - self.curve_entry_margin_m,
            )
            allowed_now = math.sqrt(
                local_speed**2
                + 2.0
                * self.comfortable_curve_deceleration_mps2
                * braking_distance
            )
            allowed_now = min(float(road_speed_mps), allowed_now)
            if allowed_now < best_target:
                best_target = allowed_now
                best_local_speed = local_speed
                best_curvature = curvature
                best_distance = distance_m

        return {
            "target_speed_mps": float(best_target),
            "local_curve_speed_mps": float(best_local_speed),
            "curvature_1pm": float(best_curvature),
            "distance_m": float(best_distance),
        }

    def sample_preview_curvatures(self, path, speed_mps=0.0):
        """Öndeki rota boyunca mesafe ve yumuşatılmış eğrilik örnekleri üretir."""
        if len(path) < 5:
            return []

        preview_distance_m = clamp(
            float(speed_mps) * 4.5 + 20.0,
            45.0,
            110.0,
        )
        distances = [0.0]
        preview = [path[0]]
        for point in path[1:]:
            previous = preview[-1]
            distance = distances[-1] + math.hypot(
                float(point.x) - float(previous.x),
                float(point.y) - float(previous.y),
            )
            preview.append(point)
            distances.append(distance)
            if distance >= preview_distance_m:
                break

        raw = []
        for middle_index in range(2, len(preview) - 2):
            curvature = abs(
                self.three_point_curvature(
                    preview[middle_index - 2],
                    preview[middle_index],
                    preview[middle_index + 2],
                )
            )
            raw.append(0.0 if curvature >= 0.50 else curvature)

        samples = []
        for index, curvature in enumerate(raw):
            start = max(0, index - 2)
            end = min(len(raw), index + 3)
            window = sorted(raw[start:end])
            smoothed = window[len(window) // 2]
            samples.append((distances[index + 2], smoothed))
        return samples

    def select_road_speed(self, speed_limit_kmh):
        """Tabela varsa tabela hızını, yoksa varsayılan 70 km/sa hedefini seçer."""
        if speed_limit_kmh is None:
            return self.cruise_speed_mps
        try:
            limit_mps = float(speed_limit_kmh) / 3.6
        except (TypeError, ValueError):
            return self.cruise_speed_mps
        if not math.isfinite(limit_mps) or limit_mps <= 0.0:
            return self.cruise_speed_mps
        return limit_mps

    def calculate_curve_speed(self, curvature, road_speed_mps=None):
        road_speed_mps = (
            self.cruise_speed_mps
            if road_speed_mps is None
            else max(0.0, float(road_speed_mps))
        )
        if curvature < 1e-4:
            return road_speed_mps
        comfortable_speed = math.sqrt(
            self.maximum_lateral_acceleration_mps2 / curvature
        )
        # 23 km/sa, yalnız bu hızda yanal ivme sınırı sağlanıyorsa doğal olarak
        # korunur. Dar virajda sabit bir taban fiziksel sınırla çelişir.
        return min(road_speed_mps, comfortable_speed)

    def calculate_lane_recovery_speed(self, state, lateral_info):
        """Şerit kenarında güvenli toparlanma hızını verir."""
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
            return 23.0 / 3.6
        if edge_ratio >= 0.75 or heading_error >= math.radians(20.0):
            return 30.0 / 3.6
        if edge_ratio >= 0.50 or heading_error >= math.radians(12.0):
            return 40.0 / 3.6
        return None

    def calculate_preview_curvature(self, path, speed_mps=0.0):
        """Hıza göre ilerideki yolun güvenilir eğrilik değerini hesaplar."""
        samples = self.sample_preview_curvatures(path, speed_mps)
        if not samples:
            return 0.0
        return max(curvature for _distance, curvature in samples)

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
        """Hedef hızın çevrimler arasında aniden değişmesini önler."""
        difference = desired_speed - self.previous_target_speed_mps
        if difference >= 0.0:
            maximum_change = self.maximum_speed_increase_mps2 * self.dt
            return self.previous_target_speed_mps + min(difference, maximum_change)

        maximum_change = self.maximum_speed_decrease_mps2 * self.dt
        return self.previous_target_speed_mps + max(difference, -maximum_change)

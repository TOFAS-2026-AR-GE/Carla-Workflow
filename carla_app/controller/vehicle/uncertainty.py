"""Ölçüm yaşı ve covariance bilgisini kontrol sınırlarına dönüştürür."""

import math


def clamp(value, minimum, maximum):
    return max(minimum, min(float(value), maximum))


class ControlUncertaintyManager:
    """Eski veya belirsiz ölçümlerde mesafeyi ve hedef hızı muhafazakâr yapar."""

    def __init__(self, dt, parameters):
        self.dt = max(0.01, float(dt))
        self.parameters = parameters

    def apply(self, state, lead_vehicle, emergency_obstacle, road_context=None):
        current_frame_id = int(state.get("frame_id", 0))
        ego_speed = max(0.0, float(state.get("speed_mps", 0.0)))
        conservative_lead = self.conservative_obstacle(
            lead_vehicle,
            current_frame_id,
            ego_speed,
            normal_path=True,
        )
        conservative_emergency = self.conservative_obstacle(
            emergency_obstacle,
            current_frame_id,
            ego_speed,
            normal_path=False,
        )
        speed_cap, reasons = self.speed_cap(state, road_context or {})
        return conservative_lead, conservative_emergency, {
            "speed_cap_mps": speed_cap,
            "reasons": reasons,
            "lead_adjusted": conservative_lead is not None,
            "emergency_adjusted": conservative_emergency is not None,
        }

    def conservative_obstacle(
        self,
        obstacle,
        current_frame_id,
        ego_speed,
        normal_path,
    ):
        if obstacle is None:
            return None

        measurement_frame_id = obstacle.get("measurement_frame_id")
        age_frames = 0
        if measurement_frame_id is not None:
            try:
                age_frames = max(
                    0,
                    int(current_frame_id) - int(measurement_frame_id),
                )
            except (TypeError, ValueError):
                age_frames = 0
        age_s = age_frames * self.dt

        if normal_path:
            hard_age = float(
                getattr(
                    self.parameters,
                    "lead_measurement_hard_age_s",
                    0.35,
                )
            )
        else:
            hard_age = float(
                getattr(
                    self.parameters,
                    "emergency_measurement_hard_age_s",
                    0.20,
                )
            )
        hard_age = max(self.dt, hard_age)
        if age_s > hard_age:
            # Süresi dolmuş bir ölçüm ne normal takipte ne de AEB yolunda
            # yeniden kullanılır. Yeni sensör kanıtı gelene kadar üst katmandaki
            # algı/lokalizasyon yaş politikası aracı güvenli hıza indirir.
            return None

        try:
            distance = float(obstacle["distance_m"])
        except (KeyError, TypeError, ValueError):
            return obstacle
        if not math.isfinite(distance):
            return obstacle

        source = str(obstacle.get("source", "unknown"))
        default_std = 0.75
        if "camera" in source:
            default_std = 1.50
        elif "radar" in source:
            default_std = 0.65

        explicit_distance_std = "distance_std_m" in obstacle
        try:
            base_distance_std = float(
                obstacle.get("distance_std_m", default_std)
            )
        except (TypeError, ValueError):
            base_distance_std = default_std
            explicit_distance_std = False
        if not math.isfinite(base_distance_std):
            base_distance_std = default_std
            explicit_distance_std = False
        distance_std = max(0.20, base_distance_std + 0.80 * age_s)

        # Sigma mesafe payı yalnız tracker/sensör açıkça covariance sağladığında
        # uygulanır. Kaynağı bilinmeyen taze eski-uyumlu girdiye varsayımsal
        # standart sapma eklemek 7.1 m'yi sebepsiz 4.1 m'ye düşürüyordu.
        sigma_multiplier = float(
            getattr(self.parameters, "control_uncertainty_sigma", 2.0)
        )
        sigma_margin = (
            max(0.0, sigma_multiplier) * distance_std
            if explicit_distance_std
            else 0.0
        )
        motion_margin = max(0.0, float(ego_speed)) * age_s
        total_margin = motion_margin + sigma_margin

        try:
            relative_speed = float(obstacle.get("relative_speed_mps", 0.0))
        except (TypeError, ValueError):
            relative_speed = 0.0
        if not math.isfinite(relative_speed):
            relative_speed = 0.0

        age_ratio = clamp(age_s / hard_age, 0.0, 1.0)
        relative_speed_uncertainty = min(
            3.0,
            age_ratio * distance_std / hard_age,
        )

        # Taze ve covariance bilgisi taşımayan girdiyi aynen döndürmek eski API
        # sözleşmesini, nesne kimliğini ve ham sensör teşhisini korur.
        if (
            age_frames == 0
            and not explicit_distance_std
            and relative_speed_uncertainty == 0.0
        ):
            return obstacle

        adjusted = dict(obstacle)
        adjusted["raw_distance_m"] = distance
        adjusted["distance_m"] = max(0.0, distance - total_margin)
        adjusted["measurement_age_frames"] = age_frames
        adjusted["measurement_age_s"] = age_s
        adjusted["distance_std_m"] = distance_std
        adjusted["uncertainty_margin_m"] = total_margin
        adjusted["relative_speed_mps"] = (
            relative_speed - relative_speed_uncertainty
        )
        adjusted["relative_speed_uncertainty_mps"] = (
            relative_speed_uncertainty
        )
        adjusted["too_old_for_control"] = False
        adjusted["distance_uncertainty_explicit"] = explicit_distance_std
        return adjusted

    def speed_cap(self, state, road_context):
        cap = math.inf
        reasons = []
        localization = state.get("localization")
        if localization:
            status = str(localization.get("status", "UNAVAILABLE"))
            localization_age = float(localization.get("sensor_age_s", math.inf))
            if status in {"UNAVAILABLE", "LOST"}:
                cap = 0.0
                reasons.append("localization_lost")
            elif status == "DEGRADED":
                cap = min(
                    cap,
                    float(
                        getattr(
                            self.parameters,
                            "localization_degraded_speed_mps",
                            10.0 / 3.6,
                        )
                    ),
                )
                reasons.append("localization_degraded")
            if localization_age > float(
                getattr(self.parameters, "localization_hard_stop_age_s", 0.60)
            ):
                cap = 0.0
                reasons.append("localization_measurement_too_old")

        perception_age_frames = int(
            road_context.get("perception_age_frames", 0) or 0
        )
        perception_age_s = perception_age_frames * self.dt
        degraded_age = float(
            getattr(self.parameters, "perception_degraded_age_s", 0.20)
        )
        stop_age = float(getattr(self.parameters, "perception_stop_age_s", 0.50))
        if perception_age_s >= stop_age:
            cap = min(cap, 0.0)
            reasons.append("perception_too_old")
        elif perception_age_s >= degraded_age:
            degraded_cap = float(
                getattr(
                    self.parameters,
                    "perception_degraded_speed_mps",
                    15.0 / 3.6,
                )
            )
            ratio = clamp(
                (perception_age_s - degraded_age) / max(0.01, stop_age - degraded_age),
                0.0,
                1.0,
            )
            cap = min(cap, degraded_cap * (1.0 - ratio))
            reasons.append("perception_age_uncertainty")

        return cap, reasons

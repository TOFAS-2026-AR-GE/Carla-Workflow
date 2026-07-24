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
        adjusted = dict(obstacle)
        measurement_frame_id = obstacle.get("measurement_frame_id")
        age_frames = 0
        if measurement_frame_id is not None:
            age_frames = max(0, current_frame_id - int(measurement_frame_id))
        age_s = age_frames * self.dt

        source = str(obstacle.get("source", "unknown"))
        default_std = 0.75
        if "camera" in source:
            default_std = 1.50
        elif "radar" in source:
            default_std = 0.65
        try:
            distance_std = float(obstacle.get("distance_std_m", default_std))
        except (TypeError, ValueError):
            distance_std = default_std
        distance_std = max(0.20, distance_std + 0.80 * age_s)

        try:
            distance = float(obstacle["distance_m"])
        except (KeyError, TypeError, ValueError):
            return adjusted
        sigma_multiplier = float(
            getattr(self.parameters, "control_uncertainty_sigma", 2.0)
        )
        motion_margin = ego_speed * age_s
        uncertainty_margin = sigma_multiplier * distance_std
        conservative_distance = max(
            0.0,
            distance - motion_margin - uncertainty_margin,
        )
        adjusted["raw_distance_m"] = distance
        adjusted["distance_m"] = conservative_distance
        adjusted["measurement_age_frames"] = age_frames
        adjusted["measurement_age_s"] = age_s
        adjusted["distance_std_m"] = distance_std
        adjusted["uncertainty_margin_m"] = motion_margin + uncertainty_margin

        try:
            relative_speed = float(obstacle.get("relative_speed_mps", 0.0))
        except (TypeError, ValueError):
            relative_speed = 0.0
        adjusted["relative_speed_mps"] = relative_speed - min(
            3.0,
            distance_std / max(self.dt, age_s + self.dt),
        )

        hard_age = float(
            getattr(
                self.parameters,
                "lead_measurement_hard_age_s",
                0.35 if normal_path else 0.20,
            )
        )
        adjusted["too_old_for_control"] = age_s > hard_age
        if normal_path and adjusted["too_old_for_control"]:
            return None
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

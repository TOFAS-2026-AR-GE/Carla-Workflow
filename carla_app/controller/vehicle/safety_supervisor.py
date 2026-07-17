"""Çarpışma süresine ve gerekli yavaşlamaya göre acil freni denetler.

Bu sınıf normal gaz-fren hesabından bağımsızdır. Yalnızca doğrulanmış kritik
tehlikede tam fren ister.
"""

import math


class EmergencyBrakeSupervisor:
    """Takip hedefi ile ham radar tehlikesinden daha riskli olanı inceler."""

    def __init__(self):
        self.stopping_clearance_m = 2.0
        self.hard_distance_m = 1.25
        self.immediate_distance_m = 0.65
        self.ttc_threshold_s = 1.50
        self.immediate_ttc_s = 0.80
        self.required_deceleration_threshold_mps2 = 4.0
        self.confirmation_ticks = 2
        self.hazard_count = 0
        self.last_hazard = None

    def evaluate_candidates(self, lead_obstacle, radar_obstacle=None):
        """İki güvenlik adayını karşılaştırıp acil fren kararını verir."""
        selected_obstacle = None
        selected_metrics = None
        selected_priority = None

        for obstacle in (lead_obstacle, radar_obstacle):
            metrics = self.calculate_metrics(obstacle)
            if metrics is None:
                continue

            priority = self.calculate_risk_priority(metrics)
            if selected_priority is None or priority < selected_priority:
                selected_obstacle = obstacle
                selected_metrics = metrics
                selected_priority = priority

        if selected_obstacle is None:
            self.hazard_count = 0
            self.last_hazard = None
            return False, self.empty_info()

        emergency, info = self.evaluate_metrics(
            selected_metrics,
            selected_obstacle,
        )
        return emergency, info

    def evaluate(self, obstacle):
        """Tek bir tehlike adayını değerlendirir."""
        metrics = self.calculate_metrics(obstacle)
        if metrics is None:
            self.hazard_count = 0
            self.last_hazard = None
            return False, self.empty_info()
        return self.evaluate_metrics(metrics, obstacle)

    def evaluate_metrics(self, metrics, obstacle=None):
        """Mesafe ve TTC değerini doğrulayıp acil fren kararına çevirir."""
        obstacle = obstacle or {}
        source = obstacle.get("source", "unknown")

        # Tek bir ham radar noktası, çok kısa TTC hesaplandığı için aracı
        # aniden durdurmasın. Fiziksel olarak çok yakın değilse ikinci, yeni
        # radar karesinde de aynı engeli görmeyi bekleriz.
        is_raw_radar = source == "radar_emergency"
        immediate_hazard = metrics["distance_m"] <= self.immediate_distance_m
        if not is_raw_radar:
            immediate_hazard = (
                immediate_hazard or metrics["ttc_s"] <= self.immediate_ttc_s
            )
        ordinary_hazard = (
            metrics["distance_m"] <= self.hard_distance_m
            or metrics["ttc_s"] <= self.ttc_threshold_s
            or metrics["required_deceleration_mps2"]
            >= self.required_deceleration_threshold_mps2
        )

        new_observation = self.is_new_observation(obstacle)
        same_hazard = self.is_same_hazard(obstacle)

        if ordinary_hazard and new_observation:
            self.hazard_count = self.hazard_count + 1 if same_hazard else 1
            self.last_hazard = self.remember_hazard(obstacle)
        elif not ordinary_hazard and new_observation:
            self.hazard_count = max(0, self.hazard_count - 1)
            if self.hazard_count == 0:
                self.last_hazard = None

        emergency = immediate_hazard or self.hazard_count >= self.confirmation_ticks
        return emergency, {
            "distance_m": metrics["distance_m"],
            "ttc_s": metrics["ttc_s"],
            "required_deceleration_mps2": metrics["required_deceleration_mps2"],
            "hazard_count": self.hazard_count,
            "immediate_hazard": immediate_hazard,
            "ordinary_hazard": ordinary_hazard,
            "new_observation": new_observation,
            "target_source": source,
            "target_track_id": obstacle.get("track_id"),
            "measurement_frame_id": obstacle.get("measurement_frame_id"),
            "bearing_deg": obstacle.get("bearing_deg"),
            "reason": self.hazard_reason(metrics, immediate_hazard),
        }

    def is_new_observation(self, obstacle):
        frame_id = obstacle.get("measurement_frame_id")
        if frame_id is None or self.last_hazard is None:
            return True
        return frame_id != self.last_hazard.get("measurement_frame_id")

    def is_same_hazard(self, obstacle):
        if self.last_hazard is None:
            return False
        if obstacle.get("source", "unknown") != self.last_hazard.get("source"):
            return False

        track_id = obstacle.get("track_id")
        old_track_id = self.last_hazard.get("track_id")
        if track_id not in (None, -1, -2) and old_track_id not in (None, -1, -2):
            return track_id == old_track_id

        try:
            bearing_change = abs(
                float(obstacle.get("bearing_deg", 0.0))
                - float(self.last_hazard.get("bearing_deg", 0.0))
            )
            distance_change = abs(
                float(obstacle["distance_m"])
                - float(self.last_hazard["distance_m"])
            )
        except (KeyError, TypeError, ValueError):
            return False
        return bearing_change <= 4.0 and distance_change <= 2.5

    def remember_hazard(self, obstacle):
        return {
            "source": obstacle.get("source", "unknown"),
            "track_id": obstacle.get("track_id"),
            "distance_m": obstacle.get("distance_m"),
            "bearing_deg": obstacle.get("bearing_deg", 0.0),
            "measurement_frame_id": obstacle.get("measurement_frame_id"),
        }

    def hazard_reason(self, metrics, immediate_hazard):
        if metrics["distance_m"] <= self.immediate_distance_m:
            return "critical_distance"
        if immediate_hazard or metrics["ttc_s"] <= self.ttc_threshold_s:
            return "short_ttc"
        if metrics["distance_m"] <= self.hard_distance_m:
            return "short_distance"
        if (
            metrics["required_deceleration_mps2"]
            >= self.required_deceleration_threshold_mps2
        ):
            return "required_deceleration"
        return None

    def calculate_metrics(self, obstacle):
        """Engel ölçümünden TTC ve gerekli yavaşlama değerlerini hesaplar."""
        if not isinstance(obstacle, dict):
            return None
        try:
            distance = float(obstacle["distance_m"])
            relative_speed = float(obstacle["relative_speed_mps"])
        except (KeyError, TypeError, ValueError):
            return None

        if not math.isfinite(distance) or not math.isfinite(relative_speed):
            return None
        if distance <= 0.0:
            return None

        closing_speed = max(0.0, -relative_speed)
        ttc = distance / closing_speed if closing_speed > 0.10 else math.inf
        usable_distance = max(distance - self.stopping_clearance_m, 0.10)
        required_deceleration = closing_speed**2 / (2.0 * usable_distance)
        return {
            "distance_m": distance,
            "ttc_s": ttc,
            "required_deceleration_mps2": required_deceleration,
        }

    def calculate_risk_priority(self, metrics):
        """En kritik adayın önce seçilmesi için sıralama değeri üretir."""
        immediate = (
            metrics["distance_m"] <= self.immediate_distance_m
            or metrics["ttc_s"] <= self.immediate_ttc_s
        )
        ordinary = (
            metrics["distance_m"] <= self.hard_distance_m
            or metrics["ttc_s"] <= self.ttc_threshold_s
            or metrics["required_deceleration_mps2"]
            >= self.required_deceleration_threshold_mps2
        )
        if immediate:
            risk_level = 0
        elif ordinary:
            risk_level = 1
        else:
            risk_level = 2
        return risk_level, metrics["ttc_s"], metrics["distance_m"]

    def empty_info(self):
        return {
            "ttc_s": None,
            "required_deceleration_mps2": None,
            "hazard_count": 0,
            "target_source": None,
            "distance_m": None,
            "measurement_frame_id": None,
            "reason": None,
            "new_observation": False,
        }

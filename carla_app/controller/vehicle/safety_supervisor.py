"""Carpisma suresine gore bagimsiz acil fren denetimi."""

import math


class EmergencyBrakeSupervisor:
    """Yalnizca carpisma riski dogrulanirsa tam fren ister."""

    def __init__(self):
        self.stopping_clearance_m = 2.0
        self.hard_distance_m = 1.25
        self.immediate_distance_m = 0.65
        self.ttc_threshold_s = 1.50
        self.immediate_ttc_s = 0.80
        self.required_deceleration_threshold_mps2 = 4.0
        self.confirmation_ticks = 2
        self.hazard_count = 0

    def evaluate_candidates(self, *obstacles):
        candidates = []
        for obstacle in obstacles:
            metrics = self.calculate_metrics(obstacle)
            if metrics is not None:
                candidates.append((obstacle, metrics))

        if not candidates:
            self.hazard_count = 0
            return False, self.empty_info()

        obstacle, metrics = min(
            candidates,
            key=lambda item: self.calculate_risk_priority(item[1]),
        )
        emergency, info = self.evaluate_metrics(metrics)
        info["target_source"] = obstacle.get("source", "unknown")
        return emergency, info

    def evaluate(self, obstacle):
        metrics = self.calculate_metrics(obstacle)
        if metrics is None:
            self.hazard_count = 0
            return False, self.empty_info()
        return self.evaluate_metrics(metrics)

    def evaluate_metrics(self, metrics):
        immediate_hazard = (
            metrics["distance_m"] <= self.immediate_distance_m
            or metrics["ttc_s"] <= self.immediate_ttc_s
        )
        ordinary_hazard = (
            metrics["distance_m"] <= self.hard_distance_m
            or metrics["ttc_s"] <= self.ttc_threshold_s
            or metrics["required_deceleration_mps2"]
            >= self.required_deceleration_threshold_mps2
        )

        if ordinary_hazard:
            self.hazard_count += 1
        else:
            self.hazard_count = max(0, self.hazard_count - 1)

        emergency = immediate_hazard or self.hazard_count >= self.confirmation_ticks
        return emergency, {
            "ttc_s": metrics["ttc_s"],
            "required_deceleration_mps2": metrics["required_deceleration_mps2"],
            "hazard_count": self.hazard_count,
        }

    def calculate_metrics(self, obstacle):
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
        risk_level = 0 if immediate else 1 if ordinary else 2
        return risk_level, metrics["ttc_s"], metrics["distance_m"]

    def empty_info(self):
        return {
            "ttc_s": None,
            "required_deceleration_mps2": None,
            "hazard_count": 0,
            "target_source": None,
        }

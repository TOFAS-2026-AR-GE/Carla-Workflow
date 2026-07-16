import math


class EmergencyBrakeSupervisor:
    """
    Normal ACC'den bagimsiz son guvenlik katmani.

    TTC ve gerekli yavaslama miktarini denetler.
    """

    def __init__(self):
        # ACC aims for a 2 m standstill gap. The emergency layer reserves the
        # same clearance while calculating stopping distance, but only uses a
        # smaller physical-clearance limit after the vehicles are stationary.
        # This keeps AEB from fighting the normal HOLD brake at 2 m.
        self.stopping_clearance_m = 2.0
        self.hard_distance_m = 1.25
        self.immediate_distance_m = 0.65

        self.ttc_threshold_s = 1.50
        self.immediate_ttc_s = 0.80

        self.required_deceleration_threshold_mps2 = 4.0

        # Tek bir gurultulu frame tam fren yaptirmasin.
        # Fakat cok yakin tehlikede bekleme yok.
        self.hazard_count = 0
        self.confirmation_ticks = 2

    def evaluate_candidates(self, *obstacles):
        """Evaluate the most urgent valid obstacle without hiding another one."""
        candidates = [obstacle for obstacle in obstacles if obstacle is not None]
        target = min(candidates, key=self._priority, default=None)
        emergency, info = self.evaluate(target)
        info["target_source"] = (
            target.get("source", "unknown") if isinstance(target, dict) else None
        )
        return emergency, info

    def _priority(self, obstacle):
        try:
            distance = float(obstacle["distance_m"])
            relative_speed = float(obstacle["relative_speed_mps"])
        except (KeyError, TypeError, ValueError):
            return 3, math.inf, math.inf

        if not math.isfinite(distance) or not math.isfinite(relative_speed):
            return 3, math.inf, math.inf

        closing_speed = max(0.0, -relative_speed)
        ttc = distance / closing_speed if closing_speed > 0.1 else math.inf
        usable_distance = max(distance - self.stopping_clearance_m, 0.10)
        required_deceleration = closing_speed**2 / (2.0 * usable_distance)

        immediate = distance <= self.immediate_distance_m or ttc <= self.immediate_ttc_s
        ordinary = (
            distance <= self.hard_distance_m
            or ttc <= self.ttc_threshold_s
            or required_deceleration >= self.required_deceleration_threshold_mps2
        )
        risk_level = 0 if immediate else 1 if ordinary else 2
        return risk_level, ttc, distance

    def evaluate(self, lead_vehicle):
        if lead_vehicle is None:
            self.hazard_count = 0

            return False, {
                "ttc_s": None,
                "required_deceleration_mps2": None,
                "hazard_count": 0,
            }

        try:
            distance = float(lead_vehicle["distance_m"])
            relative_speed = float(lead_vehicle["relative_speed_mps"])
        except (KeyError, TypeError, ValueError):
            self.hazard_count = 0
            return False, {
                "ttc_s": None,
                "required_deceleration_mps2": None,
                "hazard_count": 0,
            }

        if not math.isfinite(distance) or not math.isfinite(relative_speed):
            self.hazard_count = 0
            return False, {
                "ttc_s": None,
                "required_deceleration_mps2": None,
                "hazard_count": 0,
            }

        closing_speed = max(
            0.0,
            -relative_speed,
        )

        if closing_speed > 0.10:
            ttc = distance / closing_speed
        else:
            ttc = math.inf

        usable_distance = max(
            distance - self.stopping_clearance_m,
            0.10,
        )

        required_deceleration = closing_speed**2 / (2.0 * usable_distance)

        immediate_hazard = (
            distance <= self.immediate_distance_m or ttc <= self.immediate_ttc_s
        )

        ordinary_hazard = (
            distance <= self.hard_distance_m
            or ttc <= self.ttc_threshold_s
            or required_deceleration >= self.required_deceleration_threshold_mps2
        )

        if ordinary_hazard:
            self.hazard_count += 1
        else:
            self.hazard_count = max(
                0,
                self.hazard_count - 1,
            )

        emergency = immediate_hazard or self.hazard_count >= self.confirmation_ticks

        return emergency, {
            "ttc_s": ttc,
            "required_deceleration_mps2": (required_deceleration),
            "hazard_count": (self.hazard_count),
        }

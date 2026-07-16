import math


class EmergencyBrakeSupervisor:
    """
    Normal ACC'den bagimsiz son guvenlik katmani.

    TTC ve gerekli yavaslama miktarini denetler.
    """

    def __init__(self):
        self.hard_distance_m = 2.0
        self.immediate_distance_m = 1.25

        self.ttc_threshold_s = 1.10
        self.immediate_ttc_s = 0.60

        self.required_deceleration_threshold_mps2 = (
            6.0
        )

        # Tek bir gurultulu frame tam fren yaptirmasin.
        # Fakat cok yakin tehlikede bekleme yok.
        self.hazard_count = 0
        self.confirmation_ticks = 2

    def evaluate(self, lead_vehicle):
        if lead_vehicle is None:
            self.hazard_count = 0

            return False, {
                "ttc_s": None,
                "required_deceleration_mps2": None,
                "hazard_count": 0,
            }

        distance = float(
            lead_vehicle["distance_m"]
        )

        relative_speed = float(
            lead_vehicle[
                "relative_speed_mps"
            ]
        )

        closing_speed = max(
            0.0,
            -relative_speed,
        )

        if closing_speed > 0.10:
            ttc = (
                distance / closing_speed
            )
        else:
            ttc = math.inf

        usable_distance = max(
            distance - self.hard_distance_m,
            0.10,
        )

        required_deceleration = (
            closing_speed**2
            / (
                2.0
                * usable_distance
            )
        )

        immediate_hazard = (
            distance
            <= self.immediate_distance_m
            or ttc
            <= self.immediate_ttc_s
        )

        ordinary_hazard = (
            distance
            <= self.hard_distance_m
            or ttc
            <= self.ttc_threshold_s
            or required_deceleration
            >= self.required_deceleration_threshold_mps2
        )

        if ordinary_hazard:
            self.hazard_count += 1
        else:
            self.hazard_count = max(
                0,
                self.hazard_count - 1,
            )

        emergency = (
            immediate_hazard
            or self.hazard_count
            >= self.confirmation_ticks
        )

        return emergency, {
            "ttc_s": ttc,
            "required_deceleration_mps2": (
                required_deceleration
            ),
            "hazard_count": (
                self.hazard_count
            ),
        }
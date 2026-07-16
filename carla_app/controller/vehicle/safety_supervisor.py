import math


class EmergencyBrakeSupervisor:
    """TTC tabanlı son güvenlik katmanı."""

    def __init__(self):
        self.hard_distance_m = 3.0
        self.ttc_threshold_s = 1.40

        self.required_deceleration_threshold_mps2 = (
            5.5
        )

    def evaluate(
        self,
        lead_vehicle,
    ):
        if lead_vehicle is None:
            return False, {
                "ttc_s": None,
                "required_deceleration_mps2": None,
            }

        distance = lead_vehicle["distance_m"]

        relative_speed = lead_vehicle[
            "relative_speed_mps"
        ]

        # relative_speed negatifse ego araç yaklaşıyor.
        closing_speed = max(
            0.0,
            -relative_speed,
        )

        if closing_speed > 0.10:
            ttc = (
                distance
                / closing_speed
            )
        else:
            ttc = math.inf

        usable_distance = max(
            distance - self.hard_distance_m,
            0.10,
        )

        required_deceleration = (
            closing_speed**2
            / (2.0 * usable_distance)
        )

        emergency = (
            distance <= self.hard_distance_m

            or ttc <= self.ttc_threshold_s

            or required_deceleration
            >= (
                self
                .required_deceleration_threshold_mps2
            )
        )

        return emergency, {
            "ttc_s": ttc,
            "required_deceleration_mps2": (
                required_deceleration
            ),
        }
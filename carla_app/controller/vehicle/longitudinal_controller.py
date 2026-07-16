def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


class LongitudinalController:
    """PI hız kontrolü ve CBF takip filtresi."""

    def __init__(self, dt=0.05):
        self.dt = dt

        # PI cruise control.
        self.kp = 0.70
        self.ki = 0.12

        self.speed_deadband_mps = 0.20
        self.integral_limit = 8.0
        self.integral_error = 0.0

        # Sabit zaman aralıklı takip politikası.
        self.standstill_gap_m = 4.0
        self.time_headway_s = 1.50

        # CBF güvenlik katsayısı.
        self.cbf_alpha = 0.80

        self.max_acceleration_mps2 = 2.0
        self.max_deceleration_mps2 = 4.0

        # Konfor için jerk sınırları.
        self.acceleration_jerk_limit = 1.5
        self.braking_jerk_limit = 3.0

        self.previous_acceleration = 0.0

    def run_step(
        self,
        state,
        lead_vehicle,
        target_speed,
    ):
        nominal_acceleration, integral_candidate = (
            self._cruise_acceleration(
                current_speed=state["speed_mps"],
                target_speed=target_speed,
            )
        )

        safe_acceleration_limit = float("inf")
        desired_gap = None
        gap_error = None

        if lead_vehicle is not None:
            desired_gap = (
                self.standstill_gap_m
                + self.time_headway_s
                * state["speed_mps"]
            )

            gap_error = (
                lead_vehicle["distance_m"]
                - desired_gap
            )

            # CBF:
            #
            # h = distance - d0 - T*v
            #
            # h_dot + alpha*h >= 0
            #
            # Buradan ego ivmesi için üst sınır çıkar.
            safe_acceleration_limit = (
                lead_vehicle[
                    "relative_speed_mps"
                ]
                + self.cbf_alpha * gap_error
            ) / self.time_headway_s

        # Nominal PI komutunun güvenli olmasına izin ver.
        # Güvensizse CBF sınırı komutu aşağı çeker.
        desired_acceleration = min(
            nominal_acceleration,
            safe_acceleration_limit,
        )

        desired_acceleration = clamp(
            desired_acceleration,
            -self.max_deceleration_mps2,
            self.max_acceleration_mps2,
        )

        safety_filter_active = (
            lead_vehicle is not None
            and safe_acceleration_limit
            < nominal_acceleration
        )

        # Öndeki araç yüzünden yavaşlarken PI integratörünü
        # şişirmiyoruz. Yoksa araç kaybolunca ani gaz verir.
        if not safety_filter_active:
            self.integral_error = integral_candidate
        else:
            self.integral_error *= 0.995

        acceleration = self._limit_jerk(
            desired_acceleration
        )

        throttle, brake = self._to_pedals(
            acceleration
        )

        if lead_vehicle is None:
            mode = "CRUISE"
        elif safety_filter_active:
            mode = "FOLLOW"
        else:
            mode = "LEAD_FAR"

        info = {
            "mode": mode,
            "target_speed_mps": target_speed,
            "nominal_acceleration_mps2": (
                nominal_acceleration
            ),
            "safe_acceleration_limit_mps2": (
                safe_acceleration_limit
                if lead_vehicle is not None
                else None
            ),
            "acceleration_mps2": acceleration,
            "desired_gap_m": desired_gap,
            "gap_error_m": gap_error,
        }

        return throttle, brake, info

    def _cruise_acceleration(
        self,
        current_speed,
        target_speed,
    ):
        speed_error = (
            target_speed
            - current_speed
        )

        if (
            abs(speed_error)
            < self.speed_deadband_mps
        ):
            proportional_error = 0.0
        else:
            proportional_error = speed_error

        integral_candidate = clamp(
            self.integral_error
            + speed_error * self.dt,
            -self.integral_limit,
            self.integral_limit,
        )

        acceleration = (
            self.kp * proportional_error
            + self.ki * integral_candidate
        )

        acceleration = clamp(
            acceleration,
            -self.max_deceleration_mps2,
            self.max_acceleration_mps2,
        )

        return (
            acceleration,
            integral_candidate,
        )

    def _limit_jerk(
        self,
        desired_acceleration,
    ):
        if (
            desired_acceleration
            >= self.previous_acceleration
        ):
            maximum_change = (
                self.acceleration_jerk_limit
                * self.dt
            )
        else:
            maximum_change = (
                self.braking_jerk_limit
                * self.dt
            )

        acceleration_change = clamp(
            desired_acceleration
            - self.previous_acceleration,
            -maximum_change,
            maximum_change,
        )

        acceleration = (
            self.previous_acceleration
            + acceleration_change
        )

        self.previous_acceleration = acceleration

        return acceleration

    def _to_pedals(
        self,
        acceleration,
    ):
        if acceleration > 0.05:
            throttle = (
                acceleration
                / self.max_acceleration_mps2
            )

            return (
                clamp(
                    throttle,
                    0.0,
                    0.75,
                ),
                0.0,
            )

        if acceleration < -0.05:
            brake = (
                abs(acceleration)
                / self.max_deceleration_mps2
            )

            return (
                0.0,
                clamp(
                    brake,
                    0.0,
                    0.80,
                ),
            )

        return 0.0, 0.0
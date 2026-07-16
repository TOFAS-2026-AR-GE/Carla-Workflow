def clamp(value, minimum, maximum):
    return max(
        minimum,
        min(value, maximum),
    )


class LongitudinalController:
    """
    PI cruise control
    + constant-time-headway ACC
    + CBF safety filter.
    """

    def __init__(self, dt=0.05):
        self.dt = float(dt)

        # Serbest yol PI hiz kontrolu.
        self.kp = 0.70
        self.ki = 0.10

        self.speed_deadband_mps = 0.15
        self.integral_limit = 6.0
        self.integral_error = 0.0

        # Sabit zaman aralikli takip:
        #
        # desired_gap =
        # standstill_gap + headway * ego_speed
        #
        # 3 metre durus mesafesidir.
        # 30 km/h hizda hareketli takip mesafesi degildir.
        self.standstill_gap_m = 3.0
        self.time_headway_s = 0.90

        # ACC geri besleme.
        self.gap_gain = 0.28
        self.relative_speed_gain = 0.85

        # Uzak aracin gereksiz yere kontrole
        # girmesini engelleyen dinamik aktivasyon.
        self.activation_margin_m = 4.0
        self.activation_reaction_time_s = 1.0
        self.comfortable_deceleration_mps2 = 2.5

        # FOLLOW ve LEAD_FAR arasinda titremeyi engeller.
        self.exit_hysteresis_m = 4.0

        # CBF:
        #
        # h = distance - d0 - T*v
        # h_dot + alpha*h >= 0
        self.cbf_alpha = 0.65

        self.max_acceleration_mps2 = 2.0
        self.max_deceleration_mps2 = 4.5

        self.acceleration_jerk_limit = 1.3
        self.braking_jerk_limit = 2.8

        self.previous_acceleration = 0.0

        self.following_active = False

        # Yeni bir track tek frame gorulunce
        # normal ACC'ye girme.
        self.candidate_track_id = None
        self.candidate_seen_ticks = 0
        self.minimum_confirmation_ticks = 2

    def run_step(
        self,
        state,
        lead_vehicle,
        target_speed,
    ):
        current_speed = float(
            state["speed_mps"]
        )

        (
            nominal_acceleration,
            integral_candidate,
        ) = self._cruise_acceleration(
            current_speed=current_speed,
            target_speed=float(target_speed),
        )

        lead = self._validate_lead(
            lead_vehicle
        )

        lead_confirmed = (
            self._update_lead_confirmation(
                lead
            )
        )

        desired_gap = None
        gap_error = None
        activation_distance = None
        following_acceleration = None
        safe_acceleration_limit = None

        if lead is not None:
            distance = lead["distance_m"]
            relative_speed = lead[
                "relative_speed_mps"
            ]

            # relative_speed negatifse
            # ego arac ondeki araca yaklasiyor.
            closing_speed = max(
                0.0,
                -relative_speed,
            )

            desired_gap = (
                self.standstill_gap_m
                + self.time_headway_s
                * current_speed
            )

            gap_error = (
                distance - desired_gap
            )

            activation_distance = (
                self._activation_distance(
                    desired_gap,
                    closing_speed,
                )
            )

            if not self.following_active:
                self.following_active = (
                    lead_confirmed
                    and distance
                    <= activation_distance
                )
            else:
                should_exit = (
                    distance
                    > activation_distance
                    + self.exit_hysteresis_m
                    and closing_speed < 0.5
                )

                if should_exit:
                    self.following_active = False

            if self.following_active:
                following_acceleration = (
                    self.gap_gain
                    * gap_error
                    + self.relative_speed_gain
                    * relative_speed
                )

                # CBF'den ego ivmesine gelen
                # guvenli ust sinir.
                safe_acceleration_limit = (
                    relative_speed
                    + self.cbf_alpha
                    * gap_error
                ) / self.time_headway_s
        else:
            self.following_active = False

        desired_acceleration = (
            nominal_acceleration
        )

        if (
            self.following_active
            and following_acceleration
            is not None
        ):
            desired_acceleration = min(
                desired_acceleration,
                following_acceleration,
                safe_acceleration_limit,
            )

        desired_acceleration = clamp(
            desired_acceleration,
            -self.max_deceleration_mps2,
            self.max_acceleration_mps2,
        )

        # ACC araci sinirlarken PI integratorunun
        # sisip arac kaybolunca ani gaz vermesini engelle.
        if not self.following_active:
            self.integral_error = (
                integral_candidate
            )
        else:
            self.integral_error *= 0.98

        acceleration = self._limit_jerk(
            desired_acceleration
        )

        throttle, brake = self._to_pedals(
            acceleration
        )

        if lead is None:
            mode = "CRUISE"
        elif self.following_active:
            mode = "FOLLOW"
        else:
            mode = "LEAD_FAR"

        info = {
            "mode": mode,
            "target_speed_mps": float(
                target_speed
            ),
            "nominal_acceleration_mps2": (
                nominal_acceleration
            ),
            "following_acceleration_mps2": (
                following_acceleration
            ),
            "safe_acceleration_limit_mps2": (
                safe_acceleration_limit
            ),
            "acceleration_mps2": acceleration,
            "desired_gap_m": desired_gap,
            "gap_error_m": gap_error,
            "activation_distance_m": (
                activation_distance
            ),
            "lead_confirmed_for_control": (
                lead_confirmed
            ),
        }

        return throttle, brake, info

    def _activation_distance(
        self,
        desired_gap,
        closing_speed,
    ):
        # Goreli fren mesafesi:
        # delta_v^2 / (2 * comfortable_decel)
        relative_braking_distance = (
            closing_speed**2
            / (
                2.0
                * self.comfortable_deceleration_mps2
            )
        )

        return (
            desired_gap
            + self.activation_margin_m
            + self.activation_reaction_time_s
            * closing_speed
            + relative_braking_distance
        )

    def _validate_lead(
        self,
        lead_vehicle,
    ):
        if lead_vehicle is None:
            return None

        distance = float(
            lead_vehicle.get(
                "distance_m",
                -1.0,
            )
        )

        relative_speed = float(
            lead_vehicle.get(
                "relative_speed_mps",
                0.0,
            )
        )

        if not 0.5 < distance <= 80.0:
            return None

        if not -20.0 <= relative_speed <= 20.0:
            return None

        return {
            **lead_vehicle,
            "distance_m": distance,
            "relative_speed_mps": (
                relative_speed
            ),
        }

    def _update_lead_confirmation(
        self,
        lead,
    ):
        if lead is None:
            self.candidate_track_id = None
            self.candidate_seen_ticks = 0
            return False

        track_id = lead.get("track_id")

        if track_id == self.candidate_track_id:
            self.candidate_seen_ticks += 1
        else:
            self.candidate_track_id = track_id
            self.candidate_seen_ticks = 1

        return (
            self.candidate_seen_ticks
            >= self.minimum_confirmation_ticks
        )

    def _cruise_acceleration(
        self,
        current_speed,
        target_speed,
    ):
        speed_error = (
            target_speed - current_speed
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
            self.kp
            * proportional_error
            + self.ki
            * integral_candidate
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

        self.previous_acceleration = (
            acceleration
        )

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
                    0.85,
                ),
            )

        return 0.0, 0.0
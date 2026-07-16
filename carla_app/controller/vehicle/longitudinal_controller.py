"""Cruise control and adaptive following controller."""

import math


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


class LongitudinalController:
    """Cruise, normal following and stable low-speed stop-and-go control."""

    def __init__(self, dt=0.05):
        self.dt = float(dt)

        # Open-road cruise controller.
        self.cruise_kp = 0.70
        self.cruise_ki = 0.10
        self.speed_deadband_mps = 0.15
        self.integral_limit = 6.0
        self.integral_error = 0.0

        # Normal moving ACC. The 2 m value is only the standstill clearance;
        # the time gap adds more distance while the ego vehicle is moving.
        self.standstill_gap_m = 2.0
        self.time_headway_s = 0.9
        self.gap_gain = 0.28
        self.relative_speed_gain = 0.85
        self.cbf_alpha = 0.65

        # Low-speed traffic uses a target approach speed instead of directly
        # alternating between a distance command and a relative-speed command.
        self.stop_go_entry_ego_speed_mps = 2.5
        self.stop_go_entry_lead_speed_mps = 1.5
        self.stop_go_exit_lead_speed_mps = 2.5
        self.stop_go_entry_distance_m = 12.0
        self.stop_go_exit_distance_m = 14.0
        self.approach_gap_gain = 0.38
        self.approach_speed_gain = 1.10
        self.approach_speed_deadband_mps = 0.05
        self.maximum_closing_speed_mps = 1.20
        self.approach_resume_lead_speed_mps = 0.80
        self.minimum_creep_speed_mps = 0.20
        self.maximum_creep_speed_mps = 0.30
        self.minimum_creep_throttle = 0.12
        self.approach_acceleration_deadband_mps2 = 0.06

        # HOLD is latched at the stop line. RESTART can only be entered after
        # HOLD, so a noisy large range cannot create a false restart command.
        self.hold_entry_speed_mps = 0.20
        self.hold_entry_distance_m = 2.30
        self.hold_lead_speed_mps = 0.30
        self.hold_brake = 0.25
        self.restart_lead_speed_mps = 0.40
        self.slow_restart_lead_speed_mps = 0.15
        self.slow_restart_gap_m = 2.50
        self.restart_gap_m = 2.90
        self.restart_confirmation_ticks = 2
        self.restart_acceleration_mps2 = 1.30
        self.restart_end_speed_mps = 1.0

        # Following activation remains dynamic at road speed.
        self.activation_margin_m = 4.0
        self.activation_reaction_time_s = 1.0
        self.comfortable_deceleration_mps2 = 2.5
        self.exit_hysteresis_m = 4.0

        # Actuator limits.
        self.max_acceleration_mps2 = 2.0
        self.max_deceleration_mps2 = 4.5
        self.acceleration_jerk_limit_mps3 = 1.3
        self.approach_jerk_limit_mps3 = 2.0
        self.restart_jerk_limit_mps3 = 4.0
        self.braking_jerk_limit_mps3 = 2.8
        self.previous_acceleration = 0.0

        # Persistent controller state.
        self.following_active = False
        self.stop_go_active = False
        self.hold_active = False
        self.restart_active = False
        self.restart_evidence_ticks = 0
        # DRIVE may accelerate, BRAKE never returns to throttle for the same
        # stationary target, and CREEP is only an early-stop fallback.
        self.approach_phase = "DRIVE"

        self.candidate_track_id = None
        self.candidate_distance_m = None
        self.candidate_seen_ticks = 0
        self.minimum_confirmation_ticks = 2

        # A small asymmetric filter suppresses camera/radar source jumps. A
        # closer measurement is accepted faster than a farther measurement.
        self.filtered_distance_m = None
        self.filtered_lead_speed_mps = None

    def run_step(self, state, lead_vehicle, target_speed):
        current_speed = max(0.0, float(state["speed_mps"]))
        target_speed = max(0.0, float(target_speed))
        nominal_acceleration, integral_candidate = self._cruise_acceleration(
            current_speed=current_speed,
            target_speed=target_speed,
        )

        raw_lead = self._validate_lead(lead_vehicle)
        lead_confirmed = self._update_lead_confirmation(raw_lead)
        lead = self._filter_lead(raw_lead, current_speed)

        desired_gap = None
        gap_error = None
        activation_distance = None
        following_acceleration = None
        safe_acceleration_limit = None
        approach_target_speed = None

        if lead is None:
            self._clear_following_state()
        else:
            distance = lead["distance_m"]
            relative_speed = lead["relative_speed_mps"]
            lead_speed = lead["lead_speed_mps"]
            closing_speed = max(0.0, -relative_speed)

            desired_gap = self.standstill_gap_m + self.time_headway_s * current_speed
            gap_error = distance - desired_gap
            activation_distance = self._activation_distance(desired_gap, closing_speed)
            stop_go_candidate = (
                current_speed <= self.stop_go_entry_ego_speed_mps
                and lead_speed <= self.stop_go_entry_lead_speed_mps
                and distance <= self.stop_go_entry_distance_m
            )

            if not self.following_active:
                self.following_active = lead_confirmed and (
                    distance <= activation_distance or stop_go_candidate
                )
            elif (
                distance > activation_distance + self.exit_hysteresis_m
                and closing_speed < 0.5
            ):
                self._clear_following_state()

            if self.following_active:
                safe_acceleration_limit = (
                    relative_speed + self.cbf_alpha * gap_error
                ) / self.time_headway_s

                self._update_stop_go_active(
                    current_speed=current_speed,
                    lead_speed=lead_speed,
                    distance=distance,
                )

                if self.stop_go_active:
                    self._update_hold_and_restart(
                        current_speed=current_speed,
                        distance=distance,
                        lead_speed=lead_speed,
                        measured_lead_speed=lead["measured_lead_speed_mps"],
                    )
                    if (
                        self.approach_phase == "BRAKE"
                        and lead_speed >= self.approach_resume_lead_speed_mps
                    ):
                        self.approach_phase = "DRIVE"
                    approach_target_speed = self._approach_target_speed(
                        distance=distance,
                        lead_speed=lead_speed,
                        target_speed=target_speed,
                    )
                    if (
                        self.approach_phase == "BRAKE"
                        and current_speed <= 0.05
                        and distance > self.hold_entry_distance_m
                    ):
                        self.approach_phase = "CREEP"
                    if self.approach_phase == "CREEP":
                        approach_target_speed = clamp(
                            approach_target_speed,
                            self.minimum_creep_speed_mps,
                            self.maximum_creep_speed_mps,
                        )
                    following_acceleration = self._approach_acceleration(
                        current_speed=current_speed,
                        target_speed=approach_target_speed,
                    )
                    if self.restart_active:
                        following_acceleration = max(
                            following_acceleration,
                            self.restart_acceleration_mps2,
                        )
                else:
                    self._clear_stop_go_state()
                    following_acceleration = (
                        self.gap_gain * gap_error
                        + self.relative_speed_gain * relative_speed
                    )

        desired_acceleration = nominal_acceleration
        if self.following_active:
            desired_acceleration = min(
                nominal_acceleration,
                following_acceleration,
                safe_acceleration_limit,
            )

        desired_acceleration = clamp(
            desired_acceleration,
            -self.max_deceleration_mps2,
            self.max_acceleration_mps2,
        )

        if self.stop_go_active and abs(desired_acceleration) < (
            self.approach_acceleration_deadband_mps2
        ):
            desired_acceleration = 0.0

        stop_go_approach = (
            lead is not None
            and self.stop_go_active
            and not self.hold_active
            and not self.restart_active
        )
        if stop_go_approach:
            if (
                self.approach_phase == "DRIVE"
                and desired_acceleration < -self.approach_acceleration_deadband_mps2
            ):
                self.approach_phase = "BRAKE"
            if self.approach_phase == "BRAKE":
                desired_acceleration = min(0.0, desired_acceleration)
            elif (
                self.approach_phase == "CREEP"
                and lead["distance_m"] > self.hold_entry_distance_m
            ):
                desired_acceleration = max(0.0, desired_acceleration)

        if self.following_active:
            self.integral_error *= 0.98
        else:
            self.integral_error = integral_candidate

        if self.hold_active:
            self.previous_acceleration = 0.0
            acceleration = 0.0
            throttle, brake = 0.0, self.hold_brake
        else:
            acceleration_jerk_limit = None
            if self.restart_active:
                acceleration_jerk_limit = self.restart_jerk_limit_mps3
            elif self.stop_go_active:
                acceleration_jerk_limit = self.approach_jerk_limit_mps3

            acceleration = self._limit_jerk(
                desired_acceleration,
                acceleration_jerk_limit=acceleration_jerk_limit,
            )
            pedal_deadband = (
                self.approach_acceleration_deadband_mps2
                if self.stop_go_active
                else 0.05
            )
            minimum_throttle = (
                self.minimum_creep_throttle
                if self.approach_phase == "CREEP" and desired_acceleration > 0.0
                else 0.0
            )
            throttle, brake = self._to_pedals(
                acceleration,
                pedal_deadband,
                minimum_throttle,
            )

        mode = self._mode(lead)
        info = {
            "mode": mode,
            "target_speed_mps": target_speed,
            "nominal_acceleration_mps2": nominal_acceleration,
            "following_acceleration_mps2": following_acceleration,
            "safe_acceleration_limit_mps2": safe_acceleration_limit,
            "acceleration_mps2": acceleration,
            "desired_gap_m": desired_gap,
            "gap_error_m": gap_error,
            "activation_distance_m": activation_distance,
            "lead_confirmed_for_control": lead_confirmed,
            "raw_lead_distance_m": (
                raw_lead["distance_m"] if raw_lead is not None else None
            ),
            "filtered_lead_distance_m": (
                lead["distance_m"] if lead is not None else None
            ),
            "filtered_lead_speed_mps": (
                lead["lead_speed_mps"] if lead is not None else None
            ),
            "approach_target_speed_mps": approach_target_speed,
            "approach_phase": self.approach_phase,
            "stop_go_active": self.stop_go_active,
            "hold_active": self.hold_active,
            "restart_active": self.restart_active,
        }
        return throttle, brake, info

    def _update_stop_go_active(self, current_speed, lead_speed, distance):
        if self.hold_active or self.restart_active:
            self.stop_go_active = True
            return

        if self.stop_go_active:
            should_exit = (
                lead_speed >= self.stop_go_exit_lead_speed_mps
                or distance >= self.stop_go_exit_distance_m
            )
            if should_exit:
                self.stop_go_active = False
            return

        self.stop_go_active = (
            current_speed <= self.stop_go_entry_ego_speed_mps
            and lead_speed <= self.stop_go_entry_lead_speed_mps
            and distance <= self.stop_go_entry_distance_m
        )
        if self.stop_go_active:
            self.approach_phase = "DRIVE"

    def _update_hold_and_restart(
        self,
        current_speed,
        distance,
        lead_speed,
        measured_lead_speed,
    ):
        if self.hold_active:
            lead_started_moving = measured_lead_speed >= self.restart_lead_speed_mps
            slow_lead_opened_gap = (
                lead_speed >= self.slow_restart_lead_speed_mps
                and distance >= self.slow_restart_gap_m
            )
            gap_opened = distance >= self.restart_gap_m

            if lead_started_moving or slow_lead_opened_gap or gap_opened:
                self.restart_evidence_ticks += 1
            else:
                self.restart_evidence_ticks = 0

            if self.restart_evidence_ticks >= self.restart_confirmation_ticks:
                self.hold_active = False
                self.restart_active = True
                self.restart_evidence_ticks = 0
                self.approach_phase = "DRIVE"
            return

        if self.restart_active:
            lead_stopped_again = (
                distance <= self.hold_entry_distance_m + 0.15
                and lead_speed <= self.hold_lead_speed_mps
            )
            if current_speed >= self.restart_end_speed_mps or lead_stopped_again:
                self.restart_active = False

        can_hold = (
            not self.restart_active
            and current_speed <= self.hold_entry_speed_mps
            and distance <= self.hold_entry_distance_m
            and lead_speed <= self.hold_lead_speed_mps
        )
        if can_hold:
            self.hold_active = True
            self.restart_evidence_ticks = 0
            self.approach_phase = "HOLD"

    def _approach_target_speed(self, distance, lead_speed, target_speed):
        free_gap = max(0.0, distance - self.standstill_gap_m)
        closing_speed = min(
            self.maximum_closing_speed_mps,
            self.approach_gap_gain * free_gap,
        )
        return min(target_speed, lead_speed + closing_speed)

    def _approach_acceleration(self, current_speed, target_speed):
        speed_error = target_speed - current_speed
        if abs(speed_error) <= self.approach_speed_deadband_mps:
            return 0.0
        return self.approach_speed_gain * speed_error

    def _filter_lead(self, lead, current_speed):
        if lead is None:
            self._reset_lead_filter()
            return None

        measured_lead_speed = max(
            0.0,
            current_speed + lead["relative_speed_mps"],
        )
        if self.filtered_distance_m is None:
            self.filtered_distance_m = lead["distance_m"]
            self.filtered_lead_speed_mps = measured_lead_speed
        else:
            predicted_distance = max(
                0.5,
                self.filtered_distance_m
                + (self.filtered_lead_speed_mps - current_speed) * self.dt,
            )
            distance_error = lead["distance_m"] - predicted_distance

            if distance_error < 0.0:
                # Closer readings are safety-relevant and accepted quickly.
                limited_error = clamp(distance_error, -1.5, 0.0)
                distance_alpha = 0.55
            else:
                # Farther source-switch jumps are accepted slowly.
                limited_error = clamp(distance_error, 0.0, 0.40)
                distance_alpha = 0.20

            self.filtered_distance_m = (
                predicted_distance + distance_alpha * limited_error
            )
            self.filtered_lead_speed_mps = (
                0.75 * self.filtered_lead_speed_mps + 0.25 * measured_lead_speed
            )

        filtered = dict(lead)
        filtered["distance_m"] = self.filtered_distance_m
        filtered["lead_speed_mps"] = self.filtered_lead_speed_mps
        filtered["relative_speed_mps"] = self.filtered_lead_speed_mps - current_speed
        filtered["measured_lead_speed_mps"] = measured_lead_speed
        return filtered

    def _activation_distance(self, desired_gap, closing_speed):
        relative_braking_distance = closing_speed**2 / (
            2.0 * self.comfortable_deceleration_mps2
        )
        return (
            desired_gap
            + self.activation_margin_m
            + self.activation_reaction_time_s * closing_speed
            + relative_braking_distance
        )

    @staticmethod
    def _validate_lead(lead_vehicle):
        if lead_vehicle is None:
            return None

        try:
            distance = float(lead_vehicle.get("distance_m", -1.0))
            relative_speed = float(lead_vehicle.get("relative_speed_mps", 0.0))
        except (TypeError, ValueError):
            return None

        if not math.isfinite(distance) or not math.isfinite(relative_speed):
            return None
        if not 0.5 < distance <= 80.0:
            return None
        if not -20.0 <= relative_speed <= 20.0:
            return None

        return {
            **lead_vehicle,
            "distance_m": distance,
            "relative_speed_mps": relative_speed,
        }

    def _update_lead_confirmation(self, lead):
        if lead is None:
            self.candidate_track_id = None
            self.candidate_distance_m = None
            self.candidate_seen_ticks = 0
            return False

        track_id = lead.get("track_id")
        same_track = track_id == self.candidate_track_id
        same_physical_target = (
            self.candidate_distance_m is not None
            and abs(lead["distance_m"] - self.candidate_distance_m) <= 3.0
        )

        if same_track or same_physical_target:
            self.candidate_seen_ticks += 1
        else:
            self.candidate_seen_ticks = 1

        self.candidate_track_id = track_id
        self.candidate_distance_m = lead["distance_m"]
        return self.candidate_seen_ticks >= self.minimum_confirmation_ticks

    def _cruise_acceleration(self, current_speed, target_speed):
        speed_error = target_speed - current_speed
        proportional_error = (
            0.0 if abs(speed_error) < self.speed_deadband_mps else speed_error
        )
        integral_candidate = clamp(
            self.integral_error + speed_error * self.dt,
            -self.integral_limit,
            self.integral_limit,
        )
        acceleration = (
            self.cruise_kp * proportional_error + self.cruise_ki * integral_candidate
        )
        acceleration = clamp(
            acceleration,
            -self.max_deceleration_mps2,
            self.max_acceleration_mps2,
        )
        return acceleration, integral_candidate

    def _limit_jerk(self, desired_acceleration, acceleration_jerk_limit=None):
        if desired_acceleration >= self.previous_acceleration:
            jerk_limit = (
                self.acceleration_jerk_limit_mps3
                if acceleration_jerk_limit is None
                else float(acceleration_jerk_limit)
            )
        else:
            jerk_limit = self.braking_jerk_limit_mps3

        maximum_change = jerk_limit * self.dt
        acceleration_change = clamp(
            desired_acceleration - self.previous_acceleration,
            -maximum_change,
            maximum_change,
        )
        self.previous_acceleration += acceleration_change
        return self.previous_acceleration

    def _to_pedals(self, acceleration, deadband, minimum_throttle=0.0):
        if acceleration > deadband:
            throttle = acceleration / self.max_acceleration_mps2
            throttle = max(float(minimum_throttle), throttle)
            return clamp(throttle, 0.0, 0.75), 0.0
        if acceleration < -deadband:
            brake = abs(acceleration) / self.max_deceleration_mps2
            return 0.0, clamp(brake, 0.0, 0.85)
        return 0.0, 0.0

    def _mode(self, lead):
        if lead is None:
            return "CRUISE"
        if self.hold_active:
            return "HOLD"
        if self.restart_active:
            return "RESTART"
        if self.stop_go_active and self.following_active:
            return "APPROACH"
        if self.following_active:
            return "FOLLOW"
        return "LEAD_FAR"

    def _clear_stop_go_state(self):
        self.stop_go_active = False
        self.hold_active = False
        self.restart_active = False
        self.restart_evidence_ticks = 0
        self.approach_phase = "DRIVE"

    def _clear_following_state(self):
        self.following_active = False
        self._clear_stop_go_state()

    def _reset_lead_filter(self):
        self.filtered_distance_m = None
        self.filtered_lead_speed_mps = None

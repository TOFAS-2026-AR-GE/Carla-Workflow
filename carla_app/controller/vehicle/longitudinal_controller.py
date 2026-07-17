"""IDM tabanlı araç takip, gaz ve fren kontrolünü yapar.

Girdi olarak ego hızı, hedef hız ve seçilen ön araç gelir. Çıktı her zaman
birbirini dışlayan gaz veya fren değeridir; ikisi aynı anda verilmez.
"""

import math


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


class LongitudinalController:
    """Sabit hız, araç takibi ve durduğu yerde fren tutmayı yönetir."""

    def __init__(self, dt=0.05):
        self.dt = float(dt)

        # IDM ayarları. Her sayının fiziksel bir anlamı vardır.
        self.standstill_gap_m = 2.0
        self.time_headway_s = 1.2
        self.maximum_acceleration_mps2 = 1.5
        self.comfortable_deceleration_mps2 = 2.0
        self.acceleration_exponent = 4.0

        # CARLA pedal sınırları ve konfor sınırları.
        self.maximum_deceleration_mps2 = 5.0
        self.acceleration_jerk_mps3 = 3.0
        self.braking_jerk_mps3 = 6.0
        self.throttle_acceleration_mps2 = 3.0
        self.rolling_resistance_throttle = 0.05
        self.coasting_deceleration_mps2 = (
            self.throttle_acceleration_mps2 * self.rolling_resistance_throttle
        )
        self.brake_deadband_mps2 = 0.05
        self.breakaway_speed_mps = 0.50
        self.low_speed_throttle_floor = 0.20
        self.maximum_throttle = 0.75
        self.maximum_brake = 0.90
        self.previous_acceleration_mps2 = 0.0

        # Dur-kalk içindeki tek kesikli durum mekanik fren tutmadır.
        # Normal yaklaşma ve yeniden kalkış ivmesini her zaman IDM hesaplar.
        self.hold_speed_mps = 0.15
        self.hold_distance_m = 2.30
        self.hold_lead_speed_mps = 0.20
        self.hold_brake = 0.30
        self.hold_release_lead_speed_mps = 0.30
        self.hold_release_distance_m = 2.60
        self.hold_release_ticks = 2
        self.hold_active = False
        self.release_evidence_ticks = 0
        self.restart_ticks_remaining = 0

        # Normal adaptif hız kontrolü için iki ardışık ölçüm yeterlidir.
        # Acil fren bağımsızdır ve ilk kritik radar dönüşüne tepki verebilir.
        self.minimum_lead_ticks = 2
        self.lead_ticks = 0
        self.last_track_id = None
        self.last_raw_distance_m = None

        # Yakınlaşan ölçüm hemen, uzaklaşan kaynak değişimi daha yavaş kabul
        # edilir. Bu sayede kamera-radar geçişleri gaz-fren sıçraması üretmez.
        self.filtered_distance_m = None
        self.filtered_lead_speed_mps = None

    def run_step(self, state, lead_vehicle, target_speed):
        """Mevcut hız ve ön araçtan bu çevrimin gaz-fren değerini hesaplar."""
        ego_speed = max(0.0, float(state["speed_mps"]))
        target_speed = max(0.1, float(target_speed))

        raw_lead = self.validate_lead(lead_vehicle)
        lead_confirmed = self.confirm_lead(raw_lead)
        lead = self.filter_lead(raw_lead, ego_speed)

        free_acceleration = self.calculate_free_road_acceleration(
            ego_speed, target_speed
        )
        desired_acceleration = free_acceleration
        desired_gap = None
        interaction_ratio = None
        lead_is_far = False

        if lead is not None and lead_confirmed:
            desired_gap = self.calculate_idm_gap(
                ego_speed=ego_speed,
                relative_speed=lead["relative_speed_mps"],
            )
            interaction_ratio = desired_gap / max(lead["distance_m"], 0.25)
            lead_is_far = self.lead_is_far(
                distance=lead["distance_m"],
                desired_gap=desired_gap,
                relative_speed=lead["relative_speed_mps"],
            )
            if not lead_is_far:
                desired_acceleration = self.maximum_acceleration_mps2 * (
                    1.0
                    - (ego_speed / target_speed) ** self.acceleration_exponent
                    - interaction_ratio**2
                )

        desired_acceleration = clamp(
            desired_acceleration,
            -self.maximum_deceleration_mps2,
            self.maximum_acceleration_mps2,
        )

        measured_lead_speed = (
            lead["measured_lead_speed_mps"] if lead is not None else None
        )
        self.update_hold(
            ego_speed=ego_speed,
            lead=lead,
            measured_lead_speed=measured_lead_speed,
        )

        if self.hold_active:
            self.previous_acceleration_mps2 = 0.0
            acceleration = 0.0
            throttle, brake = 0.0, self.hold_brake
            mode = "HOLD"
        else:
            acceleration = self.limit_jerk(desired_acceleration)
            throttle, brake = self.convert_acceleration_to_pedals(
                acceleration, ego_speed
            )

            if self.restart_ticks_remaining > 0:
                self.restart_ticks_remaining -= 1
                mode = "RESTART"
            elif lead is None:
                mode = "CRUISE"
            elif not lead_confirmed or lead_is_far:
                mode = "LEAD_FAR"
            else:
                mode = "FOLLOW"

        info = {
            "mode": mode,
            "target_speed_mps": target_speed,
            "desired_acceleration_mps2": desired_acceleration,
            "acceleration_mps2": acceleration,
            "free_acceleration_mps2": free_acceleration,
            "desired_gap_m": desired_gap,
            "interaction_ratio": interaction_ratio,
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
            "hold_active": self.hold_active,
            "restart_active": mode == "RESTART",
        }
        return throttle, brake, info

    def notify_emergency_stop(self):
        """Acil fren pedalları yönetirken eski ivme hafızasını temizler."""
        self.previous_acceleration_mps2 = 0.0
        self.restart_ticks_remaining = 0

    def calculate_free_road_acceleration(self, ego_speed, target_speed):
        speed_ratio = ego_speed / target_speed
        return self.maximum_acceleration_mps2 * (
            1.0 - speed_ratio**self.acceleration_exponent
        )

    def calculate_idm_gap(self, ego_speed, relative_speed):
        """Hıza ve yaklaşma hızına göre korunması gereken mesafeyi hesaplar."""
        # Projedeki bağıl hız, ön araç hızı eksi ego hızı biçimindedir.
        # IDM pozitif yaklaşma hızı kullanır; bu nedenle işaret terslenir.
        closing_speed = -relative_speed
        braking_scale = 2.0 * math.sqrt(
            self.maximum_acceleration_mps2 * self.comfortable_deceleration_mps2
        )
        dynamic_part = (
            ego_speed * self.time_headway_s + ego_speed * closing_speed / braking_scale
        )
        return self.standstill_gap_m + max(0.0, dynamic_part)

    def lead_is_far(self, distance, desired_gap, relative_speed):
        closing_speed = max(0.0, -relative_speed)
        return distance > max(30.0, 3.0 * desired_gap) and closing_speed < 1.0

    def update_hold(self, ego_speed, lead, measured_lead_speed):
        """Tam duruş frenine girme veya bu frenden çıkma kararını günceller."""
        if lead is None:
            self.release_evidence_ticks = 0
            return

        if self.hold_active:
            release_requested = (
                measured_lead_speed >= self.hold_release_lead_speed_mps
                or lead["distance_m"] >= self.hold_release_distance_m
            )
            if release_requested:
                self.release_evidence_ticks += 1
            else:
                self.release_evidence_ticks = 0

            if self.release_evidence_ticks >= self.hold_release_ticks:
                self.hold_active = False
                self.release_evidence_ticks = 0
                self.restart_ticks_remaining = max(1, int(round(0.5 / self.dt)))
            return

        should_hold = (
            ego_speed <= self.hold_speed_mps
            and lead["distance_m"] <= self.hold_distance_m
            and lead["lead_speed_mps"] <= self.hold_lead_speed_mps
        )
        if should_hold:
            self.hold_active = True
            self.release_evidence_ticks = 0
            self.restart_ticks_remaining = 0

    def confirm_lead(self, lead):
        """Normal takibe vermeden önce ön aracı iki ölçümle doğrular."""
        if lead is None:
            self.lead_ticks = 0
            self.last_track_id = None
            self.last_raw_distance_m = None
            return False

        same_track = lead.get("track_id") == self.last_track_id
        same_range = (
            self.last_raw_distance_m is not None
            and abs(lead["distance_m"] - self.last_raw_distance_m) <= 4.0
        )
        self.lead_ticks = self.lead_ticks + 1 if same_track or same_range else 1
        self.last_track_id = lead.get("track_id")
        self.last_raw_distance_m = lead["distance_m"]
        return self.lead_ticks >= self.minimum_lead_ticks

    def filter_lead(self, lead, ego_speed):
        """Kaynak geçişlerinde mesafe ve hız sıçramalarını yumuşatır."""
        if lead is None:
            self.filtered_distance_m = None
            self.filtered_lead_speed_mps = None
            return None

        measured_lead_speed = max(
            0.0,
            ego_speed + lead["relative_speed_mps"],
        )
        if self.filtered_distance_m is None:
            self.filtered_distance_m = lead["distance_m"]
            self.filtered_lead_speed_mps = measured_lead_speed
        else:
            predicted_distance = max(
                0.25,
                self.filtered_distance_m
                + (self.filtered_lead_speed_mps - ego_speed) * self.dt,
            )
            range_error = lead["distance_m"] - predicted_distance
            if range_error < 0.0:
                correction = 0.55 * clamp(range_error, -1.5, 0.0)
            else:
                correction = 0.20 * clamp(range_error, 0.0, 0.40)
            self.filtered_distance_m = predicted_distance + correction
            self.filtered_lead_speed_mps += 0.25 * (
                measured_lead_speed - self.filtered_lead_speed_mps
            )

        filtered = dict(lead)
        filtered["distance_m"] = self.filtered_distance_m
        filtered["lead_speed_mps"] = self.filtered_lead_speed_mps
        filtered["relative_speed_mps"] = self.filtered_lead_speed_mps - ego_speed
        filtered["measured_lead_speed_mps"] = measured_lead_speed
        return filtered

    def limit_jerk(self, desired_acceleration):
        """İvmenin bir çevrimde ne kadar değişebileceğini sınırlar."""
        jerk_limit = (
            self.acceleration_jerk_mps3
            if desired_acceleration >= self.previous_acceleration_mps2
            else self.braking_jerk_mps3
        )
        maximum_change = jerk_limit * self.dt
        change = clamp(
            desired_acceleration - self.previous_acceleration_mps2,
            -maximum_change,
            maximum_change,
        )
        self.previous_acceleration_mps2 += change
        return self.previous_acceleration_mps2

    def convert_acceleration_to_pedals(self, acceleration, ego_speed):
        """İstenen ivmeyi aynı anda çakışmayan gaz ve frene çevirir."""
        throttle = (
            self.rolling_resistance_throttle
            + acceleration / self.throttle_acceleration_mps2
        )
        if throttle > 0.0:
            if ego_speed <= self.hold_speed_mps and acceleration <= 0.0:
                return 0.0, 0.0
            throttle = (
                max(throttle, self.low_speed_throttle_floor)
                if ego_speed < self.breakaway_speed_mps and acceleration > 0.0
                else throttle
            )
            return clamp(throttle, 0.0, self.maximum_throttle), 0.0

        braking_acceleration = abs(acceleration) - self.coasting_deceleration_mps2
        if braking_acceleration > self.brake_deadband_mps2:
            brake = braking_acceleration / self.maximum_deceleration_mps2
            return 0.0, clamp(brake, 0.0, self.maximum_brake)

        return 0.0, 0.0

    def validate_lead(self, lead_vehicle):
        """Geçersiz veya fiziksel sınır dışı ön araç ölçümünü reddeder."""
        if lead_vehicle is None:
            return None

        try:
            distance = float(lead_vehicle.get("distance_m", -1.0))
            relative_speed = float(lead_vehicle.get("relative_speed_mps", 0.0))
        except (TypeError, ValueError):
            return None

        if not math.isfinite(distance) or not math.isfinite(relative_speed):
            return None
        if not 0.25 < distance <= 100.0:
            return None
        if not -30.0 <= relative_speed <= 30.0:
            return None

        valid_lead = dict(lead_vehicle)
        valid_lead["distance_m"] = distance
        valid_lead["relative_speed_mps"] = relative_speed
        return valid_lead

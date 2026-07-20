"""Hedef hızı rahat gaz ve fren komutlarına çeviren PID kontrolcü."""

import math


def clamp(value, minimum, maximum):
    """Bir sayıyı verilen alt ve üst sınırlar içinde tutar."""
    return max(minimum, min(value, maximum))


class LongitudinalPIDController:
    """Hız hatasını PID ile kontrol eder ve güvenli takip hızını uygular."""

    def __init__(self, dt=0.05):
        self.dt = max(0.01, float(dt))

        # Hız PID katsayıları. Çıkış doğrudan istenen ivmedir.
        self.kp = 0.55
        self.ki = 0.10
        self.kd = 0.08
        self.integral_limit = 5.0
        self.derivative_filter_ratio = 0.20

        # Konfor sınırları.
        self.maximum_acceleration_mps2 = 1.8
        self.maximum_deceleration_mps2 = 3.5
        self.acceleration_jerk_mps3 = 1.4
        self.braking_jerk_mps3 = 3.6

        # CARLA gaz ve fren dönüşüm değerleri.
        self.throttle_acceleration_mps2 = 3.0
        self.brake_deceleration_mps2 = 5.0
        self.rolling_resistance_acceleration_mps2 = 0.12
        self.maximum_throttle = 0.75
        self.maximum_brake = 0.75
        self.breakaway_throttle = 0.18

        # Ön araç veya sanal durma noktası için güvenli mesafe.
        self.standstill_gap_m = 2.0
        self.time_headway_s = 1.5
        self.gap_gain = 0.35
        self.relative_speed_gain = 0.60

        # Tam duruşta aracın eğimde kaymasını engeller.
        self.hold_speed_mps = 0.20
        self.hold_brake = 0.35
        self.hold_active = False

        self.integral = 0.0
        self.previous_speed_mps = None
        self.filtered_derivative = 0.0
        self.previous_acceleration_mps2 = 0.0
        self.filtered_lead_distance_m = None

    def run_step(self, state, lead_vehicle, target_speed):
        """Tek çevrim için birbirini dışlayan gaz ve fren değerleri üretir."""
        ego_speed = max(0.0, float(state.get("speed_mps", 0.0)))
        requested_speed = max(0.0, float(target_speed))
        raw_lead_distance = None
        if isinstance(lead_vehicle, dict):
            try:
                raw_lead_distance = float(lead_vehicle.get("distance_m"))
            except (TypeError, ValueError):
                raw_lead_distance = None
        lead = self.validate_lead(lead_vehicle)
        lead = self.filter_lead_distance(lead)

        effective_speed, follow_info = self.calculate_effective_target(
            ego_speed,
            requested_speed,
            lead,
        )
        self.update_hold(ego_speed, effective_speed, lead, follow_info)

        if self.hold_active:
            self.reset_pid(keep_speed=ego_speed)
            throttle = 0.0
            brake = self.hold_brake
            acceleration = 0.0
            desired_acceleration = 0.0
            mode = "HOLD"
        else:
            desired_acceleration = self.calculate_pid_acceleration(
                effective_speed,
                ego_speed,
            )
            acceleration = self.limit_jerk(desired_acceleration)
            throttle, brake = self.acceleration_to_pedals(
                acceleration,
                ego_speed,
                effective_speed,
            )
            mode = "FOLLOW" if follow_info["following"] else "CRUISE"

        info = {
            "controller": "pid",
            "mode": mode,
            "target_speed_mps": requested_speed,
            "effective_target_speed_mps": effective_speed,
            "speed_error_mps": effective_speed - ego_speed,
            "desired_acceleration_mps2": desired_acceleration,
            "acceleration_mps2": acceleration,
            "integral": self.integral,
            "filtered_derivative_mps2": self.filtered_derivative,
            "desired_gap_m": follow_info["desired_gap_m"],
            "lead_speed_mps": follow_info["lead_speed_mps"],
            "raw_lead_distance_m": raw_lead_distance,
            "filtered_lead_distance_m": (
                lead["distance_m"] if lead is not None else None
            ),
            "hold_active": self.hold_active,
        }
        return throttle, brake, info

    def calculate_pid_acceleration(self, target_speed, current_speed):
        """P, I ve filtrelenmiş D terimlerinden istenen ivmeyi hesaplar."""
        error = target_speed - current_speed

        if self.previous_speed_mps is None:
            measured_derivative = 0.0
        else:
            measured_derivative = -(
                current_speed - self.previous_speed_mps
            ) / self.dt
        self.previous_speed_mps = current_speed
        self.filtered_derivative += self.derivative_filter_ratio * (
            measured_derivative - self.filtered_derivative
        )

        candidate_integral = clamp(
            self.integral + error * self.dt,
            -self.integral_limit,
            self.integral_limit,
        )
        candidate_output = (
            self.kp * error
            + self.ki * candidate_integral
            + self.kd * self.filtered_derivative
        )

        # Doygunluk yönünde integral büyütmeyerek wind-up oluşmasını önleriz.
        upper_saturated = candidate_output > self.maximum_acceleration_mps2
        lower_saturated = candidate_output < -self.maximum_deceleration_mps2
        if not ((upper_saturated and error > 0.0) or (lower_saturated and error < 0.0)):
            self.integral = candidate_integral

        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * self.filtered_derivative
        )
        return clamp(
            output,
            -self.maximum_deceleration_mps2,
            self.maximum_acceleration_mps2,
        )

    def calculate_effective_target(self, ego_speed, requested_speed, lead):
        """Ön araç varsa PID'nin izleyeceği daha güvenli hız hedefini seçer."""
        info = {
            "following": False,
            "desired_gap_m": None,
            "lead_speed_mps": None,
        }
        if lead is None:
            return requested_speed, info

        lead_speed = max(0.0, ego_speed + lead["relative_speed_mps"])
        desired_gap = self.standstill_gap_m + self.time_headway_s * ego_speed
        gap_error = lead["distance_m"] - desired_gap

        # Mesafe büyüdükçe ön aracın hızının üstüne çıkılabilir; mesafe
        # daralınca bağıl hız terimi hedefi erkenden aşağı çeker.
        follow_speed = (
            lead_speed
            + self.gap_gain * gap_error
            + self.relative_speed_gain * lead["relative_speed_mps"]
        )
        follow_speed = max(0.0, follow_speed)

        source = str(lead.get("source", ""))
        is_stop_point = source in {
            "traffic_light_red",
            "traffic_light_orange",
            "traffic_light_yellow",
            "pedestrian",
            "sensor_fault",
        }
        if is_stop_point or lead_speed < 0.30:
            remaining = max(0.0, lead["distance_m"] - self.standstill_gap_m)
            stopping_speed = math.sqrt(
                2.0 * 2.0 * remaining
            )
            follow_speed = min(follow_speed, stopping_speed)

        info = {
            "following": follow_speed < requested_speed - 0.05,
            "desired_gap_m": desired_gap,
            "lead_speed_mps": lead_speed,
        }
        return min(requested_speed, follow_speed), info

    def filter_lead_distance(self, lead):
        """Mesafe ölçümündeki küçük kare-kare sıçramaları yumuşatır."""
        if lead is None:
            self.filtered_lead_distance_m = None
            return None

        measured = lead["distance_m"]
        if self.filtered_lead_distance_m is None:
            self.filtered_lead_distance_m = measured
        else:
            ratio = 0.55 if measured < self.filtered_lead_distance_m else 0.25
            self.filtered_lead_distance_m += ratio * (
                measured - self.filtered_lead_distance_m
            )

        filtered = dict(lead)
        filtered["distance_m"] = self.filtered_lead_distance_m
        return filtered

    def update_hold(self, ego_speed, effective_speed, lead, follow_info):
        """Tam duruş frenine girme ve tekrar hareket etme kararını verir."""
        stopped_request = effective_speed <= 0.10
        close_stopped_lead = (
            lead is not None
            and lead["distance_m"] <= self.standstill_gap_m + 0.40
            and (follow_info["lead_speed_mps"] or 0.0) <= 0.30
        )

        if ego_speed <= self.hold_speed_mps and (stopped_request or close_stopped_lead):
            self.hold_active = True
        elif effective_speed >= 0.50 and not close_stopped_lead:
            self.hold_active = False

    def limit_jerk(self, desired_acceleration):
        """İvmenin çevrimler arasında sert değişmesini önler."""
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

    def acceleration_to_pedals(self, acceleration, ego_speed, target_speed):
        """İvme isteğini aynı anda çalışmayan gaz veya frene dönüştürür."""
        if target_speed <= 0.05 and ego_speed <= self.hold_speed_mps:
            return 0.0, self.hold_brake

        if acceleration >= -self.rolling_resistance_acceleration_mps2:
            throttle = (
                acceleration + self.rolling_resistance_acceleration_mps2
            ) / self.throttle_acceleration_mps2
            if target_speed > ego_speed + 0.20 and ego_speed < 0.50:
                throttle = max(throttle, self.breakaway_throttle)
            return clamp(throttle, 0.0, self.maximum_throttle), 0.0

        brake = (
            -acceleration - self.rolling_resistance_acceleration_mps2
        ) / self.brake_deceleration_mps2
        return 0.0, clamp(brake, 0.0, self.maximum_brake)

    def validate_lead(self, lead_vehicle):
        """Fiziksel olmayan ön araç ölçümlerini kontrol hesabına sokmaz."""
        if not isinstance(lead_vehicle, dict):
            return None
        try:
            distance = float(lead_vehicle.get("distance_m", -1.0))
            relative_speed = float(lead_vehicle.get("relative_speed_mps", 0.0))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(distance) or not math.isfinite(relative_speed):
            return None
        if not 0.0 < distance <= 120.0:
            return None
        if not -40.0 <= relative_speed <= 40.0:
            return None

        lead = dict(lead_vehicle)
        lead["distance_m"] = distance
        lead["relative_speed_mps"] = relative_speed
        return lead

    def reset_pid(self, keep_speed=None):
        """Duruş veya acil frenden sonra eski PID hafızasını temizler."""
        self.integral = 0.0
        self.filtered_derivative = 0.0
        self.previous_speed_mps = keep_speed
        self.previous_acceleration_mps2 = 0.0

    def notify_emergency_stop(self):
        """Acil fren devreye girdiğinde PID hafızasını temizler."""
        self.hold_active = False
        self.reset_pid()

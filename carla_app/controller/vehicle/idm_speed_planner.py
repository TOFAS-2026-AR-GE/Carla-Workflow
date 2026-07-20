"""IDM ile ön araç durumundan PID için güvenli hız referansı üretir."""

import math


def clamp(value, minimum, maximum):
    """Bir sayıyı alt ve üst sınırlar içinde tutar."""
    return max(minimum, min(value, maximum))


class IDMSpeedPlanner:
    """IDM ivmesini kısa ufuklu ve yumuşak bir hız hedefine çevirir."""

    def __init__(self, dt=0.05):
        self.dt = max(0.01, float(dt))

        # IDM değerleri. Her değerin doğrudan fiziksel bir anlamı vardır.
        self.standstill_gap_m = 2.0
        self.time_headway_s = 1.5
        self.maximum_acceleration_mps2 = 1.5
        self.comfortable_deceleration_mps2 = 2.0
        self.maximum_deceleration_mps2 = 3.5
        self.acceleration_exponent = 4.0

        # IDM ivmesini PID'nin takip edeceği hıza çeviren kısa tahmin süresi.
        # İki saniyelik ufuk, 60 km/h hızda kırmızı ışığa konforlu biçimde
        # duracak kadar erken; PID'yi ani tam fren komutuna zorlamayacak
        # kadar da yumuşaktır.
        self.reference_horizon_s = 2.0

        # Normal araç takibi iki yeni ölçümle doğrulanır. Trafik ışığı gibi
        # davranış katmanından gelen sanal engeller zaten doğrulanmıştır.
        self.minimum_lead_ticks = 2
        self.lead_ticks = 0
        self.last_track_id = None
        self.last_raw_distance_m = None

        # Mesafe ve ön araç hızı ölçümündeki kare-kare sıçramaları azaltır.
        self.filtered_distance_m = None
        self.filtered_lead_speed_mps = None
        self.filtered_identity = None

    def run_step(self, state, control_obstacle, speed_cap_mps):
        """Üst hız sınırı ve ön engelden PID için hız referansı üretir."""
        ego_speed = max(0.0, float(state.get("speed_mps", 0.0)))
        speed_cap = max(0.0, float(speed_cap_mps))
        raw_lead = self.validate_lead(control_obstacle)
        rule_obstacle = self.is_rule_obstacle(raw_lead)
        if rule_obstacle:
            # Kırmızı ışık gibi sanal engeller davranış katmanında
            # zaten doğrulanır. Eski fiziksel araç sayacını temizlemek,
            # yeşil sonrasında eski bir radar hedefini hemen kabul etmemizi önler.
            self.reset_lead_confirmation()
            lead_confirmed = True
        else:
            lead_confirmed = self.confirm_lead(raw_lead)
        lead = self.filter_lead(raw_lead, ego_speed)

        free_acceleration = self.free_road_acceleration(ego_speed, speed_cap)
        idm_acceleration = free_acceleration
        desired_gap = None
        interaction_ratio = None
        lead_is_far = False

        if lead is not None and lead_confirmed:
            desired_gap = self.calculate_desired_gap(
                ego_speed,
                lead["relative_speed_mps"],
            )
            interaction_ratio = desired_gap / max(0.25, lead["distance_m"])
            lead_is_far = self.lead_is_far(
                lead["distance_m"],
                desired_gap,
                lead["relative_speed_mps"],
                rule_obstacle,
            )
            if not lead_is_far:
                speed_ratio = self.safe_speed_ratio(ego_speed, speed_cap)
                idm_acceleration = self.maximum_acceleration_mps2 * (
                    1.0
                    - speed_ratio**self.acceleration_exponent
                    - interaction_ratio**2
                )

        idm_acceleration = clamp(
            idm_acceleration,
            -self.maximum_deceleration_mps2,
            self.maximum_acceleration_mps2,
        )
        reference_speed = self.acceleration_to_speed_reference(
            ego_speed,
            speed_cap,
            idm_acceleration,
        )

        following = bool(
            lead is not None
            and lead_confirmed
            and not lead_is_far
        )
        info = {
            "planner": "idm",
            "speed_cap_mps": speed_cap,
            "reference_speed_mps": reference_speed,
            "idm_acceleration_mps2": idm_acceleration,
            "free_acceleration_mps2": free_acceleration,
            "desired_gap_m": desired_gap,
            "interaction_ratio": interaction_ratio,
            "following": following,
            "lead_is_far": lead_is_far,
            "lead_confirmed": lead_confirmed,
            "rule_obstacle": rule_obstacle,
            "raw_lead_distance_m": (
                raw_lead["distance_m"] if raw_lead is not None else None
            ),
            "filtered_lead_distance_m": (
                lead["distance_m"] if lead is not None else None
            ),
            "filtered_lead_speed_mps": (
                lead["lead_speed_mps"] if lead is not None else None
            ),
        }
        return reference_speed, info

    def free_road_acceleration(self, ego_speed, speed_cap):
        """Önde engel yokken hız sınırına yaklaşma ivmesini hesaplar."""
        if speed_cap <= 0.05:
            return -self.maximum_deceleration_mps2
        speed_ratio = ego_speed / max(speed_cap, 0.10)
        acceleration = self.maximum_acceleration_mps2 * (
            1.0 - speed_ratio**self.acceleration_exponent
        )
        return clamp(
            acceleration,
            -self.maximum_deceleration_mps2,
            self.maximum_acceleration_mps2,
        )

    def calculate_desired_gap(self, ego_speed, relative_speed):
        """Hıza ve yaklaşma hızına göre dinamik IDM mesafesini verir."""
        closing_speed = max(0.0, -float(relative_speed))
        braking_scale = 2.0 * math.sqrt(
            self.maximum_acceleration_mps2
            * self.comfortable_deceleration_mps2
        )
        dynamic_gap = (
            ego_speed * self.time_headway_s
            + ego_speed * closing_speed / braking_scale
        )
        return self.standstill_gap_m + max(0.0, dynamic_gap)

    def acceleration_to_speed_reference(
        self,
        ego_speed,
        speed_cap,
        idm_acceleration,
    ):
        """IDM ivmesini PID için kısa ufuklu hız hedefine dönüştürür."""
        if speed_cap <= 0.05:
            return 0.0
        predicted_speed = ego_speed + idm_acceleration * self.reference_horizon_s
        return clamp(predicted_speed, 0.0, speed_cap)

    def safe_speed_ratio(self, ego_speed, speed_cap):
        """Sıfır hız sınırında bölme hatası oluşmasını engeller."""
        if speed_cap <= 0.05:
            return ego_speed / 0.10
        return ego_speed / speed_cap

    def lead_is_far(
        self,
        distance_m,
        desired_gap_m,
        relative_speed_mps,
        rule_obstacle,
    ):
        """Uzak ve yaklaşılmayan aracı gereksiz fren sebebi yapmaz."""
        if rule_obstacle:
            return False
        closing_speed = max(0.0, -relative_speed_mps)
        return (
            distance_m > max(30.0, 3.0 * desired_gap_m)
            and closing_speed < 1.0
        )

    def confirm_lead(self, lead):
        """Normal takip aracını iki uyumlu ölçümden sonra kabul eder."""
        if lead is None:
            self.reset_lead_confirmation()
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

    def reset_lead_confirmation(self):
        """Normal araç takibinin kısa doğrulama hafızasını temizler."""
        self.lead_ticks = 0
        self.last_track_id = None
        self.last_raw_distance_m = None

    def filter_lead(self, lead, ego_speed):
        """Mesafe ve ön araç hızını basit bir tahmin-düzeltme ile süzer."""
        if lead is None:
            self.filtered_distance_m = None
            self.filtered_lead_speed_mps = None
            self.filtered_identity = None
            return None

        # Aynı takip kimliğinin radar/kamera kaynağı değişebilir. Bu
        # nedenle kimlik varsa onu, sanal engellerde ise kaynağı kullanırız.
        track_id = lead.get("track_id")
        identity = track_id if track_id is not None else str(lead.get("source", ""))
        measured_lead_speed = max(
            0.0,
            ego_speed + lead["relative_speed_mps"],
        )
        new_identity = identity != self.filtered_identity
        large_jump = (
            self.filtered_distance_m is not None
            and abs(lead["distance_m"] - self.filtered_distance_m) > 6.0
        )

        if self.filtered_distance_m is None or new_identity or large_jump:
            self.filtered_distance_m = lead["distance_m"]
            self.filtered_lead_speed_mps = measured_lead_speed
        else:
            predicted_distance = max(
                0.25,
                self.filtered_distance_m
                + (self.filtered_lead_speed_mps - ego_speed) * self.dt,
            )
            error = lead["distance_m"] - predicted_distance
            ratio = 0.55 if error < 0.0 else 0.25
            self.filtered_distance_m = predicted_distance + ratio * error
            self.filtered_lead_speed_mps += 0.25 * (
                measured_lead_speed - self.filtered_lead_speed_mps
            )

        self.filtered_identity = identity
        filtered = dict(lead)
        filtered["distance_m"] = self.filtered_distance_m
        filtered["lead_speed_mps"] = self.filtered_lead_speed_mps
        filtered["relative_speed_mps"] = (
            self.filtered_lead_speed_mps - ego_speed
        )
        return filtered

    def is_rule_obstacle(self, lead):
        """Davranış katmanından gelen doğrulanmış sanal engeli ayırır."""
        if lead is None:
            return False
        return str(lead.get("source", "")) in {
            "traffic_light_red",
            "traffic_light_orange",
            "traffic_light_yellow",
            "pedestrian",
            "sensor_fault",
        }

    def validate_lead(self, lead):
        """Fiziksel olmayan engel ölçümlerini IDM hesabına sokmaz."""
        if not isinstance(lead, dict):
            return None
        try:
            distance = float(lead.get("distance_m", -1.0))
            relative_speed = float(lead.get("relative_speed_mps", 0.0))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(distance) or not math.isfinite(relative_speed):
            return None
        if not 0.0 < distance <= 120.0:
            return None
        if not -40.0 <= relative_speed <= 40.0:
            return None

        valid = dict(lead)
        valid["distance_m"] = distance
        valid["relative_speed_mps"] = relative_speed
        return valid

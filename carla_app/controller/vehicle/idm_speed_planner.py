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

        # IDM ivmesi PID'ye zaten ileri besleme olarak verilir. Hız referansını
        # iki saniye ileri taşımak, duran bir araca yaklaşırken hedefi gereğinden
        # erken sıfıra indiriyordu. Kısa ufuk PID hatasını düzeltir; asıl
        # yavaşlama kararı yine fiziksel IDM ivmesinden gelir.
        self.reference_horizon_s = 0.75

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
        self.filtered_source = None
        self.filter_reused_identity = False
        self.filtered_distance_uncertainty_m = 0.0

        # Duran bir fiziksel araca yaklaşma, normal IDM takibinden ayrılır.
        # IDM'nin dinamik aralığı ego hızı düşünce küçüldüğü için tek başına
        # kullanıldığında araç önce durup sonra standstill aralığına sürünebilir.
        # Bu durumda kalan mesafeden sürekli bir duruş hızı profili üretilir.
        self.follow_state = "FREE"
        self.stationary_lead_ticks = 0
        self.moving_lead_ticks = 0
        self.stationary_confirmation_ticks = 2
        self.moving_confirmation_ticks = 2
        self.stationary_enter_speed_mps = 0.50
        self.stationary_exit_speed_mps = 1.00
        self.stop_profile_gap_m = 2.40
        self.stop_profile_deceleration_mps2 = 2.0

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
        follow_state = self.update_follow_state(
            lead,
            lead_confirmed,
            rule_obstacle,
        )

        free_acceleration = self.free_road_acceleration(ego_speed, speed_cap)
        idm_acceleration = free_acceleration
        desired_gap = None
        interaction_ratio = None
        lead_is_far = False

        stopping_profile = bool(
            lead is not None
            and lead_confirmed
            and follow_state == "STOPPING"
        )
        reference_speed = None

        if stopping_profile:
            desired_gap = self.stop_profile_terminal_gap()
            interaction_ratio = desired_gap / max(0.25, lead["distance_m"])
            reference_speed, idm_acceleration = self.stationary_lead_reference(
                ego_speed,
                speed_cap,
                lead["distance_m"],
                desired_gap,
            )
        elif lead is not None and lead_confirmed:
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
        if reference_speed is None:
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
            "follow_state": follow_state,
            "stopping_profile": stopping_profile,
            "filter_reused_identity": self.filter_reused_identity,
            "lead_distance_uncertainty_m": self.filtered_distance_uncertainty_m,
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

    def update_follow_state(self, lead, lead_confirmed, rule_obstacle):
        """Hareketli takip ile fiziksel duran araca yaklaşmayı ayırır."""
        if lead is None or not lead_confirmed:
            self.stationary_lead_ticks = 0
            self.moving_lead_ticks = 0
            self.follow_state = "FREE" if lead is None else "PENDING"
            return self.follow_state

        if rule_obstacle:
            self.stationary_lead_ticks = 0
            self.moving_lead_ticks = 0
            self.follow_state = "RULE_STOP"
            return self.follow_state

        lead_speed = max(0.0, float(lead["lead_speed_mps"]))
        if self.follow_state == "STOPPING":
            if lead_speed >= self.stationary_exit_speed_mps:
                self.moving_lead_ticks += 1
            else:
                self.moving_lead_ticks = 0
            if self.moving_lead_ticks >= self.moving_confirmation_ticks:
                self.follow_state = "FOLLOW"
                self.stationary_lead_ticks = 0
        else:
            if lead_speed <= self.stationary_enter_speed_mps:
                self.stationary_lead_ticks += 1
            else:
                self.stationary_lead_ticks = 0
            if self.stationary_lead_ticks >= self.stationary_confirmation_ticks:
                self.follow_state = "STOPPING"
                self.moving_lead_ticks = 0
            else:
                self.follow_state = "FOLLOW"
        return self.follow_state

    def stop_profile_terminal_gap(self):
        """Ölçüm oynaklığını duran lead aralığında güvenlik payına çevirir."""
        uncertainty_margin = clamp(
            self.filtered_distance_uncertainty_m - 0.15,
            0.0,
            0.80,
        )
        return self.stop_profile_gap_m + uncertainty_margin

    def stationary_lead_reference(
        self,
        ego_speed,
        speed_cap,
        distance_m,
        terminal_gap_m,
    ):
        """Duran lead için standstill aralığında biten hız/ivme profili."""
        remaining_distance = max(0.0, float(distance_m) - terminal_gap_m)
        profile_speed = math.sqrt(
            2.0 * self.stop_profile_deceleration_mps2 * remaining_distance
        )
        reference_speed = min(speed_cap, profile_speed)

        # Profil üzerindeyken dv/dt=-b olur. Araç profilin gerisine düşerse
        # negatif ileri beslemeyi yumuşakça kaldırmak, ikinci bir erken duruşu
        # ve ardından breakaway gazıyla sürünmeyi önler.
        speed_deficit = max(0.0, reference_speed - ego_speed)
        feedforward_ratio = clamp(1.0 - speed_deficit / 0.25, 0.0, 1.0)
        feedforward = -self.stop_profile_deceleration_mps2 * feedforward_ratio
        if ego_speed <= 0.05 and reference_speed <= 0.05:
            feedforward = 0.0
        return reference_speed, feedforward

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
            self.filtered_source = None
            self.filter_reused_identity = False
            self.filtered_distance_uncertainty_m = 0.0
            return None

        # Aynı takip kimliğinin radar/kamera kaynağı değişebilir. Bu
        # nedenle kimlik varsa onu, sanal engellerde ise kaynağı kullanırız.
        track_id = lead.get("track_id")
        raw_identity = (
            track_id if track_id is not None else str(lead.get("source", ""))
        )
        source = str(lead.get("source", ""))
        measured_lead_speed = max(
            0.0,
            ego_speed + lead["relative_speed_mps"],
        )
        identity = self.resolve_filter_identity(
            raw_identity,
            source,
            lead["distance_m"],
            measured_lead_speed,
            ego_speed,
        )
        new_identity = identity != self.filtered_identity
        large_jump = (
            self.filtered_distance_m is not None
            and abs(lead["distance_m"] - self.filtered_distance_m) > 6.0
        )

        if self.filtered_distance_m is None or new_identity or large_jump:
            self.filtered_distance_m = lead["distance_m"]
            self.filtered_lead_speed_mps = measured_lead_speed
            self.filtered_distance_uncertainty_m = 0.0
        else:
            predicted_distance = max(
                0.25,
                self.filtered_distance_m
                + (self.filtered_lead_speed_mps - ego_speed) * self.dt,
            )
            raw_error = lead["distance_m"] - predicted_distance
            self.filtered_distance_uncertainty_m = (
                0.90 * self.filtered_distance_uncertainty_m
                + 0.10 * min(3.0, abs(raw_error))
            )
            # Ham güvenlik katmanı yakın tehlikeyi ayrıca izler. Normal takip
            # filtresinde tek karelik 2-4 m radar/kamera sıçramasının doğrudan
            # pedala taşınmasını önlemek için yalnız fiziksel olarak makul
            # yeniliği kabul ederiz. Yaklaşan hedefe tepki yine daha hızlıdır.
            bounded_error = clamp(raw_error, -0.50, 0.35)
            ratio = 0.08 if bounded_error < 0.0 else 0.04
            self.filtered_distance_m = (
                predicted_distance + ratio * bounded_error
            )
            speed_error = clamp(
                measured_lead_speed - self.filtered_lead_speed_mps,
                -1.5,
                1.5,
            )
            self.filtered_lead_speed_mps += 0.20 * speed_error

        self.filtered_identity = identity
        self.filtered_source = source
        filtered = dict(lead)
        filtered["distance_m"] = self.filtered_distance_m
        filtered["lead_speed_mps"] = self.filtered_lead_speed_mps
        filtered["relative_speed_mps"] = (
            self.filtered_lead_speed_mps - ego_speed
        )
        return filtered

    def resolve_filter_identity(
        self,
        raw_identity,
        source,
        measured_distance,
        measured_lead_speed,
        ego_speed,
    ):
        """Kamera kaybında aynı radar hedefinin filtre hafızasını korur."""
        self.filter_reused_identity = False
        if self.filtered_identity is None or self.filtered_distance_m is None:
            return raw_identity
        if raw_identity == self.filtered_identity:
            return raw_identity

        fallback_sources = {"radar_direct", "radar_predicted"}
        source_transition = (
            source in fallback_sources
            or self.filtered_source in fallback_sources
        )
        if not source_transition:
            return raw_identity

        predicted_distance = max(
            0.25,
            self.filtered_distance_m
            + (self.filtered_lead_speed_mps - ego_speed) * self.dt,
        )
        distance_gate = max(3.0, 0.15 * max(1.0, measured_distance))
        same_motion = (
            abs(measured_lead_speed - self.filtered_lead_speed_mps) <= 3.0
        )
        same_range = abs(measured_distance - predicted_distance) <= distance_gate
        if same_range and same_motion:
            self.filter_reused_identity = True
            return self.filtered_identity
        return raw_identity

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

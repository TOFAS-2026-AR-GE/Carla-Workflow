"""Konuma bağlı S-eğrisi trafik ışığı duruş planlayıcısı."""

from __future__ import annotations

import math


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


class SmoothStopPlanner:
    """Durma çizgisine kadar sıfır başlangıç/son ivmeli hız profili üretir.

    Kalan mesafenin normalize edilmiş değeri ``u`` için
    ``h(u) = 10u³ - 15u⁴ + 6u⁵`` kullanılır. İstenen hızın karesi bu
    fonksiyonla
    ölçeklenir. Bu seçimin iki önemli sonucu vardır:

    * Fren ivmesi planın başında sıfırdan başlar.
    * Fren ivmesi araç dururken tekrar sıfıra yaklaşır.

    Böylece sabit ``v²/(2d)`` freninin sert başlangıcı ve son metredeki ani
    fren bırakma davranışı ortadan kalkar. Profil mesafeye bağlı olduğu için
    araç fiziği, eğim veya küçük zaman gecikmeleri süre tabanlı bir planı
    bozmaz; her kare gerçek kalan mesafeden aynı yumuşak hedef yeniden okunur.
    """

    def __init__(
        self,
        dt: float = 0.05,
        stop_gap_m: float = 0.20,
    ) -> None:
        self.dt = max(1e-3, float(dt))
        self.stop_gap_m = max(0.10, float(stop_gap_m))

        self.comfort_peak_deceleration_mps2 = 1.80
        self.maximum_deceleration_mps2 = 7.50
        self.normal_jerk_limit_mps3 = 1.20
        self.emergency_jerk_limit_mps3 = 7.00

        self.speed_feedback_gain = 1.00
        self.maximum_recovery_acceleration_mps2 = 0.35
        self.hold_speed_mps = 0.12
        self.hold_capture_margin_m = 0.10
        self.coast_margin_m = 0.75
        self.final_zone_m = 7.0

        self.reset()

    def reset(self) -> None:
        self.phase = "IDLE"
        self.plan_start_distance_m = None
        self.plan_start_speed_mps = None
        self.profile_peak_deceleration_mps2 = 0.0
        self.target_speed_mps = None
        self.reference_acceleration_mps2 = 0.0
        self.required_deceleration_mps2 = 0.0
        self.emergency = False
        self.jerk_mps3 = self.normal_jerk_limit_mps3

    def update(
        self,
        speed_mps: float,
        distance_m: float,
        active: bool = True,
    ) -> tuple[float, dict]:
        speed_mps = max(0.0, float(speed_mps))
        distance_m = max(0.0, float(distance_m))
        effective_distance_m = max(0.0, distance_m - self.stop_gap_m)

        if not active:
            self.reset()
            return 0.0, self.info(distance_m, effective_distance_m)

        if (
            speed_mps <= self.hold_speed_mps
            and effective_distance_m <= self.hold_capture_margin_m
        ):
            self.phase = "HOLD"
            self.target_speed_mps = 0.0
            self.reference_acceleration_mps2 = 0.0
            self.required_deceleration_mps2 = 0.0
            self.emergency = False
            self.jerk_mps3 = self.normal_jerk_limit_mps3
            return 0.0, self.info(distance_m, effective_distance_m)

        self.required_deceleration_mps2 = (
            speed_mps * speed_mps
            / (2.0 * max(effective_distance_m, 0.15))
            if speed_mps > 0.0
            else 0.0
        )

        if self.phase in {"IDLE", "COAST"}:
            comfortable_distance_m = self.smooth_profile_distance(
                speed_mps,
                self.comfort_peak_deceleration_mps2,
            )
            if (
                effective_distance_m
                > comfortable_distance_m + self.coast_margin_m
            ):
                self.phase = "COAST"
                self.target_speed_mps = speed_mps
                self.reference_acceleration_mps2 = 0.0
                self.profile_peak_deceleration_mps2 = 0.0
                self.emergency = False
                self.jerk_mps3 = self.normal_jerk_limit_mps3
                return 0.0, self.info(distance_m, effective_distance_m)

            self.phase = "PROFILE"
            self.plan_start_distance_m = max(0.20, effective_distance_m)
            self.plan_start_speed_mps = max(0.05, speed_mps)
            self.profile_peak_deceleration_mps2 = (
                0.9375
                * self.plan_start_speed_mps
                * self.plan_start_speed_mps
                / self.plan_start_distance_m
            )

        plan_distance_m = max(
            0.20,
            float(self.plan_start_distance_m or effective_distance_m),
        )
        plan_speed_mps = max(
            0.05,
            float(self.plan_start_speed_mps or speed_mps),
        )
        normalized_distance = clamp(
            effective_distance_m / plan_distance_m,
            0.0,
            1.0,
        )

        # Quintic smootherstep: hız, ivme ve jerk profilin iki ucunda da
        # süreklidir. Kübik smoothstep yalnız ivmeyi sıfırlarken bu profil
        # jerk'i de sıfıra getirir; pedal başlangıcı ve son duruş daha doğal
        # hissedilir.
        u = normalized_distance
        smooth_value = (
            10.0 * u**3
            - 15.0 * u**4
            + 6.0 * u**5
        )
        smooth_derivative = 30.0 * u**2 * (1.0 - u) ** 2

        reference_speed_mps = (
            plan_speed_mps * math.sqrt(max(0.0, smooth_value))
        )
        reference_acceleration_mps2 = -(
            plan_speed_mps
            * plan_speed_mps
            / (2.0 * plan_distance_m)
            * smooth_derivative
        )

        acceleration_command = (
            reference_acceleration_mps2
            + self.speed_feedback_gain
            * (reference_speed_mps - speed_mps)
        )

        emergency_threshold_mps2 = max(
            3.20,
            1.35 * self.profile_peak_deceleration_mps2,
        )
        self.emergency = (
            self.profile_peak_deceleration_mps2 > 3.50
            or self.required_deceleration_mps2 > emergency_threshold_mps2
            or (
                effective_distance_m <= 0.15
                and speed_mps > 0.80
            )
        )

        if self.emergency:
            self.phase = "EMERGENCY"
            self.jerk_mps3 = self.emergency_jerk_limit_mps3
            acceleration_command = min(
                acceleration_command,
                -clamp(
                    1.08 * self.required_deceleration_mps2,
                    self.comfort_peak_deceleration_mps2,
                    self.maximum_deceleration_mps2,
                ),
            )
        else:
            self.jerk_mps3 = self.normal_jerk_limit_mps3
            self.phase = (
                "FINAL"
                if effective_distance_m <= self.final_zone_m
                else "PROFILE"
            )

        # Araç gerçek fiziği planlanandan biraz fazla yavaşlatırsa freni
        # bırakmak yetmeyebilir. Küçük pozitif komut, yalnız hedef hızın altında
        # kalındığında devreye girer ve sert tekrar hızlanma oluşturmaz.
        acceleration_command = clamp(
            acceleration_command,
            -self.maximum_deceleration_mps2,
            self.maximum_recovery_acceleration_mps2,
        )

        self.target_speed_mps = reference_speed_mps
        self.reference_acceleration_mps2 = (
            reference_acceleration_mps2
        )

        return acceleration_command, self.info(
            distance_m,
            effective_distance_m,
        )

    @staticmethod
    def smooth_profile_distance(
        speed_mps: float,
        peak_deceleration_mps2: float,
    ) -> float:
        """Smoothstep profilinin verilen tepe ivmesi için mesafesi."""
        speed_mps = max(0.0, float(speed_mps))
        peak_deceleration_mps2 = max(
            1e-6,
            float(peak_deceleration_mps2),
        )
        # Quintic smootherstep için h'(u) en fazla 1.875'tir. Profil
        # tepe ivmesi 0.9375*v0²/D olduğundan D aşağıdaki gibi bulunur.
        return 0.9375 * speed_mps * speed_mps / peak_deceleration_mps2

    @staticmethod
    def stopping_distance(
        speed_mps: float,
        peak_deceleration_mps2: float,
        jerk_mps3: float,
    ) -> float:
        """Geriye uyumlu jerk-sınırlı trapez profil mesafesi."""
        speed_mps = max(0.0, float(speed_mps))
        peak_deceleration_mps2 = max(
            1e-6,
            float(peak_deceleration_mps2),
        )
        jerk_mps3 = max(1e-6, float(jerk_mps3))
        return (
            speed_mps * speed_mps
            / (2.0 * peak_deceleration_mps2)
            + speed_mps * peak_deceleration_mps2
            / (2.0 * jerk_mps3)
        )

    def info(
        self,
        distance_m: float,
        effective_distance_m: float,
    ) -> dict:
        return {
            "phase": self.phase,
            "distance_m": float(distance_m),
            "effective_distance_m": float(effective_distance_m),
            "stop_gap_m": self.stop_gap_m,
            "target_speed_mps": self.target_speed_mps,
            "reference_acceleration_mps2": (
                self.reference_acceleration_mps2
            ),
            "required_deceleration_mps2": (
                self.required_deceleration_mps2
            ),
            "peak_deceleration_mps2": (
                self.profile_peak_deceleration_mps2
            ),
            "jerk_mps3": self.jerk_mps3,
            "emergency": self.emergency,
            "plan_start_distance_m": self.plan_start_distance_m,
            "plan_start_speed_mps": self.plan_start_speed_mps,
        }

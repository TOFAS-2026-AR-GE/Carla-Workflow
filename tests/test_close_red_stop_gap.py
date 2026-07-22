"""Kırmızı ışıkta yaklaşık 30 cm kala duruş testi."""

import math

from carla_app.controller.vehicle.longitudinal_controller import (
    LongitudinalController,
)


def simulate_stop(speed_kmh: float, distance_m: float) -> float:
    controller = LongitudinalController(dt=0.05)
    speed_mps = speed_kmh / 3.6

    for _ in range(2400):
        light = {
            "track_id": -100,
            "stop_line_id": 1,
            "distance_m": max(0.03, distance_m),
            "relative_speed_mps": -speed_mps,
            "source": "traffic_light_red",
        }
        throttle, brake, info = controller.run_step(
            {"speed_mps": speed_mps},
            light,
            30.0,
        )

        acceleration = 3.2 * throttle - 7.5 * brake
        if speed_mps > 0.02:
            acceleration -= 0.12

        speed_mps = max(
            0.0,
            speed_mps + acceleration * controller.dt,
        )
        distance_m -= speed_mps * controller.dt

        if info["hold_active"] and speed_mps <= 0.03:
            return distance_m

    raise AssertionError("Araç HOLD durumuna geçemedi.")


def test_city_speeds_stop_about_thirty_centimetres_before_line():
    for speed_kmh, initial_distance_m in (
        (10.0, 9.0),
        (30.0, 35.0),
        (50.0, 88.0),
        (60.0, 125.0),
    ):
        final_distance_m = simulate_stop(
            speed_kmh,
            initial_distance_m,
        )
        assert 0.22 <= final_distance_m <= 0.40


def test_internal_gap_and_hold_capture_sum_to_thirty_centimetres():
    controller = LongitudinalController(dt=0.05)

    assert math.isclose(
        controller.traffic_light_stop_gap_m,
        0.20,
        abs_tol=1e-9,
    )
    assert math.isclose(
        controller.traffic_light_hold_distance_m,
        0.30,
        abs_tol=1e-9,
    )

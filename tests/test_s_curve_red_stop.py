"""Kırmızı ışık S-eğrisi planlayıcısı regresyon testleri."""

from __future__ import annotations

from carla_app.controller.vehicle.longitudinal_controller import (
    LongitudinalController,
)
from carla_app.controller.vehicle.smooth_stop_planner import (
    SmoothStopPlanner,
)
from carla_app.controller.vehicle.traffic_light import TrafficLightObserver


def simulate_stop(speed_kmh: float, distance_m: float):
    controller = LongitudinalController(dt=0.05)
    speed_mps = speed_kmh / 3.6
    maximum_brake = 0.0
    previous_command_acceleration = 0.0
    maximum_command_jerk = 0.0
    phases = []

    for _ in range(2400):
        light = {
            "track_id": -100,
            "stop_line_id": 1,
            "distance_m": max(0.21, distance_m),
            "relative_speed_mps": -speed_mps,
            "source": "traffic_light_red",
        }
        throttle, brake, info = controller.run_step(
            {"speed_mps": speed_mps},
            light,
            30.0,
        )
        maximum_brake = max(maximum_brake, brake)
        command_acceleration = float(info["acceleration_mps2"])
        maximum_command_jerk = max(
            maximum_command_jerk,
            abs(command_acceleration - previous_command_acceleration)
            / controller.dt,
        )
        previous_command_acceleration = command_acceleration
        phases.append(info["red_stop_plan"]["phase"])

        # Bu basit model kasıtlı olarak controller pedal eşlemesinden daha
        # güçlü fren uygular; planın araç-model hatasına dayanıklılığını sınar.
        actual_acceleration = 3.2 * throttle - 7.5 * brake
        if speed_mps > 0.02:
            actual_acceleration -= 0.12
        speed_mps = max(
            0.0,
            speed_mps + actual_acceleration * controller.dt,
        )
        distance_m -= speed_mps * controller.dt

        if info["hold_active"] and speed_mps <= 0.03:
            break

    return {
        "distance_m": distance_m,
        "speed_mps": speed_mps,
        "maximum_brake": maximum_brake,
        "maximum_command_jerk": maximum_command_jerk,
        "phases": phases,
        "hold_active": info["hold_active"],
    }


def test_smooth_profile_has_zero_acceleration_at_both_ends():
    planner = SmoothStopPlanner(dt=0.05, stop_gap_m=0.2)
    speed_mps = 50.0 / 3.6
    distance_m = (
        planner.smooth_profile_distance(
            speed_mps,
            planner.comfort_peak_deceleration_mps2,
        )
        + planner.stop_gap_m
    )

    first_acceleration, first_info = planner.update(
        speed_mps,
        distance_m,
    )
    assert abs(first_acceleration) < 1e-9
    assert first_info["phase"] == "PROFILE"

    final_acceleration, final_info = planner.update(
        0.20,
        planner.stop_gap_m + 0.12,
    )
    assert final_info["phase"] in {"FINAL", "EMERGENCY"}
    assert final_acceleration > -0.50


def test_activation_distance_scales_with_speed_squared():
    observer = TrafficLightObserver(0.05, 800, 600, maximum_distance_m=220.0)
    distance_10 = observer.activation_distance_m(10.0 / 3.6)
    distance_50 = observer.activation_distance_m(50.0 / 3.6)
    distance_60 = observer.activation_distance_m(60.0 / 3.6)

    assert 6.0 <= distance_10 <= 9.0
    assert 105.0 <= distance_50 <= 112.0
    assert 150.0 <= distance_60 <= 158.0
    assert distance_10 < distance_50 < distance_60


def test_red_commit_survives_detection_dropout_until_green_confirmed():
    observer = TrafficLightObserver(0.05, 800, 600)

    observer._add_vote("red", confidence=0.90, speed_mps=10.0)
    observer._add_vote("red", confidence=0.90, speed_mps=10.0)
    assert observer.red_committed is True
    assert observer.confirmed_state == "red"

    # Tek yeşil karesi fren taahhüdünü bırakmaz.
    observer._add_vote("green", confidence=0.90, speed_mps=5.0)
    assert observer.red_committed is True

    # İkinci doğrulanmış yeşil kontrollü serbest bırakır.
    observer._add_vote("green", confidence=0.90, speed_mps=5.0)
    assert observer.red_committed is False
    assert observer.confirmed_state == "green"


def test_route_distance_filter_rejects_single_frame_jump():
    observer = TrafficLightObserver(0.05, 800, 600)

    first = observer.filter_stop_distance(
        actor=None,
        raw_distance_m=40.0,
        speed_mps=10.0,
    )
    second = observer.filter_stop_distance(
        actor=None,
        raw_distance_m=50.0,
        speed_mps=10.0,
    )

    assert first == 40.0
    # Ham ölçüm 10 m sıçrasa da çıktı yalnız sınırlı miktarda büyür.
    assert second < 41.0


def test_city_speed_stops_are_smooth_and_near_target_line():
    cases = (
        (10.0, 9.0),
        (30.0, 35.0),
        (50.0, 88.0),
        (60.0, 125.0),
    )

    for speed_kmh, distance_m in cases:
        result = simulate_stop(speed_kmh, distance_m)
        assert result["hold_active"] is True
        assert 0.22 <= result["distance_m"] <= 0.45
        assert result["maximum_brake"] <= 0.32
        assert result["maximum_command_jerk"] <= 1.50
        assert "EMERGENCY" not in result["phases"]


def test_late_red_is_not_ignored_and_uses_emergency_profile():
    planner = SmoothStopPlanner(dt=0.05, stop_gap_m=0.2)
    acceleration, info = planner.update(
        speed_mps=100.0 / 3.6,
        distance_m=80.0,
    )

    # İlk uzamsal profil örneği sıfır ivmeyle başlasa bile kalan mesafe
    # yetersiz olduğundan güvenlik zarfı aynı karede güçlü fren ister.
    assert info["emergency"] is True
    assert info["phase"] == "EMERGENCY"
    assert acceleration <= -5.0

def test_stop_waypoints_are_preferred_over_trigger_box():
    class Location:
        def __init__(self, x, y, z=0.0):
            self.x = x
            self.y = y
            self.z = z

    class Transform:
        def __init__(self, location):
            self.location = location

    class Waypoint:
        def __init__(self, location):
            self.transform = Transform(location)

    class Actor:
        def get_stop_waypoints(self):
            return [Waypoint(Location(25.0, 0.0))]

    locations = TrafficLightObserver.stop_locations(Actor())
    assert len(locations) == 1
    assert locations[0].x == 25.0


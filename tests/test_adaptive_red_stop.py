from carla_app.controller.vehicle.longitudinal_controller import (
    LongitudinalController,
)
from carla_app.controller.vehicle.traffic_light import TrafficLightObserver


def test_activation_distance_grows_strongly_with_speed():
    observer = TrafficLightObserver(
        dt=0.05,
        image_width=800,
        image_height=600,
    )

    distance_10 = observer.activation_distance_m(10.0 / 3.6)
    distance_50 = observer.activation_distance_m(50.0 / 3.6)
    distance_100 = observer.activation_distance_m(100.0 / 3.6)

    assert 5.0 <= distance_10 <= 7.5
    assert 50.0 <= distance_50 <= 65.0
    assert 190.0 <= distance_100 <= 210.0
    assert distance_10 < distance_50 < distance_100


def test_low_speed_does_not_accept_red_too_early():
    observer = TrafficLightObserver(0.05, 800, 600)

    assert observer.requires_stop("red", 5.0, 10.0 / 3.6)
    assert not observer.requires_stop("red", 10.0, 10.0 / 3.6)


def test_high_speed_accepts_red_far_away():
    observer = TrafficLightObserver(0.05, 800, 600)

    assert observer.requires_stop("red", 190.0, 100.0 / 3.6)
    assert not observer.requires_stop("red", 215.0, 100.0 / 3.6)


def test_high_speed_high_confidence_red_needs_one_vote():
    observer = TrafficLightObserver(0.05, 800, 600)

    observer._add_vote(
        "red",
        confidence=0.90,
        speed_mps=100.0 / 3.6,
    )

    assert observer.confirmed_state == "red"


def test_low_speed_red_still_needs_two_votes():
    observer = TrafficLightObserver(0.05, 800, 600)

    observer._add_vote(
        "red",
        confidence=0.90,
        speed_mps=10.0 / 3.6,
    )
    assert observer.confirmed_state != "red"

    observer._add_vote(
        "red",
        confidence=0.90,
        speed_mps=10.0 / 3.6,
    )
    assert observer.confirmed_state == "red"


def test_constant_deceleration_plan_stays_stable_on_ideal_path():
    controller = LongitudinalController(dt=0.05)

    first = controller.calculate_interaction_acceleration(
        distance_m=100.4,
        desired_gap_m=0.4,
        relative_speed_mps=-20.0,
        ego_speed_mps=20.0,
        source="traffic_light_red",
    )
    second = controller.calculate_interaction_acceleration(
        distance_m=90.4,
        desired_gap_m=0.4,
        relative_speed_mps=-18.97366596,
        ego_speed_mps=18.97366596,
        source="traffic_light_red",
    )

    assert -2.05 <= first <= -1.95
    assert abs(second - first) <= 0.05


def test_late_red_detection_uses_strong_braking_instead_of_ignoring():
    controller = LongitudinalController(dt=0.05)

    acceleration = controller.calculate_interaction_acceleration(
        distance_m=50.4,
        desired_gap_m=0.4,
        relative_speed_mps=-(100.0 / 3.6),
        ego_speed_mps=100.0 / 3.6,
        source="traffic_light_red",
    )

    assert acceleration <= -7.5
    assert acceleration >= -controller.traffic_light_emergency_deceleration_mps2

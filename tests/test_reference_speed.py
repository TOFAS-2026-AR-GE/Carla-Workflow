from carla_app.controller.vehicle.reference_speed import RandomReferenceSpeed


def test_random_reference_is_deterministic_and_changes():
    first = RandomReferenceSpeed(
        dt=0.1,
        minimum_speed_kmh=10,
        maximum_speed_kmh=50,
        minimum_hold_seconds=0.5,
        maximum_hold_seconds=0.5,
        seed=3,
        enabled=True,
        initial_speed_kmh=30,
    )
    second = RandomReferenceSpeed(
        dt=0.1,
        minimum_speed_kmh=10,
        maximum_speed_kmh=50,
        minimum_hold_seconds=0.5,
        maximum_hold_seconds=0.5,
        seed=3,
        enabled=True,
        initial_speed_kmh=30,
    )
    sequence_a = [first.update(0.0)[0] for _ in range(20)]
    sequence_b = [second.update(0.0)[0] for _ in range(20)]
    assert sequence_a == sequence_b
    assert len(set(round(value, 5) for value in sequence_a)) >= 3


def test_disabled_reference_uses_fallback():
    generator = RandomReferenceSpeed(dt=0.05, enabled=False)
    value, info = generator.update(12.5)
    assert value == 12.5
    assert info["enabled"] is False

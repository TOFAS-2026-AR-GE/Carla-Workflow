from carla_app.controller.vehicle.longitudinal_controller import LongitudinalController
from scripts.controller_benchmark import calculate_metrics, simulate


def plant_step(speed, acceleration, throttle, brake, dt=0.05):
    drag = 0.16 + 0.012 * speed if speed > 0.05 else 0.0
    desired = 3.2 * throttle - 4.8 * brake - drag
    acceleration += (desired - acceleration) * dt / 0.30
    speed = max(0.0, speed + acceleration * dt)
    return speed, acceleration


def test_random_reference_tracking_comfort_regression():
    rows = simulate(duration_s=60.0, dt=0.05, seed=7)
    metrics = calculate_metrics(rows)
    assert metrics["target_mae_mps"] < 0.85
    assert metrics["target_rmse_mps"] < 1.10
    assert metrics["maximum_acceleration_mps2"] < 1.50
    assert metrics["maximum_deceleration_mps2"] > -2.40
    assert metrics["p95_absolute_jerk_mps3"] < 1.35


def test_following_scenario_settles_without_crossing_ten_metre_base_gap():
    dt = 0.05
    controller = LongitudinalController(dt, follow_gap_m=10.0, follow_gap_margin_m=1.5)
    ego_speed = 15.0
    lead_speed = 8.0
    distance = 35.0
    acceleration = 0.0

    for _ in range(int(40.0 / dt)):
        lead = {
            "track_id": 1,
            "distance_m": distance,
            "relative_speed_mps": lead_speed - ego_speed,
            "source": "camera_radar_track",
        }
        throttle, brake, _ = controller.run_step(
            {"speed_mps": ego_speed},
            lead,
            target_speed=18.0,
        )
        ego_speed, acceleration = plant_step(
            ego_speed,
            acceleration,
            throttle,
            brake,
            dt,
        )
        distance += (lead_speed - ego_speed) * dt
        assert distance >= 10.0

    assert abs(ego_speed - lead_speed) < 0.15
    assert 11.0 <= distance <= 15.0


def test_red_light_scenario_stops_before_line_with_margin():
    dt = 0.05
    controller = LongitudinalController(dt, follow_gap_m=10.0, follow_gap_margin_m=1.5)
    ego_speed = 10.0
    distance = 35.0
    acceleration = 0.0

    for _ in range(int(12.0 / dt)):
        obstacle = {
            "track_id": -100,
            "distance_m": distance,
            "relative_speed_mps": -ego_speed,
            "source": "traffic_light_red",
        }
        throttle, brake, info = controller.run_step(
            {"speed_mps": ego_speed},
            obstacle,
            target_speed=15.0,
        )
        ego_speed, acceleration = plant_step(
            ego_speed,
            acceleration,
            throttle,
            brake,
            dt,
        )
        distance = max(0.25, distance - ego_speed * dt)
        if ego_speed < 0.02 and info["mode"] == "HOLD_RED":
            break

    assert ego_speed < 0.02
    assert 2.8 <= distance <= 4.2
    assert info["mode"] == "HOLD_RED"

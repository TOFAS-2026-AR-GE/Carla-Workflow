from carla_app.controller.vehicle.longitudinal_controller import LongitudinalController


def state(speed):
    return {"speed_mps": float(speed)}


def test_pedals_are_mutually_exclusive_and_jerk_limited():
    controller = LongitudinalController(dt=0.05)
    previous = 0.0
    for _ in range(40):
        throttle, brake, info = controller.run_step(state(0.0), None, 12.0)
        assert not (throttle > 0.0 and brake > 0.0)
        acceleration = info["acceleration_mps2"]
        assert acceleration - previous <= controller.acceleration_jerk_mps3 * controller.dt + 1e-9
        previous = acceleration


def test_close_vehicle_requests_deceleration_after_confirmation():
    controller = LongitudinalController(dt=0.05, follow_gap_m=10.0)
    lead = {
        "track_id": 1,
        "distance_m": 8.0,
        "relative_speed_mps": -1.0,
        "source": "camera_radar_track",
    }
    controller.run_step(state(10.0), lead, 15.0)
    throttle, brake, info = controller.run_step(state(10.0), lead, 15.0)
    assert info["lead_confirmed_for_control"] is True
    assert info["desired_gap_m"] >= 10.0
    assert info["desired_acceleration_mps2"] < 0.0
    assert throttle == 0.0 or brake == 0.0


def test_vehicle_hold_keeps_about_ten_metre_gap():
    controller = LongitudinalController(dt=0.05, follow_gap_m=10.0)
    lead = {
        "track_id": 2,
        "distance_m": 10.2,
        "relative_speed_mps": 0.0,
        "source": "camera_radar_track",
    }
    throttle, brake, info = controller.run_step(state(0.0), lead, 10.0)
    assert info["hold_active"] is True
    assert info["mode"] == "HOLD"
    assert throttle == 0.0
    assert brake > 0.0


def test_red_light_uses_short_stop_line_gap_not_vehicle_gap():
    controller = LongitudinalController(dt=0.05, follow_gap_m=10.0)
    light = {
        "track_id": -100,
        "distance_m": 12.0,
        "relative_speed_mps": -5.0,
        "source": "traffic_light_red",
    }
    _, _, info = controller.run_step(state(5.0), light, 12.0)
    assert info["mode"] == "STOP_RED"
    assert info["desired_gap_m"] < 10.0
    assert info["desired_acceleration_mps2"] < 0.0

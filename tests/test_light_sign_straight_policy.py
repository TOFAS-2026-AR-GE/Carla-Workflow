from __future__ import annotations

from types import SimpleNamespace

from carla_app.controller.vehicle.longitudinal_controller import LongitudinalController
from carla_app.controller.vehicle.speed_sign import SpeedSignObserver
from carla_app.controller.vehicle.stanley_controller import StanleyController
from carla_app.controller.vehicle.traffic_light import TrafficLightObserver


def point(x, y):
    return SimpleNamespace(x=float(x), y=float(y))


def straight_path(length=80):
    return [point(index, 0.0) for index in range(length)]


def straight_state(y=0.0, yaw=0.0, speed=8.0):
    return {
        "reference_path": straight_path(),
        "location": point(0.0, y),
        "yaw": yaw,
        "speed_mps": speed,
        "lane_width": 3.5,
        "vehicle_half_width_m": 0.95,
    }


def sign_detection(frame_id, name, confidence=0.95):
    return {
        "frame_id": frame_id,
        "signs": [
            {
                "class_name": name,
                "confidence": confidence,
                "detection_confidence": confidence,
                "bbox": (650, 170, 700, 230),
            }
        ],
    }


def test_red_is_ignored_until_ten_metres_and_yellow_green_always_go():
    observer = TrafficLightObserver(0.05, 800, 600)
    assert not observer.requires_stop("red", 10.01, 8.0)
    assert observer.requires_stop("red", 10.0, 8.0)
    assert not observer.requires_stop("yellow", 2.0, 8.0)
    assert not observer.requires_stop("green", 2.0, 8.0)


def test_go_colours_clear_old_red_without_waiting():
    observer = TrafficLightObserver(0.05, 800, 600)
    observer._add_vote("red")
    observer._add_vote("red")
    assert observer.confirmed_state == "red"
    observer._add_vote("yellow")
    assert observer.confirmed_state == "yellow"
    observer._add_vote("green")
    assert observer.confirmed_state == "green"


def test_speed_sign_requires_two_frames_then_sets_reference():
    observer = SpeedSignObserver(0.05, 800, 600, maximum_speed_kmh=90)
    first = observer.update(1, sign_detection(1, "speed_limit_20"))
    second = observer.update(2, sign_detection(2, "speed_limit_20"))
    assert first["speed_limit_kmh"] is None
    assert second["speed_limit_kmh"] == 20.0
    assert second["changed"]


def test_speed_sign_fifty_and_end_limit_are_supported():
    observer = SpeedSignObserver(0.05, 800, 600, maximum_speed_kmh=90)
    observer.update(1, sign_detection(1, "speed_limit_50"))
    result = observer.update(2, sign_detection(2, "speed_limit_50"))
    assert result["speed_limit_mps"] == 50.0 / 3.6
    cleared = observer.update(3, sign_detection(3, "end_all_speed_and_passing_limits"))
    assert cleared["speed_limit_kmh"] is None
    assert cleared["changed"]


def test_straight_road_small_noise_does_not_create_wobble():
    controller = StanleyController(dt=0.05)
    outputs = []
    for tick in range(80):
        y_noise = 0.025 if tick % 2 == 0 else -0.025
        yaw_noise = 0.25 if tick % 2 == 0 else -0.25
        outputs.append(controller.run_step(straight_state(y_noise, yaw_noise)))

    settled = outputs[20:]
    assert max(abs(value) for value in settled) < 0.035
    changes = [abs(b - a) for a, b in zip(settled, settled[1:])]
    assert max(changes) < 0.012
    assert controller.last_info["straight_mode"]


def test_red_activation_at_ten_metres_can_stop_near_line_at_thirty_kmh():
    controller = LongitudinalController(dt=0.05)
    speed = 30.0 / 3.6
    distance = 30.0

    for _ in range(1200):
        lead = None
        if distance <= 10.0:
            lead = {
                "track_id": -100,
                "distance_m": max(0.21, distance),
                "relative_speed_mps": -speed,
                "lead_speed_mps": 0.0,
                "source": "traffic_light_red",
            }
        throttle, brake, info = controller.run_step(
            {"speed_mps": speed},
            lead,
            30.0 / 3.6,
        )
        acceleration = 2.8 * throttle - 7.5 * brake
        if speed > 0.02:
            acceleration -= 0.12
        speed = max(0.0, speed + acceleration * controller.dt)
        distance -= speed * controller.dt
        if info["hold_active"] and speed <= 0.02:
            break

    assert info["hold_active"]
    assert 0.15 <= distance <= 0.75

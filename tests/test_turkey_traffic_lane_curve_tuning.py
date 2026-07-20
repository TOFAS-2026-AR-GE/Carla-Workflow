from __future__ import annotations

import math
from types import SimpleNamespace

from carla_app.controller.vehicle.lead_vehicle import LeadVehicleTracker
from carla_app.controller.vehicle.longitudinal_controller import LongitudinalController
from carla_app.controller.vehicle.stanley_controller import StanleyController


def point(x, y, z=0.0):
    return SimpleNamespace(x=float(x), y=float(y), z=float(z))


def quarter_turn(radius=10.0):
    result = []
    for index in range(41):
        angle = -0.5 * math.pi + 0.5 * math.pi * index / 40.0
        result.append(point(radius * math.cos(angle), radius + radius * math.sin(angle)))
    return result


def state(path, speed=4.0):
    return {
        "reference_path": path,
        "location": point(0.0, 0.0),
        "yaw": 0.0,
        "speed_mps": speed,
        "lane_width": 3.5,
        "vehicle_half_width_m": 0.95,
    }


def simulate_red_stop(initial_speed_kmh, initial_distance_m):
    controller = LongitudinalController(dt=0.05)
    speed = initial_speed_kmh / 3.6
    distance = float(initial_distance_m)
    for _ in range(1600):
        throttle, brake, info = controller.run_step(
            {"speed_mps": speed},
            {
                "track_id": -100,
                "distance_m": distance,
                "relative_speed_mps": -speed,
                "lead_speed_mps": 0.0,
                "source": "traffic_light_red",
            },
            max(0.1, initial_speed_kmh / 3.6),
        )
        acceleration = 2.8 * throttle - 5.0 * brake
        if speed > 0.02:
            acceleration -= 0.12
        speed = max(0.0, speed + acceleration * controller.dt)
        distance -= speed * controller.dt
        if info["hold_active"] and speed <= 0.02:
            return distance
    raise AssertionError("Araç trafik ışığı hold durumuna ulaşmadı")



class FakeWaypoint:
    def __init__(self, road_id, lane_id, yaw=0.0):
        self.road_id = road_id
        self.lane_id = lane_id
        self.transform = SimpleNamespace(
            rotation=SimpleNamespace(yaw=float(yaw))
        )


class FakeMap:
    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        lane_id = 1 if abs(float(location.y)) < 1.0 else 2
        return FakeWaypoint(road_id=10, lane_id=lane_id, yaw=0.0)


def make_lane_tracker():
    tracker = LeadVehicleTracker.__new__(LeadVehicleTracker)
    tracker.carla_map = FakeMap()
    tracker.current_route_curvature = lambda state: 0.0
    return tracker


def test_same_map_lane_is_accepted():
    tracker = make_lane_tracker()
    route_position = {
        "lateral_m": 0.25,
        "projection_x": 10.0,
        "projection_y": 0.0,
    }
    assert tracker.is_same_route_lane(10.0, 0.25, route_position, state([]))


def test_adjacent_map_lane_is_rejected_even_if_projection_is_misleading():
    tracker = make_lane_tracker()
    route_position = {
        "lateral_m": 0.25,
        "projection_x": 10.0,
        "projection_y": 0.0,
    }
    assert not tracker.is_same_route_lane(10.0, 3.5, route_position, state([]))

def test_red_light_stop_is_tighter_than_previous_setting():
    distance = simulate_red_stop(30.0, 30.0)
    assert 0.15 <= distance <= 0.60


def test_early_stop_creeps_close_to_line():
    distance = simulate_red_stop(0.0, 5.8)
    assert 0.15 <= distance <= 0.60


def test_curve_steering_remains_smooth_after_fine_tuning():
    controller = StanleyController(dt=0.05)
    outputs = [controller.run_step(state(quarter_turn())) for _ in range(25)]
    changes = [b - a for a, b in zip(outputs, outputs[1:])]
    accelerations = [b - a for a, b in zip(changes, changes[1:])]
    assert max(abs(value) for value in changes) <= 0.050
    assert max(abs(value) for value in accelerations) <= 0.013
    assert abs(outputs[-1]) > 0.10

import math

import carla


def build_reference_path(start_waypoint, point_count=10, spacing=0.5):
    reference_path = []
    current_waypoint = start_waypoint

    for _ in range(point_count):
        next_waypoints = current_waypoint.next(spacing)

        if not next_waypoints:
            break

        current_waypoint = next_waypoints[0]
        reference_path.append(current_waypoint.transform.location)

    return reference_path


def read_vehicle_state(world, vehicle):
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()

    speed_mps = math.sqrt(
        velocity.x**2
        + velocity.y**2
        + velocity.z**2
    )

    waypoint = world.get_map().get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if waypoint is None:
        raise RuntimeError("Arac icin surus seridi bulunamadi.")

    reference_path = build_reference_path(waypoint)

    if reference_path:
        next_location = reference_path[0]
    else:
        next_location = None

    return {
        "location": transform.location,
        "yaw": transform.rotation.yaw,
        "speed_mps": speed_mps,
        "speed_kmh": speed_mps * 3.6,
        "next_location": next_location,
        "reference_path": reference_path,
        "road_id": waypoint.road_id,
        "lane_id": waypoint.lane_id,
        "lane_width": waypoint.lane_width,
        "is_junction": waypoint.is_junction,
    }


def serializable_vehicle_state(state, control):
    location = state["location"]

    return {
        "x": float(location.x),
        "y": float(location.y),
        "z": float(location.z),
        "yaw": float(state["yaw"]),
        "speed_mps": float(state["speed_mps"]),
        "speed_kmh": float(state["speed_kmh"]),
        "road_id": int(state["road_id"]),
        "lane_id": int(state["lane_id"]),
        "lane_width": float(state["lane_width"]),
        "is_junction": bool(state["is_junction"]),
        "control": {
            "throttle": float(control.throttle),
            "steer": float(control.steer),
            "brake": float(control.brake),
        },
    }
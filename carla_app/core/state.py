"""CARLA aracından kontrolcülerin kullanacağı sade durum sözlüğünü üretir."""

import math

import carla


def build_reference_path(
    start_waypoint,
    point_count=80,
    spacing=1.0,
):
    """Kalıcı rota yöneticisi yoksa kısa bir referans yol üretir."""

    reference_path = []
    current_waypoint = start_waypoint

    for _ in range(point_count):
        next_waypoints = current_waypoint.next(spacing)

        if not next_waypoints:
            break

        current_yaw = math.radians(current_waypoint.transform.rotation.yaw)

        def heading_difference(waypoint):
            candidate_yaw = math.radians(waypoint.transform.rotation.yaw)
            difference = candidate_yaw - current_yaw

            return abs(
                math.atan2(
                    math.sin(difference),
                    math.cos(difference),
                )
            )

        current_waypoint = min(
            next_waypoints,
            key=heading_difference,
        )

        reference_path.append(current_waypoint.transform.location)

    return reference_path


def read_vehicle_state(
    world,
    vehicle,
    route_manager=None,
):
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()

    speed_mps = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    if route_manager is not None:
        waypoint, reference_path = route_manager.update(transform.location)
    else:
        waypoint = world.get_map().get_waypoint(
            transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if waypoint is None:
            raise RuntimeError("Arac icin surus seridi bulunamadi.")

        reference_path = build_reference_path(waypoint)

    if waypoint is None:
        raise RuntimeError("Kalici rota uzerinde waypoint bulunamadi.")

    next_location = reference_path[0] if reference_path else None

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
        # Aracın yarı genişliği, şerit merkezinden güvenli uzaklığı
        # hesaplarken kullanılır.
        "vehicle_half_width_m": float(vehicle.bounding_box.extent.y),
        "is_junction": waypoint.is_junction,
    }


def serializable_vehicle_state(
    state,
    control,
):
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
        "vehicle_half_width_m": float(state.get("vehicle_half_width_m", 0.95)),
        "is_junction": bool(state["is_junction"]),
        "control": {
            "throttle": float(control.throttle),
            "steer": float(control.steer),
            "brake": float(control.brake),
        },
    }

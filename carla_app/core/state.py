import math

import carla


def build_reference_path(
    start_waypoint,
    point_count=80,
    spacing=1.0,
):
    """
    RouteManager kullanilmayan yerler icin
    geriye uyumlu basit yol uretici.
    """

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



def distance_to_next_junction(route_manager, vehicle_location, maximum_distance_m=40.0):
    """Kalıcı rota üzerinde sıradaki kavşağa kalan mesafeyi bulur."""
    waypoints = list(getattr(route_manager, "waypoints", []) or [])
    if not waypoints:
        return None

    search_count = min(25, len(waypoints))
    nearest_index = min(
        range(search_count),
        key=lambda index: math.hypot(
            vehicle_location.x - waypoints[index].transform.location.x,
            vehicle_location.y - waypoints[index].transform.location.y,
        ),
    )

    distance_m = math.hypot(
        vehicle_location.x - waypoints[nearest_index].transform.location.x,
        vehicle_location.y - waypoints[nearest_index].transform.location.y,
    )

    for index in range(nearest_index, len(waypoints)):
        waypoint = waypoints[index]
        if waypoint.is_junction:
            return float(distance_m)

        if index + 1 >= len(waypoints):
            break

        current = waypoint.transform.location
        following = waypoints[index + 1].transform.location
        distance_m += math.hypot(
            following.x - current.x,
            following.y - current.y,
        )
        if distance_m > float(maximum_distance_m):
            break

    return None

def read_vehicle_state(
    world,
    vehicle,
    route_manager=None,
):
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()

    speed_mps = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    junction_distance_m = None
    if route_manager is not None:
        waypoint, reference_path = route_manager.update(transform.location)
        junction_distance_m = distance_to_next_junction(
            route_manager,
            transform.location,
        )
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
        # Aracin yarim genisligi, serit merkezinden ne kadar
        # uzaklasabilecegimizi hesaplarken kullanilir.
        "vehicle_half_width_m": float(vehicle.bounding_box.extent.y),
        "is_junction": waypoint.is_junction,
        "junction_distance_m": junction_distance_m,
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
        "junction_distance_m": state.get("junction_distance_m"),
        "control": {
            "throttle": float(control.throttle),
            "steer": float(control.steer),
            "brake": float(control.brake),
        },
    }

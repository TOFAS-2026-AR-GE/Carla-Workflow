"""CARLA aracından kontrolcülerin kullanacağı sade durum sözlüğünü üretir."""

import math

import carla


def traffic_light_stop_distance(vehicle, traffic_light):
    """Etkileyen ışığın CARLA stop waypoint'ine boylamasına mesafesini ölçer.

    Yeni CARLA sürümlerindeki ``get_stop_waypoints`` doğrudan tetikleme
    kutusundan hesaplanan durma çizgilerini verir. Eski sürümlerde veya eksik
    aktör verisinde ``None`` dönerek kontrol katmanının güvenli mesafe
    kestirimine geçmesine izin verir.
    """
    if traffic_light is None:
        return None
    try:
        transform = vehicle.get_transform()
        stop_waypoints = traffic_light.get_stop_waypoints()
    except (AttributeError, RuntimeError):
        return None

    location = transform.location
    yaw = math.radians(float(transform.rotation.yaw))
    forward_x = math.cos(yaw)
    forward_y = math.sin(yaw)
    right_x = -forward_y
    right_y = forward_x
    candidates = []

    for waypoint in stop_waypoints or ():
        try:
            stop_location = waypoint.transform.location
            delta_x = float(stop_location.x) - float(location.x)
            delta_y = float(stop_location.y) - float(location.y)
            forward_distance = delta_x * forward_x + delta_y * forward_y
            lateral_distance = abs(delta_x * right_x + delta_y * right_y)
            lane_width = max(2.0, float(getattr(waypoint, "lane_width", 3.5)))
        except (AttributeError, TypeError, ValueError):
            continue

        # Aynı trafik ışığı farklı şeritlerin stop waypoint'lerini döndürebilir.
        # Yalnız ego yönünün önündeki ve ego şeridine yakın adayı kullanırız.
        if forward_distance >= 0.0 and lateral_distance <= lane_width:
            candidates.append(forward_distance)

    if not candidates:
        return None
    return max(0.25, min(candidates))


def read_simulator_traffic_light(vehicle):
    """CARLA'nın bu aracı etkileyen trafik ışığı durumunu sadeleştirir.

    Bu bilgi kırmızı ışığı uzaktan bulmak için kullanılmaz. Araç çizgide
    durduğunda lamba kameranın üstünden çıkarsa yeşili doğrulayan yedek
    bilgidir. ``is_at_traffic_light`` kontrolü önemlidir; CARLA, aracı
    etkileyen ışık yokken de durum değerini Green olarak döndürebilir.
    """
    try:
        affected = bool(vehicle.is_at_traffic_light())
    except (AttributeError, RuntimeError):
        return {"available": False, "affected": False, "color": None}

    if not affected:
        return {"available": True, "affected": False, "color": None}

    try:
        raw_state = vehicle.get_traffic_light_state()
    except (AttributeError, RuntimeError):
        return {"available": False, "affected": True, "color": None}

    state_name = getattr(raw_state, "name", str(raw_state).rsplit(".", 1)[-1])
    color = str(state_name).strip().lower()
    if color == "yellow":
        color = "orange"
    if color not in {"red", "orange", "green", "off", "unknown"}:
        color = "unknown"

    traffic_light = None
    try:
        traffic_light = vehicle.get_traffic_light()
    except (AttributeError, RuntimeError):
        pass

    actor_id = getattr(traffic_light, "id", None)
    return {
        "available": True,
        "affected": True,
        "color": color,
        "estimated_distance_m": traffic_light_stop_distance(
            vehicle,
            traffic_light,
        ),
        "actor_id": int(actor_id) if actor_id is not None else None,
    }


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
        "simulator_traffic_light": read_simulator_traffic_light(vehicle),
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

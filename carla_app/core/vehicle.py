import random

import carla


def spawn_ego_vehicle(world, blueprint_name):
    blueprint = world.get_blueprint_library().find(blueprint_name)
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for point in spawn_points:
        vehicle = world.try_spawn_actor(blueprint, point)
        if vehicle is not None:
            print(f"[OK] Ego arac olusturuldu: {vehicle.type_id}")
            return vehicle

    raise RuntimeError("Ego arac icin bos spawn noktasi bulunamadi.")


def apply_vehicle_lights(vehicle, control, control_info):
    """Sinyal ve fren lambalarını mevcut diğer ışıkları bozmadan uygular."""
    try:
        current_mask = int(vehicle.get_light_state())
        left_bit = int(carla.VehicleLightState.LeftBlinker)
        right_bit = int(carla.VehicleLightState.RightBlinker)
        brake_bit = int(carla.VehicleLightState.Brake)

        light_mask = current_mask & ~(left_bit | right_bit | brake_bit)
        direction = (
            control_info.get("turn_signal", {}).get("direction", "off")
            if isinstance(control_info, dict)
            else "off"
        )

        if direction == "left":
            light_mask |= left_bit
        elif direction == "right":
            light_mask |= right_bit

        if float(getattr(control, "brake", 0.0)) > 0.03:
            light_mask |= brake_bit

        vehicle.set_light_state(carla.VehicleLightState(light_mask))
        return True
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False

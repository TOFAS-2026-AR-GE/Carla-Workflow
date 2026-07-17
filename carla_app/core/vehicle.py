import random


def spawn_ego_vehicle(world, blueprint_name, role_name="ego_vehicle"):
    blueprint = world.get_blueprint_library().find(blueprint_name)
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", role_name)

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for point in spawn_points:
        vehicle = world.try_spawn_actor(blueprint, point)
        if vehicle is not None:
            print(f"[OK] Ego arac olusturuldu: {vehicle.type_id}")
            return vehicle

    raise RuntimeError("Ego arac icin bos spawn noktasi bulunamadi.")



def load_map(client, world, map_name):
    if map_name is None:
        print("[INFO] Varsayilan CARLA haritasi kullaniliyor.")
        return world

    if str(map_name).strip() == "":
        print("[INFO] Varsayilan CARLA haritasi kullaniliyor.")
        return world

    if str(map_name).strip().lower() == "default":
        print("[INFO] Varsayilan CARLA haritasi kullaniliyor.")
        return world

    current_map = world.get_map().name

    if current_map.endswith(map_name):
        return world

    print(f"[INFO] Harita yukleniyor: {map_name}")
    client.load_world(map_name)
    return client.get_world()
  

def apply_world_settings(world, scenario):
    simulation = scenario.get("simulation", {})

    settings = world.get_settings()

    if "synchronous_mode" in simulation:
        settings.synchronous_mode = simulation["synchronous_mode"]

    if "fixed_delta_seconds" in simulation:
        settings.fixed_delta_seconds = simulation["fixed_delta_seconds"]

    world.apply_settings(settings)
    print("[OK] Senaryo dunya ayarlari uygulandi.")

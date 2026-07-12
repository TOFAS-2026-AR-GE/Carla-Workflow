import random


NPC_VEHICLES = []

def spawn_traffic(client, world, scenario):
    traffic = scenario.get("traffic", {})
    npc_count = int(traffic.get("npc_count", 0))

    if npc_count <= 0:
        print("[INFO] Trafik araci istenmedi.")
        return

    traffic_manager = client.get_trafficmanager(
        int(traffic.get("tm_port", 8000))
    )

    traffic_manager.set_synchronous_mode(
        world.get_settings().synchronous_mode
    )
    traffic_manager.set_global_distance_to_leading_vehicle(
        float(traffic.get("safe_distance_m", 3.0))
    )

    vehicle_blueprints = world.get_blueprint_library().filter(
        "vehicle.*"
    )
    spawn_points = world.get_map().get_spawn_points()

    random.shuffle(spawn_points)

    for spawn_point in spawn_points:
        if len(NPC_VEHICLES) >= npc_count:
            break

        blueprint = random.choice(vehicle_blueprints)
        vehicle = world.try_spawn_actor(
            blueprint,
            spawn_point,
        )

        if vehicle is None:
            continue

        vehicle.set_autopilot(
            True,
            traffic_manager.get_port(),
        )

        traffic_manager.vehicle_percentage_speed_difference(
            vehicle,
            float(traffic.get("speed_difference_percent", 0.0)),
        )

        NPC_VEHICLES.append(vehicle)

    print(f"[OK] {len(NPC_VEHICLES)} trafik araci olusturuldu.")


def cleanup_traffic():
    for vehicle in NPC_VEHICLES:
        try:
            vehicle.destroy()
        except Exception:
            pass

    NPC_VEHICLES.clear()


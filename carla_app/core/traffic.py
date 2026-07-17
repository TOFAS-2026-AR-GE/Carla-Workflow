"""Senaryoda istenen trafik araçlarını oluşturur ve kapanışta temizler."""

import random


class Traffic:
    """CARLA Traffic Manager ile sürülen diğer araçları yönetir."""
    def __init__(self, client, world, scenario):
        self.client = client
        self.world = world
        self.scenario = scenario
        self.actors = []
        self.manager = None

    def spawn(self):
        if self.scenario.npc_count <= 0:
            print("[INFO] Trafik kapali.")
            return

        self.manager = self.client.get_trafficmanager(
            self.scenario.traffic_manager_port
        )
        self.manager.set_synchronous_mode(self.world.get_settings().synchronous_mode)
        self.manager.set_global_distance_to_leading_vehicle(
            self.scenario.safe_distance_m
        )

        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        for point in spawn_points:
            if len(self.actors) >= self.scenario.npc_count:
                break
            actor = self.world.try_spawn_actor(random.choice(blueprints), point)
            if actor is None:
                continue
            actor.set_autopilot(True, self.manager.get_port())
            self.manager.vehicle_percentage_speed_difference(
                actor, self.scenario.speed_difference_percent
            )
            self.actors.append(actor)

        print(f"[OK] {len(self.actors)} trafik araci olusturuldu.")

    def stop(self):
        for actor in self.actors:
            try:
                actor.destroy()
            except Exception:
                pass
        self.actors.clear()

        if self.manager is not None:
            try:
                self.manager.set_synchronous_mode(False)
            except Exception:
                pass
